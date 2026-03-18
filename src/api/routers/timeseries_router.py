from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn

router = APIRouter(prefix="/timeseries", tags=["timeseries"])

# -----------------------------
# Config + paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../HealthCare
ART_DIR = PROJECT_ROOT / "artifacts" / "timeseries"

with open(ART_DIR / "ts_config.json", "r") as f:
    CFG = json.load(f)

T = int(CFG["input_shape"]["T"])
F = int(CFG["input_shape"]["F"])

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# You MUST paste your exact model classes here
# (same definitions as notebook)
# -----------------------------
class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # [B,T,H]
        h_last = out[:, -1, :]        # [B,H]
        return self.fc(h_last).squeeze(1)  # [B]

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)         # [B,T,H]
        h_last = out[:, -1, :]
        return self.fc(h_last).squeeze(1)


# -----------------------------
# Load models once
# NOTE: adjust hidden_dim/num_layers to match your notebook
# Best source: artifacts/timeseries/arch.json
# -----------------------------
def load_models():
    # TODO: set these to match your training (check arch.json)
    hidden_dim = 64
    num_layers = 1

    rnn = RNNRegressor(input_dim=F, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    lstm = LSTMRegressor(input_dim=F, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)

    rnn.load_state_dict(torch.load(ART_DIR / "los_rnn.pt", map_location=DEVICE))
    lstm.load_state_dict(torch.load(ART_DIR / "los_lstm.pt", map_location=DEVICE))

    rnn.eval()
    lstm.eval()
    return {"rnn": rnn, "lstm": lstm}

MODELS = load_models()


# -----------------------------
# Parsing helpers (same logic as notebook)
# -----------------------------
def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def parse_ts_any(s):
    if s is None:
        return None
    if isinstance(s, str):
        s = s.strip()
        if s == "":
            return None
        obj = json.loads(s)
    else:
        obj = s

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = np.array(v, dtype=np.float32).reshape(-1)
        return out

    if isinstance(obj, list):
        if len(obj) == 0:
            return None

        if isinstance(obj[0], dict):
            keys = sorted(obj[0].keys())
            numeric_keys = []
            for k in keys:
                sample_vals = [t.get(k, None) for t in obj[:5]]
                conv = [_to_float_or_nan(v) for v in sample_vals]
                if np.isfinite(conv).any():
                    numeric_keys.append(k)
            if len(numeric_keys) == 0:
                return None
            rows = [[_to_float_or_nan(t.get(k, np.nan)) for k in numeric_keys] for t in obj]
            X = np.array(rows, dtype=np.float32)
            return np.nan_to_num(X, nan=0.0)

        if isinstance(obj[0], (list, tuple)):
            return np.array(obj, dtype=np.float32)

        if isinstance(obj[0], (int, float, str)):
            vals = [_to_float_or_nan(v) for v in obj]
            X = np.array(vals, dtype=np.float32).reshape(-1, 1)
            return np.nan_to_num(X, nan=0.0)

    return None

def merge_ts(vitals_payload, labs_payload=None):
    ts = parse_ts_any(vitals_payload)
    if ts is None:
        return None

    # Dict of series -> stack keys sorted (same as your notebook style)
    if isinstance(ts, dict):
        keys = sorted(ts.keys())
        arrays = [ts[k] for k in keys]
        tmin = min(len(a) for a in arrays)
        if tmin <= 0:
            return None
        X = np.stack([a[:tmin] for a in arrays], axis=1).astype(np.float32)
    else:
        X = np.asarray(ts, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # pad/truncate to T
    if X.shape[0] >= T:
        Xout = X[-T:, :]
    else:
        pad = np.zeros((T - X.shape[0], X.shape[1]), dtype=np.float32)
        Xout = np.vstack([pad, X])

    # enforce F columns: pad or truncate columns
    if Xout.shape[1] < F:
        Xout = np.hstack([Xout, np.zeros((T, F - Xout.shape[1]), dtype=np.float32)])
    elif Xout.shape[1] > F:
        Xout = Xout[:, :F]

    return Xout.astype(np.float32)


# -----------------------------
# Request/response models
# -----------------------------
class TimeSeriesRequest(BaseModel):
    vitals_ts_json: str | dict | list
    labs_ts_json: str | dict | list | None = None

@router.post("/predict_los")
def predict_los(req: TimeSeriesRequest, model: str = "lstm"):
    model = model.lower().strip()
    if model not in MODELS:
        return {"error": f"model must be one of {list(MODELS.keys())}"}

    X = merge_ts(req.vitals_ts_json, req.labs_ts_json)
    if X is None:
        return {"error": "Could not parse time-series input"}

    xb = torch.tensor(X[None, :, :], dtype=torch.float32).to(DEVICE)  # [1,T,F]
    with torch.no_grad():
        pred = MODELS[model](xb).detach().cpu().numpy().reshape(-1)[0]

    return {
        "model": model,
        "pred_los_days": float(pred),
        "T": T,
        "F": F
    }
