from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np


def _ensure_exists(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    return p


@lru_cache(maxsize=64)
def load_joblib(path: str):
    p = _ensure_exists(Path(path))
    return joblib.load(p)


@lru_cache(maxsize=256)
def load_json(path: str) -> dict:
    p = _ensure_exists(Path(path))
    return json.loads(p.read_text())


@lru_cache(maxsize=32)
def load_npz(path: str):
    p = _ensure_exists(Path(path))
    return np.load(p, allow_pickle=True)


def load_parquet(path: str):
    import pandas as pd
    p = _ensure_exists(Path(path))
    return pd.read_parquet(p)


@lru_cache(maxsize=16)
def load_torch_model(path: str, device: str = "cpu"):
    """
    Returns: (mode, obj)
      - ("torchscript", torch.jit.ScriptModule)
      - ("state_dict", dict or checkpoint)
      - ("raw", anything else)
    """
    import torch
    p = _ensure_exists(Path(path))

    # Try TorchScript first
    try:
        m = torch.jit.load(str(p), map_location=device)
        m.eval()
        return ("torchscript", m)
    except Exception:
        pass

    # Try torch.load
    obj = torch.load(str(p), map_location=device)

    # Heuristics to identify state_dict/checkpoints
    if isinstance(obj, dict):
        # common checkpoint keys
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return ("state_dict", obj)
        # could be plain state_dict (param_name -> tensor)
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return ("state_dict", obj)

    return ("raw", obj)
