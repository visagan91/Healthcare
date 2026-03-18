from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
import json
import joblib

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# ---- paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # HealthCare/
ART_DIR = PROJECT_ROOT / "artifacts" / "sentiment"

PIPE_PATH = ART_DIR / "sentiment_tfidf_logreg.joblib"
CFG_PATH = ART_DIR / "sentiment_config.json"
LABEL_MAP_PATH = ART_DIR / "sentiment_label_map.json"

# ---- load artifacts once at import ----
assert PIPE_PATH.exists(), f"Missing {PIPE_PATH}"
pipe = joblib.load(PIPE_PATH)

cfg = {"text_col": "patient_feedback_text", "label_col": "sentiment_label"}
if CFG_PATH.exists():
    cfg = json.loads(CFG_PATH.read_text())

label_map = None
if LABEL_MAP_PATH.exists():
    label_map = json.loads(LABEL_MAP_PATH.read_text())


class SentimentRequest(BaseModel):
    text: str


@router.post("/predict")
def predict_sentiment(req: SentimentRequest):
    text = (req.text or "").strip()
    if not text:
        return {"error": "text is empty"}

    pred = pipe.predict([text])[0]

    out = {
        "sentiment": str(pred),
        "model": "tfidf_logreg"
    }

    # Optional probabilities (if available)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba([text])[0]
        classes = list(map(str, pipe.classes_))
        out["proba"] = {classes[i]: float(proba[i]) for i in range(len(classes))}

    # Optional label map (not required, but nice to expose)
    if label_map is not None:
        out["label_map"] = label_map

    return out
