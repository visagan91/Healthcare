from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

ART_DIR = Path("artifacts/chatbot")
vectorizer = joblib.load(ART_DIR / "tfidf.joblib")
doc_matrix = sparse.load_npz(ART_DIR / "doc_matrix.npz")
doc_store = pd.read_parquet(ART_DIR / "doc_store.parquet")

cfg = json.load(open(ART_DIR / "chatbot_config.json"))
TOP_K = int(cfg.get("top_k", 3))

class ChatbotRequest(BaseModel):
    question: str
    top_k: int | None = None

class ChatbotResponse(BaseModel):
    answer: str
    retrieved: list

@router.post("/ask", response_model=ChatbotResponse)
def ask(req: ChatbotRequest):
    q = (req.question or "").strip()
    if not q:
        return {"answer": "Please enter a question.", "retrieved": []}

    k = int(req.top_k or TOP_K)
    q_vec = vectorizer.transform([q])

    sims = cosine_similarity(q_vec, doc_matrix).ravel()
    top_idx = np.argsort(-sims)[:k]

    retrieved = []
    for i in top_idx:
        row = doc_store.iloc[int(i)]
        retrieved.append({
            "score": float(sims[i]),
            "encounter_id": str(row.get("encounter_id", "")),
            "patient_id": str(row.get("patient_id", "")),
            "reference_answer": str(row.get("chatbot_reference_answer", ""))[:500],
        })

    # Simple “answer”: take best doc’s reference answer if present, else summarize doc text (basic)
    best = doc_store.iloc[int(top_idx[0])]
    answer = str(best.get("chatbot_reference_answer", "")).strip()
    if not answer:
        answer = "I found a relevant clinical context, but no reference answer was provided."

    return {"answer": answer, "retrieved": retrieved}
