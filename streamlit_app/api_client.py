import os
import requests

BASE_URL = os.environ.get("API_BASE", "http://127.0.0.1:8000")

def list_models(timeout=10):
    r = requests.get(f"{BASE_URL}/models", timeout=timeout)
    r.raise_for_status()
    return r.json()

def predict(model_id: str, payload: dict, return_proba: bool = False, top_k: int = 5, timeout=60):
    req = {"model_id": model_id, "payload": payload, "return_proba": return_proba, "top_k": top_k}
    r = requests.post(f"{BASE_URL}/predict", json=req, timeout=timeout)
    if r.status_code != 200:
        # Show backend error message instead of only "500"
        raise RuntimeError(f"Backend error {r.status_code}: {r.text}")
    return r.json()