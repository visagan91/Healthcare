import os
import requests

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def list_models(timeout: int = 10):
    r = requests.get(f"{BASE_URL}/models", timeout=timeout)
    r.raise_for_status()
    return r.json()


def predict(
    model_id: str,
    payload: dict,
    return_proba: bool = False,
    top_k: int = 5,
    timeout: int = 60,
):
    req = {
        "model_id": model_id,
        "payload": payload,
        "return_proba": return_proba,
        "top_k": top_k,
    }
    r = requests.post(f"{BASE_URL}/predict", json=req, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Backend error {r.status_code}: {r.text}")
    return r.json()