from pydantic import BaseModel
from typing import Any, Optional

class PredictRequest(BaseModel):
    model_id: str
    payload: dict[str, Any]
    return_proba: Optional[bool] = False
    top_k: Optional[int] = 5  # for retrieval/chatbot

class PredictResponse(BaseModel):
    model_id: str
    output: Any
    meta: dict[str, Any] = {}
