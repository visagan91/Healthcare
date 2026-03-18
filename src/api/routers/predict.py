from fastapi import APIRouter, HTTPException
from src.core.schemas import PredictRequest, PredictResponse
from src.core.settings import settings
from src.core.registry import ModelRegistry
from src.core.predictors import predict as run_predict

router = APIRouter()
reg = ModelRegistry(settings.manifest_path)
reg.load()

@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model_id not in {m.model_id for m in reg.list()}:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {req.model_id}")

    spec = reg.get(req.model_id)
    try:
        output, meta = run_predict(spec, req.payload, return_proba=bool(req.return_proba), top_k=int(req.top_k or 5))
        return PredictResponse(model_id=req.model_id, output=output, meta=meta)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
