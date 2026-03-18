import base64
import traceback
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.core.registry import ModelRegistry
from src.core.predictors import predict

router = APIRouter(tags=["Xray CNN"])


def get_registry() -> ModelRegistry:
    project_root = Path(__file__).resolve().parents[3]
    manifest_path = project_root / "artifacts" / "manifest.json"
    registry = ModelRegistry(manifest_path)
    registry.load()
    return registry


@router.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        image_b64 = base64.b64encode(contents).decode("utf-8")

        registry = get_registry()
        spec = registry.get("xray_multilabel_resnet18_v1")

        output, meta = predict(
            spec=spec,
            payload={"image_b64": image_b64},
            return_proba=False,
            top_k=5,
        )

        return {
            "model_id": "xray_multilabel_resnet18_v1",
            "output": output,
            "meta": meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))