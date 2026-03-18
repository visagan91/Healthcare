from fastapi import APIRouter
from src.core.settings import settings
from src.core.registry import ModelRegistry

router = APIRouter()
reg = ModelRegistry(settings.manifest_path)
reg.load()

@router.get("")
def list_models():
    return {
        "models": [
            {
                "model_id": m.model_id,
                "kind": m.kind,
                "task": m.task,
                "model_path": str(m.model_path),
                "meta_paths": {k: str(v) for k, v in m.meta_paths.items()},
                "input_schema": m.input_schema
            }
            for m in reg.list()
        ]
    }
