import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str
    task: str
    model_path: Path
    meta_paths: dict[str, Path]
    input_schema: dict[str, Any]


class ModelRegistry:
    def __init__(self, manifest_path: Path, artifacts_dir: Path | None = None):
        self.manifest_path = Path(manifest_path).resolve()

        # Project root:
        # if manifest is HealthCare/artifacts/manifest.json -> project root is HealthCare
        if self.manifest_path.parent.name == "artifacts":
            self.project_root = self.manifest_path.parent.parent
        else:
            self.project_root = self.manifest_path.parent

        self.artifacts_dir = (
            Path(artifacts_dir).resolve()
            if artifacts_dir is not None
            else (self.project_root / "artifacts").resolve()
        )

        self._index: dict[str, ModelSpec] = {}

    def _resolve_path(self, p: str | Path) -> Path:
        pp = Path(p)

        if pp.is_absolute():
            return pp.resolve()

        # If manifest paths already start with "artifacts/..."
        # resolve from project root, not artifacts/artifacts
        if len(pp.parts) > 0 and pp.parts[0] == "artifacts":
            return (self.project_root / pp).resolve()

        # Otherwise resolve relative to artifacts dir
        return (self.artifacts_dir / pp).resolve()

    def load(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        idx: dict[str, ModelSpec] = {}

        for m in data.get("models", []):
            model_id = m["model_id"]

            if "model_path" not in m:
                raise KeyError(f"Missing 'model_path' for model_id={model_id}")

            idx[model_id] = ModelSpec(
                model_id=model_id,
                kind=m["kind"],
                task=m["task"],
                model_path=self._resolve_path(m["model_path"]),
                meta_paths={
                    k: self._resolve_path(v)
                    for k, v in m.get("meta_paths", {}).items()
                },
                input_schema=m.get("input_schema", {}),
            )

        self._index = idx

    def list(self) -> list[ModelSpec]:
        return list(self._index.values())

    def get(self, model_id: str) -> ModelSpec:
        if model_id not in self._index:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._index[model_id]