from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    artifacts_dir: Path = Path("artifacts")
    manifest_path: Path = Path("artifacts/manifest.json")
    device: str = "cpu"  # later: "cuda" if available

settings = Settings()
