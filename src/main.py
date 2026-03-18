from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health, models, predict, xray_router


def create_app() -> FastAPI:
    app = FastAPI(title="HealthCare Model API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router, prefix="/models")
    app.include_router(predict.router, prefix="/predict")
    app.include_router(xray_router.router, prefix="/xray")

    return app


app = create_app()