from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.model_loader import load_models


app = FastAPI(
    title="DriveGuard-X AI Backend",
    version="0.1.0",
    description=(
        "Inference backend for the DriveGuard-X research prototype. "
        "The backend currently exposes the Battery AI and IMU AI models. "
        "Road, Fusion, Attribution and Protection modules will be added later."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(
    router,
    prefix="/api/v1",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event() -> None:
    """
    Load all currently available AI models when the
    FastAPI application starts.
    """

    load_models()


# ============================================================
# ROOT
# ============================================================

@app.get("/", tags=["system"])
def root():
    return {
        "project": "DriveGuard-X",
        "service": "AI Backend",
        "status": "running",
        "version": app.version,
    }