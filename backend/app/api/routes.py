from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    BatteryFeatureVector,
    BatteryPredictionResponse,
    ModelInfoResponse,
    IMUPredictionRequest,
    IMUPredictionResponse,
)

from app.services.battery_service import predict_battery_soh
from app.services.imu_service import predict_imu


router = APIRouter()


# ============================================================
# SYSTEM
# ============================================================

@router.get("/health", tags=["system"])
def health():
    """
    Basic health check for the DriveGuard-X AI backend.
    """

    from app.services.model_loader import get_battery_model_info

    info = get_battery_model_info()

    return {
        "status": "ok",
        "battery_model_loaded": info["loaded"],
    }


# ============================================================
# BATTERY AI
# ============================================================

@router.get(
    "/battery/model-info",
    response_model=ModelInfoResponse,
    tags=["battery"],
)
def battery_model_info():
    """
    Return information about the currently loaded Battery AI model.
    """

    from app.services.model_loader import get_battery_model_info

    return get_battery_model_info()


@router.post(
    "/battery/predict",
    response_model=BatteryPredictionResponse,
    tags=["battery"],
)
def battery_predict(features: BatteryFeatureVector):
    """
    Predict battery SOH from the 21 engineered battery features.
    """

    try:
        return predict_battery_soh(features)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


# ============================================================
# IMU AI
# ============================================================

@router.post(
    "/imu/predict",
    response_model=IMUPredictionResponse,
    tags=["imu"],
)
def imu_predict(request: IMUPredictionRequest):
    """
    Predict driving behavior from exactly 4 raw IMU samples.

    The raw IMU samples are converted into the 42 statistical
    features expected by the trained XGBoost model.
    """

    try:
        return predict_imu(request)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc