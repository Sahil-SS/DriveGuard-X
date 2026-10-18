import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.api.schemas import (
    BatteryFeatureVector,
    BatteryPredictionResponse,
    ModelInfoResponse,
    IMUPredictionRequest,
    IMUPredictionResponse,
    RoadPredictRequest,
    RoadPredictionResponse,
    LiveBatteryResponse,
    LiveIMUResponse,
    LiveRoadResponse,
    LiveSnapshotResponse,
)

from app.services.battery_service import predict_battery_soh
from app.services.imu_service import predict_imu
from app.services.road_service import road_service
from app.api.schemas import CombinedInferenceRequest, AttributionResponse
from app.services.attribution_service import attribution_engine
from app.services.live_telemetry import live_telemetry


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
        "road_model_loaded": road_service.loaded,
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


# ============================================================
# ROAD AI (Audio-Only Model 3)
# ============================================================

@router.get(
    "/road/model-info",
    tags=["road"],
)
def road_model_info():
    """
    Return metadata and configuration of the loaded Audio Road model.
    """
    return road_service.get_model_info()


@router.post(
    "/road/predict",
    response_model=RoadPredictionResponse,
    tags=["road"],
)
def road_predict(request: RoadPredictRequest):
    """
    Classify road condition from audio samples captured via INMP441.
    """
    try:
        return road_service.predict(request.samples)

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
# MULTIMODAL CAUSE ATTRIBUTION
# ============================================================
@router.post(
    "/inference",
    response_model=AttributionResponse,
    tags=["Multimodal Fusion & Attribution"],
)
def combined_inference(request: CombinedInferenceRequest):
    """
    Unified DriveGuard-X endpoint:
    Inputs from Battery sensors, MPU6050 IMU, and INMP441 microphone ->
    Multimodal Fusion -> Cause Attribution -> Adaptive Protection Decision.
    """
    try:
        return attribution_engine.process_inference(request)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Attribution engine error: {str(exc)}"
        ) from exc


# ============================================================
# LIVE TELEMETRY FOR THE INSTRUMENT CLUSTER
# ============================================================

@router.post(
    "/live/battery",
    response_model=LiveBatteryResponse,
    tags=["live telemetry"],
)
def live_battery(features: BatteryFeatureVector):
    """Process and cache the latest battery telemetry/features."""
    try:
        prediction = predict_battery_soh(features)
        reading = {
            "timestamp": datetime.now(timezone.utc),
            "features": features.model_dump(),
            "soh_percent": prediction["soh_percent"],
            "stressed": (
                prediction["soh_percent"] < 80.0
                or features.voltage_drop > 0.8
                or features.load_current_abs_max > 2.0
            ),
        }
        live_telemetry.update("battery", reading)
        return reading
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/live/imu",
    response_model=LiveIMUResponse,
    tags=["live telemetry"],
)
def live_imu(request: IMUPredictionRequest):
    """Process and cache the latest four-sample IMU window."""
    try:
        prediction = predict_imu(request)
        class_names = {
            1: "normal",
            2: "aggressive_acceleration",
            3: "harsh_braking",
            4: "sharp_cornering",
        }
        reading = {
            "timestamp": datetime.now(timezone.utc),
            "samples": [sample.model_dump() for sample in request.samples],
            "predicted_class": prediction["predicted_class"],
            "driving_class": class_names.get(
                prediction["predicted_class"], "unknown"
            ),
        }
        live_telemetry.update("imu", reading)
        return reading
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/live/road",
    response_model=LiveRoadResponse,
    tags=["live telemetry"],
)
def live_road(request: RoadPredictRequest):
    """Process and cache the latest INMP441 audio window."""
    try:
        prediction = road_service.predict(request.samples)
        reading = {
            "timestamp": datetime.now(timezone.utc),
            "sample_count": len(request.samples),
            **prediction,
        }
        live_telemetry.update("road", reading)
        return reading
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/live/snapshot",
    response_model=LiveSnapshotResponse,
    tags=["live telemetry"],
)
def live_snapshot():
    """Return the latest processed battery, IMU, and road readings."""
    return live_telemetry.snapshot()


@router.websocket("/live/ws")
async def live_websocket(websocket: WebSocket):
    """Stream the latest telemetry snapshot once per second to a dashboard."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(live_telemetry.snapshot())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
