from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# BATTERY AI
# ============================================================

class BatteryFeatureVector(BaseModel):
    """
    Exact 21-feature input expected by the current final
    DriveGuard-X Battery AI model.

    Important:
    This is a cycle-level model. These are engineered
    discharge-cycle features, not instantaneous ESP32 readings.
    """

    model_config = ConfigDict(extra="forbid")

    voltage_initial: float
    voltage_final: float
    voltage_min: float
    voltage_max: float
    voltage_mean: float
    voltage_std: float
    voltage_drop: float

    current_mean: float
    current_min: float
    current_max: float
    current_std: float

    temperature_initial: float
    temperature_final: float
    temperature_max: float
    temperature_mean: float
    temperature_rise: float

    load_current_abs_mean: float
    load_current_abs_max: float
    load_voltage_mean: float
    load_voltage_min: float

    discharge_duration: float


class BatteryPredictionResponse(BaseModel):
    model: str

    soh: float = Field(
        description="Predicted State of Health on the model's 0-1 scale."
    )

    soh_percent: float

    input_feature_count: int


class ModelInfoResponse(BaseModel):
    loaded: bool

    model_type: str | None = None

    n_estimators: int | None = None

    feature_count: int | None = None

    features: list[str] = Field(
        default_factory=list
    )

    development_batteries: list[str] = Field(
        default_factory=list
    )

    untouched_test_batteries: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# IMU AI
# ============================================================

class IMUSample(BaseModel):
    """
    One raw IMU sample.

    The trained DriveGuard-X IMU model expects six raw
    signals:

        Accelerometer:
            AccX, AccY, AccZ

        Gyroscope:
            GyroX, GyroY, GyroZ
    """

    model_config = ConfigDict(extra="forbid")

    AccX: float
    AccY: float
    AccZ: float

    GyroX: float
    GyroY: float
    GyroZ: float


class IMUPredictionRequest(BaseModel):
    """
    Request body for IMU inference.

    The trained model operates on exactly four consecutive
    IMU samples.
    """

    model_config = ConfigDict(extra="forbid")

    samples: list[IMUSample] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Exactly 4 consecutive IMU samples."
    )


class IMUPredictionResponse(BaseModel):
    """
    Response returned by the IMU inference endpoint.
    """

    model: str

    predicted_class: int

    feature_count: int

    window_size: int


# ============================================================
# ROAD AI (Audio-Only Model 3)
# ============================================================

class RoadPredictRequest(BaseModel):
    """
    Request body for Road Audio inference.
    """
    model_config = ConfigDict(extra="forbid")

    samples: list[float] = Field(
        ...,
        description="1D array of audio samples captured via INMP441 I2S."
    )


class RoadPredictionResponse(BaseModel):
    """
    Response returned by the Road Audio inference endpoint.
    """
    road_class: str
    confidence: float
    probabilities: dict[str, float]


    # Add to backend/app/api/schemas.py

class CombinedInferenceRequest(BaseModel):
    """
    Combined request payload accepting inputs for all 3 sensing modalities:
    1. Battery discharge cycle features (or state indicators)
    2. Exactly 4 raw IMU samples (driving behavior)
    3. Audio samples from INMP441 (road condition)
    """
    model_config = ConfigDict(extra="forbid")
    
    battery_features: BatteryFeatureVector
    imu_samples: list[IMUSample] = Field(..., min_length=4, max_length=4)
    audio_samples: list[float] = Field(..., description="Audio window samples at 16kHz")

class AttributionResponse(BaseModel):
    battery_state: dict[str, Any]
    driving_behavior: dict[str, Any]
    road_condition: dict[str, Any]
    attribution: dict[str, Any]
    protection: dict[str, Any]


# ============================================================
# LIVE TELEMETRY
# ============================================================

class LiveBatteryResponse(BaseModel):
    timestamp: datetime
    features: BatteryFeatureVector
    soh_percent: float
    stressed: bool


class LiveIMUResponse(BaseModel):
    timestamp: datetime
    samples: list[IMUSample]
    predicted_class: int
    driving_class: str


class LiveRoadResponse(BaseModel):
    timestamp: datetime
    sample_count: int
    road_class: str
    confidence: float
    probabilities: dict[str, float]


class LiveSnapshotResponse(BaseModel):
    battery: LiveBatteryResponse | None = None
    imu: LiveIMUResponse | None = None
    road: LiveRoadResponse | None = None
