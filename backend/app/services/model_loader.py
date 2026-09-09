import json
from functools import lru_cache
from pathlib import Path

import joblib
from xgboost import XGBClassifier

from app.core.config import settings


# ============================================================
# BATTERY MODEL
# ============================================================

@lru_cache(maxsize=1)
def get_battery_model():
    path = Path(settings.battery_model_path)

    if not path.exists():
        raise RuntimeError(
            f"Battery model not found at: {path}. "
            "Copy the Battery AI artifact into backend/models/battery/."
        )

    return joblib.load(path)


@lru_cache(maxsize=1)
def get_battery_metadata() -> dict:
    path = Path(settings.battery_metadata_path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def get_battery_schema() -> list[str]:
    path = Path(settings.battery_schema_path)

    if not path.exists():
        metadata = get_battery_metadata()
        return metadata.get("features", [])

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data.get("features", [])

    return data


# ============================================================
# IMU MODEL
# ============================================================

IMU_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "imu"
    / "driveguard_xgb_final.json"
)

IMU_METADATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "imu"
    / "model_metadata.json"
)

IMU_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "imu"
    / "imu_feature_list.json"
)


@lru_cache(maxsize=1)
def get_imu_model():
    """
    Load the native XGBoost JSON model.

    We intentionally use XGBClassifier.load_model()
    instead of joblib.load() because the joblib artifact
    is not compatible with the current XGBoost runtime.
    """

    if not IMU_MODEL_PATH.exists():
        raise RuntimeError(
            f"IMU model not found at: {IMU_MODEL_PATH}. "
            "Copy driveguard_xgb_final.json into "
            "backend/models/imu/."
        )

    model = XGBClassifier()
    model.load_model(str(IMU_MODEL_PATH))

    return model


@lru_cache(maxsize=1)
def get_imu_metadata() -> dict:
    if not IMU_METADATA_PATH.exists():
        return {}

    with IMU_METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def get_imu_schema() -> list[str]:
    if not IMU_SCHEMA_PATH.exists():
        metadata = get_imu_metadata()
        return metadata.get("features", [])

    with IMU_SCHEMA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data.get("features", [])

    return data


# ============================================================
# STARTUP MODEL LOADING
# ============================================================

def load_models() -> None:
    """
    Load all currently available DriveGuard-X AI artifacts
    during application startup.
    """

    # Battery
    get_battery_model()
    get_battery_metadata()
    get_battery_schema()

    # IMU
    get_imu_model()
    get_imu_metadata()
    get_imu_schema()


# ============================================================
# BATTERY MODEL INFO
# ============================================================

def get_battery_model_info() -> dict:
    try:
        get_battery_model()

        metadata = get_battery_metadata()
        features = get_battery_schema()

        return {
            "loaded": True,
            "model_type": metadata.get("model"),
            "n_estimators": metadata.get("n_estimators"),
            "feature_count": len(features),
            "features": features,
            "development_batteries": metadata.get(
                "development_batteries", []
            ),
            "untouched_test_batteries": metadata.get(
                "untouched_test_batteries", []
            ),
            "metadata": metadata,
        }

    except Exception:
        return {
            "loaded": False,
            "model_type": None,
            "n_estimators": None,
            "feature_count": None,
            "features": [],
            "development_batteries": [],
            "untouched_test_batteries": [],
            "metadata": {},
        }


# ============================================================
# IMU MODEL INFO
# ============================================================

def get_imu_model_info() -> dict:
    try:
        model = get_imu_model()

        metadata = get_imu_metadata()
        features = get_imu_schema()

        return {
            "loaded": True,
            "model_type": metadata.get("model"),
            "feature_count": len(features),
            "features": features,
            "classes": metadata.get("classes", []),
            "test_accuracy": metadata.get("test_accuracy"),
            "test_macro_f1": metadata.get("test_macro_f1"),
            "test_weighted_f1": metadata.get(
                "test_weighted_f1"
            ),
            "metadata": metadata,
        }

    except Exception:
        return {
            "loaded": False,
            "model_type": None,
            "feature_count": None,
            "features": [],
            "classes": [],
            "test_accuracy": None,
            "test_macro_f1": None,
            "test_weighted_f1": None,
            "metadata": {},
        }