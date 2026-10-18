# backend/app/services/road_service.py
from pathlib import Path
from typing import Any
import math

import joblib

from app.preprocessing import road_features as rf

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "road" / "road_model.pkl"

class RoadService:
    def __init__(self):
        self.bundle: dict[str, Any] | None = None
        self.loaded = False

        if MODEL_PATH.is_file():
            self.bundle = joblib.load(MODEL_PATH)
            self.loaded = True

    def predict(self, samples: list[float]) -> dict[str, Any]:
        if not self.loaded or self.bundle is None:
            raise RuntimeError(f"Road model artifact not found at {MODEL_PATH}")
        if not samples:
            raise ValueError("samples must contain at least one value")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in samples):
            raise ValueError("samples must contain only finite numeric values")

        y_pp = rf.preprocess(
            samples,
            sr=rf.SR,
            normalize=self.bundle.get("config", {}).get("normalize", "none"),
        )
        label, probs = rf.predict_window(self.bundle, y_pp)
        return {
            "road_class": label,
            "confidence": probs[label],
            "probabilities": probs
        }

    def get_model_info(self) -> dict[str, Any]:
        if not self.loaded or self.bundle is None:
            return {"status": "not_loaded"}
        return {
            "model_type": self.bundle.get("model_name"),
            "classes": list(self.bundle["label_encoder"].classes_),
            "feature_count": len(self.bundle["feature_names"]),
            "config": self.bundle.get("config", {}),
        }

road_service = RoadService()
