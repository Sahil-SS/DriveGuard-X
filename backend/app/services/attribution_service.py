# backend/app/services/attribution_service.py
import numpy as np
from app.services.battery_service import predict_battery_soh
from app.services.imu_service import predict_imu
from app.services.road_service import road_service
from app.api.schemas import CombinedInferenceRequest, IMUPredictionRequest, BatteryFeatureVector

ROAD_DISTURBANCE_CONFIDENCE_THRESHOLD = 0.6

class AttributionEngine:
    def __init__(self):
        # Class mappings for IMU model (1: Normal, 2: Harsh Accel, 3: Harsh Braking, 4: Sharp Cornering)
        self.imu_class_names = {
            1: "normal",
            2: "aggressive_acceleration",
            3: "harsh_braking",
            4: "sharp_cornering"
        }

    def process_inference(self, request: CombinedInferenceRequest) -> dict:
        # 1. Run Battery AI
        batt_result = predict_battery_soh(request.battery_features)
        
        # Safely handle whether battery_service returns a dict or an object
        if isinstance(batt_result, dict):
            soh_pct = batt_result.get("soh_percent", 0.0)
        else:
            soh_pct = getattr(batt_result, "soh_percent", 0.0)
        
        # Determine battery stress condition
        is_battery_stressed = (soh_pct < 80.0) or (request.battery_features.voltage_drop > 0.8) or (request.battery_features.load_current_abs_max > 2.0)

        # 2. Run IMU AI (Driving Behavior)
        imu_req = IMUPredictionRequest(samples=request.imu_samples)
        imu_res = predict_imu(imu_req)
        
        if isinstance(imu_res, dict):
            raw_pred_class = imu_res.get("predicted_class", 1)
        else:
            raw_pred_class = getattr(imu_res, "predicted_class", 1)
            
        driving_label = self.imu_class_names.get(raw_pred_class, "normal")
        is_aggressive_driver = driving_label != "normal"

        # 3. Run Road Audio AI
        road_res = road_service.predict(request.audio_samples)
        road_class = road_res["road_class"]
        road_confidence = road_res["confidence"]
        is_road_disturbed = (
            road_class in ["rough", "moderately_rough", "impact"]
            and road_confidence >= ROAD_DISTURBANCE_CONFIDENCE_THRESHOLD
        )

        # 4. Multimodal Cause Attribution Logic (Section 17.2)
        cause = "normal"
        attribution_confidence = 0.95
        protection_action = "maintain_power"
        pwm_limit = 100

        if is_battery_stressed:
            if is_aggressive_driver and not is_road_disturbed:
                cause = "driver_induced"
                attribution_confidence = 0.88
                protection_action = "moderate_power_limitation"
                pwm_limit = 80
            elif is_road_disturbed and not is_aggressive_driver and (road_confidence > 0.6):
                cause = "road_induced"
                attribution_confidence = 0.82
                protection_action = "avoid_unnecessary_severe_intervention"
                pwm_limit = 90
            elif soh_pct < 75.0 and not is_aggressive_driver and not is_road_disturbed:
                cause = "battery_degraded"
                attribution_confidence = 0.92
                protection_action = "strong_battery_protection"
                pwm_limit = 65
            else:
                cause = "mixed_or_high_stress"
                attribution_confidence = 0.75
                protection_action = "balanced_power_limitation"
                pwm_limit = 70

        return self._result(
            soh_pct, is_battery_stressed, driving_label, raw_pred_class,
            road_class, road_confidence, cause, attribution_confidence,
            protection_action, pwm_limit,
        )

    def fuse_results(self, battery: dict, imu: dict, road: dict) -> dict:
        """Fuse already-produced live model results without rerunning AI."""
        soh_pct = float(battery.get("soh_percent", 0.0))
        stressed = bool(battery.get("stressed", False))
        raw_class = int(imu.get("predicted_class", 0))
        driving = self.imu_class_names.get(raw_class, "unknown")
        road_class = str(road.get("road_class", "unavailable"))
        confidence = float(road.get("confidence", 0.0))
        disturbed = road_class in {"rough", "moderately_rough", "impact"} and confidence >= ROAD_DISTURBANCE_CONFIDENCE_THRESHOLD
        cause, attr_conf, action, pwm = "normal", 0.95, "maintain_power", 100
        if stressed:
            if driving != "normal" and not disturbed:
                cause, attr_conf, action, pwm = "driver_induced", 0.88, "moderate_power_limitation", 80
            elif disturbed and driving == "normal" and confidence > 0.6:
                cause, attr_conf, action, pwm = "road_induced", 0.82, "avoid_unnecessary_severe_intervention", 90
            elif soh_pct < 75 and driving == "normal" and not disturbed:
                cause, attr_conf, action, pwm = "battery_degraded", 0.92, "strong_battery_protection", 65
            else:
                cause, attr_conf, action, pwm = "mixed_or_high_stress", 0.75, "balanced_power_limitation", 70
        return self._result(soh_pct, stressed, driving, raw_class, road_class, confidence, cause, attr_conf, action, pwm)

    @staticmethod
    def _result(soh_pct, stressed, driving, raw_class, road_class, confidence, cause, attr_conf, action, pwm):
        return {
            "battery_state": {
                "soh_percent": soh_pct,
                "stressed": stressed
            },
            "driving_behavior": {
                "class": driving,
                "raw_class_id": raw_class
            },
            "road_condition": {
                "class": road_class,
                "confidence": confidence
            },
            "attribution": {
                "likely_cause": cause,
                "confidence": attr_conf
            },
            "protection": {
                "action": action,
                "pwm_limit_pct": pwm
            }
        }

attribution_engine = AttributionEngine()
