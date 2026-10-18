"""End-to-end checks for the multimodal attribution endpoint.

Run from ``backend`` while the API is running:

    python tests/test_attribution_endpoint.py

The script searches several sensor combinations, prints every observed
attribution, and independently verifies the attribution rules against the
three individual model responses.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import wavfile


BASE_URL = os.getenv("DRIVEGUARD_API_URL", "http://127.0.0.1:8000/api/v1")
SAMPLE_COUNT = 1600
SAMPLE_RATE = 16000
ROAD_AUDIO_DIR = Path(__file__).resolve().parents[1] / "test_audio"
ROAD_DISTURBANCE_CONFIDENCE_THRESHOLD = 0.6


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"{path} failed: {exc}") from exc


def battery_profile(degraded: bool) -> dict[str, float]:
    if degraded:
        return {
            "voltage_initial": 3.5,
            "voltage_final": 2.0,
            "voltage_min": 1.8,
            "voltage_max": 3.8,
            "voltage_mean": 2.6,
            "voltage_std": 0.5,
            "voltage_drop": 1.5,
            "current_mean": 5.0,
            "current_min": 0.5,
            "current_max": 10.0,
            "current_std": 3.0,
            "temperature_initial": 50.0,
            "temperature_final": 90.0,
            "temperature_max": 100.0,
            "temperature_mean": 70.0,
            "temperature_rise": 40.0,
            "load_current_abs_mean": 5.0,
            "load_current_abs_max": 10.0,
            "load_voltage_mean": 2.5,
            "load_voltage_min": 1.8,
            "discharge_duration": 500.0,
        }
    return {
            "voltage_initial": 4.2,
            "voltage_final": 4.1,
            "voltage_min": 4.05,
            "voltage_max": 4.2,
            "voltage_mean": 4.15,
            "voltage_std": 0.02,
            "voltage_drop": 0.1,
            "current_mean": 0.1,
            "current_min": 0.05,
            "current_max": 0.2,
            "current_std": 0.03,
            "temperature_initial": 20.0,
            "temperature_final": 21.0,
            "temperature_max": 22.0,
            "temperature_mean": 20.5,
            "temperature_rise": 1.0,
            "load_current_abs_mean": 0.1,
            "load_current_abs_max": 0.2,
            "load_voltage_mean": 4.15,
            "load_voltage_min": 4.05,
            "discharge_duration": 5000.0,
    }


def imu_profile(aggressive: bool) -> list[dict[str, float]]:
    if aggressive:
        values = (
            (3.5, 0.2, 12.0, 180.0, 20.0, 250.0),
            (-3.0, 0.1, 7.0, -160.0, -20.0, -220.0),
            (4.0, -0.3, 14.0, 220.0, 30.0, 300.0),
            (-4.5, 0.4, 6.0, -240.0, -30.0, -280.0),
        )
    else:
        values = (
            (0.5675038, -0.0212618, 9.4735875, -0.2066940, -0.5137487, 0.1390406),
            (-0.2451993, 0.0859193, -0.2173982, 0.1891802, 0.0490866, -0.6926247),
            (0.1021900, -0.0775044, 10.1789782, 0.2022756, -0.1900728, -0.2113466),
            (-0.0068613, -0.0964664, 10.1579233, -0.0410135, -0.3276452, -0.2213815),
        )
    return [
        {
            "AccX": acc_x,
            "AccY": acc_y,
            "AccZ": acc_z,
            "GyroX": gyro_x,
            "GyroY": gyro_y,
            "GyroZ": gyro_z,
        }
        for acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z in values
    ]


def audio_profile(kind: str) -> list[float]:
    recording = ROAD_AUDIO_DIR / f"{kind}.wav"
    if recording.is_file():
        sample_rate, samples = wavfile.read(recording)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{recording.name} uses {sample_rate} Hz; expected {SAMPLE_RATE} Hz"
            )
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        return samples.tolist()

    if kind == "silence":
        return [0.0] * SAMPLE_COUNT
    if kind == "impulse":
        return [1.0 if index % 160 == 0 else 0.0 for index in range(SAMPLE_COUNT)]
    frequency = {"low": 50.0, "mid": 500.0, "high": 3000.0}[kind]
    return [
        0.1 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
        for index in range(SAMPLE_COUNT)
    ]


def expected_attribution(
    battery_features: dict[str, float],
    battery: dict[str, Any],
    imu: dict[str, Any],
    road: dict[str, Any],
) -> str:
    soh_percent = float(battery["soh_percent"])
    stressed = (
        soh_percent < 80.0
        or battery_features["voltage_drop"] > 0.8
        or battery_features["load_current_abs_max"] > 2.0
    )
    aggressive = int(imu["predicted_class"]) != 1
    confidence = float(road["confidence"])
    disturbed = (
        road["road_class"] in {"rough", "moderately_rough", "impact"}
        and confidence >= ROAD_DISTURBANCE_CONFIDENCE_THRESHOLD
    )

    if stressed:
        if aggressive and not disturbed:
            return "driver_induced"
        if disturbed and not aggressive and confidence > 0.6:
            return "road_induced"
        if soh_percent < 75.0 and not aggressive and not disturbed:
            return "battery_degraded"
        return "mixed_or_high_stress"
    return "normal"


def run_case(degraded: bool, aggressive: bool, audio_kind: str) -> dict[str, Any]:
    battery_features = battery_profile(degraded)
    imu_samples = imu_profile(aggressive)
    audio_samples = audio_profile(audio_kind)
    payload = {
        "battery_features": battery_features,
        "imu_samples": imu_samples,
        "audio_samples": audio_samples,
    }

    battery = post("/battery/predict", battery_features)
    imu = post("/imu/predict", {"samples": imu_samples})
    road = post("/road/predict", {"samples": audio_samples})
    combined = post("/inference", payload)

    expected = expected_attribution(battery_features, battery, imu, road)
    actual = combined["attribution"]["likely_cause"]
    verified = expected == actual
    print(
        f"degraded={degraded:<5} aggressive={aggressive:<5} "
        f"audio={audio_kind:<7} | "
        f"battery={battery['soh_percent']:.2f}% "
        f"imu={imu['predicted_class']} "
        f"road={road['road_class']}({road['confidence']:.3f}) | "
        f"expected={expected:<22} actual={actual:<22} "
        f"{'PASS' if verified else 'MISMATCH'}"
    )
    print(json.dumps(combined, indent=2))
    return combined


def main() -> None:
    print(f"Testing {BASE_URL}/inference")
    observed: set[str] = set()
    for degraded in (False, True):
        for aggressive in (False, True):
            for audio_kind in ("smooth", "moderately_rough", "rough", "impact"):
                result = run_case(degraded, aggressive, audio_kind)
                observed.add(result["attribution"]["likely_cause"])

    expected_types = {
        "normal",
        "driver_induced",
        "road_induced",
        "battery_degraded",
        "mixed_or_high_stress",
    }
    print("\nObserved attribution types:", sorted(observed))
    missing = expected_types - observed
    if missing:
        print("Not observed with the current model artifacts:", sorted(missing))
        print("This means the generated sensor inputs did not make the road/IMU models produce the conditions required for those branches.")
    else:
        print("All attribution branches were observed.")


if __name__ == "__main__":
    main()
