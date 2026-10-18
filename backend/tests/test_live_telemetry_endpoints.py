"""Smoke test for the live telemetry endpoints.

Run from ``backend`` while the API is running:

    .\\.venv\\Scripts\\python.exe tests\\test_live_telemetry_endpoints.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

BASE_URL = "http://127.0.0.1:8000/api/v1"


def post(path: str, payload: dict) -> dict:
    request = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        assert response.status == 200
        return json.loads(response.read().decode())


def get(path: str) -> dict:
    with urlopen(f"{BASE_URL}{path}", timeout=30) as response:
        assert response.status == 200
        return json.loads(response.read().decode())


def main() -> None:
    battery = {
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
    imu = [
        {
            "AccX": 0.5675038,
            "AccY": -0.0212618,
            "AccZ": 9.4735875,
            "GyroX": -0.2066940,
            "GyroY": -0.5137487,
            "GyroZ": 0.1390406,
        },
        {
            "AccX": -0.2451993,
            "AccY": 0.0859193,
            "AccZ": -0.2173982,
            "GyroX": 0.1891802,
            "GyroY": 0.0490866,
            "GyroZ": -0.6926247,
        },
        {
            "AccX": 0.1021900,
            "AccY": -0.0775044,
            "AccZ": 10.1789782,
            "GyroX": 0.2022756,
            "GyroY": -0.1900728,
            "GyroZ": -0.2113466,
        },
        {
            "AccX": -0.0068613,
            "AccY": -0.0964664,
            "AccZ": 10.1579233,
            "GyroX": -0.0410135,
            "GyroY": -0.3276452,
            "GyroZ": -0.2213815,
        },
    ]
    audio = [0.1 * math.sin(2 * math.pi * 3000 * i / 16000) for i in range(1600)]

    battery_result = post("/live/battery", battery)
    imu_result = post("/live/imu", {"samples": imu})
    road_result = post("/live/road", {"samples": audio})
    snapshot = get("/live/snapshot")

    assert battery_result["soh_percent"] > 0
    assert len(imu_result["samples"]) == 4
    assert road_result["sample_count"] == 1600
    assert snapshot["battery"]["soh_percent"] == battery_result["soh_percent"]
    assert snapshot["imu"]["predicted_class"] == imu_result["predicted_class"]
    assert snapshot["road"]["road_class"] == road_result["road_class"]

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/live/ws") as websocket:
            websocket_snapshot = websocket.receive_json()
            assert set(websocket_snapshot) == {"battery", "imu", "road"}

    print("Live telemetry endpoint test: PASS")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
