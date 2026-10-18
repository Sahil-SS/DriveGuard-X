# DriveGuard-X API Documentation

This document is the integration contract for Node-RED and the instrument-cluster frontend.

## 1. Server

Start the backend from `O:\\DriveGuard_X\\backend`:

```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Base URL:

```text
http://127.0.0.1:8000/api/v1
```

If Node-RED or the frontend runs on another machine, start Uvicorn with `--host 0.0.0.0` and replace `127.0.0.1` with the backend host's LAN IP.

All HTTP bodies are JSON. Swagger is available at `http://127.0.0.1:8000/docs`.

## 2. Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service/model health |
| POST | `/battery/predict` | Battery SOH and stress |
| GET | `/battery/model-info` | Battery metadata |
| POST | `/imu/predict` | Driving behavior |
| POST | `/road/predict` | Road condition |
| GET | `/road/model-info` | Road metadata |
| POST | `/inference` | All models plus attribution/protection |
| POST | `/live/battery` | Process and cache latest battery result |
| POST | `/live/imu` | Process and cache latest IMU result |
| POST | `/live/road` | Process and cache latest road result |
| GET | `/live/snapshot` | Read latest cached readings |
| WS | `/live/ws` | Receive the latest snapshot every second |

## 3. Input contracts

### Battery

POST `/battery/predict` or `/live/battery`:

```json
{
  "cycle_index": 120,
  "voltage_mean": 3.72,
  "voltage_std": 0.04,
  "voltage_min": 3.61,
  "voltage_max": 3.81,
  "current_mean": 1.2,
  "current_std": 0.35,
  "current_min": 0.2,
  "current_max": 2.4,
  "temperature_mean": 29.0,
  "temperature_std": 2.0,
  "temperature_min": 26.0,
  "temperature_max": 33.0,
  "soc_mean": 0.72,
  "soc_std": 0.08,
  "soc_min": 0.60,
  "soc_max": 0.84,
  "energy_throughput": 1.8,
  "charge_duration": 3600,
  "discharge_duration": 3000,
  "internal_resistance": 0.08
}
```

Example response from `/battery/predict`:

```json
{"soh_percent": 91.07, "stressed": false}
```

The exact SOH depends on the loaded model. This model expects cycle-level statistical features.

### IMU

POST `/imu/predict` or `/live/imu`:

```json
{
  "samples": [
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}
  ]
}
```

Exactly four samples are required. Example response:

```json
{"predicted_class": 1, "driving_class": "normal"}
```

Class mapping: 1 normal, 2 harsh_acceleration, 3 harsh_braking, 4 sharp_cornering.

### Road audio

POST `/road/predict` or `/live/road`:

```json
{
  "samples": [0.0, 0.001, -0.002, 0.004]
}
```

The example is shortened for readability. Send at least 320 finite samples; a one-second 16 kHz window (16,000 samples) is preferred. Example response:

```json
{
  "road_class": "smooth",
  "confidence": 0.82,
  "probabilities": {
    "impact": 0.03,
    "moderately_rough": 0.07,
    "rough": 0.08,
    "smooth": 0.82
  }
}
```

## 4. Combined inference

POST `/inference`:

```json
{
  "battery_features": {
    "cycle_index": 120,
    "voltage_mean": 3.72,
    "voltage_std": 0.04,
    "voltage_min": 3.61,
    "voltage_max": 3.81,
    "current_mean": 1.2,
    "current_std": 0.35,
    "current_min": 0.2,
    "current_max": 2.4,
    "temperature_mean": 29.0,
    "temperature_std": 2.0,
    "temperature_min": 26.0,
    "temperature_max": 33.0,
    "soc_mean": 0.72,
    "soc_std": 0.08,
    "soc_min": 0.60,
    "soc_max": 0.84,
    "energy_throughput": 1.8,
    "charge_duration": 3600,
    "discharge_duration": 3000,
    "internal_resistance": 0.08
  },
  "imu_samples": [
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0},
    {"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}
  ],
  "audio_samples": [0.0, 0.001, -0.002, 0.004]
}
```

Use real road audio or a full test window in actual requests.

Response shape:

```json
{
  "battery_state": {
    "soh_percent": 91.07,
    "stressed": false
  },
  "driving_behavior": {
    "class": "normal",
    "raw_class_id": 1
  },
  "road_condition": {
    "class": "smooth",
    "confidence": 0.82
  },
  "attribution": {
    "likely_cause": "normal",
    "confidence": 1.0
  },
  "protection": {
    "action": "normal_operation",
    "pwm_limit_pct": 100
  }
}
```

The numeric values vary with the model and input. The attribution branches are `normal`, `driver_induced`, `road_induced`, `battery_degraded`, and `mixed_or_high_stress`.

## 5. Live dashboard API

The live endpoints return processed readings and cache only the latest value. They are useful for an instrument cluster.

### POST `/live/battery`

Request: same body as `/battery/predict`.

Response:

```json
{
  "timestamp": "2026-10-18T12:00:00Z",
  "features": {"cycle_index": 120, "voltage_mean": 3.72},
  "soh_percent": 91.07,
  "stressed": false
}
```

The `features` object contains the complete submitted feature vector.

### POST `/live/imu`

Request: same body as `/imu/predict`.

Response:

```json
{
  "timestamp": "2026-10-18T12:00:01Z",
  "samples": [],
  "predicted_class": 1,
  "driving_class": "normal"
}
```

The response contains the complete four-sample window in `samples`.

### POST `/live/road`

Request: same body as `/road/predict`.

Response:

```json
{
  "timestamp": "2026-10-18T12:00:02Z",
  "sample_count": 16000,
  "road_class": "smooth",
  "confidence": 0.82,
  "probabilities": {
    "impact": 0.03,
    "moderately_rough": 0.07,
    "rough": 0.08,
    "smooth": 0.82
  }
}
```

### GET `/live/snapshot`

Returns:

```json
{
  "battery": null,
  "imu": null,
  "road": null
}
```

Before a source has posted data, its value is `null`. After updates, each value contains the corresponding live response. The snapshot is process-local; it resets when the server restarts.

### WebSocket `/live/ws`

Connect to:

```text
ws://127.0.0.1:8000/api/v1/live/ws
```

The server sends a JSON snapshot approximately once per second. JavaScript example:

```javascript
const socket = new WebSocket("ws://127.0.0.1:8000/api/v1/live/ws");
socket.onmessage = (event) => {
  const snapshot = JSON.parse(event.data);
  console.log(snapshot.battery, snapshot.imu, snapshot.road);
};
```

## 6. Node-RED wiring

Recommended flow:

```text
MQTT / serial / sensor input
        -> JSON node
        -> HTTP request node
        -> dashboard or debug node
```

For each live source, configure an HTTP Request node:

| Source | Method | URL |
|---|---|---|
| Battery | POST | `http://127.0.0.1:8000/api/v1/live/battery` |
| IMU | POST | `http://127.0.0.1:8000/api/v1/live/imu` |
| Road | POST | `http://127.0.0.1:8000/api/v1/live/road` |

Set `Content-Type: application/json`. For a dashboard, either poll `GET /live/snapshot` or use the WebSocket from the frontend. To display attribution and protection together, call `POST /inference` and map its response fields directly to dashboard widgets.

Suggested frontend mappings:

| Widget | JSON path |
|---|---|
| Battery SOH | `battery.soh_percent` or `battery_state.soh_percent` |
| Battery warning | `battery.stressed` or `battery_state.stressed` |
| Driving behavior | `imu.driving_class` or `driving_behavior.class` |
| Road state | `road.road_class` or `road_condition.class` |
| Road confidence | `road.confidence` or `road_condition.confidence` |
| Cause | `attribution.likely_cause` |
| Protection | `protection.action` and `protection.pwm_limit_pct` |

## 7. Errors

- `422`: invalid JSON, missing fields, wrong types, fewer than four IMU samples, or non-finite sensor values.
- `500`: model/preprocessing failure; inspect backend logs.
- `503`: model not available or startup loading failed.

## 8. Verification

With the API running:

```powershell
cd O:\DriveGuard_X\backend
.\.venv\Scripts\python.exe tests\test_attribution_endpoint.py
.\.venv\Scripts\python.exe tests\test_live_telemetry_endpoints.py
.\.venv\Scripts\python.exe -m pytest -q
```

The live test checks battery, IMU, road, snapshot, and WebSocket behavior. The attribution test checks all five attribution branches.

