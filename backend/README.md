# DriveGuard-X Backend

FastAPI inference and telemetry service for the DriveGuard-X battery, IMU, road-condition, attribution, and protection pipeline.

See [api_docs.md](./api_docs.md) for the endpoint contracts used by Node-RED and the frontend.

## Architecture and flow

```text
Battery JSON ──> validation ──> Battery Random Forest ──> SOH + stressed
IMU samples ──> 42-feature extraction ──> IMU XGBoost ──> driving class
Road audio ──> 47-feature extraction ──> Road Random Forest ──> road class/confidence
       \\                 \\                         \\
        +------------------+---------------------------+
                           v
                 Attribution engine
                           |
                           v
        likely cause + confidence + protection action

Node-RED / sensor gateway
  ├─ POST /live/battery
  ├─ POST /live/imu
  ├─ POST /live/road
  ├─ GET  /live/snapshot
  └─ WS   /live/ws
```

The combined endpoint runs all three model services and then passes their outputs to the attribution and protection logic. The live endpoints run one model at a time and keep the latest processed values in an in-process snapshot for an instrument-cluster frontend.

## Code structure

| Path | Responsibility |
|---|---|
| `app/main.py` | Creates FastAPI, configures CORS, loads models at startup, mounts `/api/v1`. |
| `app/api/routes.py` | HTTP and WebSocket routes. |
| `app/api/schemas.py` | Pydantic request/response validation. |
| `app/core/config.py` | Backend and model paths. |
| `app/services/model_loader.py` | Loads runtime model artifacts. |
| `app/services/battery_service.py` | Battery feature validation and SOH prediction. |
| `app/services/imu_service.py` | IMU prediction from four samples. |
| `app/services/road_service.py` | Road audio prediction. |
| `app/services/attribution_service.py` | Combines predictions into cause and protection. |
| `app/services/live_telemetry.py` | Stores the latest processed reading. |
| `app/preprocessing/` | Battery, IMU, and road feature extraction. |
| `models/` | Model files and metadata. |
| `tests/` | Endpoint and attribution verification scripts. |

## Model contracts

### Battery

The battery request contains these 21 features:

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

This is a cycle/statistical SOH model, not a raw high-frequency battery-stream model. Runtime artifacts are under `models/battery/`.

### IMU

Send exactly four consecutive samples. Each sample has six values:

```json
{"ax": 0.0, "ay": 0.0, "az": 9.81, "gx": 0.0, "gy": 0.0, "gz": 0.0}
```

The runtime extractor produces the 42 statistics expected by the trained classifier.

| Raw class | Meaning |
|---:|---|
| 1 | normal |
| 2 | harsh_acceleration |
| 3 | harsh_braking |
| 4 | sharp_cornering |

### Road

Send mono audio samples. The current preprocessing expects 16 kHz audio, preferably a one-second/16,000-sample window, with at least 320 samples. It extracts 47 features and predicts `impact`, `moderately_rough`, `rough`, or `smooth`.

The current road artifact was trained on synthetic audio, so it is suitable for integration tests but must be retrained and validated with real vehicle audio before production use.

## Attribution rules

| Condition | Result |
|---|---|
| Battery is not stressed | `normal` |
| Stressed + aggressive driving + no confident road disturbance | `driver_induced` |
| Stressed + non-aggressive driving + road confidence >= 0.6 | `road_induced` |
| SOH < 75% + normal driving + no confident road disturbance | `battery_degraded` |
| Other stressed combinations | `mixed_or_high_stress` |

A road disturbance requires class `rough`, `moderately_rough`, or `impact`, and confidence at least 0.6. The protection service converts the stress/attribution result to a protection action and PWM limit.

## Directory layout

```text
backend/
├── app/
│   ├── main.py
│   ├── api/{routes.py,schemas.py}
│   ├── core/config.py
│   ├── preprocessing/
│   └── services/
├── models/{battery,imu,road}/
├── test_audio/
├── tests/
├── requirements.txt
├── README.md
└── api_docs.md
```

## Install

From PowerShell:

```powershell
cd O:\DriveGuard_X\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
cd O:\DriveGuard_X\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Then open:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`
- Health: `http://127.0.0.1:8000/api/v1/health`

For a frontend on another device, use `--host 0.0.0.0` and replace `127.0.0.1` with the backend machine's LAN address. Add authentication and network controls before exposing it outside localhost.

## Endpoint groups

- Health: `GET /api/v1/health`
- Individual models: `POST /api/v1/battery/predict`, `POST /api/v1/imu/predict`, `POST /api/v1/road/predict`
- Model metadata: `GET /api/v1/battery/model-info`, `GET /api/v1/road/model-info`
- Combined decision: `POST /api/v1/inference`
- Live inputs: `POST /api/v1/live/battery`, `POST /api/v1/live/imu`, `POST /api/v1/live/road`
- Live reads: `GET /api/v1/live/snapshot`, `WS /api/v1/live/ws`

See [api_docs.md](./api_docs.md) for complete JSON examples and Node-RED wiring.

## Verify

Run these while the API is running:

```powershell
cd O:\DriveGuard_X\backend
.\.venv\Scripts\python.exe tests\test_attribution_endpoint.py
.\.venv\Scripts\python.exe tests\test_live_telemetry_endpoints.py
.\.venv\Scripts\python.exe -m pytest -q
```

The attribution script checks `normal`, `driver_induced`, `road_induced`, `battery_degraded`, and `mixed_or_high_stress`.

## Operational notes

- Live telemetry is process-local memory. For multiple workers or durable data, use Redis, a database, or a message broker.
- Live endpoints return processed predictions, not raw sensor streams; send updates periodically from Node-RED or a sensor gateway.
- The WebSocket sends the latest snapshot once per second and does not ingest sensor data.
- Use `/inference` when the frontend needs one combined attribution/protection decision.
- The battery and road artifacts require representative real-world validation before safety-critical use.

