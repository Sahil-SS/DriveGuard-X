# DriveGuard-X FastAPI Backend

**Current stage:** Battery AI + IMU AI integration

The DriveGuard-X backend is a modular inference service for the project's AI models and, eventually, the multimodal decision-making and adaptive protection pipeline.

The backend is deliberately separated from model-training code so that additional components such as the road-condition model, multimodal fusion, cause attribution, and adaptive protection can be added without rewriting the existing model services.

---

## 1. DriveGuard-X System Architecture

The complete DriveGuard-X system is designed around three sensing modalities:

1. **Battery state and electrical behaviour**
   - Voltage
   - Current
   - Temperature
   - SOC/SOH-related information

2. **Vehicle dynamics**
   - MPU6050 accelerometer
   - MPU6050 gyroscope
   - Driving-behaviour classification

3. **Road condition**
   - INMP441 microphone
   - IMU information
   - Road-surface classification

These model outputs will eventually be combined by the backend:

```text
Battery Sensors
      |
      v
  Battery AI  --------\
                        \
MPU6050                 \
   |                     ---> Multimodal Fusion
   v                    /          |
  IMU AI --------------/           v
                              Cause Attribution
                                    |
                                    v
                             Adaptive Protection
                                    |
                                    v
                               Motor / PWM
```

The current backend implements the individual model inference layers first. Fusion and protection are intentionally deferred until the individual model interfaces are stable and validated.

---

## 2. Current Backend Architecture

```text
Client / MATLAB / Simulink / ESP32
                |
                | HTTP / JSON
                v
             FastAPI
                |
        +-------+-------+
        |               |
        v               v
 Battery Service    IMU Service
        |               |
        v               v
 Feature Schema     Windowed IMU
 Validation         Feature Extraction
        |               |
        v               v
 Random Forest      XGBoost Model
        |               |
        v               v
       SOH       Driving Behaviour

Future:
        |
        +--> Road Service
        +--> Fusion Engine
        +--> Attribution Engine
        +--> Protection Engine
```

The important design principle is:

> **Each model has its own preprocessing, service, endpoint, artifact and schema.**

This allows the models to be developed and validated independently before they are connected into the final DriveGuard-X decision pipeline.

---

## 3. Current API Modules

### Battery AI

Endpoint:

```text
POST /api/v1/battery/predict
```

Purpose:

- Accept the 21 engineered battery features.
- Run the trained Random Forest model.
- Return predicted State of Health (SOH).

The current model is a **cycle-level model**, not yet a rolling-window real-time model.

### IMU AI

Endpoint:

```text
POST /api/v1/imu/predict
```

Purpose:

- Accept exactly 4 consecutive IMU samples.
- Extract the same 42 statistical features used by the trained model.
- Run the XGBoost classifier.
- Return the predicted driving-behaviour class.

The current IMU model uses:

```text
Raw 6-axis IMU
      |
      v
4-sample window
      |
      v
42 statistical features
      |
      v
XGBoost classifier
      |
      v
Driving behaviour class
```

The four project-level classes are represented as:

```text
1 -> Normal Driving
2 -> Harsh Acceleration
3 -> Harsh Braking
4 -> Sharp Cornering
```

The exact class mapping should remain consistent with the model-development notebook and its saved metadata.

---

## 4. Important Battery Model Limitation

The current Battery AI model is a **cycle-level model** trained on 21 engineered discharge-cycle features.

It does not currently accept an arbitrary instantaneous battery reading such as:

```text
voltage = 3.8 V
current = 2.0 A
temperature = 28 °C
```

and directly predict SOH.

Instead, a discharge cycle/window is summarized into the exact feature representation used during training.

The current API therefore accepts the 21 engineered features directly.

The intended future architecture is:

```text
ESP32 / Simulink Battery Stream
              |
              v
        Window / Cycle Data
              |
              v
     Battery Feature Extraction
              |
              v
       Exact 21 Features
              |
              v
        Battery AI Model
              |
              v
             SOH
```

The existing `battery_features.py` module is reserved for this future preprocessing layer.

**Important:** the preprocessing implemented there must reproduce the feature-generation logic used during model training. It should not introduce a different feature definition merely to make the API convenient.

---

## 5. Battery Model Input Schema

The Battery AI model expects exactly these 21 features:

```text
1.  voltage_initial
2.  voltage_final
3.  voltage_min
4.  voltage_max
5.  voltage_mean
6.  voltage_std
7.  voltage_drop
8.  current_mean
9.  current_min
10. current_max
11. current_std
12. temperature_initial
13. temperature_final
14. temperature_max
15. temperature_mean
16. temperature_rise
17. load_current_abs_mean
18. load_current_abs_max
19. load_voltage_mean
20. load_voltage_min
21. discharge_duration
```

The order and meaning of these features must remain stable between:

```text
Training
   |
   v
Validation
   |
   v
Saved Model
   |
   v
FastAPI Inference
   |
   v
Future ESP32 / Simulink Integration
```

---

## 6. IMU Model Input Schema

The IMU model uses 42 statistical features derived from six raw sensor channels.

### Accelerometer features

```text
AccMeanX, AccMeanY, AccMeanZ
AccSkewX, AccSkewY, AccSkewZ
AccKurtX, AccKurtY, AccKurtZ
AccMinX, AccMinY, AccMinZ
AccMaxX, AccMaxY, AccMaxZ
AccMedianX, AccMedianY, AccMedianZ
AccStdX, AccStdY, AccStdZ
```

### Gyroscope features

```text
GyroMeanX, GyroMeanY, GyroMeanZ
GyroSkewX, GyroSkewY, GyroSkewZ
GyroKurtX, GyroKurtY, GyroKurtZ
GyroMinX, GyroMinY, GyroMinZ
GyroMaxX, GyroMaxY, GyroMaxZ
GyroMedianX, GyroMedianY, GyroMedianZ
GyroStdX, GyroStdY, GyroStdZ
```

The current deployment uses the native XGBoost model artifact rather than relying on the serialized scikit-learn wrapper.

This is intentional because the native XGBoost JSON artifact is the verified deployment artifact for the current IMU model.

---

## 7. Model Artifacts

### Battery

Place the following files in:

```text
backend/models/battery/
```

```text
driveguard_x_final_battery_rf.joblib
final_model_metadata.json
feature_schema.json
```

### IMU

Place the following files in:

```text
backend/models/imu/
```

```text
driveguard_xgb_final.json
driveguard_xgb_final.joblib
imu_feature_list.json
model_metadata.json
```

For the current backend inference path, the IMU service loads:

```text
driveguard_xgb_final.json
```

using the native XGBoost model loader.

The `.joblib` artifact may be retained as a development/reference artifact, but the backend should use the verified native JSON model for the current deployment path.

---

## 8. Repository Structure

The backend is organized approximately as follows:

```text
backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── models/
│   │   └── __init__.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── battery_features.py
│   │   └── imu_features.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── model_loader.py
│       ├── battery_service.py
│       └── imu_service.py
│
├── models/
│   ├── battery/
│   │   ├── driveguard_x_final_battery_rf.joblib
│   │   ├── final_model_metadata.json
│   │   └── feature_schema.json
│   │
│   └── imu/
│       ├── driveguard_xgb_final.json
│       ├── driveguard_xgb_final.joblib
│       ├── imu_feature_list.json
│       └── model_metadata.json
│
├── tests/
│   ├── __init__.py
│   └── test_battery_api.py
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

As more models are integrated, their preprocessing and service modules should follow the same separation.

---

## 9. Environment Setup on Windows PowerShell

From the repository root:

```powershell
cd backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

Verify that the virtual environment is active:

```powershell
python --version
pip --version
```

---

## 10. Dependencies

The backend currently requires:

```text
fastapi
uvicorn[standard]
pydantic-settings
joblib
numpy
pandas
scikit-learn==1.6.1
xgboost
pytest
httpx
```

The deployed model runtime should be kept consistent with the environment used to validate the model artifacts.

In particular, do not change ML-library versions casually after deployment validation. Model serialization and numerical preprocessing can be sensitive to library/runtime differences.

---

## 11. Run the Backend

From:

```text
backend/
```

run:

```powershell
uvicorn app.main:app --reload
```

The server will normally be available at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

## 12. API Health Check

PowerShell:

```powershell
curl http://127.0.0.1:8000/api/v1/health
```

The health endpoint confirms that the FastAPI application is running and reports whether the Battery model is loaded.

---

## 13. Battery Model Information

```powershell
curl http://127.0.0.1:8000/api/v1/battery/model-info
```

This endpoint exposes deployment metadata such as:

- Model type
- Number of estimators
- Feature count
- Feature names
- Development batteries
- Untouched test batteries
- Saved model metadata

This is useful for checking that the backend is using the expected artifact and schema.

---

## 14. IMU Model Information

The IMU model information endpoint should expose the corresponding IMU metadata once the route is enabled:

```text
GET /api/v1/imu/model-info
```

The intended information includes:

- Model type
- Feature count
- Feature names
- Classes
- Test accuracy
- Macro F1
- Weighted F1
- Model metadata

This helps verify that the deployed IMU model matches the trained model.

---

## 15. Example Battery Prediction

Use Swagger at:

```text
http://127.0.0.1:8000/docs
```

or send the following JSON to:

```text
POST /api/v1/battery/predict
```

```json
{
  "voltage_initial": 4.2,
  "voltage_final": 3.2,
  "voltage_min": 3.2,
  "voltage_max": 4.2,
  "voltage_mean": 3.8,
  "voltage_std": 0.1,
  "voltage_drop": 1.0,
  "current_mean": -2.0,
  "current_min": -2.5,
  "current_max": -1.5,
  "current_std": 0.2,
  "temperature_initial": 25.0,
  "temperature_final": 30.0,
  "temperature_max": 30.5,
  "temperature_mean": 27.5,
  "temperature_rise": 5.0,
  "load_current_abs_mean": 2.0,
  "load_current_abs_max": 2.5,
  "load_voltage_mean": 3.8,
  "load_voltage_min": 3.2,
  "discharge_duration": 3600.0
}
```

These values demonstrate the API format only.

They are **not** a validated DriveGuard-X vehicle measurement and should not be interpreted as an experimentally meaningful SOH result.

---

## 16. Example IMU Prediction

The IMU endpoint expects exactly four consecutive samples:

```text
POST /api/v1/imu/predict
```

Example:

```json
{
  "samples": [
    {
      "AccX": 0.12,
      "AccY": 0.03,
      "AccZ": 9.81,
      "GyroX": 0.01,
      "GyroY": 0.02,
      "GyroZ": 0.03
    },
    {
      "AccX": 0.15,
      "AccY": 0.04,
      "AccZ": 9.79,
      "GyroX": 0.02,
      "GyroY": 0.01,
      "GyroZ": 0.04
    },
    {
      "AccX": 0.11,
      "AccY": 0.02,
      "AccZ": 9.82,
      "GyroX": 0.01,
      "GyroY": 0.03,
      "GyroZ": 0.02
    },
    {
      "AccX": 0.13,
      "AccY": 0.03,
      "AccZ": 9.80,
      "GyroX": 0.02,
      "GyroY": 0.02,
      "GyroZ": 0.03
    }
  ]
}
```

The API internally converts the four samples into the 42-feature representation expected by the XGBoost model.

**Important:** successful API execution confirms that the endpoint and model loading work. It does not by itself prove that the preprocessing is numerically identical to the training notebook. Deployment validation must compare the notebook's feature extraction and prediction against the backend implementation.

---

## 17. Current Model Status

### Battery AI

```text
Training                 DONE
Final model              DONE
Model artifact           DONE
Feature schema           DONE
FastAPI integration      DONE
Swagger testing          DONE
Deployment validation    NEXT / REQUIRED
```

### IMU AI

```text
Training                 DONE
Final XGBoost model      DONE
Feature list             DONE
Native JSON artifact     VERIFIED
FastAPI integration      DONE
Swagger testing          DONE
Exact preprocessing
validation               NEXT / REQUIRED
```

The current API tests demonstrate that both model endpoints can load their artifacts, process requests and return predictions.

The next validation step is not to change the model, but to establish **training-to-deployment equivalence** for the preprocessing pipeline.

---

## 18. Training-to-Deployment Validation

Before using the backend with real ESP32 or Simulink data, inference should be validated against the original notebook.

The validation flow should be:

```text
Untouched test cycle / window
            |
            +----------------------+
            |                      |
            v                      v
   Original notebook          FastAPI preprocessing
            |                      |
            v                      v
       Feature vector          Feature vector
            |                      |
            v                      v
     Direct model          FastAPI model inference
       prediction                 |
            |                     |
            +----------+----------+
                       |
                       v
                 Compare outputs
```

For Battery AI, verify:

- Same 21 features
- Same feature order
- Same numerical values
- Same model artifact
- Same prediction

For IMU AI, verify:

- Same four-sample window
- Same statistical calculations
- Same 42 feature order
- Same numerical feature values
- Same XGBoost model
- Same class prediction

Only after this equivalence check should the backend be treated as deployment-ready for the corresponding model.

---

## 19. Why the Backend Does Not Yet Have a Combined Inference Endpoint

The final DriveGuard-X system will eventually need a combined inference flow similar to:

```text
Battery measurements
        |
        v
    Battery AI
        |
        +-------------------+
                            |
IMU measurements            |
        |                   |
        v                   |
     IMU AI                 |
        |                   |
        +--------+----------+
                 |
Road audio + IMU |
        |        |
        v        |
     Road AI     |
        |        |
        +--------+
                 |
                 v
        Multimodal Fusion
                 |
                 v
        Cause Attribution
                 |
                 v
       Adaptive Protection
                 |
                 v
          PWM / Motor
```

However, this combined endpoint should **not** be implemented yet.

The individual model contracts need to be stable and validated first.

This prevents problems where an error in one preprocessing pipeline becomes hidden inside a large combined inference function.

---

## 20. Planned API Expansion

The backend is intended to grow toward:

```text
/api/v1/battery/predict
/api/v1/battery/model-info

/api/v1/imu/predict
/api/v1/imu/model-info

/api/v1/road/predict
/api/v1/road/model-info

/api/v1/inference
/api/v1/protection/decision
```

The exact request and response schemas will be defined when each model is trained and validated.

---

## 21. Planned DriveGuard-X Decision Pipeline

The final system should not simply predict independent labels.

The purpose of the backend is eventually to determine **why a battery-stress event is occurring** and choose an appropriate protection response.

Conceptually:

```text
Battery State / Stress
        +
Driving Behaviour
        +
Road Condition
        |
        v
Multimodal Fusion
        |
        v
Cause Attribution
        |
        v
Adaptive Protection
```

For example, the final system may distinguish between:

```text
High battery stress
+
Aggressive driving
+
Smooth road
        |
        v
Likely driver-induced stress
        |
        v
Appropriate power limitation
```

and:

```text
High battery stress
+
Normal driving
+
Rough road / bump
        |
        v
Likely road-induced event
        |
        v
Avoid unnecessary severe intervention
```

These are **system-design examples**, not experimental results.

The exact attribution rules, confidence calculations and PWM limits will be defined and validated later.

---

## 22. MATLAB / Simulink Integration

MATLAB/Simulink will be used to create the simulation environment before complete hardware integration.

The intended workflow is:

```text
MATLAB / Simulink
       |
       | simulated vehicle + sensor data
       v
    FastAPI
       |
       +--> Battery AI
       +--> IMU AI
       +--> Road AI
       |
       v
  Fusion / Attribution
       |
       v
 Protection Decision
       |
       v
 Simulated Motor / PWM
```

Later, the virtual sensor source can be replaced or complemented by the ESP32-based hardware platform:

```text
ESP32 Sensors
      |
      v
FastAPI
      |
      v
Same AI + decision pipeline
```

This allows the model-development and deployment architecture to remain consistent between simulation and hardware experiments.

---

## 23. Hardware Integration Direction

The final hardware prototype is intended to use the ESP32 platform together with:

```text
Battery sensors
    |
    +--> Voltage sensing
    +--> Current sensing
    +--> Temperature sensing

MPU6050
    |
    +--> Accelerometer
    +--> Gyroscope

INMP441
    |
    +--> Road-surface acoustic information

ESP32-CAM
    |
    +--> Event-triggered visual information
```

The backend should not assume that the final sensor source is a computer.

Its interfaces are designed around structured sensor/model data so that MATLAB/Simulink and ESP32 can eventually act as data sources.

---

## 24. Testing Strategy

Tests should be added at multiple levels.

### Unit tests

Test:

- Feature extraction
- Input validation
- Model loading
- Output formatting

### API tests

Test:

- Health endpoint
- Model-info endpoints
- Battery prediction
- IMU prediction
- Invalid input handling

### Model equivalence tests

Compare:

```text
Notebook feature extraction
vs.
Backend feature extraction
```

and:

```text
Notebook prediction
vs.
Backend prediction
```

### System tests

Later, test the complete:

```text
Battery + IMU + Road
        |
        v
Fusion
        |
        v
Attribution
        |
        v
Protection
```

pipeline.

---

## 25. Research Evaluation Direction

The backend is part of a larger research system.

The final evaluation should not rely only on individual ML accuracy.

The project should eventually compare the DriveGuard-X adaptive approach against conventional threshold-based protection.

Relevant system-level measures include:

- False protection interventions
- Missed protection events
- Protection latency
- Battery stress
- Power limitation
- Energy consumption
- Vehicle performance
- Attribution accuracy
- Model inference latency

The objective is to demonstrate whether contextual information can reduce unnecessary protection while still responding appropriately to genuine battery-risk events.

---

## 26. Separation of Training and Deployment Code

**Do not mix model-training code with API code.**

Model development should remain in the AI/model-development portion of the repository.

Training notebooks should contain:

```text
Dataset loading
Data cleaning
Feature engineering
Train/validation/test split
Model training
Evaluation
Artifact generation
```

The FastAPI backend should contain:

```text
Model loading
Input validation
Inference preprocessing
Prediction
API schemas
API routing
Deployment configuration
Tests
```

This separation makes the project easier to reproduce, debug and deploy.

---

## 27. Model Artifact Versioning

As models evolve, avoid silently replacing artifacts.

Prefer explicit versioning, for example:

```text
models/
├── battery/
│   └── v1/
│       ├── model.joblib
│       ├── metadata.json
│       └── feature_schema.json
│
└── imu/
    └── v1/
        ├── model.json
        ├── metadata.json
        └── feature_schema.json
```

The exact versioning structure can be introduced once the current models are locked.

The important principle is that the backend should always be able to identify **which model artifact and feature schema produced a prediction**.

---

## 28. Current Status

```text
==================================================
DRIVEGUARD-X BACKEND STATUS
==================================================

FastAPI foundation             DONE
Project structure              DONE

Battery AI
  Model                        DONE
  Artifact                     DONE
  Schema                       DONE
  API                          DONE
  Swagger test                 DONE

IMU AI
  Model                        DONE
  Native JSON artifact         DONE
  Feature list                 DONE
  API                          DONE
  Swagger test                 DONE
  Exact preprocessing check    NEXT

Road AI                        NEXT
Fusion Engine                  FUTURE
Cause Attribution              FUTURE
Adaptive Protection            FUTURE
Combined Inference API         FUTURE
MATLAB/Simulink integration    FUTURE
ESP32 integration              FUTURE
Real-world data validation     FUTURE
Target-domain adaptation       FUTURE
Final system evaluation       FUTURE
==================================================
```

---

## 29. Development Principle

The backend should evolve in this order:

```text
1. Lock Battery AI
        |
2. Validate Battery deployment
        |
3. Lock IMU AI
        |
4. Validate IMU deployment
        |
5. Develop Road AI
        |
6. Validate Road deployment
        |
7. Build Multimodal Fusion
        |
8. Build Cause Attribution
        |
9. Build Adaptive Protection
        |
10. Build Combined Inference API
        |
11. Connect MATLAB/Simulink
        |
12. Connect ESP32 hardware
        |
13. Collect target-domain data
        |
14. Retrain / calibrate where required
        |
15. Run final system experiments
```

This staged approach keeps each layer independently testable and makes it easier to identify whether an error comes from the sensor data, preprocessing, model, fusion logic or protection logic.

---

## 30. Project Goal

The FastAPI backend is not the final product by itself.

Its role is to provide the deployment layer connecting the trained AI models to the rest of DriveGuard-X:

```text
Sensors
   |
   v
AI Models
   |
   v
FastAPI
   |
   v
Multimodal Understanding
   |
   v
Cause-Aware Decision
   |
   v
Adaptive Battery Protection
```

The final research contribution is the **system-level cause-aware protection architecture**, where battery electrical information is interpreted together with vehicle dynamics and road context rather than relying only on fixed battery thresholds.
