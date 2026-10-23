# DriveGuard-X Node-RED gateway and input simulator

The flow sends MQTT telemetry through the existing adapter to FastAPI:

```text
MQTT driveguard/esp32/telemetry
  -> Node-RED adapter
  -> FastAPI /live/battery, /live/imu, /live/road
  -> backend fusion
  -> dashboard WebSocket
```

## Input-driven simulator

The simulator publishes sensor-like telemetry to the same MQTT topic as a device. It only sets input measurements. It does not send expected model classes, attribution labels, or protection outputs; those come from the backend.

Configure any subset of these factors with values from 0 to 1:

- `accel_demand`
- `braking_demand`
- `cornering`
- `road_vibration`
- `battery_load`
- `battery_wear`
- `battery_heat`

Example smooth/low-load run:

```powershell
$body = @{
  accel_demand = 0.1
  braking_demand = 0
  cornering = 0
  road_vibration = 0.05
  battery_load = 0.1
  battery_wear = 0.05
  battery_heat = 0.05
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://localhost:1880/simulator/config -ContentType 'application/json' -Body $body
Invoke-RestMethod -Method Post -Uri http://localhost:1880/simulator/start
```

Change conditions while it runs by posting a new factor set to `/simulator/config`. The stream runs at 4 messages per second. Stop it with:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:1880/simulator/stop
```

The simulator defaults to disabled. The inject node only publishes while enabled. Reset to low-stress inputs by posting all seven factors as `0`, then start again.

## Backend and dashboard

Set `DRIVEGUARD_API_BASE_URL` in the Node-RED environment if FastAPI is not at `http://127.0.0.1:8000/api/v1`. Configure the MQTT broker node if Mosquitto is not at `127.0.0.1:1883`.

The frontend reads live fused data from FastAPI at `/api/v1/live/ws`. Open the Next.js dashboard to see model predictions, fusion/attribution, protection recommendations, and optional session reports.

## Real ESP32 integration notes

The attached ESP32 firmware currently serves its own controller and telemetry over HTTP on its Wi-Fi access point; it does not publish MQTT. Connect an MQTT publisher to `driveguard/esp32/telemetry` before using the real device with this flow.

The ESP32 currently sends microphone RMS/dB summaries rather than raw audio. The flow creates a short synthetic audio proxy from RMS for the audio model, so road predictions from simulation are pipeline demonstrations, not a substitute for real INMP441 audio windows. Likewise, the battery model was trained on cycle-level features; the gateway constructs a rolling feature vector from the telemetry stream. Validate model quality against real target-device data before treating predictions as measured battery health.

## Manual model inputs in the Node-RED editor

The **MANUAL MODEL INPUTS** section provides three one-click Inject nodes. Edit a node by double-clicking it, change its JSON payload, click **Done**, then click **Deploy**. Click the small square on the left of the Inject node to send that modality to FastAPI:

- **Send battery model input:** edit the 21-feature battery vector; it calls `/live/battery`.
- **Send IMU window:** edit exactly four samples with `AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, and `GyroZ`; it calls `/live/imu`.
- **Send audio window:** edit `rms`, `tone_hz`, `texture`, `impact`, and `impact_position`; the function generates a 16 kHz audio window and calls `/live/road`.

These are inputs, not expected labels. The backend model responses appear in the existing **Backend inference response** debug node and in the dashboard. Trigger all three once to give the backend a complete set of modalities. Stop the periodic simulator before manual checks so its MQTT messages do not overwrite the latest readings.
