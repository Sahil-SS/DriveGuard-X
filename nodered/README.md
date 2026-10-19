# DriveGuard-X Node-RED gateway

This flow provides the ESP32-facing HTTP gateway for the existing FastAPI backend. Node-RED validates and forwards sensor payloads; it does not run AI inference, calculate attribution, or control motor PWM.

The flow also includes a clearly labelled synthetic telemetry section. Its three inject nodes send coordinated synthetic battery, IMU, and road-audio data every five seconds through FastAPI. The generator cycles through a 4 x 4 x 4 matrix (64 combinations) of battery, driving, and road conditions, then repeats. Disable those inject nodes when the real ESP32 is connected.

## Data flow

```text
ESP32 / sensor gateway
  -> Node-RED validation routes
  -> FastAPI /api/v1/live/battery
  -> FastAPI /api/v1/live/imu
  -> FastAPI /api/v1/live/road
  -> FastAPI fused result
  -> dashboard WebSocket /api/v1/live/ws
```

## Gateway routes

| Method | Node-RED route | Backend route |
|---|---|---|
| POST | `/esp32/battery` | `/api/v1/live/battery` |
| POST | `/esp32/imu` | `/api/v1/live/imu` |
| POST | `/esp32/road` | `/api/v1/live/road` |
| GET | `/inference` | `/api/v1/live/inference` |
| GET | `/health` | `/api/v1/health` |

The imported flow validates the battery feature vector, exactly four IMU samples, and at least 320 finite road-audio samples before forwarding them.

## Synthetic demo mode

With FastAPI and Node-RED running, deploy the flow and leave these nodes enabled:

- `Synthetic battery / 5s`
- `Synthetic IMU / 5s`
- `Synthetic road audio / 5s`

Open the Node-RED debug sidebar to see the three backend responses and their `synthetic/scenario/<id>` topic. The battery function advances the scenario ID, while the IMU and road functions use the same stored scenario, so each fused result represents one coordinated combination. After the first three messages arrive, FastAPI creates a fused result. Open the frontend and confirm that the battery gauge, trend graph, driving behavior, road condition, attribution, protection status, sensor lamps, and event log update. The frontend connects to FastAPI's `/api/v1/live/ws`; it does not receive fake values directly from Node-RED.

If the frontend remains disconnected, verify `NEXT_PUBLIC_WS_URL` and that FastAPI is running on the address in `DRIVEGUARD_API_BASE_URL`.

## Install and run

Install Node-RED using the official Node-RED installer or npm. From this directory:

```powershell
cd O:\DriveGuard_X\nodered
Copy-Item .env.example .env
npm install -g --unsafe-perm node-red
```

Set `DRIVEGUARD_API_BASE_URL` in `.env`, then run:

```powershell
node-red --userDir . --settings settings.js
```

Open the editor at `http://127.0.0.1:1880`, or call the gateway routes directly. Import [flows.json](./flows.json) if Node-RED does not automatically load it from the user directory.

## Example payloads

Battery uses the exact 21-feature request documented in [backend/api_docs.md](../backend/api_docs.md). IMU requires exactly four samples with `AccX`, `AccY`, `AccZ`, `GyroX`, `GyroY`, and `GyroZ`. Road audio uses:

```json
{"samples":[0.0,0.001,-0.002,0.004]}
```

Use at least 320 finite samples; a 16 kHz one-second window is preferred.

## ESP32 configuration

Point the ESP32 or its transport adapter at the Node-RED host:

```text
http://<node-red-host>:1880/esp32/battery
http://<node-red-host>:1880/esp32/imu
http://<node-red-host>:1880/esp32/road
```

Set `Content-Type: application/json`. Do not put MongoDB or Groq credentials in Node-RED or ESP32 firmware; those remain backend-only in `backend/.env`.

## Notes

- Node-RED and FastAPI must both be running.
- The backend URL is configured through `DRIVEGUARD_API_BASE_URL`.
- The frontend continues to use the FastAPI WebSocket directly unless you deliberately add a Node-RED WebSocket bridge.
- Validate the gateway on a trusted LAN before exposing it externally; authentication is not added because the existing project has no authentication layer.
