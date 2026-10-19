# DriveGuard-X vehicle infotainment dashboard

This is a read-only vehicle instrument-cluster dashboard. ESP32/Node-RED remains responsible for sending sensor data; FastAPI remains responsible for model inference, fusion, attribution, protection, MongoDB recording, and Groq reports.

## Configure

Copy `.env.example` to `.env.local` and set:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://127.0.0.1:8000/api/v1/live/ws
```

The browser never receives MongoDB credentials or the Groq API key.

## Run

```powershell
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard connects to the single backend WebSocket and displays live battery SOH, driving behavior, road condition, attribution, protection PWM, source availability, trend graphs, and freshness state.

## Recording and reports

- **Start Recording** calls `POST /api/v1/sessions`.
- Fused WebSocket results are persisted by the backend while the session is active.
- **Stop & Generate Summary** calls `POST /api/v1/sessions/{id}/stop`.
- The dashboard polls `GET /api/v1/sessions/{id}/report` until report generation completes or fails.
- Previous sessions are loaded from `GET /api/v1/sessions`, and reports reopen through the report endpoint.

The frontend does not call `/inference`, submit duplicate sensor payloads, control PWM, or fabricate telemetry.
