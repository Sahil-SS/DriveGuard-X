"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  getReport,
  getActiveSession,
  getSessions,
  liveUrl,
  startSession,
  stopSession,
  Snapshot,
} from "./api";
import { Gauge, Trend } from "./visuals";

const labels: Record<string, string> = {
  normal: "Normal operation",
  driver_induced: "Driver-induced stress",
  road_induced: "Road-induced stress",
  battery_degraded: "Battery degraded",
  mixed_or_high_stress: "Mixed / high stress",
};
const fmt = (value: any) =>
  value ? new Date(value).toLocaleTimeString() : "—";
export default function Dashboard() {
  const [snap, setSnap] = useState<Snapshot>({
    battery: null,
    imu: null,
    road: null,
    fused: null,
    sequence: 0,
  });
  const [connected, setConnected] = useState(false);
  const [session, setSession] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [report, setReport] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [samples, setSamples] = useState<Snapshot[]>([]);
  const [now, setNow] = useState(Date.now());
  const socket = useRef<WebSocket | null>(null);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let closed = false;
    let retry = 1000;
    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(liveUrl());
      socket.current = ws;
      ws.onopen = () => {
        setConnected(true);
        retry = 1000;
      };
      ws.onmessage = (e) => {
        const next = JSON.parse(e.data) as Snapshot;
        setSnap(next);
        if (next.fused) setSamples((current) => [...current.slice(-35), next]);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) {
          setTimeout(connect, retry);
          retry = Math.min(retry * 2, 10000);
        }
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      closed = true;
      socket.current?.close();
    };
  }, []);
  useEffect(() => {
    Promise.all([getSessions(), getActiveSession()])
      .then(([sessions, active]) => {
        setHistory(sessions.items || []);
        if (active?.status === "recording") setSession(active);
      })
      .catch(() => setHistory([]));
  }, []);
  const fused = snap.fused;
  const age = fused
    ? Math.round((now - new Date(fused.timestamp).getTime()) / 1000)
    : null;
  const health = useMemo(
    () =>
      !fused
        ? "Unavailable"
        : fused.battery_state?.stressed
          ? "Warning"
          : "Healthy",
    [fused],
  );
  async function begin() {
    if (busy || session) return;
    setBusy(true);
    try {
      setSession(await startSession());
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function end() {
    if (!session || busy) return;
    setBusy(true);
    try {
      const ended = await stopSession(session.session_id);
      setSession({ ...session, ...ended, status: "finalized" });
      let r;
      for (let attempt = 0; attempt < 20; attempt += 1) {
        r = await getReport(session.session_id);
        if (r.report_status === "completed" || r.report_status === "failed") break;
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
      }
      setReport(r);
      setHistory((await getSessions()).items || []);
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="cockpit">
      <header className="topbar">
        <div>
          <div className="eyebrow">DRIVEGUARD-X / VEHICLE INSTRUMENTS</div>
          <h1>4WD protection console</h1>
          <span className="muted">
            Backend-owned inference · no motor control
          </span>
        </div>
        <div className="status">
          <i className={`dot ${connected ? "live" : ""}`} />
          {connected ? "LIVE STREAM" : "DISCONNECTED"}
          {age !== null && (
            <span className={age > 5 ? "warning" : "muted"}>· {age}s old</span>
          )}
        </div>
      </header>
      <section className="grid">
        <article className="panel chart-panel wide">
          <div className="panel-heading"><div><div className="eyebrow">LIVE TREND / BATTERY AI</div><h2>State of health trajectory</h2></div><span className="live-badge">{samples.length} EVENTS</span></div>
          <Trend values={samples.map((item) => Number(item.fused?.battery_state?.soh_percent || 0))} />
          <div className="axis"><span>HISTORICAL</span><span>NOW</span></div>
        </article>
        <article className="panel chart-panel wide">
          <div className="panel-heading"><div><div className="eyebrow">LIVE TREND / FUSION</div><h2>Attribution confidence</h2></div><span className="metric-small">{fused?.attribution?.confidence != null ? `${Math.round(fused.attribution.confidence * 100)}%` : "--"}</span></div>
          <Trend amber values={samples.map((item) => Number(item.fused?.attribution?.confidence || 0) * 100)} />
          <div className="axis"><span>LOW CONFIDENCE</span><span>HIGH CONFIDENCE</span></div>
        </article>
        <article className="panel source-panel wide">
          <div className="eyebrow">ESP32 SENSOR NETWORK</div><h2>Live source availability</h2>
          <div className="source-lamps"><span><i className={`lamp ${snap.battery ? "on" : "off"}`} />Battery AI</span><span><i className={`lamp ${snap.imu ? "on" : "off"}`} />IMU / driving</span><span><i className={`lamp ${snap.road ? "on" : "off"}`} />Road audio</span><span><i className={`lamp ${fused ? "on" : "off"}`} />Fusion engine</span></div>
        </article>
        <article className="panel wide">
          <div className="row">
            <h2>BATTERY STATE OF HEALTH</h2>
            <span className={health === "Warning" ? "warning" : "healthy"}>
              {health}
            </span>
          </div>
          <div className="battery-visual"><Gauge value={fused?.battery_state?.soh_percent ?? null} label="SOH" warning={Boolean(fused?.battery_state?.stressed)} /><div><div className="metric">{fused?.battery_state?.soh_percent ?? "—"}<span className="unit">%</span></div><span className="muted">Live model output</span></div></div>
          <div className="bar">
            <i
              style={{
                width: `${Math.min(100, Math.max(0, fused?.battery_state?.soh_percent || 0))}%`,
              }}
            />
          </div>
          <span className="muted">
            Last processed {fmt(snap.battery?.timestamp)} · stress{" "}
            {fused?.battery_state?.stressed == null
              ? "unavailable"
              : fused.battery_state.stressed
                ? "detected"
                : "not detected"}
          </span>
        </article>
        <article className="panel wide">
          <h2>CAUSE ATTRIBUTION</h2>
          <div
            className={`metric ${fused?.attribution?.likely_cause !== "normal" ? "warning" : "healthy"}`}
          >
            {labels[fused?.attribution?.likely_cause] || "Unavailable"}
          </div>
          <span className="muted">
            Confidence{" "}
            {fused?.attribution?.confidence != null
              ? `${Math.round(fused.attribution.confidence * 100)}%`
              : "—"}
          </span>
        </article>
        <article className="panel third">
          <h2>DRIVING BEHAVIOR</h2>
          <div className="metric">{fused?.driving_behavior?.class || "—"}</div>
          <span className="muted">
            Class ID {fused?.driving_behavior?.raw_class_id ?? "—"} ·{" "}
            {fmt(snap.imu?.timestamp)}
          </span>
        </article>
        <article className="panel third">
          <h2>ROAD CONDITION</h2>
          <div className="metric">{fused?.road_condition?.class || "—"}</div>
          <span className="muted">
            Confidence{" "}
            {fused?.road_condition?.confidence != null
              ? `${Math.round(fused.road_condition.confidence * 100)}%`
              : "—"}
          </span>
        </article>
        <article className="panel third">
          <h2>PROTECTION DECISION</h2>
          <div className="metric">
            {fused?.protection?.pwm_limit_pct ?? "—"}
            <span className="unit">PWM %</span>
          </div>
          <span
            className={
              fused?.protection?.pwm_limit_pct < 100 ? "warning" : "healthy"
            }
          >
            {fused?.protection?.action || "Unavailable"}
          </span>
        </article>
        <article className="panel wide">
          <div className="row">
            <div>
              <h2>RECORDING</h2>
              <span className="muted">
                {session
                  ? `${session.status} · ${session.session_id}`
                  : "No active session"}
              </span>
            </div>
            <div className="controls">
              <button onClick={begin} disabled={busy || !!session}>
                Start
              </button>
              <button onClick={end} disabled={busy || !session}>
                Stop
              </button>
            </div>
          </div>
          <p className="muted">
            {session
              ? "Backend snapshots are being saved to MongoDB."
              : "Recording follows the existing inference stream."}
          </p>
        </article>
        <article className="panel wide">
          <h2>RECENT EVENTS / LIMITATIONS</h2>
          <ul className="events">
            <li>
              Attribution:{" "}
              {fused
                ? labels[fused.attribution?.likely_cause] || "Unknown"
                : "No fused event available"}
            </li>
            <li>Road: {fused?.road_condition?.class || "Unavailable"}</li>
            <li>
              Hardware action is not claimed unless confirmed by backend
              telemetry.
            </li>
          </ul>
        </article>
        {report?.report && (
          <article className="panel history">
            <h2>DRIVER REPORT · {report.report_status}</h2>
            <div className="report">
              {JSON.stringify(report.report, null, 2)}
            </div>
          </article>
        )}
        <article className="panel history">
          <h2>SESSION HISTORY</h2>
          <table>
            <thead>
              <tr>
                <th>Session</th>
                <th>Started</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Report</th>
              </tr>
            </thead>
            <tbody>
              {history.map((x) => (
                <tr key={x.session_id}>
                  <td>{x.session_id.slice(0, 8)}…</td>
                  <td>{fmt(x.started_at)}</td>
                  <td>
                    {x.duration_seconds
                      ? `${Math.round(x.duration_seconds)}s`
                      : "—"}
                  </td>
                  <td>{x.status}</td>
                  <td>
                    <button
                      onClick={() => getReport(x.session_id).then(setReport)}
                    >
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>
      </section>
    </main>
  );
}
