export type Live = {
  timestamp: string;
  battery_state?: any;
  driving_behavior?: any;
  road_condition?: any;
  attribution?: any;
  protection?: any;
  sequence?: number;
  [key: string]: any;
};
export type Snapshot = {
  battery: any | null;
  imu: any | null;
  road: any | null;
  fused: Live | null;
  sequence: number;
};
const base =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000/api/v1";
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok)
    throw new Error(
      (await res.json().catch(() => ({}))).detail ||
        `Request failed (${res.status})`,
    );
  return res.json();
}
export const startSession = () => api<any>("/sessions", { method: "POST" });
export const stopSession = (id: string) =>
  api<any>(`/sessions/${id}/stop`, { method: "POST" });
export const getSessions = () => api<any>("/sessions?limit=20");
export const getActiveSession = () => api<any | null>("/sessions/active");
export const getReport = (id: string) => api<any>(`/sessions/${id}/report`);
export const getLiveInference = () => api<Live | null>("/live/inference");
export function liveUrl() {
  const configured = process.env.NEXT_PUBLIC_WS_URL;
  if (configured) return configured;
  return base.replace(/^http/, "ws") + "/live/ws";
}
