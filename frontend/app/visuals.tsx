"use client";

export function Gauge({ value, label, warning = false }: { value: number | null; label: string; warning?: boolean }) {
  const safe = Math.max(0, Math.min(100, value ?? 0));
  const circumference = 2 * Math.PI * 46;
  return <div className="gauge-wrap"><svg className="gauge" viewBox="0 0 120 120"><circle className="gauge-track" cx="60" cy="60" r="46" /><circle className={`gauge-value ${warning ? "amber" : "cyan"}`} cx="60" cy="60" r="46" strokeDasharray={`${circumference * safe / 100} ${circumference}`} /></svg><div className="gauge-number">{value == null ? "--" : Math.round(value)}<small>%</small></div><div className="gauge-label">{label}</div></div>;
}

export function Trend({ values, amber = false }: { values: number[]; amber?: boolean }) {
  const data = values.slice(-48);
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const path = data.map((value, index) => `${index ? "L" : "M"}${(index / Math.max(data.length - 1, 1)) * 100},${92 - ((value - min) / range) * 78}`).join(" ");
  return <div className="trend-wrap"><svg className={`trend ${amber ? "amber" : "cyan"}`} viewBox="0 0 100 100" preserveAspectRatio="none"><path className="trend-fill" d={`${path} L100,100 L0,100 Z`} /><path className="trend-line" d={path} /></svg><span className="trend-last">{data.length ? Math.round(data[data.length - 1]) : "--"}</span></div>;
}

export function LiveMeter({ value, max, label, unit }: { value: number | null; max: number; label: string; unit: string }) {
  const safe = Math.max(0, Math.min(100, ((value ?? 0) / max) * 100));
  const angle = -90 + safe * 1.8;
  const dash = 157 * safe / 100;
  return <div className="live-meter">
    <div className="meter-head"><span>{label}</span><strong>{value == null ? "--" : value.toFixed(1)} {unit}</strong></div>
    <svg className="dial" viewBox="0 0 120 92" aria-label={`${label}: ${value ?? "unavailable"} ${unit}`}>
      <path className="dial-track" d="M10 80 A50 50 0 0 1 110 80" />
      <path className="dial-value" d="M10 80 A50 50 0 0 1 110 80" style={{ strokeDasharray: `${dash} 157` }} />
      <line className="dial-needle" x1="60" y1="80" x2="60" y2="31" style={{ transform: `rotate(${angle}deg)` }} />
      <circle className="dial-hub" cx="60" cy="80" r="4" />
      <text x="8" y="91">0</text><text x="56" y="16">{(max / 2).toFixed(0)}</text><text x="101" y="91">{max}</text>
    </svg>
  </div>;
}
