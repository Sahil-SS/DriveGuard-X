"use client";

export function Gauge({ value, label, warning = false }: { value: number | null; label: string; warning?: boolean }) {
  const safe = Math.max(0, Math.min(100, value ?? 0));
  const circumference = 2 * Math.PI * 46;
  return <div className="gauge-wrap"><svg className="gauge" viewBox="0 0 120 120"><circle className="gauge-track" cx="60" cy="60" r="46" /><circle className={`gauge-value ${warning ? "amber" : "cyan"}`} cx="60" cy="60" r="46" strokeDasharray={`${circumference * safe / 100} ${circumference}`} /></svg><div className="gauge-number">{value == null ? "--" : Math.round(value)}<small>%</small></div><div className="gauge-label">{label}</div></div>;
}

export function Trend({ values, amber = false }: { values: number[]; amber?: boolean }) {
  const data = values.slice(-36);
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const path = data.map((value, index) => `${index ? "L" : "M"}${(index / Math.max(data.length - 1, 1)) * 100},${92 - ((value - min) / range) * 78}`).join(" ");
  return <div className="trend-wrap"><svg className={`trend ${amber ? "amber" : "cyan"}`} viewBox="0 0 100 100" preserveAspectRatio="none"><path className="trend-fill" d={`${path} L100,100 L0,100 Z`} /><path className="trend-line" d={path} /></svg><span className="trend-last">{data.length ? Math.round(data[data.length - 1]) : "--"}</span></div>;
}
