import { useEffect, useRef, useState } from "react";
import type { Series } from "./api/types";
import { fmt, fmtInt } from "./lib/format";

export type ChartAxis = "step" | "elapsed";

/** Keep offscreen SVGs unmounted, including charts inside collapsed sections. */
export function MetricChart({ name, label, data, smoothing, color, axis, comparisonData, comparisonLabel, onRemove, description, stepLabel = "Logged step" }: {
  description?: string; stepLabel?: string;
  name: string; label: string; data: Series | null; smoothing: number;
  color: string; axis: ChartAxis; comparisonData: Series | null; comparisonLabel: string; onRemove?: () => void;
}) {
  const host = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: "200px" });
    if (host.current) observer.observe(host.current);
    const resize = new ResizeObserver(([entry]) => setWidth(Math.max(200, entry.contentRect.width)));
    if (host.current) resize.observe(host.current);
    return () => { observer.disconnect(); resize.disconnect(); };
  }, []);
  useEffect(() => { setRange(null); setHover(null); }, [axis]);
  const collect = (source: Series | null) => {
  const values = source?.series[name] ?? [];
  const firstTime = source?.timestamps.find((stamp) => stamp !== null);
  const origin = firstTime ? Date.parse(firstTime) : NaN;
  return values.flatMap((value, index) => {
    const stamp = source?.timestamps[index];
    const x = axis === "step" ? source?.steps[index] : stamp ? (Date.parse(stamp) - origin) / 60_000 : null;
    if (value === null || !Number.isFinite(value) || x == null || !Number.isFinite(x)) return [];
    const window = values.slice(Math.max(0, index - smoothing + 1), index + 1).filter((v): v is number => v !== null && Number.isFinite(v));
    return [{ x, value, min: source?.envelope?.[name]?.min[index] ?? value, max: source?.envelope?.[name]?.max[index] ?? value, smooth: window.reduce((a, b) => a + b, 0) / window.length, step: source?.steps[index] }];
  });
  };
  const points = collect(data);
  const comparison = collect(comparisonData);
  const shown = points.filter((p) => !range || (p.x >= range[0] && p.x <= range[1]));
  const compareShown = comparison.filter((p) => !range || (p.x >= range[0] && p.x <= range[1]));
  const height = expanded ? 360 : 180;
  const pad = { left: 52, right: 16, top: 18, bottom: 30 };
  const xs = [...shown, ...compareShown].map((p) => p.x), ys = [...shown, ...compareShown].flatMap((p) => [p.min, p.max, p.value, p.smooth]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymin === ymax) { ymin -= Math.abs(ymin || 1) * .1; ymax += Math.abs(ymax || 1) * .1; }
  const x = (v: number) => pad.left + (v - xmin) / (xmax - xmin || 1) * (width - pad.left - pad.right);
  const y = (v: number) => pad.top + (1 - (v - ymin) / (ymax - ymin || 1)) * (height - pad.top - pad.bottom);
  const path = (key: "value" | "smooth") => shown.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p[key])}`).join(" ");
  const cursor = hover === null ? null : shown.reduce<typeof shown[number] | null>((best, p) => !best || Math.abs(p.x - hover) < Math.abs(best.x - hover) ? p : best, null);
  const coordinate = (event: React.PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return Math.min(xmax, Math.max(xmin, xmin + ((event.clientX - bounds.left) / bounds.width * width - pad.left) / (width - pad.left - pad.right) * (xmax - xmin)));
  };
  const download = () => {
    const csv = "step,x,value,smoothed,min,max\n" + points.map((p) => `${p.step ?? ""},${p.x},${p.value},${p.smooth},${p.min},${p.max}`).join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const link = document.createElement("a"); link.href = url; link.download = `${name.replace(/\//g, "-")}.csv`; link.click(); URL.revokeObjectURL(url);
  };
  return <article data-metric={name} className={`chart-card${expanded ? " chart-expanded" : ""}`} ref={host}>
    <header><div><span title={name}>{label}</span></div><div className="chart-latest">{(name.endsWith("/count") ? fmtInt : fmt)(points[points.length - 1]?.value ?? null)}</div>
      <button type="button" title="Download raw and smoothed values" aria-label={`Download ${name}`} onClick={download}>CSV</button>
      <button type="button" aria-label={`Expand ${name}`} onClick={() => setExpanded(!expanded)}>{expanded ? "−" : "+"}</button>
      {onRemove && <button type="button" aria-label={`Remove ${name}`} onClick={onRemove}>×</button>}
    </header>
    {description && <p className="chart-context">{description}</p>}
    {visible && shown.length ? <>
      <svg style={{height}} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${name} chart`} tabIndex={0}
        onPointerMove={(e) => setHover(coordinate(e))} onPointerLeave={() => { if (drag === null) setHover(null); }}
        onPointerDown={(e) => { if (e.button === 0) { setDrag(coordinate(e)); e.currentTarget.setPointerCapture(e.pointerId); } }}
        onPointerUp={(e) => { const end = coordinate(e); if (drag !== null && Math.abs(end - drag) > (xmax - xmin) * .02) { const next: [number, number] = [Math.min(drag, end), Math.max(drag, end)]; if (shown.some((p) => p.x >= next[0] && p.x <= next[1])) setRange(next); } setDrag(null); }}
        onPointerCancel={() => setDrag(null)} onDoubleClick={() => setRange(null)}
        onKeyDown={(e) => { if (e.key === "Escape") setRange(null); if (["ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); const i = cursor ? shown.indexOf(cursor) : -1; setHover(shown[Math.max(0, Math.min(shown.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)))].x); } }}>
        {[0, .5, 1].map((fraction) => <g key={fraction}><line className="grid-line" x1={pad.left} x2={width - pad.right} y1={pad.top + fraction * (height - pad.top - pad.bottom)} y2={pad.top + fraction * (height - pad.top - pad.bottom)} /><text x={pad.left - 8} y={pad.top + fraction * (height - pad.top - pad.bottom) + 4}>{fmt(ymax - fraction * (ymax - ymin), 2)}</text></g>)}
        {data?.downsampled && <path className="metric-envelope" style={{ fill: color }} opacity={.15} d={shown.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.max)}`).join(" ") + " " + [...shown].reverse().map((p) => `L${x(p.x)},${y(p.min)}`).join(" ") + " Z"} />}
        {smoothing > 1 && <path className="metric-line raw-line" style={{ stroke: color }} d={path("value")} />}
        <path className="metric-line main-line" pathLength={1} style={{ stroke: color }} d={path("smooth")} />
        {compareShown.length > 0 && <path className="metric-line comparison-line" d={compareShown.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.smooth)}`).join(" ")} />}
        {shown.length === 1 && <circle cx={x(shown[0].x)} cy={y(shown[0].smooth)} r={4} style={{ fill: color }}><title>{`step ${shown[0].step}: ${shown[0].value}`}</title></circle>}
        {cursor && <g className="chart-cursor"><line x1={x(cursor.x)} x2={x(cursor.x)} y1={pad.top} y2={height - pad.bottom} /><circle cx={x(cursor.x)} cy={y(cursor.smooth)} r={4} style={{ fill: color }} /></g>}
        {drag !== null && hover !== null && <rect className="chart-selection" x={x(Math.min(drag, hover))} y={pad.top} width={Math.abs(x(hover) - x(drag))} height={height - pad.top - pad.bottom} />}
        <text x={pad.left} y={height - 6}>{fmt(xmin, 1)}</text><text className="x-end" x={width - pad.right} y={height - 6}>{fmt(xmax, 1)}</text>
      </svg>
      {comparisonLabel && <div className="chart-legend" title={comparisonLabel}>Dashed: {comparisonLabel}{!compareShown.length ? " · no matching metric" : ""}</div>}
      <div className="chart-readout" aria-live="polite">{cursor ? `${stepLabel} ${cursor.step ?? "–"} · ${data?.downsampled ? "bucket mean" : "raw"} ${fmt(cursor.value, 5)}${data?.downsampled ? ` · min ${fmt(cursor.min, 5)} / max ${fmt(cursor.max, 5)}` : ""}${smoothing > 1 ? ` · smoothed ${fmt(cursor.smooth, 5)}` : ""}` : `${axis === "step" ? stepLabel : "Elapsed minutes"} · drag to zoom`}{range && <button type="button" onClick={() => setRange(null)}>Reset zoom</button>}</div>
    </> : <div className="chart-empty">{visible ? "no data yet" : "Chart loads when visible"}</div>}
  </article>;
}
