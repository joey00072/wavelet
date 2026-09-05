import { useEffect, useMemo, useRef, useState } from "react";

import type { LineSeries, Point } from "../lib/series";
import { extent, logTicks, niceTicks, smooth, tickFormatter, type SmoothingType } from "../lib/series";
import { fmt, fmtAxis } from "../lib/format";
import { seriesColor } from "../lib/theme";
import { useMeasure } from "./useMeasure";

export type Reference = { y: number; label: string; colorIndex?: number };

const defaultYFormat = fmtAxis;
const defaultXFormat = (x: number) => (Number.isInteger(x) ? String(x) : x.toFixed(1));


export function LineChart({
  series,
  height = 200,
  xLabel = "step",
  yFormat = defaultYFormat,
  xFormat = defaultXFormat,
  logScale: logScaleProp = false,
  xLog = false,
  smoothing: smoothingProp = 0,
  smoothingType = "ema",
  yMode = "auto",
  yMin,
  yMax,
  yDomain,
  references = [],
  showLegend,
  emptyText = "No data yet",
  area = false,
  markers = false,
  controls = true,
  xExtent,
}: {
  series: LineSeries[];
  height?: number;
  xLabel?: string;
  yFormat?: (value: number) => string;
  xFormat?: (value: number) => string;
  logScale?: boolean;
  xLog?: boolean;
  smoothing?: number;
  smoothingType?: SmoothingType;
  /** auto fits the data, zero includes the origin, fixed uses yMin/yMax. */
  yMode?: "auto" | "zero" | "fixed";
  yMin?: number | null;
  yMax?: number | null;
  yDomain?: [number, number];
  references?: Reference[];
  showLegend?: boolean;
  emptyText?: string;
  area?: boolean;
  markers?: boolean;
  controls?: boolean;
  /** Minimum x range to show even when the data covers less (e.g. a single eval point). */
  xExtent?: [number, number];
}) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const logScale = logScaleProp;
  const smoothing = smoothingProp;
  const [xDomain, setXDomain] = useState<[number, number] | null>(null);
  const [drag, setDrag] = useState<{ start: number; current: number } | null>(null);

  const prepared = useMemo(
    () =>
      series
        .filter((s) => !hidden.has(s.id) && s.points.length > 0)
        .map((s) => {
          const base = xLog ? s.points.filter((p) => p.x > 0) : s.points;
          const envelope = s.envelope ? {
            min: s.envelope.min.filter((p) => !xLog || p.x > 0),
            max: s.envelope.max.filter((p) => !xLog || p.x > 0),
          } : undefined;
          return { ...s, envelope, raw: base, points: smoothing > 0 ? smooth(base, smoothingType, smoothing) : base };
        })
        .filter((s) => s.points.length > 0),
    [series, hidden, smoothing, smoothingType, xLog],
  );
  const visible = useMemo(() => {
    if (!xDomain) return prepared;
    return prepared
      .map((s) => ({
        ...s,
        points: s.points.filter((p) => p.x >= xDomain[0] && p.x <= xDomain[1]),
        raw: s.raw.filter((p) => p.x >= xDomain[0] && p.x <= xDomain[1]),
        envelope: s.envelope ? {
          min: s.envelope.min.filter((p) => p.x >= xDomain[0] && p.x <= xDomain[1]),
          max: s.envelope.max.filter((p) => p.x >= xDomain[0] && p.x <= xDomain[1]),
        } : undefined,
      }))
      .filter((s) => s.points.length > 0);
  }, [prepared, xDomain]);

  const legend = showLegend ?? series.length > 1;
  const margin = { top: 10, right: 16, bottom: 26, left: 52 };
  const width = Math.max(size.width, 120);
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const scales = useMemo(() => {
    // Fit axes to the raw data so smoothing changes the line, never the scale.
    const xs = visible.flatMap((s) => s.raw.map((p) => p.x));
    let ys = visible.flatMap((s) => s.raw.map((p) => p.y));
    ys = ys.concat(visible.flatMap((s) => [...(s.envelope?.min ?? []), ...(s.envelope?.max ?? [])].map((p) => p.y)));
    ys = ys.concat(references.map((r) => r.y));
    if (logScale) ys = ys.filter((y) => y > 0);
    let xe = xDomain ?? extent(xs);
    let ye = yDomain ?? extent(ys);
    if (!xe || !ye) return null;
    if (!xDomain && xExtent) xe = [Math.min(xe[0], xExtent[0]), Math.max(xe[1], xExtent[1])];
    if (!yDomain) {
      if (yMode === "fixed") {
        ye = [yMin ?? ye[0], yMax ?? ye[1]];
      } else if (yMode === "zero" && !logScale) {
        ye = [Math.min(0, ye[0]), Math.max(0, ye[1])];
      }
      if (ye[0] === ye[1]) ye = [ye[0] - Math.abs(ye[0]) * 0.1 - 0.5, ye[1] + Math.abs(ye[1]) * 0.1 + 0.5];
      if (yMode !== "fixed" && !logScale) {
        const pad = (ye[1] - ye[0]) * 0.06;
        ye = [ye[0] === 0 ? 0 : ye[0] - pad, ye[1] === 0 ? 0 : ye[1] + pad];
      }
    } else if (ye[0] === ye[1]) {
      ye = [ye[0] - 0.5, ye[1] + 0.5];
    }
    const useXLog = xLog && xe[0] > 0;
    const x0 = xe[0] === xe[1] ? (useXLog ? xe[0] / 2 : xe[0] - 1) : xe[0];
    const x1 = xe[0] === xe[1] ? (useXLog ? xe[1] * 2 : xe[1] + 1) : xe[1];
    const lx0 = Math.log10(Math.max(x0, 1e-12));
    const lx1 = Math.log10(Math.max(x1, 1e-12));
    const sx = useXLog
      ? (x: number) => margin.left + ((Math.log10(Math.max(x, 1e-12)) - lx0) / Math.max(lx1 - lx0, 1e-9)) * plotW
      : (x: number) => margin.left + ((x - x0) / (x1 - x0)) * plotW;
    const xOf = useXLog
      ? (px: number) => 10 ** (lx0 + ((px - margin.left) / plotW) * (lx1 - lx0))
      : (px: number) => x0 + ((px - margin.left) / plotW) * (x1 - x0);
    const lo = Math.log10(Math.max(ye[0], 1e-12));
    const hi = Math.log10(Math.max(ye[1], 1e-12));
    const sy = logScale
      ? (y: number) => margin.top + plotH - ((Math.log10(Math.max(y, 1e-12)) - lo) / Math.max(hi - lo, 1e-9)) * plotH
      : (y: number) => margin.top + plotH - ((y - ye![0]) / (ye![1] - ye![0])) * plotH;
    const allInt = !logScale && ys.length > 0 && ys.every((y) => Number.isInteger(y));
    let yTicks = logScale ? logTicks(ye[0], ye[1], Math.max(3, Math.floor(plotH / 34))) : niceTicks(ye[0], ye[1], Math.max(3, Math.floor(plotH / 34)));
    if (allInt) yTicks = [...new Set(yTicks.map((t) => Math.round(t)))].filter((t) => t >= ye![0] && t <= ye![1]);
    const xTickCount = Math.max(3, Math.min(8, Math.floor(plotW / 80)));
    const xTicks = useXLog ? logTicks(x0, x1, xTickCount) : niceTicks(x0, x1, xTickCount);
    return { sx, sy, x0, x1, ye, yTicks, xTicks, xOf, yTickFormat: tickFormatter(yTicks), xTickFormat: tickFormatter(xTicks) };
  }, [visible, references, logScale, xLog, yMode, yMin, yMax, yDomain, xDomain, xExtent, plotW, plotH, margin.left, margin.top]);

  const hover = useMemo(() => {
    if (!scales || hoverX === null || drag) return null;
    const xValue = scales.xOf(hoverX);
    let best: number | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const s of visible) {
      for (const p of s.points) {
        const d = Math.abs(p.x - xValue);
        if (d < bestDistance) {
          bestDistance = d;
          best = p.x;
        }
      }
    }
    if (best === null) return null;
    return { x: best, rows: visible.map((s) => ({ series: s, point: s.points.find((p) => p.x === best) ?? null })) };
  }, [scales, hoverX, visible, drag]);

  // Draw-in animation keyed by the set of series, not by every poll.
  const seriesKey = series.map((s) => s.id).join("|");
  const clipId = useRef(`clip-${Math.random().toString(36).slice(2, 9)}`).current;
  const [drawn, setDrawn] = useState(false);
  useEffect(() => {
    // Re-run only when the set of series changes; StrictMode's double invoke is harmless here.
    setDrawn(false);
    const timer = window.setTimeout(() => setDrawn(true), 30);
    return () => window.clearTimeout(timer);
  }, [seriesKey]);

  if (!scales) {
    return (
      <div ref={ref} className="flex items-center justify-center text-xs text-muted" style={{ height }}>
        {emptyText}
      </div>
    );
  }

  const { sx, sy, yTicks, xTicks, yTickFormat, xTickFormat } = scales;
  const axisY = yFormat === defaultYFormat ? yTickFormat : yFormat;
  const axisX = xFormat === defaultXFormat ? (v: number) => (Number.isInteger(v) ? String(v) : xTickFormat(v)) : xFormat;
  const tooltipLeft = hover ? Math.min(sx(hover.x) + 12, width - 190) : 0;
  const pixelX = (e: React.PointerEvent<SVGSVGElement>) => e.clientX - e.currentTarget.getBoundingClientRect().left;

  return (
    <div ref={ref} className="group relative w-full select-none">
      {controls && xDomain && (
        <button type="button" className="absolute right-1 top-0 z-10 rounded bg-raised/90 px-2 py-0.5 text-[10.5px] text-ink2 backdrop-blur transition-colors hover:text-ink" onClick={(e) => { e.stopPropagation(); setXDomain(null); }} title="Reset zoom (or double-click)">
          reset zoom
        </button>
      )}
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${series.map((item) => item.label).join(", ") || "Metric"} chart`}
        className={drag ? "cursor-col-resize" : "cursor-crosshair"}
        onPointerDown={(e) => {
          const x = pixelX(e);
          if (x >= margin.left && x <= width - margin.right) setDrag({ start: x, current: x });
        }}
        onPointerMove={(e) => {
          const x = pixelX(e);
          setHoverX(x);
          if (drag) setDrag({ start: drag.start, current: x });
        }}
        onPointerUp={() => {
          if (drag && Math.abs(drag.current - drag.start) > 6) {
            const a = scales.xOf(Math.min(drag.start, drag.current));
            const b = scales.xOf(Math.max(drag.start, drag.current));
            setXDomain([a, b]);
          }
          setDrag(null);
        }}
        onPointerLeave={() => {
          setHoverX(null);
          setDrag(null);
        }}
        onDoubleClick={() => setXDomain(null)}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={margin.left} x2={width - margin.right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={margin.left - 6} y={sy(t)} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--ink-muted)" className="tabular">
              {axisY(t)}
            </text>
          </g>
        ))}
        <line x1={margin.left} x2={width - margin.right} y1={margin.top + plotH} y2={margin.top + plotH} stroke="var(--axis)" strokeWidth={1} />
        {xTicks.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={height - 8} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" className="tabular">
            {axisX(t)}
          </text>
        ))}
        {xTicks.length > 0 && sx(xTicks[xTicks.length - 1]) < width - margin.right - 34 && (
          <text x={width - margin.right} y={height - 8} textAnchor="end" fontSize={9} fill="var(--ink-muted)" opacity={0.7}>
            {xLabel}
          </text>
        )}
        {staggerLabels(references.map((r) => ({ ...r, py: sy(r.y) }))).map((r) => (
          <g key={`ref-${r.label}`}>
            <line x1={margin.left} x2={width - margin.right} y1={r.py} y2={r.py} stroke={r.colorIndex !== undefined ? seriesColor(r.colorIndex) : "var(--ink-muted)"} strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
            <text x={width - margin.right} y={r.labelY} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {r.label}
            </text>
          </g>
        ))}
        <defs>
          <clipPath id={clipId}>
            <rect x={margin.left} y={margin.top - 1} width={drawn ? Math.max(plotW, 0) : 0} height={plotH + 2} style={{ transition: "width 700ms cubic-bezier(.4,0,.2,1)" }} />
          </clipPath>
        </defs>
        <g clipPath={`url(#${clipId})`}>
          {visible.map((s) => {
            const color = s.deemphasize ? "var(--deemph)" : seriesColor(s.colorIndex ?? 0);
            const path = linePath(s.points, sx, sy);
            const band = s.envelope ? envelopePath(s.envelope.min, s.envelope.max, sx, sy) : null;
            const areaPath = area ? `${path} L${sx(s.points[s.points.length - 1].x)},${margin.top + plotH} L${sx(s.points[0].x)},${margin.top + plotH} Z` : null;
            const showMarkers = markers || s.points.length <= 40;
            return (
              <g key={s.id}>
                {band && <path d={band} fill={color} opacity={0.1} />}
                {areaPath && <path d={areaPath} fill={color} opacity={0.1} />}
                {smoothing > 0 && <path d={linePath(s.raw, sx, sy)} fill="none" stroke={color} strokeWidth={1} opacity={0.16} strokeLinejoin="round" strokeLinecap="round" />}
                <path d={path} fill="none" stroke={color} strokeWidth={1.75} strokeLinejoin="round" strokeLinecap="round" />
                {showMarkers &&
                  s.points.map((p) => (
                    <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={s.points.length <= 40 ? 3 : 2} fill={color} stroke="var(--surface-1)" strokeWidth={1.5} />
                  ))}
              </g>
            );
          })}
        </g>
        {drag && Math.abs(drag.current - drag.start) > 2 && (
          <rect x={Math.min(drag.start, drag.current)} y={margin.top} width={Math.abs(drag.current - drag.start)} height={plotH} fill="var(--series-1)" opacity={0.12} rx={3} />
        )}
        {hover && (
          <g>
            <line x1={sx(hover.x)} x2={sx(hover.x)} y1={margin.top} y2={margin.top + plotH} stroke="var(--ink-muted)" strokeWidth={1} />
            {hover.rows.map(({ series: s, point }) =>
              point ? (
                <circle key={s.id} cx={sx(point.x)} cy={sy(point.y)} r={4} fill={s.deemphasize ? "var(--deemph)" : seriesColor(s.colorIndex ?? 0)} stroke="var(--surface-1)" strokeWidth={2} />
              ) : null,
            )}
          </g>
        )}
      </svg>
      {hover && (
        <div className="animate-fade pointer-events-none absolute top-7 z-10 rounded-md bg-raised/95 px-2.5 py-1.5 text-[11px] shadow-lg backdrop-blur" style={{ left: tooltipLeft, minWidth: 150 }}>
          <div className="mb-1 text-muted">
            {xLabel} {xFormat(hover.x)}
          </div>
          {hover.rows.map(({ series: s, point }) => (
            <div key={s.id} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-1.5 truncate text-ink2">
                <span className="inline-block h-0.5 w-3 rounded" style={{ background: s.deemphasize ? "var(--deemph)" : seriesColor(s.colorIndex ?? 0) }} />
                <span className="truncate" style={{ maxWidth: 140 }}>{s.label}</span>
              </span>
              <span className="tabular font-semibold text-ink">{point ? yFormat(point.y) : "–"}</span>
            </div>
          ))}
        </div>
      )}
      {legend && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 px-1">
          {series.map((s) => {
            const off = hidden.has(s.id);
            return (
              <button
                key={s.id}
                type="button"
                className={`flex items-center gap-1.5 text-[11px] transition-opacity ${off ? "text-muted line-through opacity-60" : "text-ink2"}`}
                onClick={() =>
                  setHidden((prev) => {
                    const next = new Set(prev);
                    if (next.has(s.id)) next.delete(s.id);
                    else next.add(s.id);
                    return next;
                  })
                }
                title="Toggle series"
              >
                <span className="inline-block h-0.5 w-3 rounded" style={{ background: s.deemphasize ? "var(--deemph)" : seriesColor(s.colorIndex ?? 0), opacity: off ? 0.4 : 1 }} />
                <span className="truncate" style={{ maxWidth: 220 }}>{s.label}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function staggerLabels<T extends { py: number }>(items: T[]): Array<T & { labelY: number }> {
  const sorted = [...items].sort((a, b) => a.py - b.py);
  let last = Number.NEGATIVE_INFINITY;
  return sorted.map((item) => {
    const labelY = Math.max(item.py - 3, last + 11);
    last = labelY;
    return { ...item, labelY };
  });
}

function linePath(points: Point[], sx: (x: number) => number, sy: (y: number) => number): string {
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
}

function envelopePath(min: Point[], max: Point[], sx: (x: number) => number, sy: (y: number) => number): string | null {
  if (min.length === 0 || max.length === 0) return null;
  const upper = linePath(max, sx, sy);
  const lower = [...min].reverse().map((point) => `L${sx(point.x).toFixed(1)},${sy(point.y).toFixed(1)}`).join(" ");
  return `${upper} ${lower} Z`;
}


export function SeriesTable({ series, xLabel = "step", yFormat = (v: number) => fmt(v, 3) }: { series: LineSeries[]; xLabel?: string; yFormat?: (v: number) => string }) {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
  const tail = xs.slice(-40);
  return (
    <div className="max-h-56 overflow-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="th">{xLabel}</th>
            {series.map((s) => (
              <th key={s.id} className="th">{s.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tail.map((x) => (
            <tr key={x} className="tr-hover">
              <td className="td tabular">{x}</td>
              {series.map((s) => {
                const p = s.points.find((q) => q.x === x);
                return (
                  <td key={s.id} className="td tabular">{p ? yFormat(p.y) : "–"}</td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
