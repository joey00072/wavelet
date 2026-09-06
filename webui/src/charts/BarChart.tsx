import { useState } from "react";

import { extent, niceTicks } from "../lib/series";
import { fmt, fmtAxis } from "../lib/format";
import { seriesColor } from "../lib/theme";
import { EmptyChart } from "./EmptyChart";
import { useMeasure } from "./useMeasure";

type BarCategory = { label: string; values: number[] };

/** Stacked column chart. `seriesLabels` names each stack layer in fixed categorical order; a single layer draws in slot 1 with no legend. */
export function BarChart({
  categories,
  seriesLabels,
  height = 180,
  yFormat = fmtAxis,
  maxBars = 80,
}: {
  categories: BarCategory[];
  seriesLabels: string[];
  height?: number;
  yFormat?: (v: number) => string;
  maxBars?: number;
}) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const cats = categories.slice(-maxBars);
  const margin = { top: 8, right: 12, bottom: 24, left: 44 };
  const width = Math.max(size.width, 120);
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const ye = extent([0, ...cats.map((c) => c.values.reduce((a, b) => a + b, 0))]);
  if (!ye || cats.length === 0) return <EmptyChart ref={ref} height={height} />;
  const yMax = ye[1] === 0 ? 1 : ye[1] * 1.05;
  const sy = (y: number) => margin.top + plotH - (y / yMax) * plotH;
  const slot = plotW / cats.length;
  const barW = Math.min(24, Math.max(2, slot * 0.7));
  const ticks = niceTicks(0, yMax, 4);
  const labelEvery = Math.max(1, Math.ceil(cats.length / Math.max(2, Math.floor(plotW / 60))));
  return (
    <div ref={ref} className="relative w-full select-none">
      <svg width={width} height={height} role="img" aria-label={`${seriesLabels.join(", ")} bar chart`} onPointerLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={margin.left} x2={width - margin.right} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={margin.left - 6} y={sy(t)} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--ink-muted)" className="tabular">
              {yFormat(t)}
            </text>
          </g>
        ))}
        <line x1={margin.left} x2={width - margin.right} y1={sy(0)} y2={sy(0)} stroke="var(--axis)" />
        {cats.map((c, i) => {
          const cx = margin.left + slot * i + slot / 2;
          let acc = 0;
          return (
            <g key={`${c.label}-${i}`} onPointerEnter={() => setHover(i)}>
              <rect x={margin.left + slot * i} y={margin.top} width={slot} height={plotH} fill="transparent" />
              {c.values.map((v, j) => {
                const top = sy(acc + v);
                const bottom = sy(acc);
                acc += v;
                const isTop = j === c.values.length - 1;
                const h = Math.max(0, bottom - top - (isTop ? 0 : 2));
                return <path key={j} d={roundedTop(cx - barW / 2, top, barW, h, isTop ? 3 : 0)} fill={seriesColor(j)} opacity={hover === null || hover === i ? 1 : 0.55} />;
              })}
              {i % labelEvery === 0 && (
                <text x={cx} y={height - 8} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" className="tabular">
                  {c.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hover !== null && cats[hover] && (
        <div className="pointer-events-none absolute top-1 z-10 rounded-md border border-edge bg-surface px-2 py-1.5 text-[11px] shadow-sm" style={{ left: Math.min(margin.left + slot * hover + slot / 2 + 10, width - 180), minWidth: 130 }}>
          <div className="mb-1 text-muted">{cats[hover].label}</div>
          {cats[hover].values.map((v, j) => (
            <div key={j} className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-1.5 text-ink2">
                <span className="inline-block h-2 w-2 rounded-sm" style={{ background: seriesColor(j) }} />
                {seriesLabels[j] ?? `series ${j + 1}`}
              </span>
              <span className="tabular font-semibold text-ink">{yFormat(v)}</span>
            </div>
          ))}
        </div>
      )}
      {seriesLabels.length > 1 && (
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 px-1">
          {seriesLabels.map((label, j) => (
            <span key={label} className="flex items-center gap-1.5 text-[11px] text-ink2">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ background: seriesColor(j) }} />
              {label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  if (h <= 0 || w <= 0) return "";
  const rr = Math.min(r, w / 2, h);
  return `M${x},${y + h} V${y + rr} Q${x},${y} ${x + rr},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr} V${y + h} Z`;
}

export function HistogramChart({ bins, counts, height = 120, format = fmt }: { bins: number[]; counts: number[]; height?: number; format?: (v: number) => string }) {
  if (!bins.length || !counts.length) return <EmptyChart height={height} text="No values" />;
  const categories = counts.map((count, i) => ({ label: format(bins.length === 2 ? bins[0] : bins[i]), values: [count] }));
  return <BarChart categories={categories} seriesLabels={["count"]} height={height} yFormat={(v) => String(Math.round(v))} maxBars={64} />;
}
