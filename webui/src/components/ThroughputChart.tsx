import { useMemo } from "react";
import type { MetricRow, RolloutEvent } from "../types";
import { bucketEvents } from "../utils/metrics";

const LINES = [
  { key: "submitted" as const, color: "#3b82f6", label: "submitted" },
  { key: "completed" as const, color: "#06b6d4", label: "completed" },
  { key: "published" as const, color: "#10b981", label: "published" },
  { key: "consumed" as const, color: "#f59e0b", label: "trainer steps" },
];

export function ThroughputChart({ events, metrics }: { events: RolloutEvent[]; metrics: MetricRow[] }) {
  const buckets = useMemo(() => bucketEvents(events, metrics), [events, metrics]);
  const width = 900;
  const height = 200;
  const padLeft = 36;
  const padRight = 8;
  const padY = 16;
  const maxValue = Math.max(1, ...buckets.flatMap((b) => LINES.map((l) => b[l.key])));

  const x = (i: number) =>
    buckets.length <= 1
      ? padLeft + (width - padLeft - padRight) / 2
      : padLeft + (i / (buckets.length - 1)) * (width - padLeft - padRight);
  const y = (v: number) => height - padY - (v / maxValue) * (height - padY * 2);

  const linePath = (key: (typeof LINES)[number]["key"]) =>
    buckets.map((b, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(b[key])}`).join(" ");

  // Y-axis tick values
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div>
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">Throughput</h3>
        <div className="flex flex-wrap gap-4">
          {LINES.map((l) => (
            <span key={l.key} className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
              <span className="inline-block h-[2px] w-3 rounded-full" style={{ background: l.color }} />
              {l.label}
            </span>
          ))}
        </div>
      </div>

      {buckets.length === 0 ? (
        <div className="flex h-36 items-center justify-center text-xs text-slate-500 dark:text-slate-400">
          No events yet
        </div>
      ) : (
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height: 160 }}>
          {/* Y-axis labels + grid */}
          {ticks.map((t) => {
            const yPos = padY + t * (height - padY * 2);
            const label = Math.round(maxValue * (1 - t));
            return (
              <g key={t}>
                <line
                  x1={padLeft}
                  x2={width - padRight}
                  y1={yPos}
                  y2={yPos}
                  stroke="currentColor"
                  className="text-slate-200 dark:text-white/[0.05]"
                  strokeWidth={1}
                />
                <text
                  x={padLeft - 8}
                  y={yPos + 3}
                  textAnchor="end"
                  className="fill-slate-400 dark:fill-slate-500"
                  style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
                >
                  {label}
                </text>
              </g>
            );
          })}
          {/* Lines */}
          {LINES.map((l) => (
            <path
              key={l.key}
              d={linePath(l.key)}
              fill="none"
              stroke={l.color}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.85"
            />
          ))}
        </svg>
      )}
    </div>
  );
}
