import { useMemo, useState } from "react";

import type { EvalMetricRow } from "../types";

// Metric keys are eval/<env>/avg@{k} and eval/<env>/pass@{k}, where k is the
// configured rollouts_per_example. pass@k is charted for the same k as avg@k.
const AVG_METRIC = /^avg@(\d+)$/;
const METRIC_COLORS = { avg: "#10b981", pass: "#3b82f6" };

type Point = { step: number; value: number };
type HoveredPoint = Point & { color: string; label: string; x: number; y: number };

function pointsFor(metrics: EvalMetricRow[], key: string): Point[] {
  return metrics
    .flatMap((row) => {
      const rawValue = row[key];
      const rawStep = row["progress/policy_step"] ?? row.step;
      if (typeof rawValue !== "number" || typeof rawStep !== "number") return [];
      return [{ step: rawStep, value: rawValue * 100 }];
    })
    .sort((left, right) => left.step - right.step);
}

export function EvalProgressChart({ metrics }: { metrics: EvalMetricRow[] }) {
  const [hoveredPoint, setHoveredPoint] = useState<HoveredPoint | null>(null);
  const series = useMemo(() => {
    const allKeys = new Set(metrics.flatMap((row) => Object.keys(row)));
    const definitions = new Map<string, string>();
    for (const key of allKeys) {
      const parts = key.split("/");
      if (parts.length !== 3 || parts[0] !== "eval") continue;
      const match = AVG_METRIC.exec(parts[2]);
      if (match === null) continue;
      definitions.set(key, METRIC_COLORS.avg);
      const passKey = `eval/${parts[1]}/pass@${match[1]}`;
      if (allKeys.has(passKey)) definitions.set(passKey, METRIC_COLORS.pass);
    }
    return [...definitions.keys()].sort().map((key) => {
      const [, envName, metricName] = key.split("/");
      return {
        key,
        color: definitions.get(key)!,
        label: `${envName} ${metricName}`,
        points: pointsFor(metrics, key),
      };
    });
  }, [metrics]);
  const references = series.flatMap((item) => {
    const baseline = item.points.find((point) => point.step === 0);
    return baseline === undefined
      ? []
      : [{ value: baseline.value, color: item.color, label: `base ${item.label} ${baseline.value.toFixed(1)}%` }];
  });
  const width = 900;
  const height = 250;
  const padLeft = 42;
  const padRight = 16;
  const padTop = 16;
  const padBottom = 30;
  const lastStep = Math.max(100, ...series.flatMap((item) => item.points.map((point) => point.step)));
  const x = (step: number) => padLeft + (step / lastStep) * (width - padLeft - padRight);
  const y = (value: number) =>
    padTop + ((100 - Math.min(100, Math.max(0, value))) / 100) * (height - padTop - padBottom);
  const pathFor = (points: Point[]) =>
    points.map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.step)} ${y(point.value)}`).join(" ");
  const ticks = [0, 20, 40, 60, 80, 100];
  const hasPoints = series.some((item) => item.points.length > 0);

  return (
    <div>
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">
            Evaluation progress
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Fixed-policy avg@k and pass@k from each configured environment
          </p>
        </div>
        <div className="flex flex-wrap gap-4">
          {[...references, ...series].map((item) => (
            <span key={item.label} className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
              <span className="inline-block h-[2px] w-3 rounded-full" style={{ background: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </div>

      <div className="relative">
        {!hasPoints && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-500 dark:text-slate-400">
            Waiting for the base-model evaluation
          </div>
        )}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height: 210 }}>
          {ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={padLeft}
                x2={width - padRight}
                y1={y(tick)}
                y2={y(tick)}
                stroke="currentColor"
                className="text-slate-200 dark:text-white/[0.05]"
              />
              <text
                x={padLeft - 8}
                y={y(tick) + 3}
                textAnchor="end"
                className="fill-slate-400 dark:fill-slate-500"
                style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
              >
                {tick}%
              </text>
            </g>
          ))}
          {references.map((reference) => (
            <line
              key={reference.label}
              x1={padLeft}
              x2={width - padRight}
              y1={y(reference.value)}
              y2={y(reference.value)}
              stroke={reference.color}
              strokeDasharray="7 5"
              strokeWidth="1.5"
              opacity="0.9"
            />
          ))}
          {series.map((definition) => {
            const points = definition.points;
            return (
              <g key={definition.key}>
                <path
                  d={pathFor(points)}
                  fill="none"
                  stroke={definition.color}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {points.map((point) => (
                  <circle
                    key={`${point.step}-${point.value}`}
                    cx={x(point.step)}
                    cy={y(point.value)}
                    r="5"
                    fill={definition.color}
                    className="cursor-pointer"
                    onMouseEnter={() =>
                      setHoveredPoint({
                        ...point,
                        color: definition.color,
                        label: definition.label,
                        x: x(point.step),
                        y: y(point.value),
                      })
                    }
                    onMouseLeave={() => setHoveredPoint(null)}
                  />
                ))}
              </g>
            );
          })}
          {hoveredPoint && (
            <g pointerEvents="none">
              <rect
                x={Math.min(width - padRight - 174, Math.max(padLeft, hoveredPoint.x - 87))}
                y={Math.max(padTop, hoveredPoint.y - 42)}
                width="174"
                height="30"
                rx="6"
                className="fill-slate-950 dark:fill-slate-100"
              />
              <text
                x={Math.min(width - padRight - 87, Math.max(padLeft + 87, hoveredPoint.x))}
                y={Math.max(padTop + 19, hoveredPoint.y - 23)}
                textAnchor="middle"
                className="fill-white dark:fill-slate-950"
                style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
              >
                {hoveredPoint.label} {hoveredPoint.value.toFixed(1)}% · step {hoveredPoint.step}
              </text>
            </g>
          )}
          <text
            x={padLeft}
            y={height - 8}
            className="fill-slate-400 dark:fill-slate-500"
            style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
          >
            0
          </text>
          <text
            x={width - padRight}
            y={height - 8}
            textAnchor="end"
            className="fill-slate-400 dark:fill-slate-500"
            style={{ fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
          >
            step {lastStep}
          </text>
        </svg>
      </div>
    </div>
  );
}
