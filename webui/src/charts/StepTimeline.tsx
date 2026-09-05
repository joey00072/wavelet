import { useState } from "react";

import type { TimelinePolicy, TimelineStep } from "../api/types";
import { fmtSeconds, fmtTime } from "../lib/format";
import { seriesColor } from "../lib/theme";
import { useMeasure } from "./useMeasure";

const ROW_H = 14;

/**
 * Per-step lifecycle bars on a shared wall-clock axis. Each queue step is a row:
 * published→claimed in slot 1, claimed→consumed in slot 2. Policy export→load
 * intervals are drawn as thin ticks in slot 3 above the axis.
 */
export function StepTimeline({ steps, policies, maxRows = 16 }: { steps: TimelineStep[]; policies: TimelinePolicy[]; maxRows?: number }) {
  const [ref, size] = useMeasure<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const rows = steps.slice(-maxRows);
  const width = Math.max(size.width, 160);
  const margin = { left: 44, right: 12, top: 18, bottom: 22 };
  const times = rows.flatMap((s) => [s.published_at, s.claimed_at, s.consumed_at]).filter((t): t is string => Boolean(t)).map((t) => new Date(t).getTime()).filter((t) => !Number.isNaN(t));
  if (rows.length === 0 || times.length === 0) {
    return <div className="flex h-24 items-center justify-center text-xs text-muted">No lifecycle events yet</div>;
  }
  const t0 = Math.min(...times);
  const t1 = Math.max(...times);
  const span = Math.max(t1 - t0, 1000);
  const plotW = width - margin.left - margin.right;
  const sx = (iso: string | undefined) => (iso ? margin.left + ((new Date(iso).getTime() - t0) / span) * plotW : null);
  const height = margin.top + rows.length * ROW_H + margin.bottom;
  const relevantPolicies = policies.filter((p) => {
    if (!p.exported_at) return false;
    const time = new Date(p.exported_at).getTime();
    return time >= t0 && time <= t1;
  });
  const tickCount = Math.max(2, Math.min(8, Math.floor(plotW / 130)));
  const ticks = Array.from({ length: tickCount }, (_, i) => t0 + (span * i) / (tickCount - 1));
  return (
    <div ref={ref} className="relative w-full select-none">
      <svg width={width} height={height} role="img" aria-label="Queue and policy lifecycle timeline" onPointerLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={margin.left + ((t - t0) / span) * plotW} x2={margin.left + ((t - t0) / span) * plotW} y1={margin.top} y2={height - margin.bottom} stroke="var(--grid)" />
            <text x={margin.left + ((t - t0) / span) * plotW} y={height - 7} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" className="tabular">
              {new Date(t).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </text>
          </g>
        ))}
        {relevantPolicies.map((p) => {
          const x0 = sx(p.exported_at);
          const x1 = sx(p.loaded_at ?? p.received_at ?? p.exported_at);
          if (x0 === null) return null;
          return (
            <g key={p.policy_step}>
              <rect x={x0} y={4} width={Math.max(2, (x1 ?? x0) - x0)} height={8} rx={2} fill={seriesColor(2)} />
            </g>
          );
        })}
        <text x={margin.left - 6} y={11} textAnchor="end" fontSize={9} fill="var(--ink-muted)">policy</text>
        {rows.map((s, i) => {
          const y = margin.top + i * ROW_H + 2;
          const pub = sx(s.published_at);
          const claim = sx(s.claimed_at);
          const cons = sx(s.consumed_at);
          const dim = hover !== null && hover !== i;
          return (
            <g key={s.queue_step} onPointerEnter={() => setHover(i)} opacity={dim ? 0.55 : 1}>
              <rect x={margin.left} y={y - 2} width={plotW} height={ROW_H} fill="transparent" />
              <text x={margin.left - 6} y={y + 5} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--ink-muted)" className="tabular">
                {s.queue_step}
              </text>
              {pub !== null && (
                <rect x={pub} y={y} width={Math.max(2, (claim ?? pub) - pub)} height={ROW_H - 4} rx={2} fill={seriesColor(0)} />
              )}
              {claim !== null && (
                <rect x={claim + 2} y={y} width={Math.max(2, (cons ?? claim + 2) - claim - 2)} height={ROW_H - 4} rx={2} fill={seriesColor(1)} />
              )}
              {pub === null && claim === null && cons !== null && <circle cx={cons} cy={y + 5} r={3} fill={seriesColor(1)} />}
            </g>
          );
        })}
      </svg>
      {hover !== null && rows[hover] && (
        <div className="pointer-events-none absolute right-2 top-4 z-10 rounded-md border border-edge bg-surface px-2 py-1.5 text-[11px] shadow-sm">
          <div className="mb-1 text-muted">queue step {rows[hover].queue_step} · policy {rows[hover].policy_step ?? "–"}</div>
          <Row label="published" value={fmtTime(rows[hover].published_at)} color={seriesColor(0)} />
          <Row label="waiting for trainer" value={fmtSeconds(rows[hover].publish_to_claim_seconds)} color={seriesColor(0)} />
          <Row label="training" value={fmtSeconds(rows[hover].claim_to_consume_seconds)} color={seriesColor(1)} />
          <Row label="consumed" value={fmtTime(rows[hover].consumed_at)} color={seriesColor(1)} />
        </div>
      )}
      <div className="mt-1 flex flex-wrap gap-x-3 px-1 text-[11px] text-ink2">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: seriesColor(0) }} />published → claimed</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: seriesColor(1) }} />claimed → consumed</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: seriesColor(2) }} />policy export → load</span>
      </div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="flex items-center gap-1.5 text-ink2">
        <span className="inline-block h-0.5 w-3 rounded" style={{ background: color }} />
        {label}
      </span>
      <span className="tabular font-semibold text-ink">{value}</span>
    </div>
  );
}
