import type { ReactNode } from "react";

import type { Point } from "../lib/series";
import { AnimatedNumber } from "./AnimatedNumber";
import { Sparkline } from "./Sparkline";

const TONE_CLASS = { good: "text-good", warning: "text-warn", serious: "text-serious", critical: "text-critical" };

export function StatTile({
  label,
  value,
  number,
  format,
  delta,
  deltaGood,
  sub,
  trend,
  tone,
}: {
  label: string;
  value?: ReactNode;
  number?: number | null;
  format?: (v: number) => string;
  delta?: string | null;
  deltaGood?: boolean | null;
  sub?: ReactNode;
  trend?: Point[];
  tone?: keyof typeof TONE_CLASS | null;
}) {
  const toneClass = tone ? TONE_CLASS[tone] : "text-ink";
  return (
    <div className="tile">
      <div className="flex items-center justify-between gap-2 text-[11px] text-muted">
        <span className="flex items-center gap-1.5 truncate">{label}</span>
        {delta && (
          <span className={`tabular ${deltaGood === null || deltaGood === undefined ? "text-muted" : deltaGood ? "text-[var(--success-text)]" : "text-critical"}`}>
            {delta}
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-2">
        <div className={`min-w-0 truncate text-2xl font-semibold leading-none tracking-tight sm:text-[26px] ${toneClass}`}>{format ? <AnimatedNumber value={number} format={format} /> : value}</div>
        {trend && trend.length > 1 && <span className="hidden shrink-0 sm:block"><Sparkline points={trend} width={72} /></span>}
      </div>
      {sub && <div className="line-clamp-2 text-[11px] text-muted sm:truncate" title={typeof sub === "string" ? sub : undefined}>{sub}</div>}
    </div>
  );
}
