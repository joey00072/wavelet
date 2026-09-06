import type { ReactNode } from "react";

const STATUS_DOT: Record<string, string> = { running: "text-good live-dot", completed: "text-ink2", failed: "text-critical", stale: "text-serious", stopped: "text-warn" };

/**
 * Run status as a small dot plus plain text. A run is a long, continuous
 * process, so "running" is a steady dot with a slow pulse rather than a
 * spinner, and no status is drawn as a filled pill.
 */
export function StatusBadge({ status, reason }: { status: string; reason?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-ink2" title={reason} aria-label={reason ? `${status}: ${reason}` : status}>
      <span aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-current ${STATUS_DOT[status] ?? "text-muted"}`} />
      {status}
    </span>
  );
}

const TAG_TONES = { neutral: "text-ink2", good: "text-good", warning: "text-warn", serious: "text-serious", critical: "text-critical", accent: "text-accent" };

export function Tag({ children, tone = "neutral" }: { children: ReactNode; tone?: keyof typeof TAG_TONES }) {
  return <span className={`chip ${TAG_TONES[tone]}`}>{children}</span>;
}
