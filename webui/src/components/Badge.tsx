import type { ReactElement } from "react";

/**
 * Run status as a small dot plus plain text. A run is a long, continuous
 * process, so "running" is a steady dot with a slow pulse rather than a
 * spinner, and no status is drawn as a filled pill.
 */
export function StatusBadge({ status, reason }: { status: string; reason?: string }) {
  const spec = statusSpec(status);
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-ink2" title={reason} aria-label={reason ? `${status}: ${reason}` : status}>
      {spec.icon}
      {status}
    </span>
  );
}

function dot(className: string): ReactElement {
  return <span aria-hidden className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-current ${className}`} />;
}

export function statusSpec(status: string): { icon: ReactElement; className: string; tone: "good" | "warning" | "serious" | "critical" | null } {
  switch (status) {
    case "running":
      return { icon: dot("text-good live-dot"), className: "text-good", tone: "good" };
    case "completed":
      return { icon: dot("text-ink2"), className: "text-ink2", tone: null };
    case "failed":
      return { icon: dot("text-critical"), className: "text-critical", tone: "critical" };
    case "stale":
      return { icon: dot("text-serious"), className: "text-serious", tone: "serious" };
    case "stopped":
      return { icon: dot("text-warn"), className: "text-warn", tone: "warning" };
    default:
      return { icon: dot("text-muted"), className: "text-muted", tone: null };
  }
}

export function Tag({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "good" | "warning" | "serious" | "critical" | "accent" }) {
  const cls = {
    neutral: "text-ink2",
    good: "text-good",
    warning: "text-warn",
    serious: "text-serious",
    critical: "text-critical",
    accent: "text-accent",
  }[tone];
  return <span className={`chip ${cls}`}>{children}</span>;
}
