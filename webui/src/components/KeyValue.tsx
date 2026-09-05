import type { CSSProperties, ReactNode } from "react";

export function KeyValue({ items, columns = 2 }: { items: Array<[string, ReactNode]>; columns?: number }) {
  const style = { "--kv-columns": columns } as CSSProperties;
  return (
    <dl className="key-value-grid grid gap-x-4 gap-y-2" style={style}>
      {items.map(([label, value]) => (
        <div key={label} className="min-w-0">
          <dt className="truncate text-[11px] text-muted">{label}</dt>
          <dd className="tabular truncate text-xs text-ink" title={typeof value === "string" ? value : undefined}>
            {value ?? "–"}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Section({ title, children, actions, className = "" }: { title: string; children: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <section className={`space-y-3 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <h2 className="eyebrow">{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1 rounded-md px-6 py-12 text-center">
      <div className="text-sm font-medium text-ink2">{title}</div>
      {hint && <div className="max-w-md text-xs text-muted">{hint}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return <div role="alert" className="rounded-md px-4 py-2.5 text-xs text-critical" style={{ background: "color-mix(in srgb, var(--status-critical) 12%, transparent)" }}>{error}</div>;
}
