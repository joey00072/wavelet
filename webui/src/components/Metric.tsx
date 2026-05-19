import type { ReactNode } from "react";

export function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className="mt-0.5 font-mono text-sm font-medium text-slate-800 dark:text-slate-200">{value}</dd>
    </div>
  );
}
