import type { ReactNode } from "react";
import { Activity, AlertCircle, CheckCircle2, Clock } from "lucide-react";

export function StatusIcon({ status }: { status: string }) {
  if (status === "running") return <Activity className="h-3.5 w-3.5 text-emerald-500" />;
  if (status === "completed") return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />;
  if (status === "failed") return <AlertCircle className="h-3.5 w-3.5 text-red-500" />;
  return <Clock className="h-3.5 w-3.5 text-slate-400" />;
}

export function Stat({
  label,
  value,
  sub,
  icon,
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col justify-center">
      <div className="flex items-center gap-1.5 text-sm font-medium text-slate-500 dark:text-slate-400">
        {icon}
        {label}
      </div>
      <div className="mt-1 text-3xl font-semibold tracking-tight text-slate-900 dark:text-slate-50">
        {value}
      </div>
      {sub && <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  );
}

export function InventoryCard({
  label,
  value,
  sub,
  icon,
  tone = "zinc",
}: {
  label: string;
  value: string;
  sub: string;
  icon?: ReactNode;
  tone?: "blue" | "emerald" | "amber" | "cyan" | "zinc";
}) {
  const labelColor = {
    blue: "text-blue-500 dark:text-blue-400",
    emerald: "text-emerald-500 dark:text-emerald-400",
    amber: "text-amber-500 dark:text-amber-400",
    cyan: "text-cyan-500 dark:text-cyan-400",
    zinc: "text-slate-500 dark:text-slate-400",
  }[tone];

  return (
    <div className="flex flex-col justify-center">
      <div className={`flex items-center gap-1.5 text-sm font-medium ${labelColor}`}>
        {icon}
        {label}
      </div>
      <div className="mt-1 font-mono text-3xl font-semibold text-slate-900 dark:text-slate-50">
        {value}
      </div>
      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sub}</div>
    </div>
  );
}
