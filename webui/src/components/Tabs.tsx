import type { ReactNode } from "react";
import { Activity, Search } from "lucide-react";

export type ActiveView = "overview" | "rollouts";

const tabs: Array<{ id: ActiveView; label: string; icon: ReactNode }> = [
  { id: "overview", label: "Overview", icon: <Activity className="h-3.5 w-3.5" /> },
  { id: "rollouts", label: "Rollouts", icon: <Search className="h-3.5 w-3.5" /> },
];

export function ViewTabs({
  activeView,
  onChange,
  layout = "horizontal",
}: {
  activeView: ActiveView;
  onChange: (view: ActiveView) => void;
  layout?: "horizontal" | "vertical";
}) {
  if (layout === "vertical") {
    return (
      <nav className="flex flex-col gap-0.5" aria-label="Views">
        {tabs.map((tab) => {
          const active = activeView === tab.id;
          return (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              type="button"
              onClick={() => onChange(tab.id)}
              className={`flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-slate-100 font-medium text-slate-900 dark:bg-white/[0.08] dark:text-slate-100"
                  : "text-slate-500 hover:bg-slate-100/60 hover:text-slate-700 dark:text-slate-400 dark:hover:bg-white/[0.04] dark:hover:text-slate-200"
              }`}
              aria-current={active ? "page" : undefined}
            >
              {tab.icon}
              {tab.label}
            </button>
          );
        })}
      </nav>
    );
  }

  return (
    <nav className="flex gap-4 border-b border-slate-200 dark:border-white/[0.06]" aria-label="Views">
      {tabs.map((tab) => {
        const active = activeView === tab.id;
        return (
          <button
            key={tab.id}
            id={`tab-${tab.id}`}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2 border-b-2 pb-2 text-sm font-medium transition-colors ${
              active
                ? "border-slate-900 text-slate-900 dark:border-slate-100 dark:text-slate-100"
                : "border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
            }`}
            aria-current={active ? "page" : undefined}
          >
            {tab.icon}
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}
