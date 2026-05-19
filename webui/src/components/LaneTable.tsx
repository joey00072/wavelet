import type { RolloutEvent } from "../types";
import { laneCounts } from "../utils/metrics";

export function LaneTable({ events }: { events: RolloutEvent[] }) {
  const lanes = laneCounts(events);

  return (
    <div>
      <div className="mb-4">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">Per chunk lane</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[480px] text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-white/[0.06]">
              {["Lane", "Submitted", "Completed", "Published", "Backlog"].map((col) => (
                <th key={col} className="pb-3 pr-8 text-left text-xs font-medium text-slate-500 dark:text-slate-400 last:pr-0">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {lanes.length === 0 ? (
              <tr>
                <td className="py-8 text-sm text-slate-500 dark:text-slate-400" colSpan={5}>
                  No lane data yet
                </td>
              </tr>
            ) : (
              lanes.map(([lane, counts]) => {
                const backlog = Math.max(counts.submitted - counts.published, 0);
                return (
                  <tr key={lane} className="border-b border-slate-100 dark:border-white/[0.04] last:border-0 hover:bg-slate-50 dark:hover:bg-white/[0.02]">
                    <td className="py-3 pr-8 font-mono text-xs text-slate-700 dark:text-slate-300">{lane}</td>
                    <td className="py-3 pr-8 font-mono text-xs text-slate-500 dark:text-slate-400">{counts.submitted}</td>
                    <td className="py-3 pr-8 font-mono text-xs text-slate-500 dark:text-slate-400">{counts.completed}</td>
                    <td className="py-3 pr-8 font-mono text-xs text-slate-500 dark:text-slate-400">{counts.published}</td>
                    <td className={`py-3 font-mono text-xs ${backlog > 0 ? "text-amber-500 dark:text-amber-400" : "text-slate-500 dark:text-slate-400"}`}>
                      {backlog}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
