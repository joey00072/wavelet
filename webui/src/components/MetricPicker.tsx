import { useMemo, useState } from "react";
import { Search } from "lucide-react";

import { groupKeys } from "../lib/series";

export function MetricPicker({ keys, selected, onToggle }: { keys: string[]; selected: string[]; onToggle: (key: string) => void; onClear?: () => void }) {
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const filtered = query ? keys.filter((k) => k.toLowerCase().includes(query.toLowerCase())) : keys;
    return groupKeys(filtered);
  }, [keys, query]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="relative">
        <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted" />
        <input className="input w-full !pl-6" aria-label="Search metrics" placeholder="Search metrics" value={query} onChange={(e) => setQuery(e.target.value)} autoFocus />
      </div>
      <div className="min-h-0 flex-1 overflow-auto pr-1">
        {groups.map(({ group, keys: list }) => {
          const hasSelected = list.some((k) => selected.includes(k));
          const isCollapsed = !query && !hasSelected && !expanded.has(group);
          return (
            <div key={group} className="mb-1">
              <button
                type="button"
                className="flex w-full items-center justify-between py-1 text-[11px] font-semibold uppercase tracking-wide text-muted hover:text-ink"
                onClick={() =>
                  setExpanded((prev) => {
                    const next = new Set(prev);
                    if (next.has(group)) next.delete(group);
                    else next.add(group);
                    return next;
                  })
                }
                aria-expanded={!isCollapsed}
              >
                <span className="flex items-center gap-1"><span className={`inline-block transition-transform ${isCollapsed ? "" : "rotate-90"}`}>›</span>{group}</span>
                <span className="tabular text-[10px]">{hasSelected ? `${list.filter((k) => selected.includes(k)).length}/${list.length}` : list.length}</span>
              </button>
              {!isCollapsed &&
                list.map((key) => {
                  const on = selected.includes(key);
                  return (
                    <label key={key} className={`flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 text-xs hover:bg-raised ${on ? "text-ink" : "text-ink2"}`}>
                      <input type="checkbox" className="h-3 w-3 accent-[var(--series-1)]" checked={on} onChange={() => onToggle(key)} />
                      <span className="truncate" title={key}>
                        {key.includes("/") ? key.slice(key.indexOf("/") + 1) : key}
                      </span>
                    </label>
                  );
                })}
            </div>
          );
        })}
        {groups.length === 0 && <div className="py-4 text-center text-xs text-muted">No metrics match</div>}
      </div>
    </div>
  );
}
