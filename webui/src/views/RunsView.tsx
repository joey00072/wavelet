import { useEffect, useMemo, useState } from "react";
import { ArrowRight, GitCompare, Radio } from "lucide-react";

import { fetchJson, runUrl, usePoll } from "../api/client";
import type { RunSummary, Series } from "../api/types";
import { Sparkline } from "../charts/Sparkline";
import { StatusBadge } from "../components/Badge";
import { Empty, ErrorNote } from "../components/KeyValue";
import { fmt, fmtAge, fmtInt, modelLabel, shortId } from "../lib/format";
import { CURRENT_RUN, navigate } from "../lib/router";
import { num } from "../lib/series";

export function RunsView({ apiBase, runs, error }: { apiBase: string; runs: RunSummary[] | null; error: string | null }) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [trends, setTrends] = useState<Record<string, Array<{ x: number; y: number }>>>({});
  const ids = useMemo(() => (runs ?? []).map((r) => r.id), [runs]);
  const current = useMemo(() => (runs ?? []).find((r) => r.is_current) ?? null, [runs]);
  const older = useMemo(() => (runs ?? []).filter((r) => !r.is_current), [runs]);
  const primaryRunLabel = current?.status === "running" ? "current" : "recent";

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(
        ids.map(async (id) => {
          try {
            const series = await fetchJson<Series>(`${runUrl(apiBase, id, "/series")}?source=orchestrator&keys=reward/all/mean&limit=150`);
            const points = series.steps.flatMap((step, i) => {
              const y = series.series["reward/all/mean"]?.[i];
              return step !== null && y !== null && y !== undefined ? [{ x: step, y }] : [];
            });
            return [id, points] as const;
          } catch {
            return [id, []] as const;
          }
        }),
      );
      if (!cancelled) setTrends(Object.fromEntries(entries));
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, ids.join("|")]);

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="space-y-8">
      {current && (
        <button type="button" className="flex w-full flex-wrap items-center gap-x-6 gap-y-2 py-2 text-left" onClick={() => navigate({ page: "run", runId: CURRENT_RUN, view: "overview" })} aria-label={`Open ${primaryRunLabel} run ${current.id}`}>
          <Radio className={`h-4 w-4 ${current.status === "running" ? "live-dot text-good" : "text-muted"}`} />
          <div className="min-w-0">
            <div className="eyebrow">{primaryRunLabel} run</div>
            <div className="truncate text-xl font-semibold tracking-tight text-ink">{current.id}</div>
          </div>
          <StatusBadge status={current.status} reason={current.status_reason} />
          <span className="tabular text-xs text-ink2">step {current.trainer_step === null ? "–" : current.trainer_step + 1}{current.target_step ? ` / ${current.target_step}` : ""}</span>
          <span className="tabular text-xs text-ink2">reward {fmt(num(current.latest.orchestrator, "reward/all/mean") ?? num(current.latest.trainer, "reward/all/mean"), 3)}</span>
          <Sparkline points={trends[current.id] ?? []} />
          <span className="text-xs text-muted">{fmtAge(current.updated_at)}</span>
          <ArrowRight className="ml-auto h-4 w-4 text-muted" />
        </button>
      )}
      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-ink">All runs</h1>
          <p className="text-xs text-muted">Every run discovered under the configured roots. Select two or more to compare.</p>
        </div>
        <button type="button" className="btn" disabled={selected.size < 2} onClick={() => navigate({ page: "compare", params: new URLSearchParams({ runs: [...selected].join(",") }) })}>
          <GitCompare className="h-3.5 w-3.5" /> Compare {selected.size > 0 ? `(${selected.size})` : ""}
        </button>
      </div>
      <ErrorNote error={error} />
      {runs && runs.length === 0 && <Empty title="No runs found" hint="Start the dashboard with --runs-root pointing at your outputs directory, or pass run directories explicitly." />}
      <div className="section space-y-1 md:hidden">
        {(current ? [current, ...older] : older).map((run) => {
          const reward = num(run.latest.orchestrator, "reward/all/mean") ?? num(run.latest.trainer, "reward/all/mean");
          const evalEntry = headlineEval(run);
          return (
            <article key={run.id} className="flex items-start gap-3 border-b border-edge py-3 last:border-0">
              <input type="checkbox" className="mt-1 h-4 w-4 shrink-0 accent-[var(--series-1)]" checked={selected.has(run.id)} onChange={() => toggle(run.id)} aria-label={`Select ${run.id}`} />
              <button type="button" className="min-w-0 flex-1 text-left" onClick={() => navigate({ page: "run", runId: run.id, view: "overview" })}>
                <span className="flex items-start justify-between gap-2">
                  <span className="min-w-0 truncate text-sm font-medium text-ink">{run.id}</span>
                  <StatusBadge status={run.status} reason={run.status_reason} />
                </span>
                <span className="mt-1 grid grid-cols-3 gap-2 text-[11px] text-muted">
                  <span className="tabular">step <span className="text-ink2">{run.trainer_step === null ? "–" : run.trainer_step + 1}</span></span>
                  <span className="tabular">reward <span className="text-ink2">{fmt(reward, 3)}</span></span>
                  <span className="tabular">eval <span className="text-ink2">{evalEntry ? fmt(evalEntry.value, 3) : "–"}</span></span>
                </span>
                <span className="mt-1 flex items-center justify-between gap-2 text-[11px] text-muted">
                  <span className="truncate">{modelLabel(run.model)} · {run.envs.join(", ") || run.eval_envs.join(", ") || "no environment"}</span>
                  <span className="shrink-0">{fmtAge(run.updated_at)}</span>
                </span>
              </button>
            </article>
          );
        })}
      </div>
      <div className="section hidden overflow-auto md:block" role="region" aria-label="Runs table" tabIndex={0}>
        <table className="w-full">
          <thead>
            <tr>
              <th className="th w-8" />
              <th className="th">Run</th>
              <th className="th">Status</th>
              <th className="th text-right">Step</th>
              <th className="th">Reward trend</th>
              <th className="th text-right">Reward</th>
              <th className="th text-right">Eval</th>
              <th className="th">Model</th>
              <th className="th">Env</th>
              <th className="th">Algo</th>
              <th className="th">Updated</th>
              <th className="th" />
            </tr>
          </thead>
          <tbody>
            {(current ? [current, ...older] : older).map((run) => {
              const reward = num(run.latest.orchestrator, "reward/all/mean") ?? num(run.latest.trainer, "reward/all/mean");
              const evalEntry = headlineEval(run);
              return (
                <tr key={run.id} className="tr-hover cursor-pointer border-b border-edge last:border-0" onClick={() => navigate({ page: "run", runId: run.id, view: "overview" })} onKeyDown={(event) => {
                  if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) {
                    event.preventDefault();
                    navigate({ page: "run", runId: run.id, view: "overview" });
                  }
                }} tabIndex={0}>
                  <td className="td" onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" className="h-3.5 w-3.5 accent-[var(--series-1)]" checked={selected.has(run.id)} onChange={() => toggle(run.id)} aria-label={`Select ${run.id}`} />
                  </td>
                  <td className="td">
                    <div className="flex items-center gap-1.5 font-medium text-ink">{run.id}{run.is_current && <span className="chip text-good">{primaryRunLabel}</span>}</div>
                    <div className="truncate text-[11px] text-muted" title={run.path}>{shortId(run.path, 48)}</div>
                  </td>
                  <td className="td"><StatusBadge status={run.status} reason={run.status_reason} /></td>
                  <td className="td tabular text-right">{run.trainer_step === null ? "–" : `${run.trainer_step + 1}${run.target_step ? ` / ${run.target_step}` : ""}`}</td>
                  <td className="td"><Sparkline points={trends[run.id] ?? []} /></td>
                  <td className="td tabular text-right">{fmt(reward, 3)}</td>
                  <td className="td tabular text-right" title={evalEntry?.key}>{evalEntry ? fmt(evalEntry.value, 3) : "–"}</td>
                  <td className="td" title={run.model ?? undefined}>{shortId(modelLabel(run.model), 30)}</td>
                  <td className="td">{run.envs.length ? run.envs.join(", ") : run.eval_envs.join(", ") || "–"}</td>
                  <td className="td">{run.algo ?? "–"}{run.lora ? " · lora" : ""}</td>
                  <td className="td">{fmtAge(run.updated_at)}</td>
                  <td className="td text-right"><ArrowRight className="inline h-3.5 w-3.5 text-muted" /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {runs && runs.length > 0 && <p className="text-[11px] text-muted">{fmtInt(runs.length)} run(s). Data refreshes every few seconds; completed runs are read from disk.</p>}
    </div>
  );
}

export function headlineEval(run: RunSummary): { key: string; value: number } | null {
  const row = run.latest.eval;
  if (!row) return null;
  const candidates = Object.entries(row).filter(([k, v]) => /^eval\/[^/]+\/(avg@\d+|pass@1)$/.test(k) && typeof v === "number") as Array<[string, number]>;
  if (candidates.length === 0) return null;
  candidates.sort(([a], [b]) => (a.includes("avg@") === b.includes("avg@") ? a.localeCompare(b) : a.includes("avg@") ? -1 : 1));
  return { key: candidates[0][0], value: candidates[0][1] };
}

export function useRuns(apiBase: string, intervalMs = 5000) {
  return usePoll<RunSummary[]>(`${apiBase}/api/runs`, intervalMs);
}
