import { useEffect, useMemo, useState } from "react";

import { fetchJson, qs, runUrl } from "../api/client";
import type { Evals, MetricKeys, RunSummary, Series } from "../api/types";
import { ChartCard, chartProps } from "../charts/ChartCard";
import { ChartSettingsMenu, useChartSettings } from "./TrainingView";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { StatusBadge } from "../components/Badge";
import { ListPlus } from "lucide-react";

import { Segmented } from "../components/Controls";
import { Drawer } from "../components/Drawer";
import { Empty } from "../components/KeyValue";
import { MetricPicker } from "../components/MetricPicker";
import { fmt, fmtPct } from "../lib/format";
import { navigate, updateParams } from "../lib/router";
import { firstTime, lastFinite, type LineSeries, type Point } from "../lib/series";
import { seriesColor } from "../lib/theme";

type Source = "trainer" | "orchestrator" | "eval";
const SOURCES: readonly Source[] = ["trainer", "orchestrator", "eval"];

const DEFAULTS: Record<Source, string[]> = {
  orchestrator: ["reward/all/mean", "generation/solve_none/rate", "is_truncated/all/mean", "off_policy/mean"],
  trainer: ["train/loss", "entropy/mean", "kl/mismatch", "optim/grad_norm"],
  eval: [],
};

export function CompareView({ apiBase, runs, params }: { apiBase: string; runs: RunSummary[]; params: URLSearchParams }) {
  const runIds = useMemo(() => (params.get("runs") ?? "").split(",").filter(Boolean), [params]);
  const live = runIds.some((id) => runs.find((run) => run.id === id)?.status === "running");
  const requestedSource = params.get("source");
  const source: Source = requestedSource && SOURCES.includes(requestedSource as Source) ? requestedSource as Source : "orchestrator";
  const [settings, update] = useChartSettings("wavelet.charts.compare");
  const xAxis = settings.xAxis;
  const smoothing = settings.smoothing;
  const [keysBySource, setKeysBySource] = useState<Record<string, string[]>>({});
  const [seriesByRun, setSeriesByRun] = useState<Record<string, Series>>({});
  const [evalsByRun, setEvalsByRun] = useState<Record<string, Evals>>({});
  const [picking, setPicking] = useState(false);
  const selectedParam = params.get("keys");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const union: Record<string, Set<string>> = { trainer: new Set(), orchestrator: new Set(), eval: new Set() };
      await Promise.all(
        runIds.map(async (id) => {
          try {
            const keys = await fetchJson<MetricKeys>(runUrl(apiBase, id, "/metrics/keys"));
            for (const src of SOURCES) keys[src].forEach((k) => union[src].add(k.key));
          } catch {
            // run may be unavailable
          }
        }),
      );
      if (!cancelled) setKeysBySource(Object.fromEntries(Object.entries(union).map(([k, v]) => [k, [...v].sort()])));
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, runIds.join("|")]);

  const available = keysBySource[source] ?? [];
  const selected = useMemo(() => {
    if (selectedParam === "-") return [];
    const requested = selectedParam ? selectedParam.split(",").filter(Boolean) : source === "eval" ? available.filter((k) => /\/(avg@\d+|pass@1)$/.test(k)).slice(0, 4) : DEFAULTS[source];
    return requested.filter((k) => available.length === 0 || available.includes(k));
  }, [selectedParam, source, available]);

  useEffect(() => {
    if (selected.length === 0) return;
    let cancelled = false;
    const load = async () => {
      const entries = await Promise.all(
        runIds.map(async (id) => {
          try {
            return [id, await fetchJson<Series>(`${runUrl(apiBase, id, "/series")}${qs({ source, keys: selected.join(","), limit: 0 })}`)] as const;
          } catch {
            return [id, null] as const;
          }
        }),
      );
      if (!cancelled) setSeriesByRun(Object.fromEntries(entries.filter((e): e is readonly [string, Series] => e[1] !== null)));
    };
    load();
    const timer = live ? window.setInterval(load, 8000) : undefined;
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearInterval(timer);
    };
  }, [apiBase, runIds.join("|"), source, selected.join("|"), live]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(runIds.map(async (id) => [id, await fetchJson<Evals>(runUrl(apiBase, id, "/evals")).catch(() => null)] as const));
      if (!cancelled) setEvalsByRun(Object.fromEntries(entries.filter((e): e is readonly [string, Evals] => e[1] !== null)));
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBase, runIds.join("|")]);

  const setSelected = (next: string[]) => updateParams((p) => p.set("keys", next.length ? next.join(",") : "-"));
  const setSource = (next: Source) => updateParams((p) => { p.set("source", next); p.delete("keys"); });

  if (runIds.length < 2) return <Empty title="Select at least two runs" hint="Go to Runs and tick the runs you want to compare, then click Compare." />;

  const latestOf = (line: LineSeries) => lastFinite(line.points.map((p) => p.y));
  const linesFor = (key: string): LineSeries[] =>
    runIds.map((id, index) => {
      const series = seriesByRun[id];
      const points: Point[] = [];
      const envelope = { min: [] as Point[], max: [] as Point[] };
      if (series) {
        const start = firstTime(series.timestamps);
        const xAt = (i: number) => xAxis === "time" ? (start !== null && series.timestamps[i] ? (new Date(series.timestamps[i]!).getTime() - start) / 60000 : null) : series.steps[i];
        series.series[key]?.forEach((y, i) => {
          if (y === null || y === undefined) return;
          const x = xAt(i);
          if (x === null || x === undefined) return;
          points.push({ x, y });
        });
        for (const bound of ["min", "max"] as const) {
          series.envelope?.[key]?.[bound].forEach((y, i) => {
            if (y === null || y === undefined) return;
            const x = xAt(i);
            if (x !== null && x !== undefined) envelope[bound].push({ x, y });
          });
        }
      }
      return { id, label: id, colorIndex: index, points, envelope: envelope.min.length && envelope.max.length ? envelope : undefined };
    });

  const evalEnvs = [...new Set(Object.values(evalsByRun).flatMap((e) => e.envs.map((env) => env.name)))].sort();

  return (
    <div className="min-w-0 space-y-6">
      <div className="flex flex-col items-start justify-between gap-3 sm:flex-row">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Compare {runIds.length} runs</h1>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
            {runIds.map((id, index) => {
              const run = runs.find((r) => r.id === id);
              return (
                <button key={id} type="button" className="flex items-center gap-1.5 text-xs text-ink2 hover:text-ink" onClick={() => navigate({ page: "run", runId: id, view: "overview" })}>
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: seriesColor(index) }} />
                  {id}
                  {run && <StatusBadge status={run.status} />}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:justify-end">
          <Segmented value={source} onChange={setSource} size="xs" options={[{ value: "orchestrator", label: "rollouts" }, { value: "trainer", label: "trainer" }, { value: "eval", label: "eval" }]} />
          <button type="button" className={`btn ${picking ? "btn-active" : ""}`} onClick={() => setPicking(true)}><ListPlus className="h-3.5 w-3.5" /> Metrics</button>
          <ChartSettingsMenu settings={settings} onChange={update} showLayout={false} />
        </div>
      </div>
      {selected.length === 0 && (
        <div className="flex flex-col items-center gap-3 py-16 text-center">
          <div className="text-sm font-medium text-ink2">No charts</div>
          <div className="max-w-sm text-xs text-muted">Each chart overlays one metric with one line per run; colors follow the run, not the rank.</div>
          <button type="button" className="btn btn-active" onClick={() => setPicking(true)}><ListPlus className="h-3.5 w-3.5" /> Add metrics</button>
        </div>
      )}
      <div className="grid gap-x-10 gap-y-9 md:grid-cols-2">
        {selected.map((key) => {
          const lines = linesFor(key);
          const isPct = /rate|ratio|is_truncated|mfu|\/(pass@\d+|pass\^\d+)$/.test(key);
          return (
            <ChartCard key={key} title={key} smoothingKey={`compare.${key}`} defaultSmoothing={smoothing} defaultLogScale={settings.logScale} height={200} subtitle={lines.map((l) => `${l.label}: ${isPct ? fmtPct(latestOf(l)) : fmt(latestOf(l), 4)}`).join(" · ")} table={<SeriesTable series={lines} xLabel={xAxis} />}>
              {(o) => <LineChart series={lines} {...chartProps(o)} xLabel={xAxis} markers={source === "eval"} yDomain={isPct ? [0, 1] : undefined} yFormat={isPct ? (v) => fmtPct(v, 0) : undefined} />}
            </ChartCard>
          );
        })}
      </div>
      {source !== "eval" && evalEnvs.length > 0 && (
        <div className="section">
          <div className="title mb-3">Latest evaluation per run</div>
          <table className="w-full">
            <thead><tr><th className="th">Run</th>{evalEnvs.map((env) => <th key={env} className="th text-right">{env}</th>)}<th className="th text-right">at step</th></tr></thead>
            <tbody>
              {runIds.map((id, index) => {
                const history = [...(evalsByRun[id]?.history ?? [])].sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
                const last = history[history.length - 1];
                return (
                  <tr key={id} className="border-b border-edge last:border-0">
                    <td className="td"><span className="mr-2 inline-block h-2 w-2 rounded-sm" style={{ background: seriesColor(index) }} />{id}</td>
                    {evalEnvs.map((env) => {
                      const metrics = last?.envs[env];
                      const key = metrics ? Object.keys(metrics).find((m) => /^avg@\d+$/.test(m)) : undefined;
                      return <td key={env} className="td tabular text-right">{key && metrics ? fmt(metrics[key], 4) : "–"}</td>;
                    })}
                    <td className="td tabular text-right">{last?.step ?? "–"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <Drawer open={picking} onClose={() => setPicking(false)} title="Metrics" subtitle={`${selected.length} selected · ${available.length} logged by ${source}`} width={420}>
        <div className="flex h-full flex-col gap-3">
          <div className="flex items-center gap-2">
            <button type="button" className="btn" onClick={() => setSelected([])} disabled={selected.length === 0}>Clear</button>
            <button type="button" className="btn" onClick={() => updateParams((p) => p.delete("keys"))} disabled={selectedParam === null}>Reset</button>
            <button type="button" className="btn btn-active ml-auto" onClick={() => setPicking(false)}>Done</button>
          </div>
          <div className="min-h-0 flex-1">
            <MetricPicker keys={available} selected={selected} onToggle={(key) => setSelected(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key])} />
          </div>
        </div>
      </Drawer>
    </div>
  );
}
