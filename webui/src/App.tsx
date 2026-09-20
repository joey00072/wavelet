import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  fetchJson,
  normalizeApiBase,
  persistApiBase,
  qs,
  resolveApiBase,
  runUrl,
  usePoll,
} from "./api/client";
import type {
  Evals,
  EvalRowsResponse,
  LogEntry,
  LogTail,
  MetricKeys,
  RolloutBatch,
  RolloutRow,
  RolloutRowsResponse,
  RowDetail,
  RunSummary,
  RunOption,
  Series,
} from "./api/types";
import {
  fmt,
  fmtAge,
  fmtBytes,
  fmtDateTime,
  fmtInt,
  fmtSeconds,
  modelLabel,
} from "./lib/format";
import { MetricChart, type ChartAxis } from "./MetricChart";
import { LiveEpisodes } from "./LiveEpisodes";

type Tab = "metrics" | "traces" | "report" | "logs" | "config";
type MetricSource = "trainer" | "orchestrator" | "eval";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "config", label: "Config" },
  { id: "metrics", label: "Metrics" },
  { id: "traces", label: "Traces" },
  { id: "report", label: "Report" },
  { id: "logs", label: "Logs" },
];

const METRIC_COLORS: Record<MetricSource, string> = {
  trainer: "var(--trainer)",
  orchestrator: "var(--orchestrator)",
  eval: "var(--eval)",
};

const OVERVIEW_KEYS: Record<MetricSource, string[]> = {
  trainer: [
    "reward/all/mean",
    "rollout/count",
    "train/loss",
    "train/policy_loss",
    "optim/grad_norm",
    "perf/tokens_per_second",
    "perf/throughput",
    "perf/train_seconds",
    "perf/rollout_wait_seconds",
    "perf/rollout_load_seconds",
    "perf/policy_export_seconds",
    "entropy/mean",
    "time/step",
  ],
  orchestrator: [
    "reward/episodes/all/mean",
    "reward/episodes/all/count",
    "reward/episodes/effective/mean",
    "reward/episodes/effective/count",
    "reward/all/mean",
    "generation/reward/mean",
    "generation/tokens_per_second",
    "generation/output_tokens_per_second",
    "off_policy/mean",
    "policy/lag",
    "generation/rollouts/cancelled_total",
    "generation/effective_groups/rate",
    "generation/groups/completed",
    "generation/groups/rejected",
    "fate/all/filtered_rate",
    "fate/all/errored_rate",
    "off_policy/in_flight/mean",
    "off_policy/in_queue/mean",
    "is_truncated/all/mean",
    "time/step",
  ],
  eval: [],
};

type Theme = "dark" | "light";

function initialTheme(): Theme {
  try { return localStorage.getItem("wavelet.theme") === "light" ? "light" : "dark"; } catch { return "dark"; }
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  root.setAttribute("data-theme-switching", "");
  root.setAttribute("data-theme", theme);
  window.setTimeout(() => root.removeAttribute("data-theme-switching"), 220);
  try { localStorage.setItem("wavelet.theme", theme); } catch { /* storage unavailable */ }
}

function initialQuery(): { run: string; tab: Tab } {
  const params = new URLSearchParams(window.location.search);
  const requestedTab = params.get("tab");
  const rawTab = (
    requestedTab === "rollouts"
      ? "traces"
      : requestedTab === "evals"
        ? "report"
        : requestedTab
  ) as Tab | null;
  return {
    run: params.get("run") ?? "",
    tab: TABS.some(({ id }) => id === rawTab) ? (rawTab as Tab) : "metrics",
  };
}

export function App() {
  const initial = useMemo(initialQuery, []);
  const [apiBase, setApiBase] = useState<string | null>(null);
  const [apiInput, setApiInput] = useState("");
  const [requestedRun, setRequestedRun] = useState(initial.run);
  const [tab, setTab] = useState<Tab>(initial.tab);
  const [live, setLive] = useState(true);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, []);

  useEffect(() => {
    let active = true;
    resolveApiBase().then((base) => {
      if (!active) return;
      setApiBase(base);
      setApiInput(base);
    });
    return () => {
      active = false;
    };
  }, []);

  const interval = live ? 3000 : 0;
  const runs = usePoll<RunOption[]>(
    apiBase === null ? null : `${apiBase}/api/runs?compact=true`,
    interval ? 15000 : 0,
  );
  const runId = useMemo(() => {
    if (requestedRun && runs.data?.some((run) => run.id === requestedRun)) {
      return requestedRun;
    }
    return runs.data?.find((run) => run.is_current)?.id ?? runs.data?.[0]?.id ?? null;
  }, [requestedRun, runs.data]);
  const summary = usePoll<RunSummary>(
    runId && apiBase !== null ? runUrl(apiBase, runId, "/summary") : null,
    interval,
    { resourceKey: runId },
  );

  useEffect(() => {
    if (apiBase !== null) persistApiBase(apiBase);
  }, [apiBase]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (runId) params.set("run", runId);
    else params.delete("run");
    params.set("tab", tab);
    if (params.get("api") === "same") params.delete("api");
    window.history.replaceState(null, "", `${window.location.pathname}?${params}`);
    document.title = runId ? `${runId} · Wavelet` : "Wavelet Dashboard";
  }, [runId, tab]);

  const applyApi = () => setApiBase(normalizeApiBase(apiInput));
  const current = summary.data;

  return (
    <div className="app-shell" data-run-status={current?.status ?? "idle"}>
      <a className="skip-link" href="#content">Skip to dashboard content</a>
      <aside className="sidebar">
        <div className="sidebar-top">
          <a className="brand" href="?tab=metrics" aria-label="Wavelet dashboard">
            <WaveMark />
            <span>Wavelet</span>
          </a>
          <div className="sidebar-actions">
            <label className="live-toggle">
              <input
                type="checkbox"
                checked={live}
                onChange={(event) => setLive(event.target.checked)}
              />
              <span className="live-dot" />
              live
            </label>
            <button
              type="button"
              className="theme-toggle"
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              title={theme === "dark" ? "Light mode" : "Dark mode"}
              onClick={() => { const next: Theme = theme === "dark" ? "light" : "dark"; setTheme(next); applyTheme(next); }}
            >
              <svg className="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
              <svg className="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
            </button>
          </div>
        </div>
        <label className="run-picker">
          <span className="sr-only">Run</span>
          <select
            value={runId ?? ""}
            onChange={(event) => { const params = new URLSearchParams(location.search); params.delete("step"); params.delete("row"); history.replaceState(null, "", `${location.pathname}?${params}`); setRequestedRun(event.target.value); }}
            disabled={!runs.data?.length}
          >
            {!runs.data?.length && <option value="">no runs</option>}
            {runs.data?.map((run) => (
              <option key={run.id} value={run.id}>
                {run.status === "running" ? "● " : ""}{run.id}
              </option>
            ))}
          </select>
        </label>
        <nav className="tabs" aria-label="Dashboard sections">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              className={tab === id ? "active" : ""}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        {current && <RunHeader summary={current} />}
      </aside>

      <main id="content">
        {apiBase === null || runs.loading ? (
          <Empty title="Connecting to dashboard…" />
        ) : runs.error ? (
          <Empty title="Cannot reach the API" detail={runs.error}>
            <div className="api-connect">
              <input
                aria-label="API base URL"
                value={apiInput}
                onChange={(event) => setApiInput(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && applyApi()}
                placeholder="http://host:8766"
              />
              <button type="button" onClick={applyApi}>Connect</button>
            </div>
          </Empty>
        ) : !runs.data?.length ? (
          <Empty title="No runs found" detail="Point the dashboard at a run directory or a runs root." />
        ) : !runId || !current ? (
          <Empty title={summary.error ? "Run unavailable" : "Loading run…"} detail={summary.error} />
        ) : (
          <>
            {tab === "metrics" && <MetricsView key={runId} apiBase={apiBase} runId={runId} interval={interval} summary={current} runs={runs.data ?? []} />}
            {tab === "traces" && <TracesView key={runId} apiBase={apiBase} runId={runId} interval={interval} />}
            {tab === "report" && <ReportView key={runId} apiBase={apiBase} runId={runId} interval={interval} />}
            {tab === "logs" && <LogsView key={runId} apiBase={apiBase} runId={runId} interval={interval} />}
            {tab === "config" && <ConfigView key={runId} apiBase={apiBase} runId={runId} />}
          </>
        )}
      </main>
    </div>
  );
}

function WaveMark() {
  return (
    <svg className="wave-mark" viewBox="0 0 34 20" aria-hidden="true">
      <path d="M1 12c4-10 8-10 12 0s8 10 12 0 6-7 8-2" />
    </svg>
  );
}

function RunHeader({ summary }: { summary: RunSummary }) {
  const step = summary.trainer_step ?? numberValue(summary.latest.orchestrator?.step);
  const runType = summary.algo ? "RL" : "RUN";
  return (
    <section className="run-header" aria-label="Run overview">
      <HeaderField label="status">
        <span className={`status status-${summary.status}`}>{summary.status}</span>
      </HeaderField>
      <HeaderField label="type"><span>{runType}</span></HeaderField>
      <HeaderField label="step" primary>
        <span>{fmtInt(step)} <small>/ {fmtInt(summary.target_step)}</small></span>
        <div className="run-progress" aria-hidden="true" data-done={summary.status === "completed"}>
          <span style={{ transform: `scaleX(${progress(step, summary.target_step)})` }} />
        </div>
      </HeaderField>
      <HeaderField label="model">
        <span title={summary.model ?? ""}>{modelLabel(summary.model)}</span>
      </HeaderField>
      <HeaderField label="train envs"><span>{summary.envs.join(", ") || "–"}</span></HeaderField>
      <HeaderField label="eval envs"><span>{summary.eval_envs.join(", ") || "–"}</span></HeaderField>
      <HeaderField label="duration"><span>{runDuration(summary)}</span></HeaderField>
      <HeaderField label="started"><span>{fmtAge(summary.started_at)}</span></HeaderField>
    </section>
  );
}

function HeaderField({ label, primary, children }: { label: string; primary?: boolean; children: ReactNode }) {
  return <div className={primary ? "header-field primary" : "header-field"}><span>{label}</span>{children}</div>;
}

function progress(step: number | null, target: number | null | undefined): number {
  if (step === null || !target || target <= 0) return 0;
  return Math.min(1, Math.max(0, step / target));
}

type ViewProps = { apiBase: string; runId: string; interval: number };

function MetricsView({ apiBase, runId, interval, summary, runs }: ViewProps & { summary: RunSummary; runs: RunOption[] }) {
  const keysState = usePoll<MetricKeys>(runUrl(apiBase, runId, "/metrics/keys"), interval ? 15000 : 0, { resourceKey: runId });
  const [mode, setMode] = useState<"overview" | "all">("overview");
  const [search, setSearch] = useState("");
  const [smoothing, setSmoothing] = useState(1);
  const [sectionsOpen, setSectionsOpen] = useState(true);
  const [axis, setAxis] = useState<ChartAxis>("step");
  const [compareRun, setCompareRun] = useState("");
  const saved = useMemo(() => {
    try { const value = JSON.parse(localStorage.getItem(`wavelet.metrics.${runId}`) ?? "null"); return value && ["trainer", "orchestrator", "eval"].every((k) => Array.isArray(value[k]) && value[k].every((v: unknown) => typeof v === "string")) ? value as Record<MetricSource, string[]> : null; } catch { return null; }
  }, [runId]);
  const [selected, setSelected] = useState<Record<MetricSource, string[]>>(saved ?? { trainer: [], orchestrator: [], eval: [] });
  const [customized, setCustomized] = useState(Boolean(saved));
  useEffect(() => { if (customized) localStorage.setItem(`wavelet.metrics.${runId}`, JSON.stringify(selected)); }, [selected, customized, runId]);

  useEffect(() => {
    const available = keysState.data;
    if (!available || customized) return;
    const next = { trainer: [], orchestrator: [], eval: [] } as Record<MetricSource, string[]>;
    for (const source of Object.keys(next) as MetricSource[]) {
      const names = new Set((available[source] ?? []).map(({ key }) => key));
      next[source] = OVERVIEW_KEYS[source].filter((key) => names.has(key));
    }
    if (next.orchestrator.includes("reward/episodes/all/mean")) {
      next.trainer = next.trainer.filter((key) => key !== "reward/all/mean");
      next.orchestrator = next.orchestrator.filter((key) => !["reward/all/mean", "generation/reward/mean"].includes(key));
    } else if (next.trainer.includes("reward/all/mean")) {
      // Historical trainer reward is episode weighted; the orchestrator key is problem weighted.
      next.orchestrator = next.orchestrator.filter((key) => !["reward/all/mean", "generation/reward/mean"].includes(key));
    }
    next.orchestrator.push(...(available.orchestrator ?? []).map(({ key }) => key).filter(key => /^inference\/replica_\d+\/(kv_cache_usage|requests_running|requests_waiting|preemptions_delta|generation_tokens_per_second|prompt_tokens_per_second)$/.test(key)));
    next.eval = (available.eval ?? []).map(({ key }) => key).filter((key) => /avg@|pass@|reward/.test(key)).slice(0, 2);
    setSelected(next);
  }, [keysState.data, customized]);

  const trainer = useMetricSeries(apiBase, runId, "trainer", selected.trainer, interval);
  const orchestrator = useMetricSeries(apiBase, runId, "orchestrator", selected.orchestrator, interval);
  const evaluation = useMetricSeries(apiBase, runId, "eval", selected.eval, interval);
  const compareTrainer = useMetricSeries(apiBase, compareRun, "trainer", compareRun ? selected.trainer : [], interval);
  const compareOrchestrator = useMetricSeries(apiBase, compareRun, "orchestrator", compareRun ? selected.orchestrator : [], interval);
  const compareEval = useMetricSeries(apiBase, compareRun, "eval", compareRun ? selected.eval : [], interval);
  const comparison = { trainer: compareTrainer.data, orchestrator: compareOrchestrator.data, eval: compareEval.data };
  const series: Record<MetricSource, Series | null> = { trainer: trainer.data, orchestrator: orchestrator.data, eval: evaluation.data };
  const errors = [keysState.error, trainer.error, orchestrator.error, evaluation.error, compareTrainer.error, compareOrchestrator.error, compareEval.error].filter(Boolean);

  const toggleMetric = (source: MetricSource, key: string) => {
    setCustomized(true);
    setSelected((current) => {
      const active = current[source].includes(key);
      return { ...current, [source]: active ? current[source].filter((item) => item !== key) : [...current[source], key] };
    });
  };

  return (
    <section className="view" data-view="metrics">
      <div className="controls">
        <Segmented value={mode} values={["overview", "all"]} onChange={(value) => setMode(value as "overview" | "all")} />
        <input className="search" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="regex filter, e.g. reward or ^perf/" aria-label="Filter metrics" />
        <label className="range-control">smooth
          <input type="range" min="1" max="20" value={smoothing} onChange={(event) => setSmoothing(Number(event.target.value))} />
          <span>{smoothing === 1 ? "off" : smoothing}</span>
        </label>
        <select aria-label="Compare run" value={compareRun} onChange={(e) => setCompareRun(e.target.value)}><option value="">Compare run…</option>{runs.filter((r) => r.id !== runId && r.model).map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}</select>
        <label className="control-label">x axis <select aria-label="Chart x axis" value={axis} onChange={(e) => setAxis(e.target.value as ChartAxis)}><option value="step">step</option><option value="elapsed">elapsed time</option></select></label>
        <span className="muted">{keysState.data ? `${Object.values(keysState.data).flat().length} signals` : "loading signals…"}</span>
        <div className="control-spacer" />
        <button type="button" onClick={() => { localStorage.removeItem(`wavelet.metrics.${runId}`); setCustomized(false); }}>Reset metrics</button>
        <button type="button" onClick={() => setSectionsOpen(false)}>Collapse</button>
        <button type="button" onClick={() => setSectionsOpen(true)}>Expand</button>
      </div>

      {mode === "all" && keysState.data && (
        <div className="metric-browser">
          {(Object.keys(METRIC_COLORS) as MetricSource[]).map((source) => {
            const matching = (keysState.data?.[source] ?? []).map(({ key }) => key).filter((key) => !search || Boolean(compileRegex(search)?.test(key)));
            return (
              <details key={source} open={search.length > 0}>
                <summary><span className={`source-dot source-${source}`} />{source}<span>{matching.length}</span></summary>
                <div className="metric-keys">
                  {matching.map((key) => (
                    <button key={key} type="button" className={selected[source].includes(key) ? "selected" : ""} onClick={() => toggleMetric(source, key)}>{key}</button>
                  ))}
                </div>
              </details>
            );
          })}
        </div>
      )}

      {search && !compileRegex(search) && <InlineError>Invalid regular expression.</InlineError>}
      {errors.length > 0 && <InlineError>{errors.join(" · ")}</InlineError>}
      <MetricSections
        showBlocks={mode === "overview" && !customized && !search}
        selected={selected}
        series={series}
        comparison={comparison}
        compareRun={compareRun}
        search={search}
        smoothing={smoothing}
        axis={axis}
        open={sectionsOpen}
        trainEnvs={summary.envs}
        evalEnvs={summary.eval_envs}
        onRemove={mode === "all" ? toggleMetric : undefined}
      />
      {Object.values(selected).flat().length === 0 && <Empty title="No metrics selected" detail="Open All and select any logged signal." />}
    </section>
  );
}

const SECTION_ORDER = ["train", "eval", "stability", "inference", "performance", "queue", "other"];

function metricSection(source: MetricSource, key: string): string {
  if (source === "eval" || key.startsWith("eval/")) return "eval";
  if (key.startsWith("inference/")) return "inference";
  if (/^(optim|entropy|mismatch|kl|ipo|dppo|grad|train\/(loss|policy_loss))/.test(key)) return "stability";
  if (/^(perf|time|memory|system|node)/.test(key) || key.includes("tokens_per_second")) return "performance";
  if (/^(off_policy|policy\/lag|generation\/(rollouts|groups|effective_groups))/.test(key)) return "queue";
  if (source === "orchestrator" || /^(train|reward|loss|generation|advantage|fate|off_policy|policy)/.test(key)) return "train";
  return "other";
}

function MetricSections({ showBlocks, selected, series, comparison, compareRun, search, smoothing, axis, open, trainEnvs, evalEnvs, onRemove }: {
  showBlocks: boolean;
  selected: Record<MetricSource, string[]>;
  series: Record<MetricSource, Series | null>;
  comparison: Record<MetricSource, Series | null>;
  compareRun: string;
  search: string;
  smoothing: number;
  axis: ChartAxis;
  open: boolean;
  trainEnvs: string[];
  evalEnvs: string[];
  onRemove?: (source: MetricSource, key: string) => void;
}) {
  const terms = search.trim();
  const filter = compileRegex(terms);
  const charts = (Object.keys(selected) as MetricSource[]).flatMap((source) =>
    selected[source]
      .filter((key) => !terms || Boolean(filter?.test(key)))
      .map((key) => ({ source, key })),
  ).sort((a, b) => {
    const priority = (key: string) => key === "reward/episodes/effective/mean" ? 3 : key === "reward/episodes/all/mean" ? 2 : key === "reward/all/mean" ? 1 : 0;
    return priority(b.key) - priority(a.key);
  });
  const groups = new Map<string, typeof charts>();
  for (const chart of charts) {
    const section = metricSection(chart.source, chart.key);
    groups.set(section, [...(groups.get(section) ?? []), chart]);
  }
  return (
    <div className="metric-sections">
      {SECTION_ORDER.filter((section) => groups.has(section) || (showBlocks && ["train", "eval", "inference"].includes(section))).map((section) => (
        <details className="metric-section" key={`${section}:${open}`} open={open}>
          <summary>{sectionTitle(section)}{["train", "eval"].includes(section) && <span className="section-env" title={(section === "train" ? trainEnvs : evalEnvs).join(", ")}>{(section === "train" ? trainEnvs : evalEnvs).join(", ") || "No environment configured"}</span>}</summary>
          <div className="chart-grid">
            {groups.get(section)?.map(({ source, key }) => (
              <MetricChart
                key={`${source}:${key}`}
                color={METRIC_COLORS[source]}
                axis={axis}
                label={metricLabel(key, source)}
                description={metricDescription(key, source)}
                stepLabel={source === "trainer" ? "Optimizer step" : source === "eval" ? "Evaluation step" : "Rollout queue step"}
                name={key}
                data={series[source]}
                comparisonData={comparison[source]}
                comparisonLabel={compareRun}
                smoothing={smoothing}
                onRemove={onRemove ? () => onRemove(source, key) : undefined}
              />
            ))}
          </div>
          {!groups.has(section) && <div className="section-empty">{section === "eval" ? "No evaluation metrics recorded" : section === "inference" ? "No inference telemetry recorded" : "No training metrics recorded"}</div>}
        </details>
      ))}
      {charts.length === 0 && search && <Empty title="No metrics match" detail={search} />}
    </div>
  );
}

function useMetricSeries(apiBase: string, runId: string, source: MetricSource, keys: string[], interval: number) {
  const url = keys.length ? `${runUrl(apiBase, runId, "/series")}${qs({ source, keys: keys.join(","), points: 1200 })}` : null;
  return usePoll<Series>(url, interval, { resourceKey: `${runId}:${source}` });
}


function TracesView({ apiBase, runId, interval }: ViewProps) {
  const batches = usePoll<RolloutBatch[]>(runUrl(apiBase, runId, "/rollouts?limit=500"), interval, { resourceKey: runId });
  const [mode, setMode] = useState<"latest" | "step">(new URLSearchParams(location.search).has("step") ? "step" : "latest");
  const [step, setStep] = useState<number | null>(() => { const v = new URLSearchParams(location.search).get("step"); return v !== null && /^\d+$/.test(v) ? Number(v) : null; });
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("row_index:desc");
  const [env, setEnv] = useState("");
  const [detailIndex, setDetailIndex] = useState<number | null>(() => { const v = new URLSearchParams(location.search).get("row"); return v !== null && /^\d+$/.test(v) ? Number(v) : null; });
  const [page, setPage] = useState(0);
  const [status, setStatus] = useState("");
  const activeStep = mode === "latest" ? batches.data?.[0]?.queue_step ?? null : step ?? batches.data?.[0]?.queue_step ?? null;
  const [sortKey, sortOrder] = sort.split(":");
  const rowsUrl = activeStep === null ? null : `${runUrl(apiBase, runId, "/rollouts/rows")}${qs({ step: activeStep, sort: sortKey, order: sortOrder, limit: 50, include_text: false, offset: page * 50, search, env, has_error: status === "error" ? true : status === "ok" ? false : null, truncated: status === "truncated" ? true : null })}`;
  const rows = usePoll<RolloutRowsResponse>(rowsUrl, interval, { resourceKey: `${runId}:${activeStep}` });
  const detail = usePoll<RowDetail>(detailIndex === null || activeStep === null ? null : runUrl(apiBase, runId, `/rollouts/${activeStep}/rows/${detailIndex}`), 0, { resourceKey: `${runId}:${activeStep}:${detailIndex}` });
  useEffect(() => { setPage(0); }, [activeStep, search, env, sort, status]);
  useEffect(() => { const params = new URLSearchParams(location.search); if (mode === "step" && activeStep !== null) params.set("step", String(activeStep)); else params.delete("step"); if (detailIndex !== null) params.set("row", String(detailIndex)); else params.delete("row"); history.replaceState(null, "", `${location.pathname}?${params}`); }, [mode, activeStep, detailIndex]);
  const selectRow = (index: number | null) => { if (index !== null) { setStep(activeStep); setMode("step"); } setDetailIndex(index); };
  const activeBatch = batches.data?.find((batch) => batch.queue_step === activeStep);
  const batchIndex = batches.data?.findIndex((batch) => batch.queue_step === activeStep) ?? -1;
  const moveStep = (direction: number) => {
    const next = batches.data?.[batchIndex + direction];
    if (next) {
      setStep(next.queue_step);
      setDetailIndex(null);
    }
  };
  return (
    <section className="view traces-view" data-view="traces">
      <LiveEpisodes apiBase={apiBase} runId={runId} interval={interval} />
      <div className="controls" id="trace-bar">
        <Segmented value={mode} values={["latest", "step"]} onChange={(value) => { setMode(value as "latest" | "step"); setDetailIndex(null); }} />
        <details className="filter-menu">
          <summary>filter{env || search ? <span className="filter-count">{Number(Boolean(env)) + Number(Boolean(search))}</span> : null}</summary>
          <div className="filter-popover">
            <span className="eyebrow">filter</span>
            <label className="control-label">kind <span className="kind-chip active">train</span></label>
            <label className="control-label">env
              <select value={env} onChange={(event) => setEnv(event.target.value)}>
                <option value="">all envs</option>
                {Object.keys(rows.data?.stats?.envs ?? {}).map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
            <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="prompt or completion" aria-label="Search traces" />
          </div>
        </details>
        <select aria-label="Sort traces" value={sort} onChange={(event) => setSort(event.target.value)}>
          <option value="row_index:desc">arrival ↓</option>
          <option value="row_index:asc">arrival ↑</option>
          <option value="group_key:asc">group</option>
          <option value="reward:desc">reward ↓</option>
          <option value="reward:asc">reward ↑</option>
          <option value="completion_token_count:desc">tokens ↓</option>
        </select>
        <select aria-label="Trace status" value={status} onChange={(e) => setStatus(e.target.value)}><option value="">all outcomes</option><option value="ok">no error</option><option value="error">error</option><option value="truncated">truncated</option></select>
        <span className="muted">{rows.data ? `${rows.data.filtered} sequence${rows.data.filtered === 1 ? "" : "s"}` : "loading…"}</span>
      </div>
      {mode === "step" && (
        <div className="step-bar">
          <button type="button" className="step-nav" disabled={batchIndex >= (batches.data?.length ?? 0) - 1} onClick={() => moveStep(1)}>‹</button>
          <select value={activeStep ?? ""} onChange={(event) => { setStep(Number(event.target.value)); setDetailIndex(null); }}>
            {batches.data?.map((batch) => <option key={batch.queue_step} value={batch.queue_step}>step {batch.queue_step} · {batch.status}</option>)}
          </select>
          <button type="button" className="step-nav" disabled={batchIndex <= 0} onClick={() => moveStep(-1)}>›</button>
          <span className="eyebrow">step {activeStep ?? "–"}</span>
        </div>
      )}
      {rows.error && <InlineError>{rows.error}</InlineError>}
      {rows.data?.available ? (
          <DataTable headers={["#", "arrived", "took", "kind", "env", "group", "reward", "tok", "turns", "stop", "status"]}>
            {rows.data.rows.map((row) => (
              <tr key={row.row_index} tabIndex={0} onClick={() => selectRow(row.row_index)} onKeyDown={(event) => event.key === "Enter" && selectRow(row.row_index)}>
                <td className="muted">{row.row_index}</td><td className="muted">{fmtDateTime(activeBatch?.created_at)}</td><td className="muted">{fmtSeconds(row.duration_seconds)}</td><td className="muted">train</td>
                <td>{row.env ?? "?"}</td><td className="muted" title={row.group_key ?? ""}>{short(row.group_key, 8)}</td>
                <td className={rewardClass(row.reward)}>{fmt(row.reward)}</td><td><span className="muted">in</span> {fmtInt(row.input_token_count)} <span className="muted">· out</span> {fmtInt(row.completion_token_count)}</td>
                <td>{fmtInt(row.turn_count)}</td><td className="muted">{row.is_truncated ? "truncated" : row.stop_condition ?? ""}</td>
                <td className={row.error ? "negative" : "positive"}>{row.error ? "err" : "ok"}</td>
              </tr>
            ))}
          </DataTable>
      ) : <Empty title="No traces yet" detail={rows.data?.reason} />}
      <div className="controls trace-pagination"><button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous page</button><span>{rows.data?.filtered ? `${page * 50 + 1}–${Math.min((page + 1) * 50, rows.data.filtered)} of ${rows.data.filtered}` : "0 rows"}</span><button type="button" disabled={!rows.data || (page + 1) * 50 >= rows.data.filtered} onClick={() => setPage(page + 1)}>Next page</button><span className="muted">Retained batches only; consumed payloads may expire.</span></div>
      {detailIndex !== null && rows.data && (
        <TraceViewer
          step={activeStep}
          rows={rows.data.rows}
          selectedIndex={detailIndex}
          detail={detail.data}
          error={detail.error}
          onSelect={selectRow}
          onClose={() => selectRow(null)}
        />
      )}
    </section>
  );
}

function ReportView({ apiBase, runId, interval }: ViewProps) {
  const evals = usePoll<Evals>(runUrl(apiBase, runId, "/evals"), interval, { resourceKey: runId });
  const [selectedSet, setSelectedSet] = useState("");
  const [detailIndex, setDetailIndex] = useState<number | null>(null);
  const sets = evals.data?.sets ?? [];
  const newestSet = sets.length > 0 ? sets[sets.length - 1] : null;
  const setKey = selectedSet || (newestSet ? `${newestSet.step}|${newestSet.env}` : "");
  const [stepText, env = ""] = setKey.split("|");
  const step = stepText ? Number(stepText) : null;
  const rows = usePoll<EvalRowsResponse>(step === null || !env ? null : `${runUrl(apiBase, runId, `/evals/${step}/${encodeURIComponent(env)}/rows`)}${qs({ sort: "reward", order: "desc", limit: 200 })}`, 0, { resourceKey: `${runId}:${setKey}` });
  const detail = usePoll<RowDetail>(detailIndex === null || step === null || !env ? null : runUrl(apiBase, runId, `/evals/${step}/${encodeURIComponent(env)}/rows/${detailIndex}`), 0, { resourceKey: `${runId}:${setKey}:${detailIndex}` });
  return (
    <section className="view" data-view="report">
      <div className="controls">
        <label className="control-label">evaluation
          <select value={setKey} onChange={(event) => { setSelectedSet(event.target.value); setDetailIndex(null); }}>
            {sets.map((entry) => <option key={`${entry.step}|${entry.env}`} value={`${entry.step}|${entry.env}`}>{entry.env} · policy {entry.step}</option>)}
          </select>
        </label>
        <span className="muted">{evals.data ? `${evals.data.history.length} metric snapshots · ${sets.length} saved sets` : "loading…"}</span>
      </div>
      {evals.error && <InlineError>{evals.error}</InlineError>}
      {evals.data?.history.length ? <EvalHistory history={evals.data.history} /> : null}
      {rows.data?.available ? (
        <>
          <div className="stats-grid compact">
            <Stat label="reward mean" value={fmt(rows.data.stats?.reward.mean)} />
            <Stat label="reward std" value={fmt(rows.data.stats?.reward.std)} />
            <Stat label="examples" value={fmtInt(rows.data.examples.length)} />
            <Stat label="attempts" value={fmtInt(rows.data.total)} />
            <Stat label="errors" value={fmtInt(rows.data.stats?.errors)} />
            <Stat label="truncated" value={fmtInt(rows.data.stats?.truncated)} />
          </div>
          <DataTable headers={["#", "example", "reward", "tokens", "stop", "answer", "output"]}>
            {rows.data.rows.map((row) => (
              <tr key={row.row_index} tabIndex={0} onClick={() => setDetailIndex(row.row_index)} onKeyDown={(event) => event.key === "Enter" && setDetailIndex(row.row_index)}>
                <td>{row.row_index}</td><td>{short(row.example_id, 20)}</td><td className={rewardClass(row.reward)}>{fmt(row.reward)}</td>
                <td>{fmtInt(row.completion_token_count)}</td><td>{row.has_error ? "error" : row.is_truncated ? "truncated" : row.stop_condition ?? "–"}</td>
                <td className="preview-cell">{row.answer ?? "–"}</td><td className="preview-cell">{row.completion ?? "–"}</td>
              </tr>
            ))}
          </DataTable>
        </>
      ) : <Empty title="No saved evaluation samples" detail="Evaluation metrics still appear above when only aggregate history was retained." />}
      {detailIndex !== null && <SampleDrawer title={`Eval ${detailIndex} · ${env} · policy ${step}`} detail={detail.data} error={detail.error} onClose={() => setDetailIndex(null)} />}
    </section>
  );
}

function EvalHistory({ history }: { history: Evals["history"] }) {
  const rows = history.slice().reverse().slice(0, 20);
  return (
    <details className="history" open>
      <summary>Evaluation history <span>{history.length}</span></summary>
      <DataTable headers={["step", "policy", "environment", "metric", "value"]}>
        {rows.flatMap((row, rowIndex) => Object.entries(row.envs).flatMap(([env, metrics]) => Object.entries(metrics).map(([metric, value]) => (
          <tr key={`${rowIndex}:${env}:${metric}`}><td>{fmtInt(row.step)}</td><td>{fmtInt(row.policy_step)}</td><td>{env}</td><td>{metric}</td><td>{fmt(value)}</td></tr>
        ))))}
      </DataTable>
    </details>
  );
}

function LogsView({ apiBase, runId, interval }: ViewProps) {
  const logs = usePoll<LogEntry[]>(runUrl(apiBase, runId, "/logs"), interval, { resourceKey: runId });
  const [selected, setSelected] = useState<string[]>([]);
  const [selectionRun, setSelectionRun] = useState("");
  const [view, setView] = useState<"merge" | "split">("merge");
  const levels = ["all", "debug", "info", "warn", "error"];
  const [levelIndex, setLevelIndex] = useState(0);
  const [search, setSearch] = useState("");
  const [lineLimit, setLineLimit] = useState(800);
  useEffect(() => {
    if (!logs.data || selectionRun === runId) return;
    setSelected(logs.data.map((log) => log.name));
    setSelectionRun(runId);
  }, [logs.data, runId, selectionRun]);
  const tails = useLogTails(apiBase, runId, selected, lineLimit, interval);
  const regex = compileRegex(search);
  const level = levels[levelIndex];
  const visible = Object.fromEntries(Object.entries(tails.data).map(([name, lines]) => [
    name,
    lines.filter((line) => logLevelMatches(line, level) && (!regex || regex.test(line))),
  ]));
  const merged = Object.entries(visible).flatMap(([name, lines]) => lines.map((line) => `[${name}] ${line}`));
  const toggleComponent = (name: string) => setSelected((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name]);
  return (
    <section className="view logs-view" data-view="logs">
      <div className="controls">
        <label className="control-label">attempt
          <select value="latest" disabled><option>latest</option></select>
        </label>
        <details className="filter-menu component-menu">
          <summary>components<span className="filter-count">{selected.length}</span></summary>
          <div className="filter-popover">
            {logs.data?.map((log) => <label className="component-option" key={log.name}><input type="checkbox" checked={selected.includes(log.name)} onChange={() => toggleComponent(log.name)} /><span>{log.name}</span><small>{fmtBytes(log.bytes)}</small></label>)}
          </div>
        </details>
        <Segmented value={view} values={["merge", "split"]} onChange={(value) => setView(value as "merge" | "split")} />
        <div className="level-stepper"><button type="button" aria-label="More verbose" disabled={levelIndex === 0} onClick={() => setLevelIndex((index) => index - 1)}>‹</button><span>{level}</span><button type="button" aria-label="Less verbose" disabled={levelIndex === levels.length - 1} onClick={() => setLevelIndex((index) => index + 1)}>›</button></div>
        <input className="search" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="regex filter, e.g. ERROR|WARN" aria-label="Filter logs" />
        <button type="button" onClick={() => setLineLimit((limit) => Math.min(limit * 2, 5000))} disabled={lineLimit >= 5000}>Load older</button>
        <span className="muted">{tails.loading ? "loading…" : `${view === "merge" ? merged.length : Object.values(visible).flat().length} lines`}</span>
      </div>
      {search && !regex && <InlineError>Invalid regular expression.</InlineError>}
      {tails.error && <InlineError>{tails.error}</InlineError>}
      {!logs.data?.length ? <Empty title="No logs found" /> : view === "merge" ? (
        <pre className="log-output">{merged.join("\n") || "No lines match the current filters."}</pre>
      ) : (
        <div className="log-panes">{selected.map((name) => <section className="log-pane" key={name}><header><span>{name}</span><span>{visible[name]?.length ?? 0} lines</span></header><pre>{visible[name]?.join("\n") || "No matching lines."}</pre></section>)}</div>
      )}
    </section>
  );
}

function ConfigView({ apiBase, runId }: Omit<ViewProps, "interval">) {
  const config = usePoll<Record<string, unknown>>(runUrl(apiBase, runId, "/config"), 0, { resourceKey: runId });
  const [search, setSearch] = useState("");
  const json = JSON.stringify(config.data, null, 2);
  const regex = compileRegex(search);
  const matching = search && regex ? json.split("\n").filter((line) => regex.test(line)) : json.split("\n");
  const lines = matching.join("\n");
  return (
    <section className="view config-view" data-view="config">
      <div className="controls">
        <label className="control-label">attempt <select value="latest" disabled><option>latest</option></select></label>
        <div className="segmented"><button type="button" className="active">JSON</button></div>
        <input className="search" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="regex search, e.g. lora|lr" aria-label="Filter configuration" />
        <span className="muted">{search && regex ? `${matching.length} hits` : "resolved and redacted"}</span>
        <div className="control-spacer" />
        <button type="button" onClick={() => navigator.clipboard.writeText(json)}>Copy JSON</button>
      </div>
      {search && !regex && <InlineError>Invalid regular expression.</InlineError>}
      {config.error ? <InlineError>{config.error}</InlineError> : <pre className="config-output">{lines || "No config lines match."}</pre>}
    </section>
  );
}

function useLogTails(apiBase: string, runId: string, names: string[], lineLimit: number, interval: number): { data: Record<string, string[]>; error: string | null; loading: boolean } {
  const [data, setData] = useState<Record<string, string[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const key = names.slice().sort().join("|");
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let pending = false;
    let loaded = false;
    const controller = new AbortController();
    const load = async () => {
      if (cancelled || pending) return;
      if (document.hidden) return;
      if (!names.length) {
        setData({});
        setLoading(false);
        return;
      }
      pending = true;
      try {
        const payloads = await Promise.all(names.map((name) => fetchJson<LogTail>(`${runUrl(apiBase, runId, `/logs/${encodeURIComponent(name)}`)}?lines=${lineLimit}`, controller.signal)));
        if (!cancelled) {
          loaded = true;
          setData(Object.fromEntries(payloads.map((payload) => [payload.name, payload.lines])));
          setError(null);
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        pending = false;
        if (!cancelled) {
          setLoading(false);
          if (interval > 0) timer = window.setTimeout(load, interval);
        }
      }
    };
    setLoading(true);
    const onVisible = () => { if (!document.hidden && (interval > 0 || !loaded)) { window.clearTimeout(timer); void load(); } };
    document.addEventListener("visibilitychange", onVisible);
    load();
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      controller.abort();
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase, runId, key, lineLimit, interval]);
  return { data, error, loading };
}

function TraceViewer({ step, rows, selectedIndex, detail, error, onSelect, onClose }: {
  step: number | null;
  rows: RolloutRow[];
  selectedIndex: number;
  detail: RowDetail | null;
  error: string | null;
  onSelect: (index: number) => void;
  onClose: () => void;
}) {
  const selected = rows.find((row) => row.row_index === selectedIndex) ?? null;
  const position = rows.findIndex((row) => row.row_index === selectedIndex);
  const move = (delta: number) => {
    const next = rows[position + delta];
    if (next) onSelect(next.row_index);
  };
  useEffect(() => {
    const navigate = (event: KeyboardEvent) => {
      if (event.key === "Escape") { onClose(); return; }
      if ((event.target as HTMLElement)?.closest("input, textarea, select, button")) return;
      if (event.key.startsWith("Arrow")) event.preventDefault();
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowDown" || event.key === "ArrowRight") move(1);
    };
    window.addEventListener("keydown", navigate);
    return () => window.removeEventListener("keydown", navigate);
  });
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="trace-modal" role="dialog" aria-modal="true" aria-label={`Trace Viewer · sequence ${selectedIndex}`}>
        <aside className="trace-list-pane">
          <div className="trace-pane-head step-head"><button type="button" disabled={position <= 0} onClick={() => move(-1)}>‹</button><span>step {step ?? "–"}</span><button type="button" disabled={position < 0 || position >= rows.length - 1} onClick={() => move(1)}>›</button></div>
          <div className="trace-pane-head"><span className="eyebrow">sequences</span><span className="filter-count">{rows.length}</span></div>
          <div className="trace-list">
            {rows.map((row) => (
              <button key={row.row_index} type="button" className={`trace-list-item${row.row_index === selectedIndex ? " active" : ""}${row.error ? " error" : ""}`} onClick={() => onSelect(row.row_index)}>
                <span>#{row.row_index}</span><span title={row.env ?? ""}>{row.env ?? "?"}</span><span className={rewardClass(row.reward)}>{fmt(row.reward)}</span>
              </button>
            ))}
          </div>
        </aside>
        <section className="trace-main-pane">
          <div className="trace-pane-head trace-toolbar">
            <span className="trace-title">Trace Viewer</span>
            <span className="muted">Transcript</span>
            <div className="control-spacer" />
            <button type="button" onClick={() => navigator.clipboard.writeText(location.href)}>Copy link</button>
            <button type="button" onClick={onClose} aria-label="Close detail">✕</button>
          </div>
          <div className="trace-messages">
            {error ? <InlineError>{error}</InlineError> : detail ? <TraceMessages key={`${step}:${selectedIndex}`} detail={detail} /> : <Empty title="Loading sequence…" />}
          </div>
        </section>
        <aside className="trace-overview-pane">
          <div className="trace-pane-head"><span className="trace-title">Overview</span></div>
          <div className="trace-overview">
            <div className={`reward-big ${rewardClass(selected?.reward ?? null)}`}>{fmt(selected?.reward)}</div>
            <span className="eyebrow">reward</span>
            <dl>
              <TraceField label="sequence" value={`#${selectedIndex}`} />
              <TraceField label="kind" value="train" />
              <TraceField label="environment" value={selected?.env} />
              <TraceField label="step" value={fmtInt(step)} />
              <TraceField label="policy" value={fmtInt(selected?.policy_step)} />
              <TraceField label="group" value={selected?.group_key} />
              <TraceField label="input tokens" value={fmtInt(selected?.input_token_count)} />
              <TraceField label="output tokens" value={fmtInt(selected?.completion_token_count)} />
              <TraceField label="turns" value={fmtInt(selected?.turn_count)} />
              <TraceField label="tool calls" value={fmtInt(selected?.tool_calls)} />
              <TraceField label="duration" value={fmtSeconds(selected?.duration_seconds)} />
              <TraceField label="advantage" value={fmt(selected?.advantage)} />
              <TraceField label="stop" value={selected?.is_truncated ? "truncated" : selected?.stop_condition} />
              <TraceField label="status" value={selected?.error ? "error" : "ok"} />
            </dl>
          </div>
        </aside>
      </section>
    </div>
  );
}

function TraceMessages({ detail }: { detail: RowDetail }) {
  const entries = [
    ...messageEntries(detail.prompt, "user"),
    ...messageEntries(detail.completion, "assistant"),
  ];
  const [count, setCount] = useState(12);
  const sentinel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => { if (entry.isIntersecting) setCount((n) => Math.min(n + 12, entries.length)); }, { rootMargin: "100px" });
    if (sentinel.current) observer.observe(sentinel.current);
    return () => observer.disconnect();
  }, [count, entries.length]);
  return (
    <>
      {entries.slice(0, count).map((message, index) => (
        <details className={`trace-entry role-${message.role}`} key={`${message.role}:${index}`} open>
          <summary><span>{String(index + 1).padStart(2, "0")}</span><strong>{message.role}</strong><span className="entry-preview">{short(message.content, 72)}</span><span className="entry-chevron">›</span></summary>
          <TraceText text={message.content} />
        </details>
      ))}
      {count < entries.length && <button ref={sentinel} type="button" onClick={() => setCount(count + 12)}>Load more messages ({entries.length - count} remaining)</button>}
      {detail.metadata && <details className="trace-entry"><summary><strong>Metadata</strong><span className="entry-chevron">›</span></summary><pre>{JSON.stringify(detail.metadata, null, 2)}</pre></details>}
      {detail.arrays && Object.keys(detail.arrays).length > 0 && <details className="trace-entry"><summary><strong>Token arrays</strong><span className="entry-chevron">›</span></summary><pre>{JSON.stringify(detail.arrays, null, 2)}</pre></details>}
    </>
  );
}

function messageEntries(value: RowDetail["prompt"], fallbackRole: string): Array<{ role: string; content: string }> {
  if (Array.isArray(value)) {
    return value.map((message) => ({
      role: message.role ?? fallbackRole,
      content: [message.reasoning_content ? `Reasoning\n${message.reasoning_content}` : "", typeof message.content === "string" ? message.content : message.content == null ? "" : JSON.stringify(message.content, null, 2), message.tool_calls ? `Tool calls\n${JSON.stringify(message.tool_calls, null, 2)}` : "", message.tool_call_id ? `Tool call ID: ${message.tool_call_id}` : ""].filter(Boolean).join("\n\n"),
    }));
  }
  if (value === undefined) return [];
  return [{ role: fallbackRole, content: typeof value === "string" ? value : JSON.stringify(value, null, 2) }];
}

function TraceText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return <><pre>{expanded ? text : text.slice(0, 4000)}</pre>{text.length > 4000 && <button type="button" onClick={() => setExpanded(!expanded)}>{expanded ? "Collapse text" : `Show full text (${fmtInt(text.length)} characters)`}</button>}</>;
}

function TraceField({ label, value }: { label: string; value: string | null | undefined }) {
  return <div><dt>{label}</dt><dd title={value ?? ""}>{value || "–"}</dd></div>;
}

function SampleDrawer({ title, detail, error, onClose }: { title: string; detail: RowDetail | null; error: string | null; onClose: () => void }) {
  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  return (
    <div className="drawer-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={title}>
        <header><div><span className="eyebrow">evaluation sample</span><h2>{title}</h2></div><button type="button" onClick={onClose} aria-label="Close detail">×</button></header>
        {error ? <InlineError>{error}</InlineError> : detail ? (
          <div className="detail-body">
            <div className="detail-stats"><Stat label="reward" value={fmt(numberValue(detail.reward))} /><Stat label="row" value={fmtInt(numberValue(detail.row_index))} /></div>
            <Transcript label="Prompt" value={detail.prompt} />
            <Transcript label="Completion" value={detail.completion} />
            {detail.metadata && <details><summary>Metadata</summary><pre>{JSON.stringify(detail.metadata, null, 2)}</pre></details>}
          </div>
        ) : <Empty title="Loading sample…" />}
      </aside>
    </div>
  );
}

function Transcript({ label, value }: { label: string; value: RowDetail["prompt"] }) {
  const messages = Array.isArray(value) ? value : null;
  return (
    <section className="transcript">
      <h3>{label}</h3>
      {messages ? messages.map((message, index) => (
        <div className="message" key={index}><span>{message.role ?? "message"}</span><pre>{typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2)}</pre></div>
      )) : <pre>{typeof value === "string" ? value : JSON.stringify(value, null, 2)}</pre>}
    </section>
  );
}

function DataTable({ headers, children }: { headers: string[]; children: ReactNode }) {
  return <div className="table-wrap"><table><thead><tr>{headers.map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{children}</tbody></table></div>;
}

function Segmented({ value, values, onChange }: { value: string; values: string[]; onChange: (value: string) => void }) {
  return <div className="segmented">{values.map((item) => <button key={item} type="button" className={value === item ? "active" : ""} onClick={() => onChange(item)}>{item}</button>)}</div>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="stat"><span>{label}</span><strong>{value}</strong></div>;
}

function Empty({ title, detail, children }: { title: string; detail?: string | null; children?: ReactNode }) {
  return <div className="empty"><strong>{title}</strong>{detail && <span>{detail}</span>}{children}</div>;
}

function InlineError({ children }: { children: ReactNode }) {
  return <div className="inline-error">{children}</div>;
}


function metricLabel(name: string, source: MetricSource): string {
  const labels: Record<string, string> = {
    "reward/episodes/all/mean": "all/agent/reward",
    "reward/episodes/all/count": "all/agent/count",
    "reward/episodes/effective/mean": "effective/agent/reward",
    "reward/episodes/effective/count": "effective/agent/count",
    "reward/all/mean": source === "trainer" ? "all/agent/reward" : "Reward · problem average",
    "generation/reward/mean": "Reward · all scored candidates",
    "rollout/count": "Episodes in optimizer batch",
  };
  return labels[name] ?? name
    .replace(/^train\//, "")
    .replace(/^generation\//, "gen/")
    .replace(/^inference\//, "infer/");
}

function metricDescription(name: string, source: MetricSource): string {
  const origin = source === "trainer" ? "Trainer" : source === "eval" ? "Evaluation" : "Orchestrator";
  if (name.startsWith("reward/episodes/effective/")) return `${origin} · Episodes with training tokens, excluding filtered, dummy and errored rows. Continuation branches do not count twice. This selected subset is not an overall solve rate.`;
  if (name.startsWith("reward/episodes/all/")) return `${origin} · Episode-weighted reward over the published batch, including filtered episodes. The count is the reward denominator, not the number of token rows.`;
  if (name === "reward/all/mean") return source === "trainer"
    ? "Trainer · Episode-weighted reward including filtered episodes. Optimizer step 1 consumes the first published batch."
    : "Orchestrator · Mean of per-problem rollout rewards, including filtered episodes. The first queue batch is step 0; this is not a filtered training reward.";
  if (name === "generation/reward/mean") return "Orchestrator · All scored candidate episodes, including groups rejected before publication.";
  return origin;
}

function sectionTitle(section: string): string {
  return ({ train: "Training", eval: "Evaluation", stability: "Stability", inference: "Inference", performance: "Performance", queue: "Rollout queue & policy freshness", other: "Other metrics" } as Record<string, string>)[section] ?? section;
}

function runDuration(summary: RunSummary): string {
  if (!summary.started_at) return "–";
  const start = new Date(summary.started_at).getTime();
  const end = summary.status === "running" ? Date.now() : new Date(summary.updated_at ?? summary.started_at).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "–";
  const seconds = Math.max(0, Math.floor((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function rewardClass(value: number | null): string {
  return value === null ? "" : value > 0 ? "positive" : value < 0 ? "negative" : "";
}

function short(value: string | null, size: number): string {
  if (!value) return "–";
  return value.length > size ? `${value.slice(0, size)}…` : value;
}

function compileRegex(pattern: string): RegExp | null {
  if (!pattern) return null;
  try {
    return new RegExp(pattern, "i");
  } catch {
    return null;
  }
}

function logLevelMatches(line: string, minimum: string): boolean {
  if (minimum === "all") return true;
  const order: Record<string, number> = { debug: 1, info: 2, warn: 3, warning: 3, error: 4, critical: 4 };
  const match = line.match(/\b(DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL)\b/i);
  const actual = match ? order[match[1].toLowerCase()] ?? 0 : 0;
  return actual >= (order[minimum] ?? 0);
}
