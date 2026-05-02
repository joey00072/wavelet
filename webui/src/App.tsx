import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  Boxes,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  Gauge,
  Loader2,
  Moon,
  PackageCheck,
  RefreshCw,
  Send,
  Server,
  Settings,
  Sun,
  Wifi,
  WifiOff,
} from "lucide-react";
import "./main.css";

type RolloutEvent = {
  type: "submitted" | "completed" | "published";
  queue_step: number;
  optimizer_step?: number | null;
  chunk_index?: number | null;
  timestamp: string;
  path?: string;
};

type RunState = {
  status: string;
  phase: string;
  started_at: string;
  updated_at: string;
  target_step: number;
  output_dir: string;
  launcher_mode: string;
  rollouts: {
    next_queue_step_to_submit: number;
    next_queue_step_to_publish: number;
    pending_count: number;
    completed_count: number;
    submitted_tail: RolloutEvent[];
    completed_tail: RolloutEvent[];
    published_tail: RolloutEvent[];
  };
  policy: {
    loaded_step: number | null;
    pending_load: boolean;
    requested_step: number | null;
    available_tail: number[];
  };
  errors: Array<{ type: string; message: string; timestamp: string }>;
};

type MetricRow = {
  timestamp?: string;
  step?: number;
  "progress/step"?: number;
  reward_mean?: number;
  "reward/all/mean"?: number;
  loss?: number;
  lr?: number;
  "optim/lr"?: number;
  "rollout/count"?: number;
  "tokens/train"?: number;
  "perf/train_tokens_per_second"?: number;
  "perf/step_tokens_per_second"?: number;
};

const DEFAULT_API_BASE = "http://127.0.0.1:8765";
const EVENT_LIMIT = 2000;
const METRIC_LIMIT = 200;
const POLL_MS = 2000;
const FALLBACK_CHUNKS_PER_STEP = 8;

type Theme = "dark" | "light";

type PipelineInventory = {
  submitted: number;
  publishedWatermark: number;
  generating: number;
  completedWaitingPublish: number;
  consumedEstimate: number;
  readyForTrainer: number;
  trainerUsingChunks: number;
  trainerUsingRollouts: number;
  chunksPerStep: number;
  rolloutsPerChunk: number;
  trainerStep: number | null;
};

function initialApiBase(): string {
  const params = new URLSearchParams(window.location.search);
  const query = params.get("api");
  if (query) {
    return query.replace(/\/$/, "");
  }
  const stored = window.localStorage.getItem("wavelet.apiBase");
  if (stored) {
    return stored.replace(/\/$/, "");
  }
  const envBase = import.meta.env.VITE_WAVELET_API_BASE as string | undefined;
  return (envBase || DEFAULT_API_BASE).replace(/\/$/, "");
}

function initialTheme(): Theme {
  const stored = window.localStorage.getItem("wavelet.theme");
  return stored === "light" ? "light" : "dark";
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  if (Math.abs(value) >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatRate(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${formatNumber(value, 1)}/min`;
}

function formatTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleTimeString();
}

function latestMetric(metrics: MetricRow[]): MetricRow | null {
  return metrics.length > 0 ? metrics[metrics.length - 1] : null;
}

function metricStep(row: MetricRow | null): number | null {
  if (!row) {
    return null;
  }
  return row.step ?? row["progress/step"] ?? null;
}

function elapsedMinutes(first: string | undefined, last: string | undefined): number {
  if (!first || !last) {
    return 0;
  }
  const start = new Date(first).getTime();
  const end = new Date(last).getTime();
  if (Number.isNaN(start) || Number.isNaN(end) || end <= start) {
    return 0;
  }
  return (end - start) / 60000;
}

function rateForEvents(events: RolloutEvent[], type: RolloutEvent["type"]): number {
  const filtered = events.filter((event) => event.type === type);
  if (filtered.length < 2) {
    return 0;
  }
  const minutes = elapsedMinutes(filtered[0].timestamp, filtered[filtered.length - 1].timestamp);
  return minutes > 0 ? filtered.length / minutes : 0;
}

function rateForMetrics(metrics: MetricRow[]): number {
  const withSteps = metrics.filter((row) => row.timestamp && metricStep(row) !== null);
  if (withSteps.length < 2) {
    return 0;
  }
  const first = withSteps[0];
  const last = withSteps[withSteps.length - 1];
  const minutes = elapsedMinutes(first.timestamp, last.timestamp);
  const firstStep = metricStep(first) ?? 0;
  const lastStep = metricStep(last) ?? 0;
  return minutes > 0 ? Math.max(lastStep - firstStep, 0) / minutes : 0;
}

function eventCounts(events: RolloutEvent[]) {
  return events.reduce(
    (counts, event) => {
      counts[event.type] += 1;
      return counts;
    },
    { submitted: 0, completed: 0, published: 0 },
  );
}

function inferChunksPerStep(events: RolloutEvent[]): number {
  const chunkIndexes = events
    .map((event) => event.chunk_index)
    .filter((index): index is number => typeof index === "number" && index >= 0);
  if (chunkIndexes.length === 0) {
    return FALLBACK_CHUNKS_PER_STEP;
  }
  return Math.max(...chunkIndexes) + 1;
}

function pipelineInventory(
  state: RunState | null,
  events: RolloutEvent[],
  metrics: MetricRow[],
): PipelineInventory {
  const latest = latestMetric(metrics);
  const trainerStep = metricStep(latest);
  const chunksPerStep = inferChunksPerStep(events);
  const rolloutsPerStep = latest?.["rollout/count"] ?? 128;
  const rolloutsPerChunk = Math.max(1, Math.round(rolloutsPerStep / chunksPerStep));
  const consumedEstimate = Math.max(0, (trainerStep ?? 0) * chunksPerStep);
  const publishedWatermark = state?.rollouts.next_queue_step_to_publish ?? eventCounts(events).published;
  const readyForTrainer = Math.max(0, publishedWatermark - consumedEstimate);
  const running = state?.status === "running";
  const trainerUsingChunks = running ? Math.min(chunksPerStep, readyForTrainer) : 0;

  return {
    submitted: state?.rollouts.next_queue_step_to_submit ?? eventCounts(events).submitted,
    publishedWatermark,
    generating: state?.rollouts.pending_count ?? 0,
    completedWaitingPublish: state?.rollouts.completed_count ?? 0,
    consumedEstimate,
    readyForTrainer,
    trainerUsingChunks,
    trainerUsingRollouts: trainerUsingChunks * rolloutsPerChunk,
    chunksPerStep,
    rolloutsPerChunk,
    trainerStep,
  };
}

function bucketEvents(events: RolloutEvent[], metrics: MetricRow[]) {
  const times = [
    ...events.map((event) => new Date(event.timestamp).getTime()),
    ...metrics
      .filter((row) => row.timestamp && metricStep(row) !== null)
      .map((row) => new Date(row.timestamp as string).getTime()),
  ].filter((time) => !Number.isNaN(time));
  if (times.length === 0) {
    return [];
  }
  const min = Math.min(...times);
  const max = Math.max(...times);
  const span = Math.max(max - min, 1);
  const bucketCount = Math.min(80, Math.max(12, Math.ceil(span / 15000)));
  const buckets = Array.from({ length: bucketCount }, (_, index) => ({
    index,
    start: min + (span * index) / bucketCount,
    submitted: 0,
    completed: 0,
    published: 0,
    consumed: 0,
  }));
  const bucketIndex = (timestamp: string | undefined) => {
    if (!timestamp) {
      return 0;
    }
    const time = new Date(timestamp).getTime();
    if (Number.isNaN(time)) {
      return 0;
    }
    return Math.min(bucketCount - 1, Math.max(0, Math.floor(((time - min) / span) * bucketCount)));
  };
  for (const event of events) {
    buckets[bucketIndex(event.timestamp)][event.type] += 1;
  }
  for (const row of metrics) {
    if (metricStep(row) !== null) {
      buckets[bucketIndex(row.timestamp)].consumed += 1;
    }
  }
  return buckets;
}

function laneCounts(events: RolloutEvent[]) {
  const lanes = new Map<number, { submitted: number; completed: number; published: number }>();
  for (const event of events) {
    const lane = event.chunk_index ?? 0;
    const counts = lanes.get(lane) ?? { submitted: 0, completed: 0, published: 0 };
    counts[event.type] += 1;
    lanes.set(lane, counts);
  }
  return [...lanes.entries()].sort((a, b) => a[0] - b[0]);
}

function StatusIcon({ status }: { status: string }) {
  if (status === "running") {
    return <Activity className="h-4 w-4 text-emerald-500" />;
  }
  if (status === "completed") {
    return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
  }
  if (status === "failed") {
    return <AlertCircle className="h-4 w-4 text-red-500" />;
  }
  return <Clock className="h-4 w-4 text-zinc-500" />;
}

function Stat({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {icon}
        {label}
      </div>
      <div className="mt-3 text-2xl font-semibold text-zinc-950 dark:text-zinc-50">{value}</div>
      {sub ? <div className="mt-1 truncate text-sm text-zinc-500 dark:text-zinc-400">{sub}</div> : null}
    </div>
  );
}

function InventoryCard({
  icon,
  label,
  value,
  sub,
  tone = "zinc",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  tone?: "blue" | "emerald" | "amber" | "cyan" | "zinc";
}) {
  const toneClass = {
    blue: "text-blue-600 dark:text-blue-400",
    emerald: "text-emerald-600 dark:text-emerald-400",
    amber: "text-amber-600 dark:text-amber-400",
    cyan: "text-cyan-600 dark:text-cyan-400",
    zinc: "text-zinc-600 dark:text-zinc-300",
  }[tone];
  return (
    <div className="border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className={`flex items-center gap-2 text-xs font-medium uppercase tracking-wide ${toneClass}`}>
        {icon}
        {label}
      </div>
      <div className="mt-3 font-mono text-3xl font-semibold text-zinc-950 dark:text-zinc-50">{value}</div>
      <div className="mt-1 min-h-5 text-sm text-zinc-500 dark:text-zinc-400">{sub}</div>
    </div>
  );
}

function ThroughputChart({ events, metrics }: { events: RolloutEvent[]; metrics: MetricRow[] }) {
  const buckets = useMemo(() => bucketEvents(events, metrics), [events, metrics]);
  const width = 900;
  const height = 280;
  const pad = 28;
  const maxValue = Math.max(
    1,
    ...buckets.flatMap((bucket) => [
      bucket.submitted,
      bucket.completed,
      bucket.published,
      bucket.consumed,
    ]),
  );
  const x = (index: number) =>
    pad + (buckets.length <= 1 ? 0 : (index / (buckets.length - 1)) * (width - pad * 2));
  const y = (value: number) => height - pad - (value / maxValue) * (height - pad * 2);
  const pathFor = (key: "submitted" | "completed" | "published" | "consumed") =>
    buckets
      .map((bucket, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(bucket[key])}`)
      .join(" ");

  if (buckets.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center border border-zinc-200 bg-white text-sm text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
        No events yet
      </div>
    );
  }

  return (
    <div className="overflow-hidden border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <div>
          <div className="text-sm font-semibold text-zinc-950 dark:text-zinc-50">Production vs consumption</div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">Rollout chunks per time bucket</div>
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-zinc-600 dark:text-zinc-300">
          <Legend color="#2563eb" label="submitted" />
          <Legend color="#0891b2" label="completed" />
          <Legend color="#16a34a" label="published" />
          <Legend color="#f59e0b" label="trainer steps" />
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-72 w-full">
        <rect x="0" y="0" width={width} height={height} className="fill-white dark:fill-zinc-900" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line
              x1={pad}
              x2={width - pad}
              y1={pad + tick * (height - pad * 2)}
              y2={pad + tick * (height - pad * 2)}
              className="stroke-zinc-200 dark:stroke-zinc-800"
            />
          </g>
        ))}
        <path d={pathFor("submitted")} fill="none" stroke="#2563eb" strokeWidth="2.5" />
        <path d={pathFor("completed")} fill="none" stroke="#0891b2" strokeWidth="2.5" />
        <path d={pathFor("published")} fill="none" stroke="#16a34a" strokeWidth="2.5" />
        <path d={pathFor("consumed")} fill="none" stroke="#f59e0b" strokeWidth="2.5" />
      </svg>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2.5 w-2.5" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

function LaneTable({ events }: { events: RolloutEvent[] }) {
  const lanes = laneCounts(events);
  return (
    <div className="border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
      <div className="border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
        <div className="text-sm font-semibold text-zinc-950 dark:text-zinc-50">Per chunk lane</div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400">Used as the available per-producer lane signal</div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-sm">
          <thead className="border-b border-zinc-200 text-xs uppercase text-zinc-500 dark:border-zinc-800 dark:text-zinc-400">
            <tr>
              <th className="px-4 py-3 font-medium">Lane</th>
              <th className="px-4 py-3 font-medium">Submitted</th>
              <th className="px-4 py-3 font-medium">Completed</th>
              <th className="px-4 py-3 font-medium">Published</th>
              <th className="px-4 py-3 font-medium">Backlog</th>
            </tr>
          </thead>
          <tbody>
            {lanes.length === 0 ? (
              <tr>
                <td className="px-4 py-6 text-zinc-500 dark:text-zinc-400" colSpan={5}>
                  No lane data yet
                </td>
              </tr>
            ) : (
              lanes.map(([lane, counts]) => (
                <tr key={lane} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
                  <td className="px-4 py-3 font-mono text-zinc-950 dark:text-zinc-50">{lane}</td>
                  <td className="px-4 py-3">{counts.submitted}</td>
                  <td className="px-4 py-3">{counts.completed}</td>
                  <td className="px-4 py-3">{counts.published}</td>
                  <td className="px-4 py-3">{Math.max(counts.submitted - counts.published, 0)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function App() {
  const [apiBase, setApiBase] = useState(initialApiBase);
  const [apiInput, setApiInput] = useState(initialApiBase);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [state, setState] = useState<RunState | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [events, setEvents] = useState<RolloutEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchAt, setLastFetchAt] = useState<string | null>(null);

  useEffect(() => {
    window.localStorage.setItem("wavelet.apiBase", apiBase);
  }, [apiBase]);

  useEffect(() => {
    window.localStorage.setItem("wavelet.theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      const controller = new AbortController();
      try {
        const [nextState, nextMetrics, nextEvents] = await Promise.all([
          fetchJson<RunState>(`${apiBase}/state`, controller.signal),
          fetchJson<MetricRow[]>(`${apiBase}/metrics?limit=${METRIC_LIMIT}`, controller.signal),
          fetchJson<RolloutEvent[]>(`${apiBase}/events?limit=${EVENT_LIMIT}`, controller.signal),
        ]);
        if (!cancelled) {
          setState(nextState);
          setMetrics(nextMetrics);
          setEvents(nextEvents);
          setError(null);
          setLastFetchAt(new Date().toISOString());
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(poll, POLL_MS);
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [apiBase]);

  const latest = latestMetric(metrics);
  const counts = eventCounts(events);
  const consumedRate = rateForMetrics(metrics);
  const publishedRate = rateForEvents(events, "published");
  const submittedRate = rateForEvents(events, "submitted");
  const step = metricStep(latest);
  const inventory = pipelineInventory(state, events, metrics);
  const progress =
    state && step !== null && state.target_step > 0
      ? Math.min(100, Math.max(0, (step / state.target_step) * 100))
      : 0;

  const applyApiBase = () => {
    const next = apiInput.trim().replace(/\/$/, "");
    if (next) {
      setApiBase(next);
    }
  };

  return (
    <main className="min-h-screen bg-[#f6f7f4] text-zinc-950 dark:bg-zinc-950 dark:text-zinc-100">
      <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-zinc-200 pb-5 dark:border-zinc-800 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-sm text-zinc-500 dark:text-zinc-400">
              <Server className="h-4 w-4" />
              Wavelet RL
            </div>
            <h1 className="mt-1 text-2xl font-semibold text-zinc-950 dark:text-zinc-50">Run State</h1>
          </div>
          <div className="flex w-full flex-col gap-2 sm:flex-row lg:w-auto">
            <button
              type="button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="inline-flex items-center justify-center border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:hover:bg-zinc-800"
              aria-label="Toggle dark mode"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            <div className="flex min-w-0 flex-1 border border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900 lg:w-[420px]">
              <input
                value={apiInput}
                onChange={(event) => setApiInput(event.target.value)}
                className="min-w-0 flex-1 bg-transparent px-3 py-2 text-sm text-zinc-950 outline-none placeholder:text-zinc-400 dark:text-zinc-100"
                aria-label="State API base URL"
              />
              <button
                type="button"
                onClick={applyApiBase}
                className="border-l border-zinc-300 px-3 text-sm font-medium text-zinc-800 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-100 dark:hover:bg-zinc-800"
              >
                Connect
              </button>
            </div>
            <div className="inline-flex items-center gap-2 border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900">
              {error ? <WifiOff className="h-4 w-4 text-red-500" /> : <Wifi className="h-4 w-4 text-emerald-500" />}
              <span className="truncate">{error ? error : `Updated ${formatTime(lastFetchAt)}`}</span>
            </div>
          </div>
        </header>

        <section className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            icon={<StatusIcon status={state?.status ?? "starting"} />}
            label="Status"
            value={state?.status ?? "-"}
            sub={state?.phase ?? "waiting for state"}
          />
          <Stat
            icon={<Gauge className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />}
            label="Trainer Step"
            value={step === null ? "-" : `${step}/${state?.target_step ?? "-"}`}
            sub={`${formatNumber(progress, 1)}% complete`}
          />
          <Stat
            icon={<RefreshCw className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />}
            label="Published Rate"
            value={formatRate(publishedRate)}
            sub={`submitted ${formatRate(submittedRate)}`}
          />
          <Stat
            icon={<Activity className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />}
            label="Consumed Rate"
            value={formatRate(consumedRate)}
            sub={`latest reward ${formatNumber(latest?.reward_mean ?? latest?.["reward/all/mean"])}`}
          />
        </section>

        <section className="mt-5">
          <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-zinc-950 dark:text-zinc-50">Pipeline Inventory</h2>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Current rollout chunks moving between inference, queue, and trainer
              </p>
            </div>
            <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
              {inventory.chunksPerStep} chunks/step · {inventory.rolloutsPerChunk} rollouts/chunk
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <InventoryCard
              icon={<Loader2 className="h-4 w-4" />}
              label="Generating"
              value={formatNumber(inventory.generating, 0)}
              sub={`${formatNumber(inventory.generating * inventory.rolloutsPerChunk, 0)} rollouts inflight`}
              tone="blue"
            />
            <InventoryCard
              icon={<PackageCheck className="h-4 w-4" />}
              label="Ready for Trainer"
              value={formatNumber(inventory.readyForTrainer, 0)}
              sub={`${formatNumber(inventory.readyForTrainer * inventory.rolloutsPerChunk, 0)} rollouts queued`}
              tone="emerald"
            />
            <InventoryCard
              icon={<Cpu className="h-4 w-4" />}
              label="Trainer Using Now"
              value={formatNumber(inventory.trainerUsingChunks, 0)}
              sub={`${formatNumber(inventory.trainerUsingRollouts, 0)} rollouts for step ${
                inventory.trainerStep === null ? "-" : inventory.trainerStep + 1
              }`}
              tone="amber"
            />
            <InventoryCard
              icon={<Boxes className="h-4 w-4" />}
              label="Waiting Publish"
              value={formatNumber(inventory.completedWaitingPublish, 0)}
              sub={`published watermark ${formatNumber(inventory.publishedWatermark, 0)}`}
              tone="cyan"
            />
            <InventoryCard
              icon={<Send className="h-4 w-4" />}
              label="Submitted Total"
              value={formatNumber(inventory.submitted, 0)}
              sub={`trainer consumed est. ${formatNumber(inventory.consumedEstimate, 0)}`}
            />
          </div>
        </section>

        <section className="mt-5 grid gap-4 lg:grid-cols-[2fr_1fr]">
          <ThroughputChart events={events} metrics={metrics} />
          <div className="grid gap-4">
            <div className="border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-2 text-sm font-semibold">
                <Database className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
                Rollout Totals
              </div>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <Metric label="Submitted" value={counts.submitted} />
                <Metric label="Completed" value={counts.completed} />
                <Metric label="Published" value={counts.published} />
                <Metric label="Pending" value={state?.rollouts.pending_count ?? 0} />
              </dl>
            </div>
            <div className="border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
              <div className="flex items-center gap-2 text-sm font-semibold">
                <Settings className="h-4 w-4 text-zinc-500 dark:text-zinc-400" />
                Policy
              </div>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <Metric label="Loaded" value={state?.policy.loaded_step ?? "-"} />
                <Metric label="Requested" value={state?.policy.requested_step ?? "-"} />
                <Metric label="Pending load" value={state?.policy.pending_load ? "yes" : "no"} />
                <Metric
                  label="Latest export"
                  value={
                    state?.policy.available_tail.length
                      ? state.policy.available_tail[state.policy.available_tail.length - 1]
                      : "-"
                  }
                />
              </dl>
            </div>
          </div>
        </section>

        <section className="mt-4 grid gap-4 lg:grid-cols-[1.2fr_1fr]">
          <LaneTable events={events} />
          <div className="border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
            <div className="border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
              <div className="text-sm font-semibold text-zinc-950 dark:text-zinc-50">Recent Metrics</div>
              <div className="text-xs text-zinc-500 dark:text-zinc-400">Latest trainer log row</div>
            </div>
            <dl className="grid grid-cols-2 gap-3 p-4 text-sm">
              <Metric label="Loss" value={formatNumber(latest?.loss)} />
              <Metric label="LR" value={formatNumber(latest?.lr ?? latest?.["optim/lr"], 8)} />
              <Metric label="Train tokens" value={formatNumber(latest?.["tokens/train"], 0)} />
              <Metric
                label="Step tok/s"
                value={formatNumber(latest?.["perf/step_tokens_per_second"], 0)}
              />
            </dl>
          </div>
        </section>
      </div>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className="mt-1 font-mono text-base text-zinc-950 dark:text-zinc-50">{value}</dd>
    </div>
  );
}

document.documentElement.classList.toggle("dark", initialTheme() === "dark");

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
