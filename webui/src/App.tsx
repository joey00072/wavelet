import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Moon, Server, Sun, Wifi, WifiOff } from "lucide-react";

import {
  EVENT_LIMIT,
  EVAL_METRIC_LIMIT,
  METRIC_LIMIT,
  POLL_MS,
  ROLLOUT_INSPECT_POLL_MS,
  fetchJson,
  initialApiBase,
  initialTheme,
  normalizeApiBase,
} from "./api";
import { type ActiveView, ViewTabs } from "./components/Tabs";
import type {
  EvalMetricRow,
  MetricRow,
  RolloutEvent,
  RolloutInspection,
  RolloutSnapshot,
  RunState,
  Theme,
} from "./types";
import { OverviewView } from "./views/OverviewView";
import { RolloutsView } from "./views/RolloutsView";
import { formatTime } from "./utils/format";
import {
  eventCounts,
  latestMetric,
  metricStep,
  pipelineInventory,
  rateForEvents,
  rateForMetrics,
} from "./utils/metrics";
import {
  ROLLOUT_BUFFER_LIMIT,
  SAVED_ROLLOUT_LIMIT,
  appendBufferedSnapshot,
  makeRolloutSnapshot,
  prependSnapshot,
} from "./utils/rolloutSnapshots";
import "./main.css";

function viewFromHash(): ActiveView {
  return window.location.hash === "#rollouts" ? "rollouts" : "overview";
}

function App() {
  const [apiBase, setApiBase] = useState(initialApiBase);
  const [apiInput, setApiInput] = useState(initialApiBase);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [activeView, setActiveView] = useState<ActiveView>(viewFromHash);
  const [state, setState] = useState<RunState | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [evalMetrics, setEvalMetrics] = useState<EvalMetricRow[]>([]);
  const [events, setEvents] = useState<RolloutEvent[]>([]);
  const [rolloutInspection, setRolloutInspection] = useState<RolloutInspection | null>(null);
  const [rolloutInspectionError, setRolloutInspectionError] = useState<string | null>(null);
  const [rolloutInspectionAt, setRolloutInspectionAt] = useState<string | null>(null);
  const [rolloutInspectRefresh, setRolloutInspectRefresh] = useState(0);
  const [rolloutAutoFollow, setRolloutAutoFollow] = useState(
    () => window.localStorage.getItem("wavelet.rollouts.autoFollow") !== "false",
  );
  const [rolloutSnapshots, setRolloutSnapshots] = useState<RolloutSnapshot[]>([]);
  const [savedRolloutSnapshots, setSavedRolloutSnapshots] = useState<RolloutSnapshot[]>([]);
  const [selectedRolloutSnapshotId, setSelectedRolloutSnapshotId] = useState<string | null>(null);
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
    window.localStorage.setItem("wavelet.rollouts.autoFollow", String(rolloutAutoFollow));
  }, [rolloutAutoFollow]);

  useEffect(() => {
    const onHashChange = () => setActiveView(viewFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    let controller: AbortController | undefined;
    const poll = async () => {
      controller = new AbortController();
      try {
        const [nextState, nextMetrics, nextEvalMetrics, nextEvents] = await Promise.all([
          fetchJson<RunState>(`${apiBase}/state`, controller.signal),
          fetchJson<MetricRow[]>(`${apiBase}/metrics?limit=${METRIC_LIMIT}`, controller.signal),
          fetchJson<EvalMetricRow[]>(
            `${apiBase}/eval-metrics?limit=${EVAL_METRIC_LIMIT}`,
            controller.signal,
          ),
          fetchJson<RolloutEvent[]>(`${apiBase}/events?limit=${EVENT_LIMIT}`, controller.signal),
        ]);
        if (!cancelled) {
          setState(nextState);
          setMetrics(nextMetrics);
          setEvalMetrics(nextEvalMetrics);
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
      controller?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [apiBase]);

  useEffect(() => {
    if (activeView !== "rollouts") return;
    let cancelled = false;
    let timer: number | undefined;
    let controller: AbortController | undefined;
    const poll = async () => {
      controller = new AbortController();
      try {
        const seed = Math.floor(Date.now() / ROLLOUT_INSPECT_POLL_MS);
        const nextInspection = await fetchJson<RolloutInspection>(
          `${apiBase}/rollouts/inspect?random_count=3&seed=${seed}&max_scan_rows=5000`,
          controller.signal,
        );
        if (!cancelled) {
          const capturedAt = new Date().toISOString();
          setRolloutInspection(nextInspection);
          setRolloutInspectionError(null);
          setRolloutInspectionAt(capturedAt);
          setRolloutSnapshots((s) => appendBufferedSnapshot(s, nextInspection, capturedAt));
        }
      } catch (caught) {
        if (!cancelled) {
          setRolloutInspectionError(caught instanceof Error ? caught.message : String(caught));
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(poll, ROLLOUT_INSPECT_POLL_MS);
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
      controller?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeView, apiBase, rolloutInspectRefresh]);

  const latest = latestMetric(metrics);
  const counts = eventCounts(events);
  const consumedRate = rateForMetrics(metrics);
  const publishedRate = rateForEvents(events, "published");
  const submittedRate = rateForEvents(events, "submitted");
  const step = metricStep(latest);
  const inventory = pipelineInventory(state, events, metrics);
  const selectedRolloutSnapshot = [...savedRolloutSnapshots, ...rolloutSnapshots].find(
    (s) => s.id === selectedRolloutSnapshotId,
  );
  const displayedRolloutInspection = rolloutAutoFollow
    ? rolloutInspection
    : selectedRolloutSnapshot?.inspection ?? rolloutInspection;
  const displayedRolloutInspectionAt = rolloutAutoFollow
    ? rolloutInspectionAt
    : selectedRolloutSnapshot?.captured_at ?? rolloutInspectionAt;
  const progress =
    state && step !== null && state.target_step > 0
      ? Math.min(100, Math.max(0, (step / state.target_step) * 100))
      : 0;

  const applyApiBase = () => {
    const next = normalizeApiBase(apiInput);
    if (next) {
      setApiBase(next);
      setSelectedRolloutSnapshotId(null);
      setRolloutSnapshots([]);
      setSavedRolloutSnapshots([]);
    }
  };

  const selectView = (view: ActiveView) => {
    setActiveView(view);
    const base = `${window.location.pathname}${window.location.search}`;
    window.history.replaceState(null, "", view === "rollouts" ? `${base}#rollouts` : base);
  };

  const setAutoFollowRollouts = (follow: boolean) => {
    if (follow) {
      setRolloutAutoFollow(true);
      setSelectedRolloutSnapshotId(null);
      return;
    }
    setRolloutAutoFollow(false);
    if (rolloutInspection?.available) {
      const snapshot = makeRolloutSnapshot(rolloutInspection, new Date().toISOString(), "reader");
      setRolloutSnapshots((s) => prependSnapshot(s, snapshot, ROLLOUT_BUFFER_LIMIT));
      setSelectedRolloutSnapshotId(snapshot.id);
    }
  };

  const selectRolloutSnapshot = (snapshotId: string) => {
    setRolloutAutoFollow(false);
    setSelectedRolloutSnapshotId(snapshotId);
  };

  const saveCurrentRolloutSnapshot = () => {
    if (!displayedRolloutInspection?.available) return;
    const snapshot = makeRolloutSnapshot(displayedRolloutInspection, new Date().toISOString(), "saved");
    setSavedRolloutSnapshots((s) => prependSnapshot(s, snapshot, SAVED_ROLLOUT_LIMIT));
    setRolloutAutoFollow(false);
    setSelectedRolloutSnapshotId(snapshot.id);
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 dark:bg-[#0a0a0f] dark:text-slate-100">
      <div className="mx-auto flex max-w-[1440px] min-h-screen">

        {/* ── Sidebar ── */}
        <aside className="hidden w-56 shrink-0 flex-col border-r border-slate-200 dark:border-white/[0.06] lg:flex">
          <div className="sticky top-0 flex flex-col gap-6 p-5">
            {/* Brand */}
            <div className="flex items-center gap-2.5">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-slate-900 dark:bg-white">
                <Server className="h-3.5 w-3.5 text-white dark:text-slate-900" />
              </div>
              <span className="text-sm font-semibold tracking-tight">Wavelet RL</span>
            </div>

            {/* Nav */}
            <ViewTabs activeView={activeView} onChange={selectView} layout="vertical" />

            {/* Status block */}
            <div className="rounded-lg border border-slate-200 p-3 dark:border-white/[0.06]">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500 dark:text-slate-400">{state?.status ?? "waiting"}</span>
                {error ? (
                  <span className="flex items-center gap-1 text-xs text-red-500">
                    <WifiOff className="h-3 w-3" /> offline
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
                    <span className="live-dot h-1.5 w-1.5 rounded-full bg-emerald-500" />
                    live
                  </span>
                )}
              </div>
              <div className="mt-2.5 h-1 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/[0.08]">
                <div className="progress-bar h-full rounded-full" style={{ width: `${progress}%` }} />
              </div>
              <div className="mt-1.5 flex justify-between text-[11px] text-slate-500 dark:text-slate-400">
                <span>step {step ?? "–"}</span>
                <span>{progress.toFixed(0)}%</span>
              </div>
            </div>

            {/* Controls */}
            <div className="flex flex-col gap-2">
              <div className="flex overflow-hidden rounded-md border border-slate-200 dark:border-white/[0.06]">
                <input
                  id="api-base-input"
                  value={apiInput}
                  onChange={(e) => setApiInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && applyApiBase()}
                  className="min-w-0 flex-1 bg-transparent px-2.5 py-1.5 text-xs text-slate-900 outline-none placeholder:text-slate-500 dark:text-slate-100"
                  placeholder="API base URL"
                  aria-label="State API base URL"
                />
                <button
                  type="button"
                  id="connect-btn"
                  onClick={applyApiBase}
                  className="border-l border-slate-200 px-2.5 text-xs font-medium text-slate-600 hover:bg-slate-50 dark:border-white/[0.06] dark:text-slate-300 dark:hover:bg-white/[0.04]"
                >
                  Go
                </button>
              </div>
              <button
                type="button"
                id="theme-toggle"
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-white/[0.04]"
                aria-label="Toggle theme"
              >
                {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </button>
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                {error ? <WifiOff className="h-3 w-3 shrink-0 text-red-400" /> : <Wifi className="h-3 w-3 shrink-0" />}
                <span className="truncate">{error ? error : `Updated ${formatTime(lastFetchAt)}`}</span>
              </div>
            </div>
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="min-w-0 flex-1 px-4 py-6 sm:px-6 lg:px-8">
          {/* Page title */}
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                {activeView === "rollouts" ? "Rollout Inspector" : "Training Overview"}
              </h1>
              <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">
                {activeView === "rollouts"
                  ? "Sampled completions without disturbing the training loop."
                  : "Training progress, queue flow, policy state, and throughput."}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2 lg:hidden">
              <ViewTabs activeView={activeView} onChange={selectView} layout="horizontal" />
            </div>
          </div>

          {activeView === "overview" ? (
            <OverviewView
              state={state}
              latest={latest}
              events={events}
              metrics={metrics}
              evalMetrics={evalMetrics}
              counts={counts}
              inventory={inventory}
              step={step}
              progress={progress}
              publishedRate={publishedRate}
              submittedRate={submittedRate}
              consumedRate={consumedRate}
            />
          ) : (
            <RolloutsView
              state={state}
              events={events}
              counts={counts}
              inventory={inventory}
              inspection={displayedRolloutInspection}
              liveInspection={rolloutInspection}
              inspectionError={rolloutInspectionError}
              inspectionUpdatedAt={displayedRolloutInspectionAt}
              autoFollow={rolloutAutoFollow}
              bufferedSnapshots={rolloutSnapshots}
              savedSnapshots={savedRolloutSnapshots}
              selectedSnapshotId={selectedRolloutSnapshotId}
              onSetAutoFollow={setAutoFollowRollouts}
              onRefreshInspection={() => setRolloutInspectRefresh((v) => v + 1)}
              onSelectSnapshot={selectRolloutSnapshot}
              onSaveSnapshot={saveCurrentRolloutSnapshot}
            />
          )}
        </main>
      </div>
    </div>
  );
}

document.documentElement.classList.toggle("dark", initialTheme() === "dark");

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
