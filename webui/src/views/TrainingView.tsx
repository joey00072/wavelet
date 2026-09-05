import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { ListPlus, Settings2 } from "lucide-react";

import { ChartCard, chartProps } from "../charts/ChartCard";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { Popover, Segmented, Slider } from "../components/Controls";
import { Drawer } from "../components/Drawer";
import { Empty } from "../components/KeyValue";
import { MetricPicker } from "../components/MetricPicker";
import { fmt } from "../lib/format";
import { updateParams } from "../lib/router";
import { lastFinite, seriesToLines } from "../lib/series";
import { runUrl, usePoll } from "../api/client";
import type { ExternalStatus } from "../api/types";
import { useMetricKeys, useSeries } from "./useRunData";

const PRESETS: Record<string, string[]> = {
  loss: ["train/loss", "train/policy_loss", "train/kl_loss"],
  "kl & masks": ["kl/mismatch", "kl/masked_mismatch", "kl/unmasked_mismatch", "ipo/is_masked", "ipo/is_masked_low", "ipo/is_masked_high", "dppo/is_masked", "dppo/is_masked_low", "dppo/is_masked_high"],
  optimizer: ["optim/grad_norm", "optim/lr", "entropy/mean"],
  "reward & advantage": ["reward/all/mean", "reward/all/min", "reward/all/max", "advantage/all/std", "advantage/token/mean"],
  tokens: ["tokens/train", "tokens/model", "seq_len/all/mean", "seq_len/all/max", "micro_batch/count", "rollout/count"],
  performance: ["perf/tokens_per_second", "perf/mfu", "perf/peak_memory_gib", "time/wait_for_batch", "time/train_until", "time/load_data", "time/export_policy"],
  moe: ["moe/max_vio", "moe/routing_confidence"],
};

const ORCH_PRESETS: Record<string, string[]> = {
  reward: ["reward/all/mean", "reward/all/std", "reward/all/min", "reward/all/max", "generation/reward/mean"],
  outcomes: ["generation/solve_all/rate", "generation/solve_none/rate", "generation/groups/admission_rate", "generation/groups/admitted", "generation/groups/rejected", "solve_all/all", "solve_none/all"],
  fate: ["fate/all/produced", "fate/all/trainable", "fate/all/filtered", "fate/all/truncated", "fate/all/errored", "fate/all/zero_loss"],
  lengths: ["decode_len/all/mean", "decode_len/all/max", "prefill_len/all/mean", "seq_len/all/mean", "is_truncated/all/mean", "num_turns/all/mean"],
  freshness: ["off_policy/mean", "off_policy/max", "off_policy/in_flight/max", "off_policy/in_queue/max", "policy/lag", "policy/step"],
  timing: ["time/step", "time/generate_completions", "time/publish", "time/rollout/generation/mean", "time/rollout/scoring/mean", "generation/policy_update_wait_seconds"],
  inference: ["generation/concurrency/limit", "generation/executor_concurrency", "inference/replica_0/kv_cache_usage", "inference/replica_0/requests_running", "inference/replica_0/requests_waiting", "inference/replica_0/preemptions_delta"],
  data: ["generation/data/cursor", "generation/data/epoch", "progress/samples", "progress/tokens"],
};

export type ChartSettings = { xAxis: "step" | "time"; smoothing: number; logScale: boolean; window: number; columns: number; overlay: boolean };
const DEFAULT_SETTINGS: ChartSettings = { xAxis: "step", smoothing: 0, logScale: false, window: 0, columns: 3, overlay: false };

export function useChartSettings(storageKey: string): [ChartSettings, (patch: Partial<ChartSettings>) => void] {
  const [settings, setSettings] = useState<ChartSettings>(() => {
    try {
      const stored = window.localStorage.getItem(storageKey);
      return stored ? { ...DEFAULT_SETTINGS, ...(JSON.parse(stored) as Partial<ChartSettings>) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });
  useEffect(() => window.localStorage.setItem(storageKey, JSON.stringify(settings)), [storageKey, settings]);
  return [settings, (patch) => setSettings((prev) => ({ ...prev, ...patch }))];
}

/** W&B-style panel settings: one gear, everything inside. */
export function ChartSettingsMenu({ settings, onChange, showLayout = true }: { settings: ChartSettings; onChange: (patch: Partial<ChartSettings>) => void; showLayout?: boolean }) {
  const active = settings.smoothing > 0 || settings.logScale || settings.window > 0 || settings.xAxis !== "step";
  return (
    <Popover
      width={320}
      trigger={(open) => (
        <button type="button" className={`btn ${open || active ? "btn-active" : ""}`} title="Chart settings">
          <Settings2 className="h-3.5 w-3.5" /> Settings
        </button>
      )}
    >
      <div className="space-y-4">
        <Slider label="Smoothing" min={0} max={0.99} step={0.01} value={settings.smoothing} onChange={(v) => onChange({ smoothing: v })} format={(v) => (v === 0 ? "off" : v.toFixed(2))} />
        <div className="flex items-center justify-between text-[11px] text-muted">
          <span className="w-20">X axis</span>
          <Segmented value={settings.xAxis} onChange={(v) => onChange({ xAxis: v })} size="xs" options={[{ value: "step", label: "step" }, { value: "time", label: "minutes" }]} />
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted">
          <span className="w-20">Y scale</span>
          <Segmented value={settings.logScale ? "log" : "linear"} onChange={(v) => onChange({ logScale: v === "log" })} size="xs" options={[{ value: "linear", label: "linear" }, { value: "log", label: "log" }]} />
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted">
          <span className="w-20">Window</span>
          <Segmented value={String(settings.window)} onChange={(v) => onChange({ window: Number(v) })} size="xs" options={[{ value: "0", label: "all" }, { value: "500", label: "500" }, { value: "200", label: "200" }, { value: "50", label: "50" }]} />
        </div>
        {showLayout && (
          <>
            <Slider label="Columns" min={1} max={4} step={1} value={settings.columns} onChange={(v) => onChange({ columns: v })} />
            <div className="flex items-center justify-between text-[11px] text-muted">
              <span className="w-20">Layout</span>
              <Segmented value={settings.overlay ? "overlay" : "grid"} onChange={(v) => onChange({ overlay: v === "overlay" })} size="xs" options={[{ value: "grid", label: "one per chart" }, { value: "overlay", label: "overlay" }]} />
            </div>
          </>
        )}
        <p className="text-[10.5px] leading-relaxed text-muted">Drag on a chart to zoom, double-click to reset. Hover a chart for its own smoothing and log toggles.</p>
      </div>
    </Popover>
  );
}

const NONE = "-";

function MetricExplorer({ apiBase, runId, source, params, defaultKeys, title, live }: { apiBase: string; runId: string; source: "trainer" | "orchestrator"; params: URLSearchParams; defaultKeys: string[]; title: string; live: boolean }) {
  const keys = useMetricKeys(apiBase, runId);
  const external = usePoll<ExternalStatus[]>(runUrl(apiBase, runId, "/external"), live ? 15000 : 0);
  const externalReady = (external.data ?? []).filter((e) => e.status === "ready").map((e) => e.source);
  const requestedSource = params.get("source");
  const activeSource = requestedSource && (requestedSource === source || externalReady.includes(requestedSource)) ? requestedSource : source;
  const available = useMemo(() => (keys.data?.[activeSource] ?? []).map((k) => k.key), [keys.data, activeSource]);
  const selectedParam = params.get("keys");
  const selected = useMemo(() => {
    if (selectedParam === NONE) return [];
    const requested = selectedParam ? selectedParam.split(",").filter(Boolean) : defaultKeys;
    return requested.filter((k) => available.length === 0 || available.includes(k));
  }, [selectedParam, defaultKeys, available]);
  const [settings, update] = useChartSettings(`wavelet.charts.${source}`);
  const [picking, setPicking] = useState(false);
  const series = useSeries(apiBase, runId, activeSource as "trainer" | "orchestrator", selected, settings.window, live ? 5000 : 0);
  const setSelected = (next: string[]) => updateParams((p) => p.set("keys", next.length ? next.join(",") : NONE));
  const resetSelected = () => updateParams((p) => p.delete("keys"));
  const toggle = (key: string) => setSelected(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  const lines = seriesToLines(series.data, selected, { xAxis: settings.xAxis });
  const presets = source === "trainer" ? PRESETS : ORCH_PRESETS;
  const usablePresets = Object.entries(presets).map(([name, list]) => [name, list.filter((k) => available.includes(k))] as const).filter(([, list]) => list.length > 0);
  const gridStyle = { "--chart-columns": settings.columns } as CSSProperties;
  const isDefault = selectedParam === null;

  return (
    <div className="min-w-0 space-y-6">
      <div className="flex flex-col items-start justify-between gap-3 sm:flex-row">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
          <p className="text-xs text-muted">
            {selected.length} chart{selected.length === 1 ? "" : "s"}{isDefault ? " · default set" : ""}{activeSource !== source ? ` · from ${activeSource}` : ""}{settings.window ? ` · last ${settings.window} steps` : ""}
          </p>
        </div>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:justify-end">
          {(externalReady.length > 0 || (external.data ?? []).some((e) => e.status === "loading")) && (
            <Segmented
              value={activeSource}
              onChange={(v) => updateParams((p) => { if (v === source) p.delete("source"); else p.set("source", v); p.delete("keys"); })}
              size="xs"
              options={[{ value: source, label: "local" }, ...externalReady.map((name) => ({ value: name, label: name === "wandb" ? "W&B" : "Trackio" }))]}
            />
          )}
          <button type="button" className={`btn ${picking ? "btn-active" : ""}`} onClick={() => setPicking(true)}>
            <ListPlus className="h-3.5 w-3.5" /> Metrics
          </button>
          <ChartSettingsMenu settings={settings} onChange={update} />
        </div>
      </div>
      {(external.data ?? []).some((e) => e.status === "error" || e.status === "loading") && (
        <p className="text-[11px] text-muted">
          {(external.data ?? []).filter((e) => e.status !== "unavailable" && e.status !== "ready").map((e) => `${e.source}: ${e.status}${e.error ? ` (${e.error})` : ""}`).join(" · ")}
        </p>
      )}
      {selected.length === 0 && (
        <div className="flex flex-col items-center gap-3 py-16 text-center">
          <div className="text-sm font-medium text-ink2">No charts</div>
          <div className="max-w-sm text-xs text-muted">Pick any logged metric, or bring back the default set for this view.</div>
          <div className="flex gap-2">
            <button type="button" className="btn btn-active" onClick={() => setPicking(true)}><ListPlus className="h-3.5 w-3.5" /> Add metrics</button>
            <button type="button" className="btn" onClick={resetSelected}>Reset to defaults</button>
          </div>
        </div>
      )}
      {settings.overlay && selected.length > 0 && (
        <ChartCard title={`${selected.length} metrics`} subtitle="shared axis; switch to one per chart when scales differ" refetching={series.refetching} smoothingKey={`${source}.overlay`} defaultSmoothing={settings.smoothing} defaultLogScale={settings.logScale} height={360} table={<SeriesTable series={lines} xLabel={settings.xAxis} />}>
          {(o) => <LineChart series={lines} {...chartProps(o)} xLabel={settings.xAxis} />}
        </ChartCard>
      )}
      {!settings.overlay && selected.length > 0 && (
        <div className="metric-grid grid gap-x-10 gap-y-9" style={gridStyle}>
          {lines.map((line) => (
            <ChartCard key={line.id} title={line.id} value={fmt(lastFinite(series.data?.series[line.id]), 4)} refetching={series.refetching} smoothingKey={`${source}.${line.id}`} defaultSmoothing={settings.smoothing} defaultLogScale={settings.logScale} height={settings.columns >= 4 ? 150 : 180} table={<SeriesTable series={[line]} xLabel={settings.xAxis} />}>
              {(o) => <LineChart series={[{ ...line, colorIndex: 0 }]} {...chartProps(o)} xLabel={settings.xAxis} />}
            </ChartCard>
          ))}
        </div>
      )}
      <Drawer open={picking} onClose={() => setPicking(false)} title="Metrics" subtitle={`${selected.length} selected · ${available.length} logged by ${activeSource}`} width={420}>
        <div className="flex h-full flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <select className="select" value="" onChange={(e) => { const found = usablePresets.find(([name]) => name === e.target.value); if (found) setSelected([...found[1]]); }} aria-label="Preset">
              <option value="">Add a preset…</option>
              {usablePresets.map(([name, list]) => (
                <option key={name} value={name}>{name} ({list.length})</option>
              ))}
            </select>
            <button type="button" className="btn" onClick={() => setSelected([])} disabled={selected.length === 0} title="Remove every chart">Clear</button>
            <button type="button" className="btn" onClick={resetSelected} disabled={isDefault} title="Back to this view's default charts">Reset</button>
            <button type="button" className="btn btn-active ml-auto" onClick={() => setPicking(false)}>Done</button>
          </div>
          <div className="min-h-0 flex-1">
            <MetricPicker keys={available} selected={selected} onToggle={toggle} />
          </div>
        </div>
      </Drawer>
    </div>
  );
}

export function TrainingView(props: { apiBase: string; runId: string; params: URLSearchParams; live: boolean }) {
  return <MetricExplorer {...props} source="trainer" title="Trainer" defaultKeys={PRESETS.loss.concat(["entropy/mean", "optim/grad_norm", "kl/mismatch", "ipo/is_masked", "dppo/is_masked"])} />;
}

export function RolloutMetricsView(props: { apiBase: string; runId: string; params: URLSearchParams; live: boolean }) {
  return <MetricExplorer {...props} source="orchestrator" title="Generation" defaultKeys={["reward/all/mean", "generation/reward/mean", "advantage/all/std", "is_truncated/all/mean", "decode_len/all/mean", "generation/groups/admission_rate", "generation/solve_none/rate", "off_policy/mean", "time/generate_completions"]} />;
}
