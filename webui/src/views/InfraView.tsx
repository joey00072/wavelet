import { useEffect, useMemo, useState } from "react";

import { qs, runUrl, usePoll } from "../api/client";
import type { RunSummary } from "../api/types";
import { BarChart } from "../charts/BarChart";
import { ChartCard } from "../charts/ChartCard";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { StatTile } from "../charts/StatTile";
import { Field, Segmented, Toolbar } from "../components/Controls";
import { NodesPanel } from "../components/NodesPanel";
import { Disclosure } from "../components/Disclosure";
import { Empty, ErrorNote, KeyValue, Section } from "../components/KeyValue";
import { fmt, fmtAge, fmtBytes, fmtInt, fmtPct, fmtSeconds } from "../lib/format";
import { STEP_SECONDS_KEYS, TOKENS_PER_SECOND_KEYS, TRAIN_SECONDS_KEYS, num, numAny, seriesToLines } from "../lib/series";
import { useMetricKeys, useSeries } from "./useRunData";

const GIB = 2 ** 30;
const TRAINER_KEYS = ["perf/train_tokens_per_second", "perf/step_seconds", "perf/train_seconds", "perf/throughput", "cuda_memory_allocated_bytes", "cuda_memory_reserved_bytes", "cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes", "disk_used_bytes", "disk_free_bytes", "disk_total_bytes", "disk_free_ratio", "checkpoint_disk_free_ratio", "perf/tokens_per_second", "perf/mfu", "perf/peak_memory_gib", "time/wait_for_batch", "time/load_data", "time/train_until", "time/export_policy", "time/update_weights"];

export function InfraView({ apiBase, runId, summary }: { apiBase: string; runId: string; summary: RunSummary | null }) {
  const keys = useMetricKeys(apiBase, runId);
  const inferenceKeys = useMemo(() => (keys.data?.orchestrator ?? []).map((k) => k.key).filter((k) => k.startsWith("inference/") || k.startsWith("generation/concurrency") || k === "generation/executor_concurrency" || k.startsWith("time/")), [keys.data]);
  const live = summary?.status === "running";
  const trainer = useSeries(apiBase, runId, "trainer", TRAINER_KEYS, 2000, live ? 5000 : 0);
  const orch = useSeries(apiBase, runId, "orchestrator", inferenceKeys, 2000, live ? 5000 : 0);
  const latest = summary?.latest.trainer;
  const latestO = summary?.latest.orchestrator;
  const historical = summary?.status !== "running";
  const runtimeStatus = summary?.status === "stale" ? "stale" : summary?.status ?? summary?.heartbeat?.status ?? "–";
  const allocatedMemory = num(latest, "cuda_memory_allocated_bytes");
  const reservedMemory = num(latest, "cuda_memory_reserved_bytes");

  const memory = seriesToLines(trainer.data, ["cuda_memory_allocated_bytes", "cuda_memory_reserved_bytes", "cuda_max_memory_allocated_bytes"], { labels: { cuda_memory_allocated_bytes: "allocated", cuda_memory_reserved_bytes: "reserved", cuda_max_memory_allocated_bytes: "peak allocated" } }).map((l) => ({ ...l, points: l.points.map((p) => ({ x: p.x, y: p.y / GIB })) }));
  const disk = seriesToLines(trainer.data, ["disk_free_ratio", "checkpoint_disk_free_ratio"], { labels: { disk_free_ratio: "output volume free", checkpoint_disk_free_ratio: "checkpoint volume free" } });
  const throughputKey = TOKENS_PER_SECOND_KEYS.find((k) => trainer.data?.series[k]?.some((v) => v !== null)) ?? "perf/tokens_per_second";
  const throughput = seriesToLines(trainer.data, [throughputKey]);
  const mfu = seriesToLines(trainer.data, ["perf/mfu"]);
  const timeKeys = ["time/wait_for_batch", "time/load_data", "time/train_until", "time/export_policy", "time/update_weights", "perf/train_seconds"].filter((k) => trainer.data?.series[k]?.some((v) => v !== null));
  const timeCats = (trainer.data?.steps ?? []).map((s, i) => ({ label: String(s ?? i), values: timeKeys.map((k) => trainer.data?.series[k]?.[i] ?? 0) }));
  const replicas = [...new Set(inferenceKeys.flatMap((k) => (k.match(/^inference\/(replica_\d+)\//) ? [k.match(/^inference\/(replica_\d+)\//)![1]] : [])))].sort();
  const kv = seriesToLines(orch.data, replicas.map((r) => `inference/${r}/kv_cache_usage`), { labels: Object.fromEntries(replicas.map((r) => [`inference/${r}/kv_cache_usage`, r])) });
  const requests = seriesToLines(orch.data, replicas.flatMap((r) => [`inference/${r}/requests_running`, `inference/${r}/requests_waiting`]), { labels: Object.fromEntries(replicas.flatMap((r) => [[`inference/${r}/requests_running`, `${r} running`], [`inference/${r}/requests_waiting`, `${r} waiting`]])) });
  const preemptions = seriesToLines(orch.data, replicas.map((r) => `inference/${r}/preemptions_delta`), { labels: Object.fromEntries(replicas.map((r) => [`inference/${r}/preemptions_delta`, r])) });
  const concurrency = seriesToLines(orch.data, ["generation/concurrency/limit", "generation/executor_concurrency"], { labels: { "generation/concurrency/limit": "rollout concurrency cap", "generation/executor_concurrency": "verifier executors" } });
  const orchTime = seriesToLines(orch.data, ["time/generate_completions", "time/step", "time/publish"].filter((k) => inferenceKeys.includes(k)), { labels: { "time/generate_completions": "generate", "time/step": "orchestrator step", "time/publish": "publish" } });

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Infrastructure</h1>
        <p className="mt-1 text-xs text-muted">{historical ? "Last recorded resource and process telemetry for this run." : "Live resource and process telemetry."}</p>
      </header>
      <div className="grid grid-cols-2 gap-x-8 gap-y-6 md:grid-cols-3 2xl:grid-cols-6">
        <StatTile label="Trainer status" value={runtimeStatus} sub={`heartbeat ${fmtAge(summary?.heartbeat?.timestamp)} · pid ${summary?.heartbeat?.pid ?? "–"}`} tone={summary?.status === "stale" ? "serious" : summary?.status === "failed" ? "critical" : null} />
        <StatTile label={historical ? "Last GPU memory" : "GPU memory"} value={allocatedMemory === null ? "–" : `${fmt(allocatedMemory / GIB, 1)} GiB`} sub={allocatedMemory === null ? "no GPU telemetry" : `reserved ${reservedMemory === null ? "–" : fmt(reservedMemory / GIB, 1)} · peak ${fmt(num(latest, "perf/peak_memory_gib"), 1)}`} />
        <StatTile label={historical ? "Last throughput" : "Throughput"} number={numAny(latest, TOKENS_PER_SECOND_KEYS)} format={(v) => `${fmtInt(v)} tok/s`} sub={num(latest, "perf/mfu") !== null ? `MFU ${fmtPct(num(latest, "perf/mfu"))}` : num(latest, "perf/model_tokens_per_second") !== null ? `${fmtInt(num(latest, "perf/model_tokens_per_second"))} model tok/s` : "trainer tokens per second"} />
        <StatTile label="Disk free" value={fmtPct(num(latest, "disk_free_ratio"))} sub={`${fmtBytes(num(latest, "disk_free_bytes"))} of ${fmtBytes(num(latest, "disk_total_bytes"))}`} tone={(num(latest, "disk_free_ratio") ?? 1) < 0.1 ? "serious" : null} />
        <StatTile label="Step time" number={numAny(latest, STEP_SECONDS_KEYS)} format={(v) => fmtSeconds(v)} sub={`train ${fmtSeconds(numAny(latest, TRAIN_SECONDS_KEYS))}${num(latest, "time/wait_for_batch") !== null ? ` · waiting ${fmtSeconds(num(latest, "time/wait_for_batch"))}` : ""}`} />
        <StatTile label="Inference KV cache" value={replicas.length ? fmtPct(Math.max(...replicas.map((r) => num(latestO, `inference/${r}/kv_cache_usage`) ?? 0))) : "–"} sub={replicas.length ? `${replicas.length} replica(s) · waiting ${fmtInt(replicas.reduce((a, r) => a + (num(latestO, `inference/${r}/requests_waiting`) ?? 0), 0))}` : "no scrape data"} />
      </div>

      <Section title="Trainer" className="section">
        <div className="grid gap-x-10 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
          <ChartCard title="GPU memory (GiB)" refetching={trainer.refetching} table={<SeriesTable series={memory} />}><LineChart series={memory} height={170} yFormat={(v) => fmt(v, 1)} /></ChartCard>
          <ChartCard title="Tokens per second" refetching={trainer.refetching} table={<SeriesTable series={throughput} />}><LineChart series={throughput} height={170} yFormat={(v) => fmtInt(v)} /></ChartCard>
          {mfu.some((l) => l.points.length) ? (
            <ChartCard title="Model FLOP utilization" refetching={trainer.refetching} table={<SeriesTable series={mfu} />}><LineChart series={mfu} height={170} yFormat={(v) => fmtPct(v, 0)} yDomain={[0, 1]} /></ChartCard>
          ) : (
            <ChartCard title="Disk free ratio" refetching={trainer.refetching} table={<SeriesTable series={disk} />}><LineChart series={disk} height={170} yFormat={(v) => fmtPct(v, 0)} yDomain={[0, 1]} /></ChartCard>
          )}
          {timeKeys.length > 0 && (
            <ChartCard title="Trainer step time breakdown" subtitle="seconds per optimizer step" refetching={trainer.refetching} className="md:col-span-2">
              <BarChart categories={timeCats} seriesLabels={timeKeys.map((k) => k.replace("time/", "").replace("perf/", ""))} height={190} yFormat={(v) => fmtSeconds(v)} />
            </ChartCard>
          )}
          {mfu.some((l) => l.points.length) && (
            <ChartCard title="Disk free ratio" refetching={trainer.refetching} table={<SeriesTable series={disk} />}><LineChart series={disk} height={170} yFormat={(v) => fmtPct(v, 0)} yDomain={[0, 1]} /></ChartCard>
          )}
        </div>
      </Section>

      <Disclosure id="infra.inference" title="Inference and orchestrator" summary={replicas.length ? `${replicas.length} replica(s)` : "timing only"} className="section">
        {replicas.length === 0 && orchTime.every((l) => l.points.length === 0) ? (
          <Empty title="No inference metrics" hint="vLLM replica load appears when orchestrator.concurrency scraping is enabled; timing metrics appear after the first published batch." />
        ) : (
          <div className="grid gap-x-10 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
            {replicas.length > 0 && (
              <>
                <ChartCard title="KV cache usage" subtitle="per replica" refetching={orch.refetching} table={<SeriesTable series={kv} yFormat={(v) => fmtPct(v)} />}><LineChart series={kv} height={170} yDomain={[0, 1]} yFormat={(v) => fmtPct(v, 0)} /></ChartCard>
                <ChartCard title="Requests" refetching={orch.refetching} table={<SeriesTable series={requests} />}><LineChart series={requests} height={170} yFormat={(v) => fmtInt(v)} /></ChartCard>
                <ChartCard title="Preemptions per scrape" refetching={orch.refetching} table={<SeriesTable series={preemptions} />}><LineChart series={preemptions} height={170} yFormat={(v) => fmtInt(v)} /></ChartCard>
              </>
            )}
            {concurrency.some((l) => l.points.length) && <ChartCard title="Concurrency" refetching={orch.refetching} table={<SeriesTable series={concurrency} />}><LineChart series={concurrency} height={170} yFormat={(v) => fmtInt(v)} /></ChartCard>}
            <ChartCard title="Orchestrator timing" subtitle="seconds per step" refetching={orch.refetching} table={<SeriesTable series={orchTime} yFormat={(v) => fmtSeconds(v)} />}><LineChart series={orchTime} height={170} yFormat={(v) => fmtSeconds(v)} /></ChartCard>
          </div>
        )}
      </Disclosure>

      <NodesPanel apiBase={apiBase} runId={runId} />

      <Disclosure id="infra.process" title="Process" summary={`${summary?.launcher_mode ?? "–"} · pid ${summary?.heartbeat?.pid ?? "–"}`} className="section">
        <div>
          <KeyValue columns={4} items={[["Launcher", summary?.launcher_mode ?? "–"], ["World", summary?.world ? `${summary.world.world_size} rank(s) on ${summary.world.device}` : "–"], ["Trainer pid", String(summary?.heartbeat?.pid ?? "–")], ["Resumed from", summary?.resumed_from ?? "fresh start"], ["Started", summary?.started_at ?? "–"], ["Last metric", fmtAge(summary?.updated_at)], ["Output dir", summary?.path ?? "–"], ["Config", summary?.has_config ? "resolved config found" : "no resolved config"]]} />
        </div>
      </Disclosure>

      <LogsPanel apiBase={apiBase} runId={runId} logs={summary?.logs ?? []} live={live} />
      <ErrorNote error={trainer.error ?? orch.error} />
    </div>
  );
}

function LogsPanel({ apiBase, runId, logs, live }: { apiBase: string; runId: string; logs: RunSummary["logs"]; live: boolean }) {
  const [name, setName] = useState<string | null>(null);
  const [lines, setLines] = useState(200);
  const [follow, setFollow] = useState(live);
  const [grep, setGrep] = useState("");
  useEffect(() => {
    if (live) setFollow(true);
  }, [live]);
  const active = name ?? logs[0]?.name ?? null;
  const tail = usePoll<{ name: string; lines: string[] }>(active ? `${runUrl(apiBase, runId, `/logs/${encodeURIComponent(active)}`)}${qs({ lines })}` : null, follow ? 3000 : 0);
  const shown = (tail.data?.lines ?? []).filter((line) => !grep || line.toLowerCase().includes(grep.toLowerCase()));
  return (
    <Disclosure id="infra.logs" title="Role logs" summary={logs.map((l) => l.name.replace(/\.log$/, "")).join(" · ") || "none"} className="section">
      <div>
        <Toolbar>
          <Segmented value={active ?? ""} onChange={setName} options={logs.map((l) => ({ value: l.name, label: `${l.name.replace(/\.log$/, "")} (${fmtBytes(l.bytes)})` }))} size="xs" />
          <Field label="tail">
            <Segmented value={String(lines)} onChange={(v) => setLines(Number(v))} size="xs" options={[{ value: "100", label: "100" }, { value: "200", label: "200" }, { value: "1000", label: "1000" }]} />
          </Field>
          <input className="input w-full sm:w-48" aria-label="Filter log lines" placeholder="Filter logs" value={grep} onChange={(e) => setGrep(e.target.value)} />
          <button type="button" className={`btn !py-0.5 ${follow ? "btn-active" : ""}`} onClick={() => setFollow((v) => !v)}>{follow ? "auto-refresh on" : live ? "auto-refresh off" : "snapshot"}</button>
          <span className="text-[11px] text-muted">{shown.length} lines · {fmtAge(tail.updatedAt)}</span>
        </Toolbar>
        {logs.length === 0 && <p className="mt-3 text-xs text-muted">No role logs under logs/latest.</p>}
        <ErrorNote error={tail.error} />
        {active && (
          <pre className="transcript mt-3 max-h-[28rem] overflow-auto rounded-md bg-raised px-3 py-2.5 font-mono text-[11px] leading-relaxed text-ink2">
            {shown.map((line, i) => <div key={i} className={/error|traceback|exception/i.test(line) ? "text-critical" : /warn/i.test(line) ? "text-warn" : ""}>{line}</div>)}
          </pre>
        )}
      </div>
    </Disclosure>
  );
}
