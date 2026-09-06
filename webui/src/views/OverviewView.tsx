import { useMemo } from "react";
import { AlertTriangle, CheckCircle2, Info, OctagonAlert } from "lucide-react";

import type { RunSummary } from "../api/types";
import { BarChart } from "../charts/BarChart";
import { ChartCard, chartProps } from "../charts/ChartCard";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { StatTile } from "../charts/StatTile";
import { StatusBadge } from "../components/Badge";
import { Disclosure } from "../components/Disclosure";
import { Empty, KeyValue } from "../components/KeyValue";
import { fmt, fmtAge, fmtInt, fmtPct, fmtSeconds, modelLabel } from "../lib/format";
import { healthFindings, type Finding } from "../lib/health";
import { navigate } from "../lib/router";
import { STEP_SECONDS_KEYS, TOKENS_PER_SECOND_KEYS, lastFinite, num, numAny, seriesToLines, trailingMean, windowDelta } from "../lib/series";
import { ORCH_OVERVIEW_KEYS, TRAINER_OVERVIEW_KEYS, useSeries } from "./useRunData";

export function OverviewView({ apiBase, runId, summary }: { apiBase: string; runId: string; summary: RunSummary | null }) {
  const interval = summary?.status === "running" ? 5000 : 0;
  const trainer = useSeries(apiBase, runId, "trainer", TRAINER_OVERVIEW_KEYS, 2000, interval);
  const orch = useSeries(apiBase, runId, "orchestrator", ORCH_OVERVIEW_KEYS, 2000, interval);
  const evalKeys = useMemo(() => (summary?.eval_envs ?? []).flatMap((env) => Object.keys(summary?.latest.eval ?? {}).filter((k) => k.startsWith(`eval/${env}/`) && /^eval\/[^/]+\/avg@\d+$/.test(k))), [summary]);
  const evals = useSeries(apiBase, runId, "eval", evalKeys, 2000, interval);
  const findings = useMemo(() => healthFindings(summary), [summary]);

  if (!summary) return <Empty title="Loading run…" />;
  const latestT = summary.latest.trainer;
  const latestO = summary.latest.orchestrator;
  const step = summary.trainer_step;
  const target = summary.target_step;
  const progress = step !== null && target ? Math.min(1, (step + 1) / target) : null;
  const orchestratorReward = orch.data?.series["reward/all/mean"];
  const rewardFromOrchestrator = orchestratorReward?.some((value) => value !== null && Number.isFinite(value));
  const rewardSeries = rewardFromOrchestrator ? orchestratorReward : trainer.data?.series["reward/all/mean"];
  const rewardMean = trailingMean(rewardSeries, 5);
  const rewardDelta = windowDelta(rewardSeries, 5);
  const rewardPoints = pointsOf(rewardFromOrchestrator ? orch.data?.steps ?? [] : trainer.data?.steps ?? [], rewardSeries);
  const lag = num(latestO, "policy/lag");
  const loss = numAny(latestT, ["train/loss", "loss"]);
  const solveNone = numAny(latestO, ["generation/solve_none/rate", "solve_none/all"]);
  const evalHeadline = evalKeys[0] ? lastFinite(evals.data?.series[evalKeys[0]]) : null;
  const evalDelta = evalKeys[0] ? windowDelta(evals.data?.series[evalKeys[0]], 1) : null;
  const evalIsRate = Boolean(evalKeys[0] && /\/(pass@\d+|pass\^\d+)$/.test(evalKeys[0]));

  const rewardLines = rewardFromOrchestrator
    ? seriesToLines(orch.data, ["reward/all/mean", "generation/reward/mean"], { labels: { "reward/all/mean": "train batch (after filtering)", "generation/reward/mean": "all generated (before filtering)" } })
    : seriesToLines(trainer.data, ["reward/all/mean"], { labels: { "reward/all/mean": "trainer reward" } });
  const lossLines = seriesToLines(trainer.data, ["train/policy_loss", "train/kl_loss"], { labels: { "train/policy_loss": "policy loss", "train/kl_loss": "kl loss" } });
  const entropyLines = seriesToLines(trainer.data, ["entropy/mean"]);
  const maskKey = trainer.data?.series["ipo/is_masked"]?.some((v) => v !== null) ? "ipo/is_masked" : "dppo/is_masked";
  const mismatchLines = seriesToLines(trainer.data, ["kl/mismatch", maskKey], { labels: { "kl/mismatch": "kl mismatch", [maskKey]: "masked token fraction" } });
  const gradLines = seriesToLines(trainer.data, ["optim/grad_norm"]);
  const has = (key: string) => Boolean(orch.data?.series[key]?.some((v) => v !== null));
  const solveKeys = [
    has("generation/solve_all/rate") ? "generation/solve_all/rate" : "solve_all/all",
    has("generation/solve_none/rate") ? "generation/solve_none/rate" : "solve_none/all",
    ...(has("generation/groups/admission_rate") ? ["generation/groups/admission_rate"] : []),
  ];
  const solveLines = seriesToLines(orch.data, solveKeys, { labels: { "generation/solve_all/rate": "solve all", "solve_all/all": "solve all", "generation/solve_none/rate": "solve none", "solve_none/all": "solve none", "generation/groups/admission_rate": "admitted" } });
  const lagLines = seriesToLines(orch.data, ["off_policy/mean", "off_policy/max", "policy/lag"], { labels: { "off_policy/mean": "rollout age mean", "off_policy/max": "rollout age max", "policy/lag": "policy lag" } });
  const lenLines = seriesToLines(orch.data, ["decode_len/all/mean"], { labels: { "decode_len/all/mean": "completion tokens" } });
  const truncLines = seriesToLines(orch.data, ["is_truncated/all/mean"], { labels: { "is_truncated/all/mean": "truncated fraction" } });
  const evalLines = seriesToLines(evals.data, evalKeys, { labels: Object.fromEntries(evalKeys.map((k) => [k, k.replace(/^eval\//, "")])) });
  const fateCats = (orch.data?.steps ?? []).map((s, i) => ({
    label: String(s ?? i),
    values: ["fate/all/trainable", "fate/all/filtered", "fate/all/truncated", "fate/all/errored", "fate/all/zero_loss"].map((k) => orch.data?.series[k]?.[i] ?? 0),
  }));
  const hasFate = fateCats.some((c) => c.values.some((v) => v > 0));

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-ink">{summary.id}</h1>
          <StatusBadge status={summary.status} reason={summary.status_reason} />
          <span className="tabular text-xs text-muted">updated {fmtAge(summary.updated_at)}</span>
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-xs text-ink2">
          <span className="tabular"><span className="text-muted">step </span>{step === null ? "–" : step + 1}{target ? ` / ${target}` : ""}</span>
          <Kv label="policy" value={`exported ${summary.policy?.latest_exported_step ?? "–"} · lag ${lag ?? "–"}`} />
          <Kv label="queue" value={summary.queue_summary ? `${summary.queue_summary.ready_count} ready · ${summary.queue_summary.claimed_count} claimed · ${summary.queue_summary.stale_ready_count} stale` : "not recorded"} />
          <Kv label="mode" value={summary.launcher_mode ?? "–"} />
          <Kv label="model" value={modelLabel(summary.model)} />
        </div>
        <div className="h-px w-full overflow-hidden bg-raised">
          <div className="h-full bg-accent transition-[width]" style={{ width: `${(progress ?? 0) * 100}%` }} />
        </div>
      </header>

      <div className="grid grid-cols-2 gap-x-8 gap-y-6 md:grid-cols-4">
        <StatTile label="Train reward" number={rewardMean} format={(v) => fmt(v, 3)} delta={rewardDelta === null ? null : `${rewardDelta >= 0 ? "+" : ""}${fmt(rewardDelta, 3)}`} deltaGood={rewardDelta === null ? null : rewardDelta >= 0} trend={rewardPoints} sub="last 5 vs first 5 steps" />
        <StatTile label={evalKeys[0] ? `Eval ${evalKeys[0].split("/").pop()}` : "Eval"} number={evalHeadline} format={(v) => evalIsRate ? fmtPct(v) : fmt(v, 3)} delta={evalDelta === null ? null : `${evalDelta >= 0 ? "+" : ""}${evalIsRate ? fmtPct(evalDelta) : fmt(evalDelta, 3)}`} deltaGood={evalDelta === null ? null : evalDelta >= 0} sub={summary.eval_step === null ? "no evals yet" : `${evalKeys[0]?.split("/")[1] ?? ""} · policy ${summary.eval_step}`} />
        <StatTile label="Entropy" number={num(latestT, "entropy/mean")} format={(v) => fmt(v, 3)} sub={`kl mismatch ${fmt(num(latestT, "kl/mismatch"), 4)}`} tone={(num(latestT, "entropy/mean") ?? 1) < 0.05 ? "warning" : null} />
        <StatTile label="Truncated" number={num(latestO, "is_truncated/all/mean")} format={(v) => fmtPct(v)} sub={`${fmtInt(num(latestO, "decode_len/all/mean"))} tokens mean · solve none ${fmtPct(solveNone)}`} tone={(num(latestO, "is_truncated/all/mean") ?? 0) > 0.25 ? "warning" : null} />
      </div>

      <div className="section grid gap-x-10 gap-y-8 lg:grid-cols-3">
        <div className="min-w-0 lg:col-span-2">
          <ChartCard title="Reward" subtitle="orchestrator reward per optimizer step · hover for smoothing" refetching={orch.refetching} smoothingKey="overview.reward" height={220} table={<SeriesTable series={rewardLines} />}>
            {(o) => <LineChart series={rewardLines} {...chartProps(o)} />}
          </ChartCard>
        </div>
        <Findings findings={findings} runId={runId} />
      </div>

      {evalKeys.length > 0 && (
        <div className="section">
        <ChartCard title="Fixed-policy evaluation" subtitle="mean reward (avg@k) per environment; dashed line marks the step-0 baseline" refetching={evals.refetching} smoothable={false} table={<SeriesTable series={evalLines} />}>
          <LineChart series={evalLines} height={200} markers xExtent={[0, Math.max(1, step ?? 1)]} references={evalLines.flatMap((l) => (l.points.length && l.points[0].x === 0 ? [{ y: l.points[0].y, label: `${l.label.split("/")[0]} step 0`, colorIndex: l.colorIndex }] : []))} />
        </ChartCard>
        </div>
      )}

      <Disclosure id="overview.trainer" title="Trainer signals" summary={`loss ${fmt(loss, 4)} · grad norm ${fmt(num(latestT, "optim/grad_norm"), 3)} · lr ${fmt(num(latestT, "optim/lr") ?? num(latestT, "lr"), 2)} · ${fmtInt(numAny(latestT, TOKENS_PER_SECOND_KEYS))} tok/s`} className="section">
        <div className="mb-6 grid grid-cols-2 gap-x-8 gap-y-6 md:grid-cols-4">
          <StatTile label="Loss" number={loss} format={(v) => fmt(v, 4)} sub={`policy ${fmt(num(latestT, "train/policy_loss"), 4)}`} />
          <StatTile label="Grad norm" number={num(latestT, "optim/grad_norm")} format={(v) => fmt(v, 3)} sub={`lr ${fmt(num(latestT, "optim/lr"), 2)}`} />
          <StatTile label="Solve none" number={solveNone} format={(v) => fmtPct(v)} sub={`solve all ${fmtPct(num(latestO, "generation/solve_all/rate") ?? num(latestO, "solve_all/all"))}`} />
          <StatTile label="Throughput" number={numAny(latestT, TOKENS_PER_SECOND_KEYS)} format={(v) => fmtInt(v)} sub={`tok/s · step ${fmtSeconds(numAny(latestO, STEP_SECONDS_KEYS) ?? numAny(latestT, STEP_SECONDS_KEYS))}`} />
        </div>
      <div className="grid gap-x-10 gap-y-8 md:grid-cols-2 xl:grid-cols-3">
        <ChartCard title="Loss" refetching={trainer.refetching} table={<SeriesTable series={lossLines} />}><LineChart series={lossLines} height={160} /></ChartCard>
        <ChartCard title="Entropy" refetching={trainer.refetching} table={<SeriesTable series={entropyLines} />}><LineChart series={entropyLines} height={160} /></ChartCard>
        <ChartCard title="Trainer vs inference mismatch" refetching={trainer.refetching} table={<SeriesTable series={mismatchLines} />}><LineChart series={mismatchLines} height={160} /></ChartCard>
        <ChartCard title="Gradient norm" refetching={trainer.refetching} table={<SeriesTable series={gradLines} />}><LineChart series={gradLines} height={160} /></ChartCard>
        <ChartCard title="Group outcomes" subtitle="fraction of groups" refetching={orch.refetching} table={<SeriesTable series={solveLines} yFormat={(v) => fmtPct(v)} />}><LineChart series={solveLines} height={160} yDomain={[0, 1]} yFormat={(v) => fmtPct(v, 0)} /></ChartCard>
        <ChartCard title="Off-policy distance" subtitle="policy versions between rollout generation and training" refetching={orch.refetching} table={<SeriesTable series={lagLines} />}><LineChart series={lagLines} height={160} /></ChartCard>
        <ChartCard title="Completion length" subtitle="mean decode tokens" refetching={orch.refetching} table={<SeriesTable series={lenLines} />}><LineChart series={lenLines} height={160} /></ChartCard>
        <ChartCard title="Truncation" refetching={orch.refetching} table={<SeriesTable series={truncLines} yFormat={(v) => fmtPct(v)} />}><LineChart series={truncLines} height={160} yDomain={[0, 1]} yFormat={(v) => fmtPct(v, 0)} /></ChartCard>
        {hasFate && (
          <ChartCard title="Rollout fate" subtitle="rows per step by outcome" refetching={orch.refetching}>
            <BarChart categories={fateCats} seriesLabels={["trainable", "filtered", "truncated", "errored", "zero loss"]} height={160} yFormat={(v) => fmtInt(v)} />
          </ChartCard>
        )}
      </div>

      </Disclosure>

      <Disclosure id="overview.run" title="Run details" summary={`${modelLabel(summary.model)} · ${summary.algo ?? "–"} · ${summary.envs.join(", ") || "–"}`} className="section">
        <div className="py-1">
          <KeyValue
            columns={4}
            items={[
              ["Output dir", summary.path],
              ["Model", modelLabel(summary.model)],
              ["Algorithm", `${summary.algo ?? "–"} · ${summary.loss ?? "–"}${summary.lora ? " · LoRA" : ""}`],
              ["Environments", summary.envs.join(", ") || "–"],
              ["Batch", summary.batch ? `${summary.batch.examples_per_step ?? "token-based"} × ${summary.batch.rollouts_per_example ?? "–"}` : "–"],
              ["Async", summary.batch ? `level ${summary.batch.max_async_level} · off-policy ≤ ${summary.batch.max_off_policy_steps}` : "–"],
              ["Started", summary.started_at ?? "–"],
              ["World", summary.world ? `${summary.world.world_size} rank(s) · ${summary.world.device}` : "–"],
            ]}
          />
        </div>
      </Disclosure>
    </div>
  );
}

function Kv({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-xs">
      <span className="text-muted">{label} </span>
      <span className="tabular text-ink2">{value}</span>
    </div>
  );
}

function pointsOf(steps: Array<number | null>, values: Array<number | null> | undefined) {
  if (!values) return [];
  return steps.flatMap((s, i) => (s !== null && values[i] !== null && values[i] !== undefined ? [{ x: s, y: values[i] as number }] : []));
}

function Findings({ findings, runId }: { findings: Finding[]; runId: string }) {
  return (
    <div className="flex flex-col">
      <div className="title mb-3">Health checks</div>
      <ul className="min-h-0 flex-1 space-y-1.5 overflow-auto">
        {findings.length === 0 && <li className="text-xs text-muted">No findings for a finished run.</li>}
        {findings.map((f) => (
          <li key={f.title} className="flex items-start gap-2 text-xs">
            <FindingIcon level={f.level} />
            <div className="min-w-0">
              <button type="button" className="text-left font-medium text-ink hover:underline" onClick={() => f.view && navigate({ page: "run", runId, view: f.view })}>
                {f.title}
              </button>
              <div className="text-[11px] text-muted">{f.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

const FINDING_ICONS = { critical: [OctagonAlert, "text-critical"], serious: [AlertTriangle, "text-serious"], warning: [Info, "text-warn"], good: [CheckCircle2, "text-good"] } as const;

function FindingIcon({ level }: { level: Finding["level"] }) {
  const [Icon, color] = FINDING_ICONS[level];
  return <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${color}`} />;
}
