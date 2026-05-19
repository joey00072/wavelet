import {
  Activity,
  Boxes,
  Cpu,
  Gauge,
  Loader2,
  PackageCheck,
  RefreshCw,
  Send,
} from "lucide-react";

import { Metric } from "../components/Metric";
import { InventoryCard, Stat, StatusIcon } from "../components/StatusCards";
import { ThroughputChart } from "../components/ThroughputChart";
import type { MetricRow, PipelineInventory, RolloutEvent, RunState } from "../types";
import { formatNumber, formatRate } from "../utils/format";

export function OverviewView({
  state,
  latest,
  events,
  metrics,
  counts,
  inventory,
  step,
  progress,
  publishedRate,
  submittedRate,
  consumedRate,
}: {
  state: RunState | null;
  latest: MetricRow | null;
  events: RolloutEvent[];
  metrics: MetricRow[];
  counts: { submitted: number; completed: number; published: number };
  inventory: PipelineInventory;
  step: number | null;
  progress: number;
  publishedRate: number;
  submittedRate: number;
  consumedRate: number;
}) {
  return (
    <div className="space-y-12 pb-12">
      {/* ── Key stats ── */}
      <section className="grid grid-cols-2 gap-8 sm:grid-cols-4">
        <Stat
          icon={<StatusIcon status={state?.status ?? "starting"} />}
          label="Status"
          value={state?.status ?? "–"}
          sub={state?.phase ?? "waiting"}
        />
        <Stat
          icon={<Gauge className="h-4 w-4 text-slate-400" />}
          label="Trainer step"
          value={step === null ? "–" : `${step} / ${state?.target_step ?? "–"}`}
          sub={`${formatNumber(progress, 1)}% complete`}
        />
        <Stat
          icon={<RefreshCw className="h-4 w-4 text-slate-400" />}
          label="Published rate"
          value={formatRate(publishedRate)}
          sub={`submitted ${formatRate(submittedRate)}`}
        />
        <Stat
          icon={<Activity className="h-4 w-4 text-slate-400" />}
          label="Consumed rate"
          value={formatRate(consumedRate)}
          sub={`reward ${formatNumber(latest?.reward_mean ?? latest?.["reward/all/mean"])}`}
        />
      </section>

      <div className="h-px w-full bg-slate-200 dark:bg-white/[0.04]" />

      {/* ── Throughput chart ── */}
      <section>
        <ThroughputChart events={events} metrics={metrics} />
      </section>

      <div className="h-px w-full bg-slate-200 dark:bg-white/[0.04]" />

      {/* ── Pipeline inventory ── */}
      <section>
        <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <SectionHeading>Pipeline Inventory</SectionHeading>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {inventory.chunksPerStep} batch/step · {inventory.rolloutsPerChunk} rollouts/batch
          </span>
        </div>
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 xl:grid-cols-5">
          <InventoryCard
            icon={<Loader2 className="h-4 w-4" />}
            label="Generating"
            value={formatNumber(inventory.generating, 0)}
            sub={inventory.generating > 0 ? `batch ${inventory.nextGenerateStep ?? "-"}` : "not active"}
            tone="blue"
          />
          <InventoryCard
            icon={<PackageCheck className="h-4 w-4" />}
            label="Ready"
            value={formatNumber(inventory.readyForTrainer, 0)}
            sub={`${formatNumber(inventory.readyForTrainer * inventory.rolloutsPerChunk, 0)} queued rollouts`}
            tone="emerald"
          />
          <InventoryCard
            icon={<Cpu className="h-4 w-4" />}
            label="Training"
            value={formatNumber(inventory.trainerUsingChunks, 0)}
            sub={
              inventory.trainerUsingChunks > 0
                ? `last trainer step ${inventory.trainerStep ?? "-"}`
                : `next step ${inventory.nextTrainStep ?? "-"}`
            }
            tone="amber"
          />
          <InventoryCard
            icon={<Boxes className="h-4 w-4" />}
            label="Awaiting publish"
            value={formatNumber(inventory.completedWaitingPublish, 0)}
            sub={`watermark ${formatNumber(inventory.publishedWatermark, 0)}`}
            tone="cyan"
          />
          <InventoryCard
            icon={<Send className="h-4 w-4" />}
            label="Submitted"
            value={formatNumber(inventory.submitted, 0)}
            sub={`consumed ${formatNumber(inventory.consumedEstimate, 0)}`}
          />
        </div>
      </section>

      <div className="h-px w-full bg-slate-200 dark:bg-white/[0.04]" />

      {/* ── Data + policy + metrics grids ── */}
      <section className="grid gap-12 sm:grid-cols-3">
        {/* Rollout Totals */}
        <div>
          <SectionHeading>Rollout Totals</SectionHeading>
          <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-6">
            <Metric label="Submitted" value={counts.submitted} />
            <Metric label="Completed" value={counts.completed} />
            <Metric label="Published" value={counts.published} />
            <Metric label="Pending" value={state?.rollouts.pending_count ?? 0} />
          </dl>
        </div>

        {/* Policy */}
        <div>
          <SectionHeading>Policy</SectionHeading>
          <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-6">
            <Metric label="Loaded step" value={state?.policy.loaded_step ?? "–"} />
            <Metric label="Requested" value={state?.policy.requested_step ?? "–"} />
            <Metric label="Pending load" value={state?.policy.pending_load ? "yes" : "no"} />
            <Metric
              label="Latest export"
              value={
                state?.policy.available_tail.length
                  ? state.policy.available_tail[state.policy.available_tail.length - 1]
                  : "–"
              }
            />
          </dl>
        </div>

        {/* Recent Metrics */}
        <div>
          <SectionHeading>Recent Metrics</SectionHeading>
          <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-6">
            <Metric label="Loss" value={formatNumber(latest?.loss)} />
            <Metric label="LR" value={formatNumber(latest?.lr ?? latest?.["optim/lr"], 8)} />
            <Metric label="Train tokens" value={formatNumber(latest?.["tokens/train"], 0)} />
            <Metric label="Step tok/s" value={formatNumber(latest?.["perf/step_tokens_per_second"], 0)} />
          </dl>
        </div>
      </section>
    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">
      {children}
    </h2>
  );
}
