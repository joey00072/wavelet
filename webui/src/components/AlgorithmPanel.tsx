import type { AlgorithmSource, AlgorithmTopology } from "../types";
import { formatNumber } from "../utils/format";

export function AlgorithmPanel({ topology }: { topology: AlgorithmTopology | undefined }) {
  if (!topology) {
    return null;
  }
  const totalWeight = topology.sources.reduce((total, source) => total + source.weight, 0);

  return (
    <section>
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500 dark:text-slate-400">
            Algorithm mix
          </h2>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
            {topology.sources.length} source{topology.sources.length === 1 ? "" : "s"} update one student
            {topology.student.lora_enabled ? " and one adapter" : " policy"}.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          {topology.loss_components.map((component) => (
            <LossBadge key={component} component={component} />
          ))}
          <span className="text-slate-500 dark:text-slate-400">
            {topology.teacher_count} teacher{topology.teacher_count === 1 ? "" : "s"}
            {topology.observed_step === null ? "" : ` · observed step ${topology.observed_step}`}
          </span>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {topology.sources.map((source) => (
          <SourceCard key={source.name} source={source} totalWeight={totalWeight} />
        ))}
      </div>
    </section>
  );
}

function SourceCard({ source, totalWeight }: { source: AlgorithmSource; totalWeight: number }) {
  const configuredShare = totalWeight > 0 ? source.weight / totalWeight : 0;
  const observedShare = source.observed.batch_fraction;
  const lossObservation = sourceLossObservation(source);

  return (
    <article className="rounded-xl border border-slate-200 bg-white/60 p-5 dark:border-white/[0.06] dark:bg-white/[0.015]">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
              {source.name}
            </h3>
            <span className="rounded bg-slate-900 px-2 py-0.5 font-mono text-[10px] font-medium uppercase text-white dark:bg-slate-100 dark:text-slate-900">
              {source.algorithm.name ?? source.algorithm.type}
            </span>
            <span className="font-mono text-[10px] text-slate-400">{source.algorithm.scope}</span>
          </div>
          <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400">
            {source.environment ?? "custom rollout source"}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <div className="font-mono text-sm font-medium text-slate-800 dark:text-slate-200">
            {formatPercent(observedShare ?? configuredShare)}
          </div>
          <div className="text-[10px] text-slate-400">
            {observedShare === null ? "configured" : "observed"} share
          </div>
        </div>
      </div>

      {source.algorithm.teacher && (
        <div className="mt-4 border-t border-slate-100 pt-4 dark:border-white/[0.04]">
          <div className="text-[10px] font-medium uppercase tracking-widest text-slate-400">Teacher</div>
          <div className="mt-1 flex items-center justify-between gap-4">
            <span className="truncate font-mono text-xs text-slate-700 dark:text-slate-300">
              {source.algorithm.teacher.name}
            </span>
            <span className="shrink-0 text-[10px] text-slate-400">
              {source.algorithm.teacher.replica_count} endpoint
              {source.algorithm.teacher.replica_count === 1 ? "" : "s"}
            </span>
          </div>
        </div>
      )}

      <dl className="mt-4 grid grid-cols-3 gap-4 border-t border-slate-100 pt-4 dark:border-white/[0.04]">
        <Observation label="Reward" value={formatNumber(source.observed.reward_mean)} />
        <Observation label="Trainable" value={formatPercent(source.observed.trainable_rate)} />
        <Observation label={lossObservation.label} value={lossObservation.value} />
      </dl>
    </article>
  );
}

function Observation({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="mt-1 font-mono text-xs font-medium text-slate-700 dark:text-slate-300">{value}</dd>
    </div>
  );
}

function LossBadge({ component }: { component: "rl" | "ce" | "ref_kl" }) {
  const label = component === "ref_kl" ? "REF-KL" : component.toUpperCase();
  return (
    <span className="rounded border border-slate-200 px-2 py-0.5 font-mono text-slate-600 dark:border-white/10 dark:text-slate-300">
      {label}
    </span>
  );
}

function formatPercent(value: number | null): string {
  return value === null ? "-" : `${formatNumber(value * 100, 1)}%`;
}

function sourceLossObservation(source: AlgorithmSource): { label: string; value: string } {
  if (source.algorithm.type === "opd") {
    return { label: "Ref aligned", value: formatPercent(source.observed.ref_logprobs_rate) };
  }
  const streams = [
    { component: "RL", rate: source.observed.rl_loss_rate },
    { component: "CE", rate: source.observed.ce_loss_rate },
    { component: "REF-KL", rate: source.observed.ref_kl_loss_rate },
  ].filter(({ rate }) => rate !== null && rate > 0);
  if (streams.length === 1) {
    return { label: `${streams[0].component} active`, value: formatPercent(streams[0].rate) };
  }
  if (streams.length > 1) {
    return { label: "Loss streams", value: `${streams.length} active` };
  }
  if (source.algorithm.loss_components.includes("rl")) {
    return { label: "RL active", value: "-" };
  }
  return { label: "Loss active", value: "-" };
}
