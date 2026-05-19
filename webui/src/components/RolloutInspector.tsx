import { useEffect, useMemo, useState } from "react";
import { RefreshCw, Sparkles } from "lucide-react";

import type { RolloutInspection, RolloutSample } from "../types";
import { formatNumber, formatTime } from "../utils/format";
import { Metric } from "./Metric";

export function RolloutInspector({
  title = "Rollout Inspector",
  inspection,
  error,
  updatedAt,
  onRefresh,
}: {
  title?: string;
  inspection: RolloutInspection | null;
  error: string | null;
  updatedAt: string | null;
  onRefresh: () => void;
}) {
  const sampleOptions = useMemo(() => buildSampleOptions(inspection), [inspection]);
  const [selectedSampleKey, setSelectedSampleKey] = useState("random-0");
  const selectedOption =
    sampleOptions.find((o) => o.key === selectedSampleKey && o.sample) ??
    sampleOptions.find((o) => o.sample) ??
    null;
  const selectedSample = selectedOption?.sample ?? null;

  useEffect(() => {
    if (!sampleOptions.some((o) => o.key === selectedSampleKey && o.sample)) {
      setSelectedSampleKey(sampleOptions.find((o) => o.sample)?.key ?? "random-0");
    }
  }, [sampleOptions, selectedSampleKey]);

  return (
    <section>
      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-slate-400 dark:text-slate-500" />
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            batch {inspection?.queue_step ?? "–"} · {formatNumber(inspection?.scanned_rows, 0)} rows
            {inspection?.truncated ? " · bounded" : ""}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {error ? <span className="text-red-400">{error}</span> : `Updated ${formatTime(updatedAt)}`}
          </span>
          <button
            type="button"
            onClick={onRefresh}
            className="flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-white/5 dark:bg-transparent dark:text-slate-300 dark:hover:bg-white/5"
          >
            <RefreshCw className="h-3 w-3" />
            Refresh
          </button>
        </div>
      </div>

      <div>
        {!inspection?.available ? (
          <p className="py-6 text-sm text-slate-500 dark:text-slate-400">
            {inspection?.reason ?? "No rollout batch available yet"}
          </p>
        ) : (
          <div className="space-y-8">
            {/* Controls Row */}
            <div className="flex flex-col gap-6 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex flex-wrap gap-2">
                {sampleOptions.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    onClick={() => setSelectedSampleKey(option.key)}
                    disabled={!option.sample}
                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-30 ${
                      selectedOption?.key === option.key
                        ? "bg-slate-900 text-white dark:bg-slate-200 dark:text-slate-900"
                        : "bg-slate-50 text-slate-600 hover:bg-slate-100 dark:bg-white/5 dark:text-slate-400 dark:hover:bg-white/10"
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <dl className="flex flex-wrap gap-x-8 gap-y-3">
                <Metric label="Reward mean" value={formatNumber(inspection.stats.reward.mean)} />
                <Metric label="Reward std" value={formatNumber(inspection.stats.reward.std)} />
                <Metric label="Rows" value={inspection.manifest?.rows ?? inspection.scanned_rows} />
                <Metric label="Policy" value={inspection.manifest?.policy_step ?? "–"} />
              </dl>
            </div>

            {selectedSample ? (
              <div className="space-y-6">
                <SampleHeader label={selectedOption?.label ?? "Sample"} sample={selectedSample} />

                {/* Aligned Prompt/Response Grid */}
                <div className="grid gap-8 xl:grid-cols-2">
                  <RolloutText label="Prompt" value={selectedSample.prompt} maxHeight="max-h-[32rem]" />
                  <RolloutText label="Response" value={selectedSample.completion} maxHeight="max-h-[32rem]" />
                </div>

                <div className="mt-8 border-t border-slate-200 pt-6 dark:border-white/[0.04]">
                  <p className="mb-3 break-all font-mono text-[11px] text-slate-400 dark:text-slate-500">
                    {inspection.path}
                  </p>
                  <dl className="flex gap-8">
                    <Metric label="Producer" value={inspection.manifest?.producer_id ?? "–"} />
                    <Metric label="Scanned" value={inspection.scanned_rows} />
                  </dl>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500 dark:text-slate-400">No rollout samples available</p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function buildSampleOptions(inspection: RolloutInspection | null) {
  const random = inspection?.samples.random ?? [];
  return [
    ...random.map((sample, i) => ({ key: `random-${i}`, label: `Random ${i + 1}`, sample })),
    { key: "min", label: "Min reward", sample: inspection?.samples.min_reward ?? null },
    { key: "mean", label: "Near mean", sample: inspection?.samples.near_mean_reward ?? null },
    { key: "max", label: "Max reward", sample: inspection?.samples.max_reward ?? null },
  ];
}

function SampleHeader({ label, sample }: { label: string; sample: RolloutSample }) {
  return (
    <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 dark:border-white/[0.04] sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-4">
        <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">{label}</span>
        <span className="font-mono text-xs text-slate-500 dark:text-slate-400">
          reward{" "}
          <span className="font-medium text-slate-900 dark:text-slate-200">{formatNumber(sample.reward)}</span>
          {" · "}adv{" "}
          <span className="font-medium text-slate-900 dark:text-slate-200">{formatNumber(sample.advantage)}</span>
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {[
          `example ${sample.example_id ?? "–"}`,
          `${sample.completion_token_count ?? "–"} tokens`,
          sample.stop_condition ?? "–",
        ].map((t) => (
          <span key={t} className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-mono text-[10px] text-slate-500 dark:border-white/[0.06] dark:bg-white/[0.02] dark:text-slate-400">
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

function RolloutText({
  label,
  value,
  maxHeight,
}: {
  label: string;
  value?: string;
  maxHeight: string;
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400">{label}</div>
      <pre className={`code-block flex-1 overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-5 dark:border-white/[0.04] dark:bg-white/[0.01] ${maxHeight}`}>{value || "–"}</pre>
    </div>
  );
}
