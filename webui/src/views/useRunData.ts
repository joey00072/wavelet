import { qs, runUrl, usePoll, type PollState } from "../api/client";
import type { MetricKeys, Nodes, RunSummary, Series } from "../api/types";

export function useSummary(apiBase: string, runId: string | null, intervalMs = 3000): PollState<RunSummary> {
  return usePoll<RunSummary>(runId ? runUrl(apiBase, runId, "/summary") : null, intervalMs);
}

/** Default cap on points per series; the server bucket-averages beyond it. */
const SERIES_POINTS = 1500;

export function useSeries(apiBase: string, runId: string | null, source: "trainer" | "orchestrator" | "eval", keys: string[], limit = 0, intervalMs = 5000, points = SERIES_POINTS): PollState<Series> {
  const url = runId && keys.length > 0 ? `${runUrl(apiBase, runId, "/series")}${qs({ source, keys: keys.join(","), limit, points })}` : null;
  return usePoll<Series>(url, intervalMs);
}

export function useNodes(apiBase: string, runId: string | null, intervalMs = 5000): PollState<Nodes> {
  return usePoll<Nodes>(runId ? runUrl(apiBase, runId, "/nodes") : null, intervalMs);
}

export function useMetricKeys(apiBase: string, runId: string, intervalMs = 15000): PollState<MetricKeys> {
  return usePoll<MetricKeys>(runUrl(apiBase, runId, "/metrics/keys"), intervalMs);
}

export const TRAINER_OVERVIEW_KEYS = [
  "reward/all/mean",
  "train/loss",
  "train/policy_loss",
  "train/kl_loss",
  "kl/mismatch",
  "entropy/mean",
  "ipo/is_masked",
  "dppo/is_masked",
  "optim/grad_norm",
  "optim/lr",
  "perf/tokens_per_second",
  "seq_len/all/mean",
  "tokens/train",
];

export const ORCH_OVERVIEW_KEYS = [
  "reward/all/mean",
  "reward/all/std",
  "generation/reward/mean",
  "advantage/all/std",
  "is_truncated/all/mean",
  "decode_len/all/mean",
  "generation/groups/admission_rate",
  "generation/solve_all/rate",
  "generation/solve_none/rate",
  "solve_all/all",
  "solve_none/all",
  "off_policy/mean",
  "off_policy/max",
  "policy/lag",
  "fate/all/trainable",
  "fate/all/filtered",
  "fate/all/truncated",
  "fate/all/errored",
  "fate/all/zero_loss",
  "time/generate_completions",
  "time/step",
];
