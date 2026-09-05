import type { RunSummary } from "../api/types";
import { num, numAny } from "./series";

export type Finding = { level: "critical" | "serious" | "warning" | "good"; title: string; detail: string; view?: string };

/** Deterministic health checks over the latest state a run has written. */
export function healthFindings(summary: RunSummary | null): Finding[] {
  if (!summary) return [];
  const findings: Finding[] = [];
  const trainer = summary.latest.trainer;
  const orch = summary.latest.orchestrator;
  const queue = summary.queue_summary;
  const batch = summary.batch;

  if (summary.status === "failed") findings.push({ level: "critical", title: "Run failed", detail: summary.status_reason || "heartbeat reports failed", view: "infra" });
  if (summary.status === "stale") findings.push({ level: "serious", title: "Heartbeat stale", detail: summary.status_reason, view: "infra" });
  if (summary.config_error) findings.push({ level: "warning", title: "Resolved config did not validate", detail: summary.config_error, view: "config" });

  if (queue) {
    if (queue.stale_ready_count > 0) findings.push({ level: "serious", title: `${queue.stale_ready_count} stale ready batch(es)`, detail: summary.status === "running" ? "Published batches exceed the off-policy window; the trainer will reject them." : "These batches exceeded the off-policy window in the final recorded queue state.", view: "pipeline" });
    if (queue.abandoned_claim_count > 0) findings.push({ level: "critical", title: `${queue.abandoned_claim_count} abandoned claim(s)`, detail: "A trainer claimed a batch and never consumed it.", view: "pipeline" });
    if (queue.event_parse_error_count > 0) findings.push({ level: "warning", title: "Queue event parse errors", detail: `${queue.event_parse_error_count} malformed lifecycle events`, view: "pipeline" });
    if (summary.status === "running" && queue.oldest_ready_age_seconds !== null && queue.oldest_ready_age_seconds > 300) {
      findings.push({ level: "warning", title: "Trainer is behind", detail: `Oldest ready batch has waited ${Math.round(queue.oldest_ready_age_seconds)}s.`, view: "pipeline" });
    }
  }

  const lag = num(orch, "policy/lag");
  if (batch && lag !== null && lag > batch.max_off_policy_steps) findings.push({ level: "serious", title: "Policy lag exceeds window", detail: `policy/lag=${lag} > max_off_policy_steps=${batch.max_off_policy_steps}`, view: "pipeline" });

  const truncated = num(orch, "is_truncated/all/mean") ?? num(orch, "fate/all/truncated_rate");
  if (truncated !== null && truncated > 0.25) findings.push({ level: "warning", title: `${(truncated * 100).toFixed(0)}% of rollouts truncated`, detail: "Completions hit max tokens; rewards on truncated samples are usually zero.", view: "rollouts" });

  const admission = num(orch, "generation/groups/admission_rate");
  if (admission !== null && admission < 0.5) findings.push({ level: "warning", title: "Low group admission", detail: `Only ${(admission * 100).toFixed(0)}% of groups carry reward signal; the rest are resampled.`, view: "rollouts" });

  const solveNone = num(orch, "generation/solve_none/rate") ?? num(orch, "solve_none/all");
  if (solveNone !== null && solveNone > 0.6) findings.push({ level: "warning", title: "Most groups unsolved", detail: `solve_none rate ${(solveNone * 100).toFixed(0)}%; task may be too hard for this policy.`, view: "rollouts" });

  const errored = num(orch, "fate/all/errored_rate");
  if (errored !== null && errored > 0.02) findings.push({ level: "serious", title: "Rollout errors", detail: `${(errored * 100).toFixed(1)}% of rollouts errored in the verifier.`, view: "inspector" });

  const entropy = num(trainer, "entropy/mean");
  if (entropy !== null && entropy < 0.05) findings.push({ level: "warning", title: "Entropy collapse", detail: `entropy/mean=${entropy.toFixed(3)}; the policy may have stopped exploring.`, view: "training" });

  const masked = numAny(trainer, ["ipo/is_masked", "dppo/is_masked"]);
  if (masked !== null && masked > 0.2) findings.push({ level: "warning", title: "High loss masking", detail: `${(masked * 100).toFixed(0)}% of tokens masked; trainer and inference policies disagree.`, view: "training" });

  const kl = num(trainer, "kl/mismatch");
  if (kl !== null && kl > 0.05) findings.push({ level: "serious", title: "Trainer/inference logprob mismatch", detail: `kl/mismatch=${kl.toFixed(4)}; check precision parity and sampling replay.`, view: "training" });

  const gradNorm = num(trainer, "optim/grad_norm");
  if (gradNorm !== null && !Number.isFinite(gradNorm)) findings.push({ level: "critical", title: "Non-finite gradient norm", detail: "Optimizer step produced NaN or Inf.", view: "training" });

  const failedEval = Object.entries(summary.latest.eval ?? {}).filter(([k, v]) => k.endsWith("/failed_rollouts") && typeof v === "number" && v > 0);
  for (const [key, value] of failedEval) findings.push({ level: "warning", title: "Eval rollouts failed", detail: `${key} = ${((value as number) * 100).toFixed(1)}%; failures count as incorrect in avg@k.`, view: "evals" });

  const diskRatio = num(trainer, "disk_free_ratio");
  if (diskRatio !== null && diskRatio < 0.1) findings.push({ level: "serious", title: "Disk nearly full", detail: `${(diskRatio * 100).toFixed(1)}% free on the output volume.`, view: "infra" });

  const kv = Object.entries(orch ?? {}).filter(([k, v]) => /^inference\/.*\/kv_cache_usage$/.test(k) && typeof v === "number" && v > 0.95);
  for (const [key] of kv) findings.push({ level: "warning", title: "KV cache saturated", detail: `${key.split("/")[1]} at >95% KV usage; expect preemptions.`, view: "infra" });

  if (findings.length === 0 && summary.status === "running") findings.push({ level: "good", title: "No anomalies detected", detail: "All health checks pass on the latest metrics." });
  return findings;
}
