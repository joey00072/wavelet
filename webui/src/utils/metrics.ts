import type { MetricRow, PipelineInventory, RolloutEvent, RunState } from "../types";

export function latestMetric(metrics: MetricRow[]): MetricRow | null {
  return metrics.length > 0 ? metrics[metrics.length - 1] : null;
}

export function metricStep(row: MetricRow | null): number | null {
  if (!row) {
    return null;
  }
  return row.step ?? row["progress/step"] ?? null;
}

export function elapsedMinutes(first: string | undefined, last: string | undefined): number {
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

export function rateForEvents(events: RolloutEvent[], type: RolloutEvent["type"]): number {
  const filtered = events.filter((event) => event.type === type);
  if (filtered.length < 2) {
    return 0;
  }
  const minutes = elapsedMinutes(filtered[0].timestamp, filtered[filtered.length - 1].timestamp);
  return minutes > 0 ? filtered.length / minutes : 0;
}

export function rateForMetrics(metrics: MetricRow[]): number {
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

export function eventCounts(events: RolloutEvent[]) {
  return events.reduce(
    (counts, event) => {
      counts[event.type] += 1;
      return counts;
    },
    { submitted: 0, completed: 0, published: 0 },
  );
}

export function inferChunksPerStep(events: RolloutEvent[]): number {
  if (events.length > 0 && events.every((event) => event.chunk_index === null || event.chunk_index === undefined)) {
    return 1;
  }
  const chunkIndexes = events
    .map((event) => event.chunk_index)
    .filter((index): index is number => typeof index === "number" && index >= 0);
  if (chunkIndexes.length === 0) {
    return 1;
  }
  return Math.max(...chunkIndexes) + 1;
}

function isPhase(state: RunState | null, pattern: string): boolean {
  return (state?.phase ?? "").toLowerCase().includes(pattern);
}

function ageMs(timestamp: string | undefined): number | null {
  if (!timestamp) {
    return null;
  }
  const time = new Date(timestamp).getTime();
  if (Number.isNaN(time)) {
    return null;
  }
  return Math.max(0, Date.now() - time);
}

export function pipelineInventory(
  state: RunState | null,
  events: RolloutEvent[],
  metrics: MetricRow[],
): PipelineInventory {
  const latest = latestMetric(metrics);
  const trainerStep = metricStep(latest);
  const queueSummary = state?.queue_summary ?? null;
  const chunksPerStep = inferChunksPerStep(events);
  const rolloutsPerStep = latest?.["rollout/count"] ?? 128;
  const rolloutsPerChunk = Math.max(1, Math.round(rolloutsPerStep / chunksPerStep));
  const consumedEstimate =
    queueSummary?.consumed_count ?? Math.max(0, (trainerStep === null ? 0 : trainerStep + 1) * chunksPerStep);
  const publishedWatermark =
    state?.rollouts.next_queue_step_to_publish ??
    (queueSummary?.latest_queue_step === null || queueSummary?.latest_queue_step === undefined
      ? eventCounts(events).published
      : queueSummary.latest_queue_step + 1);
  const readyForTrainer =
    queueSummary === null
      ? Math.max(0, publishedWatermark - consumedEstimate)
      : queueSummary.ready_count + queueSummary.stale_ready_count;
  const isRunning = state?.status === "running";
  const generating = Math.max(state?.rollouts.pending_count ?? 0, queueSummary?.incomplete_count ?? 0);
  const recentTrainerMetric = isRunning && (ageMs(latest?.timestamp) ?? Number.POSITIVE_INFINITY) < 30_000;
  const trainerUsingChunks =
    queueSummary?.claimed_count && queueSummary.claimed_count > 0
      ? queueSummary.claimed_count
      : recentTrainerMetric
        ? chunksPerStep
        : 0;
  const isGenerating = isRunning && (isPhase(state, "inference") || generating > 0);
  const isTraining =
    isRunning && (isPhase(state, "train") || isPhase(state, "trainer") || trainerUsingChunks > 0);
  const activeStage =
    state?.status === "failed"
      ? "failed"
      : state?.status === "completed"
        ? "completed"
        : isPhase(state, "train") || isPhase(state, "trainer")
          ? "training"
          : isGenerating
            ? "generating"
            : isTraining
              ? "training"
            : readyForTrainer > 0
              ? "ready"
              : "idle";
  const nextTrainStep =
    queueSummary?.next_expected_trainer_queue_step ?? (trainerStep === null ? 0 : trainerStep + 1);
  const latestPublishedStep = publishedWatermark > 0 ? publishedWatermark - 1 : null;
  const submitted = state?.rollouts.next_queue_step_to_submit ?? eventCounts(events).submitted;
  const currentGenerateStep = generating > 0 ? submitted - 1 : submitted;
  const claimedCount = queueSummary?.claimed_count ?? 0;

  return {
    submitted,
    publishedWatermark,
    generating,
    completedWaitingPublish: state?.rollouts.completed_count ?? 0,
    consumedEstimate,
    readyForTrainer,
    trainerUsingChunks,
    trainerUsingRollouts: trainerUsingChunks * rolloutsPerChunk,
    chunksPerStep,
    rolloutsPerChunk,
    trainerStep,
    activeStage,
    activeStageLabel: {
      generating: "Generating",
      ready: "Ready",
      training: "Training",
      idle: "Waiting",
      completed: "Completed",
      failed: "Failed",
    }[activeStage],
    activeStageDetail: {
      generating: `building batch ${currentGenerateStep >= 0 ? currentGenerateStep : "-"}`,
      ready: `${readyForTrainer} batch${readyForTrainer === 1 ? "" : "es"} ready for trainer`,
      training:
        claimedCount > 0
          ? `trainer claimed ${claimedCount} batch${claimedCount === 1 ? "" : "es"}`
          : `last trainer step ${trainerStep ?? "-"}`,
      idle: "no active queue movement",
      completed: "run finished",
      failed: state?.errors.length ? state.errors[state.errors.length - 1].message : "run failed",
    }[activeStage],
    nextGenerateStep: currentGenerateStep >= 0 ? currentGenerateStep : null,
    nextTrainStep,
    latestPublishedStep,
  };
}

export function bucketEvents(events: RolloutEvent[], metrics: MetricRow[]) {
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

export function laneCounts(events: RolloutEvent[]) {
  const lanes = new Map<number, { submitted: number; completed: number; published: number }>();
  for (const event of events) {
    const lane = event.chunk_index ?? 0;
    const counts = lanes.get(lane) ?? { submitted: 0, completed: 0, published: 0 };
    counts[event.type] += 1;
    lanes.set(lane, counts);
  }
  return [...lanes.entries()].sort((a, b) => a[0] - b[0]);
}
