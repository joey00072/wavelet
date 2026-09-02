export type RolloutEvent = {
  type: "submitted" | "completed" | "published";
  queue_step: number;
  optimizer_step?: number | null;
  chunk_index?: number | null;
  timestamp: string;
  path?: string;
};

export type RunState = {
  status: string;
  phase: string;
  started_at: string;
  updated_at: string;
  target_step: number;
  output_dir: string;
  launcher_mode: string;
  rollouts: {
    next_queue_step_to_submit: number;
    next_queue_step_to_publish: number;
    pending_count: number;
    completed_count: number;
    submitted_tail: RolloutEvent[];
    completed_tail: RolloutEvent[];
    published_tail: RolloutEvent[];
  };
  policy: {
    loaded_step: number | null;
    pending_load: boolean;
    requested_step: number | null;
    available_tail: number[];
  };
  queue_summary?: {
    ready_count: number;
    claimed_count: number;
    consumed_count: number;
    incomplete_count: number;
    unknown_count: number;
    stale_ready_count: number;
    abandoned_claim_count: number;
    oldest_ready_age_seconds: number | null;
    latest_queue_step: number | null;
    latest_consumed_queue_step: number | null;
    next_expected_trainer_queue_step: number;
    event_parse_error_count: number;
  } | null;
  queue_rates?: {
    rollouts_published_per_second: number;
    rollouts_consumed_per_second: number;
  } | null;
  errors: Array<{ type: string; message: string; timestamp: string }>;
};

export type MetricRow = {
  timestamp?: string;
  step?: number;
  "progress/step"?: number;
  reward_mean?: number;
  "reward/all/mean"?: number;
  loss?: number;
  lr?: number;
  "optim/lr"?: number;
  "rollout/count"?: number;
  "tokens/train"?: number;
  "perf/train_tokens_per_second"?: number;
  "perf/step_tokens_per_second"?: number;
};

export type EvalMetricRow = {
  timestamp?: string;
  step?: number;
  "progress/policy_step"?: number;
  [key: string]: string | number | undefined;
};

export type RolloutSample = {
  row_index: number;
  reward?: number;
  advantage?: number;
  source?: string;
  env_name?: string;
  task?: string;
  example_id?: string;
  group_key?: string;
  rollout_key?: string;
  stop_condition?: string;
  is_truncated?: boolean;
  completion_token_count?: number;
  turn_count?: number;
  prompt?: string;
  completion?: string;
  target_completion?: string;
};

export type NumericStats = {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
};

export type RolloutInspection = {
  available: boolean;
  reason: string | null;
  queue_step: number | null;
  path: string | null;
  manifest: {
    optimizer_step?: number | null;
    policy_step?: number | null;
    producer_id?: string | null;
    rows?: number | null;
    created_at?: string;
  } | null;
  scanned_rows: number;
  truncated: boolean;
  stats: {
    reward: NumericStats;
    advantage: NumericStats;
  };
  samples: {
    random: RolloutSample[];
    min_reward: RolloutSample | null;
    max_reward: RolloutSample | null;
    near_mean_reward: RolloutSample | null;
  };
};

export type RolloutSnapshot = {
  id: string;
  batch_key: string;
  captured_at: string;
  inspection: RolloutInspection;
  source: "buffer" | "reader" | "saved";
};

export type Theme = "dark" | "light";

export type PipelineInventory = {
  submitted: number;
  publishedWatermark: number;
  generating: number;
  completedWaitingPublish: number;
  consumedEstimate: number;
  readyForTrainer: number;
  trainerUsingChunks: number;
  trainerUsingRollouts: number;
  chunksPerStep: number;
  rolloutsPerChunk: number;
  trainerStep: number | null;
  activeStage: "generating" | "ready" | "training" | "idle" | "completed" | "failed";
  activeStageLabel: string;
  activeStageDetail: string;
  nextGenerateStep: number | null;
  nextTrainStep: number | null;
  latestPublishedStep: number | null;
};
