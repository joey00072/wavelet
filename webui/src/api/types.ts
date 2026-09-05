export type Json = Record<string, unknown>;

export type MetricRow = Record<string, number | string | null | undefined>;

export type RunSummary = {
  id: string;
  path: string;
  is_current?: boolean;
  status: string;
  status_reason: string;
  started_at: string | null;
  updated_at: string | null;
  heartbeat: { timestamp?: string; pid?: number; status?: string; step?: number } | null;
  world: { rank: number; world_size: number; device: string } | null;
  resumed_from: string | null;
  config_error: string | null;
  has_config: boolean;
  target_step: number | null;
  trainer_step: number | null;
  orchestrator_step: number | null;
  eval_step: number | null;
  model: string | null;
  algo: string | null;
  loss: string | null;
  launcher_mode: string | null;
  envs: string[];
  lora: boolean | null;
  batch: {
    examples_per_step: number | null;
    rollouts_per_example: number | null;
    token_batch_size: number | null;
    max_async_level: number;
    max_off_policy_steps: number;
  } | null;
  latest: { trainer: MetricRow | null; orchestrator: MetricRow | null; eval: MetricRow | null };
  queue_summary: QueueSummary | null;
  queue_rates: { rollouts_published_per_second: number; rollouts_consumed_per_second: number } | null;
  policy: { latest_exported_step: number | null; steps: number[]; incomplete_steps: number[] } | null;
  queue_errors: Record<string, unknown> | null;
  eval_envs: string[];
  logs: LogEntry[];
};

export type QueueSummary = {
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
};

export type LogEntry = { name: string; bytes: number; modified_at: string };

export type MetricKeys = Record<string, Array<{ key: string; count: number }>>;

export type ExternalStatus = { source: string; status: "unavailable" | "loading" | "ready" | "error"; reason: string; error: string | null; rows: number; keys: number; fetched_at: string | null; refreshing: boolean };

export type Series = {
  source: string;
  steps: Array<number | null>;
  timestamps: Array<string | null>;
  series: Record<string, Array<number | null>>;
  /** Present when the server bucket-averaged the rows (`points` was exceeded). */
  envelope?: Record<string, { min: Array<number | null>; max: Array<number | null> }>;
  downsampled?: boolean;
  bucket?: number;
  rows: number;
  total_rows?: number;
  parse_errors?: number;
};

export type NodeStats = { name: string } & Record<string, number | string>;
export type RankStats = {
  rank: number;
  local_rank: number;
  node: string;
  device: string;
  tokens: number;
  seconds: number;
  tokens_per_second: number;
  memory_allocated_gib: number;
  peak_memory_gib: number;
};
export type Nodes = {
  step: number | null;
  timestamp: string | null;
  nodes: NodeStats[];
  ranks: RankStats[];
  replicas: Array<{ name: string } & Record<string, number | string>>;
  world: { rank: number; world_size: number; local_world_size?: number; device: string } | null;
};

export type EvalHistoryRow = {
  step: number | null;
  policy_step: number | null;
  timestamp: string | null;
  envs: Record<string, Record<string, number>>;
};

export type Evals = {
  envs: Array<{ name: string; metrics: string[] }>;
  history: EvalHistoryRow[];
  sets: Array<{ step: number; env: string; path: string; bytes: number }>;
};

export type NumericStats = {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
};

export type Histogram = { bins: number[]; counts: number[]; min: number | null; max: number | null };

export type RolloutRow = {
  row_index: number;
  reward: number | null;
  advantage: number | null;
  env: string | null;
  example_id: string | null;
  group_key: string | null;
  rollout_key: string | null;
  policy_step: number | null;
  stop_condition: string | null;
  is_truncated: boolean;
  completion_token_count: number | null;
  input_token_count: number | null;
  turn_count: number | null;
  tool_calls: number | null;
  error: string | null;
  sequence_tokens: number | null;
  trainable_tokens: number | null;
  logprob_mean: number | null;
  logprob_min: number | null;
  has_inference_logprobs: boolean;
  has_teacher_logprobs: boolean;
  prompt: string | null;
  completion: string | null;
};

export type RolloutGroup = {
  group_key: string;
  env: string | null;
  example_id: string | null;
  policy_step: number | null;
  size: number;
  reward_mean: number | null;
  reward_std: number | null;
  reward_min: number | null;
  reward_max: number | null;
  advantage_abs_mean: number | null;
  solve_all: boolean;
  solve_none: boolean;
  zero_signal: boolean;
  truncated: number;
  completion_tokens_mean: number | null;
  row_indexes: number[];
  prompt: string | null;
};

export type RolloutRowsResponse = {
  available: boolean;
  reason?: string;
  queue_step: number | null;
  stable?: boolean;
  path?: string;
  manifest?: Json | null;
  total: number;
  scanned?: number;
  scan_limited?: boolean;
  filtered: number;
  stats?: {
    reward: NumericStats;
    advantage: NumericStats;
    completion_tokens: NumericStats;
    reward_histogram: Histogram;
    advantage_histogram: Histogram;
    completion_token_histogram: Histogram;
    truncated: number;
    errors: number;
    stop_conditions: Record<string, number>;
    envs: Record<string, number>;
    policy_steps: Record<string, number>;
  };
  groups: RolloutGroup[];
  rows: RolloutRow[];
};

export type RolloutBatch = {
  queue_step: number;
  status: string;
  stable: boolean;
  optimizer_step: number | null;
  chunk_index: number | null;
  policy_step: number | null;
  rows: number | null;
  tokens: number | null;
  reward_mean: number | null;
  producer_id: string | null;
  created_at: string | null;
  payload_bytes: number | null;
  age_seconds: number | null;
  claimed_at: string | null;
  consumed_at: string | null;
  parse_errors: string[];
};

export type EvalRow = {
  row_index: number;
  example_id: string | null;
  reward: number | null;
  has_error: boolean;
  error: string | null;
  is_truncated: boolean;
  stop_condition: string | null;
  completion_token_count: number | null;
  task: string | null;
  answer: string | null;
  metrics: Record<string, number>;
  prompt: string | null;
  completion: string | null;
};

export type EvalExample = {
  example_id: string;
  attempts: number;
  scored: number;
  errors: number;
  reward_mean: number | null;
  solved_any: boolean;
  solved_all: boolean;
  truncated: number;
  row_indexes: number[];
  prompt: string | null;
  answer: string | null;
};

export type EvalRowsResponse = {
  available: boolean;
  reason?: string;
  step: number;
  env: string;
  total: number;
  scanned?: number;
  scan_limited?: boolean;
  filtered: number;
  stats?: {
    reward: NumericStats;
    reward_histogram: Histogram;
    errors: number;
    truncated: number;
  };
  examples: EvalExample[];
  rows: EvalRow[];
};

export type Message = { role?: string; content?: string; tool_calls?: unknown };

export type RowDetail = Json & {
  row_index?: number;
  prompt?: Message[] | string;
  completion?: Message[] | string;
  target_completion?: Message[] | null;
  reward?: number | null;
  advantage?: number | null;
  metadata?: Json | null;
  arrays?: Record<string, { length: number; true_count?: number; min?: number; max?: number; mean?: number; sum?: number }>;
};

export type QueueItem = {
  queue_step: number;
  status: string;
  stable: boolean;
  manifest: Json | null;
  claim: Json | null;
  consumed: Json | null;
  age_seconds: number | null;
  parse_errors: string[];
};

export type QueueReport = {
  summary: QueueSummary | null;
  policy: { latest_exported_step: number | null; steps: number[]; incomplete_steps: number[] } | null;
  rates: { rollouts_published_per_second: number; rollouts_consumed_per_second: number } | null;
  errors: Record<string, unknown>;
  items: QueueItem[];
};

export type TimelineStep = {
  queue_step: number;
  optimizer_step: number | null;
  policy_step: number | null;
  published_at?: string;
  received_at?: string;
  claimed_at?: string;
  consumed_at?: string;
  payload_bytes?: number | null;
  trainer_wait_seconds?: number | null;
  publish_to_claim_seconds: number | null;
  claim_to_consume_seconds: number | null;
};

export type TimelinePolicy = {
  policy_step: number;
  exported_at?: string;
  received_at?: string;
  loaded_at?: string;
  payload_bytes?: number | null;
  inference_wait_seconds?: number | null;
  load_seconds?: number | null;
  export_to_load_seconds: number | null;
};

export type Timeline = {
  queue_steps: TimelineStep[];
  policies: TimelinePolicy[];
  event_count: number;
  parse_errors: number;
  dropped_events: number;
};

export type RunEvent = { timestamp: string; event: string; step: number | null; payload: Json };
