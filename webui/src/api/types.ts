export type Json = Record<string, unknown>;

export type MetricRow = Record<string, number | string | null | undefined>;

type PolicySnapshot = { latest_exported_step: number | null; steps: number[]; incomplete_steps: number[] } | null;

export type RunSummary = {
  id: string;
  path: string;
  is_current?: boolean;
  status: string;
  status_reason: string;
  started_at: string | null;
  updated_at: string | null;
  heartbeat: { timestamp?: string; pid?: number; status?: string } | null;
  world: { world_size: number; device: string } | null;
  resumed_from: string | null;
  config_error: string | null;
  has_config: boolean;
  target_step: number | null;
  trainer_step: number | null;
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
    max_async_level: number;
    max_off_policy_steps: number;
  } | null;
  latest: { trainer: MetricRow | null; orchestrator: MetricRow | null; eval: MetricRow | null };
  queue_summary: QueueSummary | null;
  policy: PolicySnapshot;
  eval_envs: string[];
  logs: Array<{ name: string; bytes: number }>;
};

type QueueSummary = {
  ready_count: number;
  claimed_count: number;
  consumed_count: number;
  incomplete_count: number;
  stale_ready_count: number;
  abandoned_claim_count: number;
  oldest_ready_age_seconds: number | null;
  latest_consumed_queue_step: number | null;
  event_parse_error_count: number;
};

export type MetricKeys = Record<string, Array<{ key: string }>>;

export type ExternalStatus = { source: string; status: "unavailable" | "loading" | "ready" | "error"; error: string | null };

export type Series = {
  steps: Array<number | null>;
  timestamps: Array<string | null>;
  series: Record<string, Array<number | null>>;
  /** Present when the server bucket-averaged the rows (`points` was exceeded). */
  envelope?: Record<string, { min: Array<number | null>; max: Array<number | null> }>;
  downsampled?: boolean;
  bucket?: number;
};

export type RankStats = {
  rank: number;
  node: string;
  device: string;
  tokens: number;
  seconds: number;
  tokens_per_second: number;
  memory_allocated_gib: number;
  peak_memory_gib: number;
};
export type Nodes = { step: number | null; timestamp: string | null; ranks: RankStats[] };

export type Evals = {
  envs: Array<{ name: string; metrics: string[] }>;
  history: Array<{ step: number | null; policy_step: number | null; envs: Record<string, Record<string, number>> }>;
  sets: Array<{ step: number; env: string }>;
};

type NumericStats = { mean: number | null; std: number | null };

type Histogram = { bins: number[]; counts: number[] };

/** Batch manifest fields the UI reads; the server writes more. */
type Manifest = Json & { optimizer_step?: number; policy_step?: number; rows?: number; reward_mean?: number; tokens?: number; payload_bytes?: number; created_at?: string; producer_id?: string };

export type RolloutRow = {
  row_index: number;
  reward: number | null;
  advantage: number | null;
  env: string | null;
  group_key: string | null;
  policy_step: number | null;
  stop_condition: string | null;
  is_truncated: boolean;
  completion_token_count: number | null;
  logprob_mean: number | null;
  completion: string | null;
};

export type RolloutGroup = {
  group_key: string;
  env: string | null;
  example_id: string | null;
  size: number;
  reward_mean: number | null;
  reward_std: number | null;
  reward_min: number | null;
  reward_max: number | null;
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
  manifest?: Manifest | null;
  total: number;
  filtered: number;
  stats?: {
    reward: NumericStats;
    advantage: NumericStats;
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
  optimizer_step: number | null;
  policy_step: number | null;
  rows: number | null;
  reward_mean: number | null;
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
  answer: string | null;
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
};

export type EvalRowsResponse = {
  available: boolean;
  total: number;
  scanned?: number;
  scan_limited?: boolean;
  filtered: number;
  stats?: { reward: NumericStats; reward_histogram: Histogram; errors: number; truncated: number };
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

export type QueueReport = {
  summary: QueueSummary | null;
  policy: PolicySnapshot;
  rates: { rollouts_published_per_second: number; rollouts_consumed_per_second: number } | null;
  items: Array<{
    queue_step: number;
    status: string;
    manifest: Manifest | null;
    claim: (Json & { claimed_at?: string; consumer_id?: string }) | null;
    consumed: (Json & { consumed_at?: string }) | null;
    age_seconds: number | null;
    parse_errors: string[];
  }>;
};

export type TimelineStep = {
  queue_step: number;
  optimizer_step: number | null;
  policy_step: number | null;
  published_at?: string;
  claimed_at?: string;
  consumed_at?: string;
  payload_bytes?: number | null;
  publish_to_claim_seconds: number | null;
  claim_to_consume_seconds: number | null;
};

export type TimelinePolicy = {
  policy_step: number;
  exported_at?: string;
  received_at?: string;
  loaded_at?: string;
  payload_bytes?: number | null;
  load_seconds?: number | null;
  export_to_load_seconds: number | null;
};

export type Timeline = { queue_steps: TimelineStep[]; policies: TimelinePolicy[]; parse_errors: number; dropped_events: number };

export type RunEvent = { timestamp: string; event: string; step: number | null; payload: Json };
