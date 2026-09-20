export type Json = Record<string, unknown>;
export type MetricRow = Record<string, number | string | null | undefined>;

type QueueSummary = {
  ready_count: number;
  claimed_count: number;
  consumed_count: number;
  incomplete_count: number;
  stale_ready_count: number;
  abandoned_claim_count: number;
};

type PolicySnapshot = {
  latest_exported_step: number | null;
  steps: number[];
  incomplete_steps: number[];
} | null;

export type RunSummary = {
  id: string;
  path: string;
  is_current?: boolean;
  status: string;
  status_reason: string;
  started_at: string | null;
  updated_at: string | null;
  target_step: number | null;
  trainer_step: number | null;
  model: string | null;
  algo: string | null;
  envs: string[];
  eval_envs: string[];
  world: { world_size: number; device: string } | null;
  latest: {
    trainer: MetricRow | null;
    orchestrator: MetricRow | null;
    eval: MetricRow | null;
  };
  queue_summary: QueueSummary | null;
  policy: PolicySnapshot;
};

export type RunOption = Pick<RunSummary, "id" | "status" | "is_current" | "model" | "trainer_step" | "target_step">;

export type MetricKeys = Record<string, Array<{ key: string }>>;

export type Series = {
  downsampled?: boolean;
  envelope?: Record<string, { min: Array<number | null>; max: Array<number | null> }>;
  steps: Array<number | null>;
  timestamps: Array<string | null>;
  series: Record<string, Array<number | null>>;
};

export type Nodes = {
  step: number | null;
  ranks: Array<Record<string, unknown>>;
  nodes: Array<Record<string, unknown>>;
  replicas: Array<Record<string, unknown>>;
};

type NumericStats = { mean: number | null; std: number | null };

export type RolloutBatch = {
  queue_step: number;
  status: string;
  optimizer_step: number | null;
  policy_step: number | null;
  rows: number | null;
  reward_mean: number | null;
  created_at: string | null;
};

export type RolloutRow = {
  row_index: number;
  reward: number | null;
  advantage: number | null;
  env: string | null;
  example_id: string | null;
  group_key: string | null;
  policy_step: number | null;
  stop_condition: string | null;
  is_truncated: boolean;
  completion_token_count: number | null;
  input_token_count: number | null;
  turn_count: number | null;
  tool_calls: number | null;
  duration_seconds: number | null;
  error: string | null;
  prompt: string | null;
  completion: string | null;
};

export type RolloutRowsResponse = {
  available: boolean;
  reason?: string;
  total: number;
  filtered: number;
  stats?: {
    reward: NumericStats;
    advantage: NumericStats;
    completion_tokens?: NumericStats;
    truncated: number;
    errors: number;
    envs?: Record<string, number>;
  };
  groups: Array<Record<string, unknown>>;
  rows: RolloutRow[];
};

export type Evals = {
  history: Array<{
    step: number | null;
    policy_step: number | null;
    envs: Record<string, Record<string, number>>;
  }>;
  sets: Array<{ step: number; env: string }>;
};

export type EvalRowsResponse = {
  available: boolean;
  total: number;
  filtered: number;
  stats?: {
    reward: NumericStats;
    errors: number;
    truncated: number;
  };
  examples: Array<Record<string, unknown>>;
  rows: Array<{
    row_index: number;
    example_id: string | null;
    reward: number | null;
    completion_token_count: number | null;
    stop_condition: string | null;
    has_error: boolean;
    is_truncated: boolean;
    answer: string | null;
    completion: string | null;
  }>;
};

export type Message = {
  role?: string;
  content?: unknown;
  tool_calls?: unknown;
  reasoning_content?: string;
  tool_call_id?: string;
};

export type RowDetail = Json & {
  row_index?: number;
  prompt?: Message[] | string;
  completion?: Message[] | string;
  reward?: number | null;
  advantage?: number | null;
  metadata?: Json | null;
  arrays?: Record<string, Json>;
};

export type LogEntry = {
  name: string;
  bytes: number;
  modified_at: string;
};

export type LogTail = {
  name: string;
  lines: string[];
};
