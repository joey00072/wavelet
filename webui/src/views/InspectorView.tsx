import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronRight, RefreshCw } from "lucide-react";

import { qs, runUrl, usePoll } from "../api/client";
import type { RolloutBatch, RolloutGroup, RolloutRow, RolloutRowsResponse, RowDetail } from "../api/types";
import { HistogramChart } from "../charts/BarChart";
import { ChartCard } from "../charts/ChartCard";
import { Tag } from "../components/Badge";
import { Field, Segmented, Toolbar } from "../components/Controls";
import { DataTable, Pager, type Column } from "../components/DataTable";
import { Drawer } from "../components/Drawer";
import { Disclosure } from "../components/Disclosure";
import { Empty, ErrorNote, KeyValue } from "../components/KeyValue";
import { Transcript } from "../components/Transcript";
import { fmt, fmtAge, fmtBytes, fmtInt, fmtPct, shortId } from "../lib/format";
import { updateParams } from "../lib/router";

const PAGE = 50;

type Filters = {
  env: string;
  group_key: string;
  min_reward: string;
  max_reward: string;
  truncated: string;
  stop_condition: string;
  advantage: string;
  has_error: string;
  search: string;
};

const EMPTY_FILTERS: Filters = { env: "", group_key: "", min_reward: "", max_reward: "", truncated: "", stop_condition: "", advantage: "", has_error: "", search: "" };

/** Group keys may be JSON blobs from verifier environments; show the identifying part. */
function groupLabel(key: string | null | undefined): string {
  if (!key) return "–";
  if (key.startsWith("{")) {
    try {
      const parsed = JSON.parse(key) as Record<string, unknown>;
      const example = parsed.example_id ?? parsed.id;
      const group = typeof parsed.rollout_group_id === "string" ? parsed.rollout_group_id.split(":").pop() : null;
      return [example !== undefined ? `ex ${example}` : null, group ? `g${group}` : null].filter(Boolean).join(" · ") || shortId(key, 16);
    } catch {
      return shortId(key, 16);
    }
  }
  return shortId(key, 18);
}

export function InspectorView({ apiBase, runId, params, live }: { apiBase: string; runId: string; params: URLSearchParams; live: boolean }) {
  const batches = usePoll<RolloutBatch[]>(`${runUrl(apiBase, runId, "/rollouts")}?limit=500`, live ? 5000 : 0);
  const follow = params.get("step") === null;
  const stepParam = params.get("step");
  const step = stepParam !== null ? Number(stepParam) : null;
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: "reward", desc: true });
  const [offset, setOffset] = useState(0);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [mode, setMode] = useState<"rows" | "groups">("groups");
  const [groupOutcome, setGroupOutcome] = useState("all");
  const [groupSort, setGroupSort] = useState("signal");
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const [detail, setDetail] = useState<{ step: number; index: number } | null>(null);

  const rowsUrl = `${runUrl(apiBase, runId, "/rollouts/rows")}${qs({ step, sort: sort.key, order: sort.desc ? "desc" : "asc", offset, limit: PAGE, ...filters })}`;
  const rows = usePoll<RolloutRowsResponse>(
    rowsUrl,
    follow && live ? 5000 : 0,
    [],
    { resourceKey: `${runId}:${step ?? "latest"}` },
  );
  const data = rows.data;
  const selectStep = (next: number | null) => {
    setOffset(0);
    updateParams((p) => (next === null ? p.delete("step") : p.set("step", String(next))));
  };
  const onSort = (key: string) => {
    setOffset(0);
    setSort((prev) => (prev.key === key ? { key, desc: !prev.desc } : { key, desc: true }));
  };
  const setFilter = (key: keyof Filters, value: string) => {
    setOffset(0);
    setFilters((prev) => ({ ...prev, [key]: value }));
  };
  const activeFilters = Object.values(filters).filter(Boolean).length;
  const activeGroupFilters = [filters.search, filters.env, filters.group_key, groupOutcome === "all" ? "" : groupOutcome].filter(Boolean).length;
  const currentStep = data?.queue_step ?? step;
  const selectedBatch = (batches.data ?? []).find((batch) => batch.queue_step === currentStep) ?? null;
  const groupSizes = (data?.groups ?? []).map((group) => group.size);
  const minGroupSize = groupSizes.length > 0 ? Math.min(...groupSizes) : null;
  const maxGroupSize = groupSizes.length > 0 ? Math.max(...groupSizes) : null;
  const uniformGroupSize = minGroupSize !== null && minGroupSize === maxGroupSize ? minGroupSize : null;

  const groups = useMemo(() => {
    const query = filters.search.trim().toLowerCase();
    const list = (data?.groups ?? []).filter((group) => {
      if (filters.env && group.env !== filters.env) return false;
      if (filters.group_key && group.group_key !== filters.group_key) return false;
      if (query) {
        const text = [group.group_key, group.example_id, group.env, group.prompt]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        if (!text.includes(query)) return false;
      }
      if (groupOutcome === "mixed" && (group.solve_all || group.solve_none || group.zero_signal)) return false;
      if (groupOutcome === "solve_all" && !group.solve_all) return false;
      if (groupOutcome === "solve_none" && !group.solve_none) return false;
      if (groupOutcome === "zero_signal" && !group.zero_signal) return false;
      if (groupOutcome === "truncated" && group.truncated === 0) return false;
      return true;
    });
    return [...list].sort((a, b) => {
      if (groupSort === "reward") return (b.reward_mean ?? -Infinity) - (a.reward_mean ?? -Infinity);
      if (groupSort === "size") return b.size - a.size;
      if (groupSort === "example") return groupLabel(a.group_key).localeCompare(groupLabel(b.group_key));
      return (b.reward_std ?? -1) - (a.reward_std ?? -1);
    });
  }, [data?.groups, filters.env, filters.group_key, filters.search, groupOutcome, groupSort]);

  useEffect(() => {
    const first = groups[0]?.group_key;
    setExpandedGroups(first ? new Set([first]) : new Set());
  }, [data?.queue_step]);

  const multiEnv = Object.keys(data?.stats?.envs ?? {}).length > 1;
  const columns: Column<RolloutRow>[] = [
    { key: "row_index", label: "#", sortable: true, render: (r) => <span className="tabular text-muted">{r.row_index}</span>, width: "3rem" },
    { key: "reward", label: "Reward", sortable: true, align: "right", render: (r) => <RewardCell value={r.reward} /> },
    { key: "advantage", label: "Adv", sortable: true, align: "right", render: (r) => <span className={r.advantage === null ? "text-muted" : r.advantage > 0 ? "text-[var(--success-text)]" : r.advantage < 0 ? "text-critical" : "text-muted"}>{fmt(r.advantage, 3)}</span> },
    { key: "completion_token_count", label: "Tokens", sortable: true, align: "right", render: (r) => fmtInt(r.completion_token_count) },
    { key: "logprob_mean", label: "Logp mean", sortable: true, align: "right", title: "mean inference logprob over trainable tokens", render: (r) => fmt(r.logprob_mean, 3) },
    { key: "is_truncated", label: "Stop", sortable: true, render: (r) => (r.is_truncated ? <Tag tone="warning">truncated</Tag> : <span className="text-muted">{r.stop_condition ?? "stop"}</span>) },
    { key: "group_key", label: "Group", sortable: true, render: (r) => <button type="button" className="hover:underline" onClick={(e) => { e.stopPropagation(); setFilter("group_key", r.group_key ?? ""); setMode("rows"); }}>{groupLabel(r.group_key)}</button> },
    ...(multiEnv ? [{ key: "env", label: "Env", sortable: true, render: (r: RolloutRow) => r.env ?? "–" } as Column<RolloutRow>] : []),
    { key: "policy_step", label: "Policy", sortable: true, align: "right", render: (r) => r.policy_step ?? "–" },
    { key: "completion", label: "Completion", render: (r) => <span className="block max-w-[26rem] truncate text-ink2" title={r.completion ?? undefined}>{(r.completion ?? "").replace(/\s+/g, " ").slice(0, 160)}</span> },
  ];

  return (
    <div className="min-w-0 space-y-8">
      <header className="border-b border-edge pb-4">
        <h1 className="sr-only">Rollout inspector</h1>
        <ErrorNote error={batches.error} />
        <div>
          <div className="mb-1 flex items-center justify-between gap-3">
            <span className="eyebrow">Rollout batch</span>
            <button
              type="button"
              className={`inline-flex items-center gap-1.5 text-[11px] transition-colors ${follow ? "text-ink" : "text-muted hover:text-ink"}`}
              onClick={() => selectStep(null)}
              aria-pressed={follow}
              title={follow ? "New rollout batches will appear automatically" : `Currently pinned to step ${currentStep ?? "–"}; click to resume following`}
            >
              <span className={`flex h-3.5 w-3.5 items-center justify-center border ${follow ? "border-accent bg-accent text-white" : "border-muted"}`} aria-hidden>
                {follow && <Check className="h-2.5 w-2.5" strokeWidth={3} />}
              </span>
              Keep viewing latest
            </button>
          </div>
          <BatchStepRail batches={batches.data ?? []} currentStep={currentStep} follow={follow} onSelect={selectStep} />
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink2">
            <span className={data?.stable ? "text-[var(--success-text)]" : "text-warn"}>{data?.stable ? "stable" : selectedBatch?.status ?? "loading"}</span>
            <span className="tabular"><span className="text-muted">optimizer </span>{String((data?.manifest as { optimizer_step?: number } | null)?.optimizer_step ?? selectedBatch?.optimizer_step ?? "–")}</span>
            <span className="tabular"><span className="text-muted">policy </span>{String((data?.manifest as { policy_step?: number } | null)?.policy_step ?? selectedBatch?.policy_step ?? "–")}</span>
            <span className="tabular"><span className="text-muted">rollouts </span>{fmtInt(data?.total ?? selectedBatch?.rows)}</span>
            <span className="tabular"><span className="text-muted">reward </span>{fmt(data?.stats?.reward.mean ?? selectedBatch?.reward_mean, 4)}{data?.stats ? ` ± ${fmt(data.stats.reward.std, 4)}` : ""}</span>
            {rows.refetching && <span className="inline-flex items-center gap-1 text-muted"><RefreshCw className="h-3 w-3 animate-spin" /> updating</span>}
          </div>
          {data?.available && data.groups.length > 0 && (
            <div
              className="mt-3 border-y border-edge py-2 text-sm text-ink2"
              role="status"
              aria-label={uniformGroupSize !== null
                ? `Batch composition: ${data.groups.length} prompt groups times ${uniformGroupSize} samples per group equals ${data.total} total rollouts`
                : `Batch composition: ${data.groups.length} prompt groups, ${minGroupSize} to ${maxGroupSize} samples per group, ${data.total} total rollouts`}
            >
              <span className="mr-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-muted">Batch composition</span>
              <strong className="tabular font-semibold text-ink">{fmtInt(data.groups.length)}</strong> prompt groups
              {uniformGroupSize !== null ? (
                <>
                  <span className="mx-2 text-muted" aria-hidden>×</span>
                  <strong className="tabular font-semibold text-ink">{fmtInt(uniformGroupSize)}</strong> samples per group
                  <span className="mx-2 text-muted" aria-hidden>=</span>
                </>
              ) : (
                <>
                  <span className="mx-2 text-muted" aria-hidden>·</span>
                  <strong className="tabular font-semibold text-ink">{fmtInt(minGroupSize)}–{fmtInt(maxGroupSize)}</strong> samples per group
                  <span className="mx-2 text-muted" aria-hidden>·</span>
                </>
              )}
              <strong className="tabular font-semibold text-ink">{fmtInt(data.total)}</strong> total rollouts
            </div>
          )}
        </div>
        {data?.available && (
          <details className="mt-3 text-[11px] text-muted">
            <summary className="cursor-pointer select-none hover:text-ink">Batch details</summary>
            <div className="mt-3">
              <KeyValue
                columns={6}
                items={[
                  ["Advantage std", fmt(data.stats?.advantage.std, 3)],
                  ["Truncated", `${data.stats?.truncated ?? 0} (${fmtPct((data.stats?.truncated ?? 0) / Math.max(data.total, 1), 0)})`],
                  ["Errors", String(data.stats?.errors ?? 0)],
                  ["Payload", fmtBytes((data.manifest as { payload_bytes?: number } | null)?.payload_bytes ?? null)],
                  ["Created", fmtAge((data.manifest as { created_at?: string } | null)?.created_at ?? null)],
                  ["Producer", shortId((data.manifest as { producer_id?: string } | null)?.producer_id ?? null, 30)],
                  ["Environments", Object.entries(data.stats?.envs ?? {}).map(([key, value]) => `${key} ${value}`).join(", ") || "–"],
                  ["Policies in batch", Object.entries(data.stats?.policy_steps ?? {}).map(([key, value]) => `${key}:${value}`).join(", ") || "–"],
                  ["Path", shortId(data.path, 80)],
                ]}
              />
            </div>
          </details>
        )}
      </header>

      <div className="min-w-0 space-y-8">
        {!data && rows.loading && <div className="py-8 text-center text-xs text-muted">Loading rollout batch…</div>}
        {!data?.available && data && <Empty title="Batch unavailable" hint={data.reason} />}
        {data?.available && (
          <>
            <Disclosure id="inspector.distributions" title="Distributions" summary={`reward ${fmt(data.stats?.reward.mean, 3)} ± ${fmt(data.stats?.reward.std, 3)} · advantage std ${fmt(data.stats?.advantage.std, 3)}`} className="section">
            <div className="grid gap-x-10 gap-y-8 md:grid-cols-3">
              <ChartCard title="Reward distribution"><HistogramChart bins={data.stats?.reward_histogram.bins ?? []} counts={data.stats?.reward_histogram.counts ?? []} format={(v) => fmt(v, 2)} /></ChartCard>
              <ChartCard title="Advantage distribution"><HistogramChart bins={data.stats?.advantage_histogram.bins ?? []} counts={data.stats?.advantage_histogram.counts ?? []} format={(v) => fmt(v, 2)} /></ChartCard>
              <ChartCard title="Completion tokens"><HistogramChart bins={data.stats?.completion_token_histogram.bins ?? []} counts={data.stats?.completion_token_histogram.counts ?? []} format={(v) => fmtInt(v)} /></ChartCard>
            </div>
            </Disclosure>

            <div className="section">
              <Toolbar>
                <Segmented value={mode} onChange={setMode} options={[{ value: "rows", label: `Samples (${fmtInt(data.filtered)})` }, { value: "groups", label: `Prompt groups (${fmtInt(data.groups.length)})` }]} />
                {mode === "groups" ? (
                  <>
                    <input className="input w-full sm:w-56" aria-label="Search rollout groups" placeholder="Search prompt / group / id" value={filters.search} onChange={(e) => setFilter("search", e.target.value)} />
                    <Field label="env">
                      <select className="select" value={filters.env} onChange={(e) => setFilter("env", e.target.value)}>
                        <option value="">all</option>
                        {Object.keys(data.stats?.envs ?? {}).map((env) => <option key={env} value={env}>{env}</option>)}
                      </select>
                    </Field>
                    <Field label="outcome">
                      <select className="select" value={groupOutcome} onChange={(e) => setGroupOutcome(e.target.value)}>
                        <option value="all">all</option>
                        <option value="mixed">mixed</option>
                        <option value="solve_all">solve all</option>
                        <option value="solve_none">solve none</option>
                        <option value="zero_signal">zero signal</option>
                        <option value="truncated">has truncation</option>
                      </select>
                    </Field>
                    <Field label="order">
                      <select className="select" value={groupSort} onChange={(e) => setGroupSort(e.target.value)}>
                        <option value="signal">reward signal</option>
                        <option value="reward">reward mean</option>
                        <option value="size">group size</option>
                        <option value="example">example</option>
                      </select>
                    </Field>
                    {activeGroupFilters > 0 && (
                      <button type="button" className="btn !py-0.5" onClick={() => { setFilters(EMPTY_FILTERS); setGroupOutcome("all"); setOffset(0); }}>
                        clear {activeGroupFilters}
                      </button>
                    )}
                  </>
                ) : (
                  <>
                    <input className="input w-full sm:w-56" aria-label="Search rollouts" placeholder="Search prompt / completion / id" value={filters.search} onChange={(e) => setFilter("search", e.target.value)} />
                    <Field label="env">
                      <select className="select" value={filters.env} onChange={(e) => setFilter("env", e.target.value)}>
                        <option value="">all</option>
                        {Object.keys(data.stats?.envs ?? {}).map((env) => <option key={env} value={env}>{env}</option>)}
                      </select>
                    </Field>
                    <Field label="reward">
                      <input className="input w-16" aria-label="Minimum reward" inputMode="decimal" placeholder="min" value={filters.min_reward} onChange={(e) => setFilter("min_reward", e.target.value)} />
                      <input className="input w-16" aria-label="Maximum reward" inputMode="decimal" placeholder="max" value={filters.max_reward} onChange={(e) => setFilter("max_reward", e.target.value)} />
                    </Field>
                    <Field label="advantage">
                      <select className="select" value={filters.advantage} onChange={(e) => setFilter("advantage", e.target.value)}>
                        <option value="">any</option>
                        <option value="positive">positive</option>
                        <option value="negative">negative</option>
                        <option value="zero">zero</option>
                        <option value="nonzero">nonzero</option>
                      </select>
                    </Field>
                    <Field label="truncated">
                      <select className="select" value={filters.truncated} onChange={(e) => setFilter("truncated", e.target.value)}>
                        <option value="">any</option>
                        <option value="true">yes</option>
                        <option value="false">no</option>
                      </select>
                    </Field>
                    <Field label="stop">
                      <select className="select" value={filters.stop_condition} onChange={(e) => setFilter("stop_condition", e.target.value)}>
                        <option value="">any</option>
                        {Object.keys(data.stats?.stop_conditions ?? {}).filter((k) => k !== "none").map((k) => <option key={k} value={k}>{k}</option>)}
                      </select>
                    </Field>
                    {filters.group_key && <Tag tone="accent">group {groupLabel(filters.group_key)}</Tag>}
                    {activeFilters > 0 && (
                      <button type="button" className="btn !py-0.5" onClick={() => { setFilters(EMPTY_FILTERS); setOffset(0); }}>
                        clear {activeFilters}
                      </button>
                    )}
                  </>
                )}
              </Toolbar>
              <div className="mt-3">
                {mode === "rows" ? (
                  <>
                    <DataTable rows={data.rows} columns={columns} rowKey={(r) => String(r.row_index)} sort={sort} onSort={onSort} onRowClick={(r) => setDetail({ step: data.queue_step!, index: r.row_index })} dense empty="No rollouts match the filters" label={`Rollouts in batch ${data.queue_step}`} />
                    <div className="mt-2">
                      <Pager offset={offset} limit={PAGE} total={data.filtered} onChange={setOffset} />
                    </div>
                  </>
                ) : (
                  <GroupedRollouts
                    apiBase={apiBase}
                    runId={runId}
                    step={data.queue_step!}
                    groups={groups}
                    total={data.groups.length}
                    expanded={expandedGroups}
                    onToggle={(key) => setExpandedGroups((previous) => {
                      const next = new Set(previous);
                      if (next.has(key)) next.delete(key);
                      else next.add(key);
                      return next;
                    })}
                    onOpen={(index) => setDetail({ step: data.queue_step!, index })}
                    onShowTable={(group) => { setFilters({ ...EMPTY_FILTERS, group_key: group.group_key }); setOffset(0); setMode("rows"); }}
                  />
                )}
              </div>
            </div>
          </>
        )}
        <ErrorNote error={rows.error} />
      </div>

      <RowDetailDrawer apiBase={apiBase} runId={runId} target={detail} onClose={() => setDetail(null)} siblings={data?.groups ?? []} onOpen={(index) => detail && setDetail({ step: detail.step, index })} />
    </div>
  );
}

function BatchStepRail({
  batches,
  currentStep,
  follow,
  onSelect,
}: {
  batches: RolloutBatch[];
  currentStep: number | null;
  follow: boolean;
  onSelect: (step: number | null) => void;
}) {
  const railRef = useRef<HTMLDivElement | null>(null);
  const ordered = useMemo(
    () => [...batches].sort((left, right) => left.queue_step - right.queue_step),
    [batches],
  );
  const newestStep = ordered[ordered.length - 1]?.queue_step ?? null;

  useEffect(() => {
    const rail = railRef.current;
    if (!rail || currentStep === null) return;
    const active = rail.querySelector<HTMLElement>(`[data-batch-step="${currentStep}"]`);
    if (!active) return;
    const left = active.offsetLeft - rail.clientWidth / 2 + active.clientWidth / 2;
    rail.scrollTo({ left: Math.max(0, left), behavior: "smooth" });
  }, [currentStep, ordered.length]);

  if (ordered.length === 0) {
    return <div className="border-y border-edge py-3 text-xs text-muted">No rollout batches yet</div>;
  }
  return (
    <div ref={railRef} className="data-table-scroll overflow-x-auto pb-1" role="navigation" aria-label="Rollout batches">
      <div className="relative flex min-w-max py-1">
        <span className="pointer-events-none absolute left-8 right-8 top-[1.9rem] h-px bg-axis" aria-hidden />
        {ordered.map((batch) => {
          const active = batch.queue_step === currentStep;
          const latest = batch.queue_step === newestStep;
          const statusTone = batch.status === "ready" ? "bg-good" : batch.status === "stale" || batch.status === "abandoned_claim" ? "bg-serious" : "bg-muted";
          return (
            <button
              key={batch.queue_step}
              type="button"
              data-batch-step={batch.queue_step}
              className={`group relative flex w-16 shrink-0 flex-col items-center py-1 text-center transition-colors sm:w-20 ${active ? "text-ink" : "text-muted hover:text-ink2"}`}
              onClick={() => onSelect(latest ? null : batch.queue_step)}
              aria-current={active ? "step" : undefined}
              aria-label={`Batch step ${batch.queue_step}, ${batch.status}, policy ${batch.policy_step ?? "unknown"}, ${batch.rows ?? "unknown"} rollouts${latest ? ", latest" : ""}`}
              title={`step ${batch.queue_step} · ${batch.status} · policy ${batch.policy_step ?? "–"} · ${batch.rows ?? "–"} rollouts`}
            >
              <span className={`tabular text-xs ${active ? "font-semibold" : ""}`}>{batch.queue_step}</span>
              <span className={`relative z-[1] mt-1 h-2.5 w-2.5 border-2 border-surface ${active ? "bg-accent ring-2 ring-accent/25" : statusTone}`} aria-hidden />
              <span className="tabular mt-1 text-[9px] text-muted">p{batch.policy_step ?? "–"}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function GroupedRollouts({
  apiBase,
  runId,
  step,
  groups,
  total,
  expanded,
  onToggle,
  onOpen,
  onShowTable,
}: {
  apiBase: string;
  runId: string;
  step: number;
  groups: RolloutGroup[];
  total: number;
  expanded: Set<string>;
  onToggle: (key: string) => void;
  onOpen: (index: number) => void;
  onShowTable: (group: RolloutGroup) => void;
}) {
  if (groups.length === 0) {
    return <Empty title="No rollout groups match" hint="Clear the group search or outcome filters to see the whole batch." />;
  }
  return (
    <div className="space-y-3" role="region" aria-label={`Rollout groups in batch ${step}`}>
      <p className="text-[11px] text-muted">
        {groups.length === total ? `${total} prompt groups` : `${groups.length} of ${total} prompt groups`} · each group contains one shared prompt and all samples generated from it
      </p>
      {groups.map((group) => (
        <RolloutGroupCard
          key={group.group_key}
          apiBase={apiBase}
          runId={runId}
          step={step}
          group={group}
          open={expanded.has(group.group_key)}
          onToggle={() => onToggle(group.group_key)}
          onOpen={onOpen}
          onShowTable={() => onShowTable(group)}
        />
      ))}
    </div>
  );
}

function RolloutGroupCard({
  apiBase,
  runId,
  step,
  group,
  open,
  onToggle,
  onOpen,
  onShowTable,
}: {
  apiBase: string;
  runId: string;
  step: number;
  group: RolloutGroup;
  open: boolean;
  onToggle: () => void;
  onOpen: (index: number) => void;
  onShowTable: () => void;
}) {
  const memberUrl = open
    ? `${runUrl(apiBase, runId, "/rollouts/rows")}${qs({ step, group_key: group.group_key, sort: "row_index", order: "asc", limit: Math.min(500, Math.max(1, group.size)) })}`
    : null;
  const members = usePoll<RolloutRowsResponse>(memberUrl, 0);
  const columns: Column<RolloutRow>[] = [
    { key: "row_index", label: "#", render: (row) => <span className="tabular text-muted">{row.row_index}</span>, width: "3rem" },
    { key: "reward", label: "Reward", align: "right", render: (row) => <RewardCell value={row.reward} /> },
    { key: "advantage", label: "Adv", align: "right", render: (row) => <span className={row.advantage === null ? "text-muted" : row.advantage > 0 ? "text-[var(--success-text)]" : row.advantage < 0 ? "text-critical" : "text-muted"}>{fmt(row.advantage, 3)}</span> },
    { key: "completion_token_count", label: "Tokens", align: "right", render: (row) => fmtInt(row.completion_token_count) },
    { key: "is_truncated", label: "Stop", render: (row) => row.is_truncated ? <Tag tone="warning">truncated</Tag> : <span className="text-muted">{row.stop_condition ?? "stop"}</span> },
    { key: "completion", label: "Completion", render: (row) => <span className="block max-w-[34rem] truncate text-ink2" title={row.completion ?? undefined}>{(row.completion ?? "").replace(/\s+/g, " ").slice(0, 220)}</span> },
  ];
  return (
    <section className="overflow-hidden rounded-md border border-edge bg-surface">
      <button type="button" className="flex w-full items-start gap-3 px-3 py-3 text-left hover:bg-raised/60" onClick={onToggle} aria-expanded={open}>
        <ChevronRight className={`mt-0.5 h-4 w-4 shrink-0 text-muted transition-transform ${open ? "rotate-90" : ""}`} />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-ink" title={group.group_key}>{groupLabel(group.group_key)}</span>
            <GroupOutcome group={group} />
            {group.env && <span className="chip">{group.env}</span>}
            <span className="tabular text-[11px] text-muted">{group.size} samples in this group</span>
          </span>
          <span className="mt-1 grid grid-cols-2 gap-x-5 gap-y-1 text-[11px] text-muted sm:grid-cols-4">
            <span>reward <strong className="font-medium text-ink2">{fmt(group.reward_mean, 3)} ± {fmt(group.reward_std, 3)}</strong></span>
            <span>range <strong className="font-medium text-ink2">{fmt(group.reward_min, 2)} – {fmt(group.reward_max, 2)}</strong></span>
            <span>tokens <strong className="font-medium text-ink2">{fmtInt(group.completion_tokens_mean)} mean</strong></span>
            <span>truncated <strong className="font-medium text-ink2">{group.truncated}/{group.size}</strong></span>
          </span>
          {!open && group.prompt && <span className="mt-1 block truncate text-[11px] text-muted">{group.prompt.replace(/\s+/g, " ")}</span>}
        </span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-edge px-3 py-3">
          {group.prompt && <div><div className="eyebrow mb-1">Shared prompt</div><p className="line-clamp-3 whitespace-pre-wrap text-xs leading-relaxed text-ink2">{group.prompt}</p></div>}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="eyebrow">All {group.size} samples in this group</div>
            <button type="button" className="btn !py-0.5" onClick={onShowTable}>Open filtered rollout table</button>
          </div>
          {members.loading && !members.data && <div className="py-4 text-center text-xs text-muted">Loading group rollouts…</div>}
          <ErrorNote error={members.error} />
          {members.data?.available && (
            <>
              <DataTable rows={members.data.rows} columns={columns} rowKey={(row) => String(row.row_index)} onRowClick={(row) => onOpen(row.row_index)} dense label={`Rollouts in group ${groupLabel(group.group_key)}`} />
              {members.data.filtered > members.data.rows.length && <p className="text-[11px] text-serious">Showing the first {members.data.rows.length} of {members.data.filtered} rollouts in this unusually large group. Use the filtered rollout table to page through all members.</p>}
            </>
          )}
        </div>
      )}
    </section>
  );
}

function GroupOutcome({ group }: { group: RolloutGroup }) {
  if (group.solve_all) return <Tag tone="good">solve all</Tag>;
  if (group.solve_none) return <Tag tone="critical">solve none</Tag>;
  if (group.zero_signal) return <Tag>zero signal</Tag>;
  return <Tag tone="accent">mixed</Tag>;
}

export function RewardCell({ value }: { value: number | null }) {
  if (value === null) return <span className="text-muted">–</span>;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-1.5 w-10 overflow-hidden rounded-full bg-raised">
        <span className="block h-full rounded-full bg-accent" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
      </span>
      <span className="tabular">{fmt(value, 2)}</span>
    </span>
  );
}

function RowDetailDrawer({ apiBase, runId, target, onClose, siblings, onOpen }: { apiBase: string; runId: string; target: { step: number; index: number } | null; onClose: () => void; siblings: RolloutGroup[]; onOpen: (index: number) => void }) {
  const [loadedIndex, setLoadedIndex] = useState<number | null>(null);
  const detail = usePoll<RowDetail>(
    target ? runUrl(apiBase, runId, `/rollouts/${target.step}/rows/${target.index}`) : null,
    0,
    [],
    { resourceKey: target ? `${runId}:${target.step}:rollout-detail` : null },
  );
  const row = detail.data;
  useEffect(() => {
    if (detail.data && target) setLoadedIndex(detail.data.row_index ?? target.index);
  }, [detail.updatedAt]);
  useEffect(() => {
    if (target === null) setLoadedIndex(null);
  }, [target]);
  const displayedIndex = row?.row_index ?? loadedIndex ?? target?.index ?? null;
  const pendingIndex = target && displayedIndex !== target.index ? target.index : null;
  const metadata = (row?.metadata ?? {}) as Record<string, unknown>;
  const group = displayedIndex !== null ? siblings.find((g) => g.row_indexes.includes(displayedIndex)) : undefined;
  return (
    <Drawer open={target !== null} onClose={onClose} title={target && displayedIndex !== null ? `Rollout ${displayedIndex} · batch ${target.step}` : ""} subtitle={group ? <span className="flex flex-wrap items-center gap-x-2"><span>group {groupLabel(group.group_key)} · {group.size} rollouts · reward mean {fmt(group.reward_mean, 3)}</span>{pendingIndex !== null && <span className="inline-flex items-center gap-1 text-accent"><RefreshCw className="h-3 w-3 animate-spin" /> loading #{pendingIndex}</span>}</span> : undefined}>
      {detail.error && <ErrorNote error={detail.error} />}
      {row && (
        <div className="space-y-4">
          <KeyValue
            columns={4}
            items={[
              ["Reward", fmt(row.reward as number | null, 4)],
              ["Advantage", fmt(row.advantage as number | null, 4)],
              ["Source", String(row.source ?? "–")],
              ["Policy step", String(metadata.policy_step ?? "–")],
              ["Stop", String(metadata.stop_condition ?? "–")],
              ["Truncated", String(metadata.is_truncated ?? false)],
              ["Completion tokens", String(metadata.completion_token_count ?? "–")],
              ["Turns", String(metadata.turn_count ?? "–")],
              ...Object.entries(row.arrays ?? {}).map(([name, s]) => [name, s.true_count !== undefined ? `${s.true_count}/${s.length} true` : s.mean !== undefined ? `n=${s.length} mean ${fmt(s.mean, 3)} min ${fmt(s.min, 3)}` : `n=${s.length}`] as [string, string]),
            ]}
          />
          {group && group.size > 1 && (
            <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted">
              same group:
              {group.row_indexes.map((index) => (
                <button key={index} type="button" className={`btn !py-0.5 ${index === displayedIndex ? "btn-active" : ""}`} onClick={() => onOpen(index)} aria-pressed={index === displayedIndex}>
                  #{index}
                </button>
              ))}
            </div>
          )}
          <Transcript title="Prompt" messages={row.prompt as never} />
          <Transcript title="Completion" messages={row.completion as never} />
          {row.target_completion ? <Transcript title="Target completion" messages={row.target_completion as never} /> : null}
          <details className="text-xs">
            <summary className="cursor-pointer text-muted">Raw metadata</summary>
            <pre className="transcript mt-2 rounded-md border border-edge p-2 font-mono text-[11px] text-ink2">{JSON.stringify(metadata, null, 2)}</pre>
          </details>
        </div>
      )}
    </Drawer>
  );
}
