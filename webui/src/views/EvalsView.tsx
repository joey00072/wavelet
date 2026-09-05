import { useEffect, useMemo, useState } from "react";

import { qs, runUrl, usePoll } from "../api/client";
import type { EvalExample, EvalRow, EvalRowsResponse, Evals, RowDetail } from "../api/types";
import { HistogramChart } from "../charts/BarChart";
import { ChartCard } from "../charts/ChartCard";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { Tag } from "../components/Badge";
import { Field, Segmented, Toolbar } from "../components/Controls";
import { DataTable, Pager, type Column } from "../components/DataTable";
import { Drawer } from "../components/Drawer";
import { Disclosure } from "../components/Disclosure";
import { Empty, ErrorNote, KeyValue, Section } from "../components/KeyValue";
import { TextBlock, Transcript } from "../components/Transcript";
import { fmt, fmtInt, fmtPct, fmtSeconds, shortId } from "../lib/format";
import { updateParams } from "../lib/router";
import type { LineSeries } from "../lib/series";
import { RewardCell } from "./InspectorView";

const PAGE = 50;
const HEADLINE = /^(avg@\d+|pass@\d+|pass\^\d+)$/;

export function EvalsView({ apiBase, runId, params, trainerStep = null, live }: { apiBase: string; runId: string; params: URLSearchParams; trainerStep?: number | null; live: boolean }) {
  const evals = usePoll<Evals>(runUrl(apiBase, runId, "/evals"), live ? 5000 : 0);
  const data = evals.data;
  const envs = data?.envs ?? [];
  const envParam = params.get("env");
  const env = envParam && envs.some((e) => e.name === envParam) ? envParam : envs[0]?.name ?? null;
  const setEnv = (next: string) => updateParams((p) => p.set("env", next));

  if (data && envs.length === 0) return <Empty title="No evaluations recorded" hint="Enable eval.env in the run config. Fixed-policy avg@k and pass@k appear here as soon as the first evaluation finishes." />;
  if (!data || !env) return <Empty title="Loading evaluations…" />;

  const envMeta = envs.find((e) => e.name === env)!;
  const headlineMetrics = envMeta.metrics.filter((m) => HEADLINE.test(m)).sort();
  const history = [...data.history].filter((h) => h.envs[env]).sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
  const lines: LineSeries[] = headlineMetrics.map((metric, index) => ({
    id: metric,
    label: metric,
    colorIndex: index,
    points: history.flatMap((h) => (h.step !== null && h.envs[env]?.[metric] !== undefined ? [{ x: h.step, y: h.envs[env][metric] }] : [])),
  }));
  const baseline = history.find((h) => h.step === 0);
  const references = baseline ? headlineMetrics.slice(0, 1).map((m) => ({ y: baseline.envs[env][m], label: `${m} at step 0`, colorIndex: 0 })) : [];
  const auxLines: LineSeries[] = ["completion_len/mean", "is_truncated/mean", "failed_rollouts"].filter((m) => envMeta.metrics.includes(m)).map((metric, index) => ({
    id: metric,
    label: metric,
    colorIndex: index,
    points: history.flatMap((h) => (h.step !== null && h.envs[env]?.[metric] !== undefined ? [{ x: h.step, y: h.envs[env][metric] }] : [])),
  }));

  return (
    <div className="space-y-8">
      <Toolbar>
        <h1 className="text-sm font-semibold">Evaluation</h1>
        {envs.length > 1 && <Segmented value={env} onChange={setEnv} options={envs.map((e) => ({ value: e.name, label: e.name }))} />}
        <span className="text-[11px] text-muted">{history.length} evaluation(s) · failures count as incorrect in primary metrics</span>
      </Toolbar>
      {history.length === 0 && <p className="text-xs text-muted">Metric history is unavailable, but saved evaluation samples are still browsable below.</p>}

      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <ChartCard title={`${env}: fixed-policy metrics`} subtitle="avg@k is mean reward; pass@k is a 0–1 rate" refetching={evals.refetching} table={<SeriesTable series={lines} />}>
            <LineChart series={lines} height={240} markers references={references} xExtent={[0, Math.max(1, trainerStep ?? 1)]} />
          </ChartCard>
        </div>
        {history.length >= 2 ? (
          <ChartCard title="Length, truncation, failures" refetching={evals.refetching} table={<SeriesTable series={auxLines} />}>
            <LineChart series={auxLines.filter((l) => l.id !== "completion_len/mean")} height={110} yDomain={[0, 1]} yFormat={(v) => fmtPct(v, 0)} markers xExtent={[0, Math.max(1, trainerStep ?? 1)]} />
            <LineChart series={auxLines.filter((l) => l.id === "completion_len/mean")} height={100} yFormat={(v) => fmtInt(v)} markers xExtent={[0, Math.max(1, trainerStep ?? 1)]} />
          </ChartCard>
        ) : (
          <div>
            <div className="title mb-3">Latest evaluation</div>
            <KeyValue
              columns={2}
              items={[
                ["Policy step", String(history[history.length - 1]?.policy_step ?? "–")],
                ["Completion length", fmtInt(history[history.length - 1]?.envs[env]?.["completion_len/mean"])],
                ["Truncated", fmtPct(history[history.length - 1]?.envs[env]?.["is_truncated/mean"])],
                ["Failed rollouts", fmtPct(history[history.length - 1]?.envs[env]?.["failed_rollouts"])],
                ["Wall time", fmtSeconds(history[history.length - 1]?.envs[env]?.["time"])],
                ["Trend", "needs another evaluation"],
              ]}
            />
          </div>
        )}
      </div>

      <Disclosure id="evals.history" title="Evaluation history" summary={`${history.length} evaluations`} className="section">
        <div className="card">
          <DataTable
            rows={[...history].reverse()}
            rowKey={(h) => String(h.step)}
            dense
            columns={[
              { key: "step", label: "Step", align: "right", render: (h) => h.step },
              { key: "policy", label: "Policy", align: "right", render: (h) => h.policy_step ?? "–" },
              ...headlineMetrics.map((m) => ({ key: m, label: m, align: "right" as const, render: (h: (typeof history)[number]) => /^pass(@|\^)/.test(m) ? fmtPct(h.envs[env]?.[m]) : fmt(h.envs[env]?.[m], 3) })),
              { key: "len", label: "Len mean", align: "right", render: (h) => fmtInt(h.envs[env]?.["completion_len/mean"]) },
              { key: "trunc", label: "Truncated", align: "right", render: (h) => fmtPct(h.envs[env]?.["is_truncated/mean"]) },
              { key: "failed", label: "Failed", align: "right", render: (h) => fmtPct(h.envs[env]?.["failed_rollouts"]) },
              { key: "time", label: "Time", align: "right", render: (h) => fmtSeconds(h.envs[env]?.["time"]) },
              { key: "set", label: "Samples", render: (h) => (data.sets.some((s) => s.step === h.step && s.env === env) ? <Tag tone="accent">on disk</Tag> : <span className="text-muted">pruned</span>) },
            ]}
          />
        </div>
      </Disclosure>

      <EvalSamples apiBase={apiBase} runId={runId} env={env} sets={data.sets.filter((s) => s.env === env).map((s) => s.step).sort((a, b) => b - a)} params={params} />
    </div>
  );
}

function EvalSamples({ apiBase, runId, env, sets, params }: { apiBase: string; runId: string; env: string; sets: number[]; params: URLSearchParams }) {
  const stepParam = params.get("evalStep");
  const step = stepParam !== null && sets.includes(Number(stepParam)) ? Number(stepParam) : sets[0];
  const compareParam = params.get("compareStep");
  const compareStep = compareParam !== null && sets.includes(Number(compareParam)) ? Number(compareParam) : null;
  const [mode, setMode] = useState<"examples" | "rows">("examples");
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: "reward", desc: false });
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<{ search: string; truncated: string; has_error: string; min_reward: string; max_reward: string }>({ search: "", truncated: "", has_error: "", min_reward: "", max_reward: "" });
  const [detail, setDetail] = useState<number | null>(null);
  const [exampleFilter, setExampleFilter] = useState<"all" | "unsolved" | "partial" | "solved" | "errors">("all");

  const url = step !== undefined ? `${runUrl(apiBase, runId, `/evals/${step}/${encodeURIComponent(env)}/rows`)}${qs({ sort: sort.key, order: sort.desc ? "desc" : "asc", offset, limit: PAGE, ...filter })}` : null;
  const rows = usePoll<EvalRowsResponse>(url, 0, [], {
    resourceKey: `${runId}:${env}:${step ?? "none"}`,
  });
  const compareUrl = compareStep !== null ? `${runUrl(apiBase, runId, `/evals/${compareStep}/${encodeURIComponent(env)}/rows`)}?limit=1` : null;
  const compare = usePoll<EvalRowsResponse>(compareUrl, 0);
  const data = rows.data;

  const examples = useMemo(() => {
    const list = data?.examples ?? [];
    const other = new Map((compare.data?.examples ?? []).map((e) => [e.example_id, e]));
    return list
      .map((e) => ({ ...e, other: other.get(e.example_id) ?? null }))
      .filter((e) => {
        if (exampleFilter === "unsolved") return !e.solved_any;
        if (exampleFilter === "partial") return e.solved_any && !e.solved_all;
        if (exampleFilter === "solved") return e.solved_all;
        if (exampleFilter === "errors") return e.errors > 0;
        return true;
      })
      .sort((a, b) => (a.reward_mean ?? -1) - (b.reward_mean ?? -1));
  }, [data?.examples, compare.data?.examples, exampleFilter]);

  const flips = useMemo(() => {
    if (!compare.data) return null;
    let gained = 0;
    let lost = 0;
    for (const e of examples) {
      if (!e.other) continue;
      if (e.solved_any && !e.other.solved_any) gained += 1;
      if (!e.solved_any && e.other.solved_any) lost += 1;
    }
    return { gained, lost };
  }, [examples, compare.data]);

  if (sets.length === 0) return <Empty title="No evaluation samples on disk" hint="Per-example eval rollouts are pruned to eval.keep_last_rollout_sets; metrics history above is complete." />;

  const rowColumns: Column<EvalRow>[] = [
    { key: "row_index", label: "#", sortable: true, render: (r) => <span className="text-muted">{r.row_index}</span> },
    { key: "example_id", label: "Example", sortable: true, render: (r) => shortId(r.example_id, 26) },
    { key: "reward", label: "Reward", sortable: true, align: "right", render: (r) => <RewardCell value={r.reward} /> },
    { key: "is_truncated", label: "Stop", sortable: true, render: (r) => (r.has_error ? <Tag tone="critical">error</Tag> : r.is_truncated ? <Tag tone="warning">truncated</Tag> : <span className="text-muted">{r.stop_condition ?? "stop"}</span>) },
    { key: "completion_token_count", label: "Tokens", sortable: true, align: "right", render: (r) => fmtInt(r.completion_token_count) },
    { key: "answer", label: "Answer", render: (r) => <span className="block max-w-[10rem] truncate">{r.answer ?? "–"}</span> },
    { key: "completion", label: "Completion", render: (r) => <span className="block max-w-[24rem] truncate text-ink2">{(r.completion ?? r.error ?? "").replace(/\s+/g, " ").slice(0, 160)}</span> },
  ];

  type ExampleRow = EvalExample & { other: EvalExample | null };
  const exampleColumns: Column<ExampleRow>[] = [
    { key: "example_id", label: "Example", render: (e) => shortId(e.example_id, 28) },
    { key: "reward_mean", label: `Reward mean (step ${step})`, align: "right", render: (e) => <RewardCell value={e.reward_mean} /> },
    ...(compareStep !== null
      ? [
          { key: "other", label: `Reward mean (step ${compareStep})`, align: "right" as const, render: (e: ExampleRow) => <RewardCell value={e.other?.reward_mean ?? null} /> },
          {
            key: "flip",
            label: "Change",
            render: (e: ExampleRow) => (!e.other ? <span className="text-muted">–</span> : e.solved_any && !e.other.solved_any ? <Tag tone="good">newly solved</Tag> : !e.solved_any && e.other.solved_any ? <Tag tone="critical">regressed</Tag> : <span className="text-muted">same</span>),
          },
        ]
      : []),
    { key: "attempts", label: "k", align: "right", render: (e) => `${e.scored}/${e.attempts}` },
    { key: "outcome", label: "Outcome", render: (e) => (e.solved_all ? <Tag tone="good">all</Tag> : e.solved_any ? <Tag tone="accent">some</Tag> : <Tag tone="critical">none</Tag>) },
    { key: "truncated", label: "Trunc", align: "right", render: (e) => e.truncated },
    { key: "errors", label: "Err", align: "right", render: (e) => e.errors },
    { key: "prompt", label: "Prompt", render: (e) => <span className="block max-w-[28rem] truncate text-ink2">{(e.prompt ?? "").replace(/\s+/g, " ").slice(0, 160)}</span> },
  ];

  return (
    <Section title="Evaluation samples" className="section">
      <div>
        <Toolbar>
          <Field label="eval step">
            <select className="select" value={step} onChange={(e) => { setOffset(0); updateParams((p) => p.set("evalStep", e.target.value)); }}>
              {sets.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </Field>
          <Field label="compare with">
            <select className="select" value={compareStep ?? ""} onChange={(e) => updateParams((p) => (e.target.value ? p.set("compareStep", e.target.value) : p.delete("compareStep")))}>
              <option value="">none</option>
              {sets.filter((s) => s !== step).map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </Field>
          <Segmented value={mode} onChange={setMode} options={[{ value: "examples", label: `Examples (${data?.examples.length ?? 0})` }, { value: "rows", label: `Rollouts (${data?.filtered ?? 0})` }]} />
          {mode === "examples" ? (
            <Segmented value={exampleFilter} onChange={setExampleFilter} size="xs" options={[{ value: "all", label: "all" }, { value: "unsolved", label: "unsolved" }, { value: "partial", label: "partial" }, { value: "solved", label: "solved" }, { value: "errors", label: "errors" }]} />
          ) : (
            <>
              <input className="input w-full sm:w-52" aria-label="Search evaluation rollouts" placeholder="Search" value={filter.search} onChange={(e) => { setOffset(0); setFilter({ ...filter, search: e.target.value }); }} />
              <Field label="truncated">
                <select className="select" value={filter.truncated} onChange={(e) => { setOffset(0); setFilter({ ...filter, truncated: e.target.value }); }}>
                  <option value="">any</option><option value="true">yes</option><option value="false">no</option>
                </select>
              </Field>
              <Field label="errors">
                <select className="select" value={filter.has_error} onChange={(e) => { setOffset(0); setFilter({ ...filter, has_error: e.target.value }); }}>
                  <option value="">any</option><option value="true">only</option><option value="false">exclude</option>
                </select>
              </Field>
            </>
          )}
          {flips && <span className="text-[11px] text-muted">vs step {compareStep}: <span className="text-[var(--success-text)]">+{flips.gained} solved</span>, <span className="text-critical">−{flips.lost} regressed</span></span>}
        </Toolbar>
        {data?.stats && (
          <div className="mt-3 grid gap-3 md:grid-cols-[1fr_2fr]">
            <ChartCard title="Reward distribution"><HistogramChart bins={data.stats.reward_histogram.bins} counts={data.stats.reward_histogram.counts} height={90} format={(v) => fmt(v, 2)} /></ChartCard>
            <div className="py-1">
              <KeyValue columns={4} items={[["Rollouts", fmtInt(data.total)], ["Mean reward", fmt(data.stats.reward.mean, 3)], ["Truncated", fmtInt(data.stats.truncated)], ["Errors", fmtInt(data.stats.errors)]]} />
              {data.scan_limited && <p className="mt-2 text-[11px] text-warn">Stats, filters, and examples cover the first {fmtInt(data.scanned)} rows only.</p>}
            </div>
          </div>
        )}
        <div className="mt-3">
          <ErrorNote error={rows.error} />
          {mode === "examples" ? (
            <DataTable rows={examples} columns={exampleColumns} rowKey={(e) => e.example_id} dense onRowClick={(e) => setDetail(e.row_indexes[0])} empty="No examples" maxHeight={520} />
          ) : (
            <>
              <DataTable rows={data?.rows ?? []} columns={rowColumns} rowKey={(r) => String(r.row_index)} sort={sort} onSort={(key) => { setOffset(0); setSort((prev) => (prev.key === key ? { key, desc: !prev.desc } : { key, desc: true })); }} onRowClick={(r) => setDetail(r.row_index)} dense empty="No rollouts match" />
              <div className="mt-2"><Pager offset={offset} limit={PAGE} total={data?.filtered ?? 0} onChange={setOffset} /></div>
            </>
          )}
        </div>
      </div>
      <EvalDetailDrawer apiBase={apiBase} runId={runId} env={env} step={step} index={detail} onClose={() => setDetail(null)} examples={data?.examples ?? []} onOpen={setDetail} />
    </Section>
  );
}

function EvalDetailDrawer({ apiBase, runId, env, step, index, onClose, examples, onOpen }: { apiBase: string; runId: string; env: string; step: number | undefined; index: number | null; onClose: () => void; examples: EvalExample[]; onOpen: (index: number) => void }) {
  const [loadedIndex, setLoadedIndex] = useState<number | null>(null);
  const detail = usePoll<RowDetail>(
    index !== null && step !== undefined ? runUrl(apiBase, runId, `/evals/${step}/${encodeURIComponent(env)}/rows/${index}`) : null,
    0,
    [],
    { resourceKey: index !== null && step !== undefined ? `${runId}:${env}:${step}:eval-detail` : null },
  );
  const row = detail.data as (RowDetail & { answer?: unknown; error?: string; metrics?: Record<string, number>; info?: unknown }) | null;
  useEffect(() => {
    if (detail.data && index !== null) setLoadedIndex(detail.data.row_index ?? index);
  }, [detail.updatedAt]);
  useEffect(() => {
    if (index === null) setLoadedIndex(null);
  }, [index]);
  const displayedIndex = row?.row_index ?? loadedIndex ?? index;
  const pendingIndex = index !== null && displayedIndex !== index ? index : null;
  const example = displayedIndex !== null ? examples.find((e) => e.row_indexes.includes(displayedIndex)) : undefined;
  return (
    <Drawer open={index !== null} onClose={onClose} title={displayedIndex !== null ? `Eval rollout ${displayedIndex} · ${env} · step ${step}` : ""} subtitle={example ? <span className="flex flex-wrap items-center gap-x-2"><span>example {example.example_id} · {example.scored}/{example.attempts} scored · mean {fmt(example.reward_mean, 3)}</span>{pendingIndex !== null && <span className="text-accent">loading #{pendingIndex}</span>}</span> : undefined}>
      {detail.error && <ErrorNote error={detail.error} />}
      {row && (
        <div className="space-y-4">
          <KeyValue columns={3} items={[["Reward", fmt(row.reward as number | null, 4)], ["Truncated", String(row.is_truncated ?? false)], ["Stop", String(row.stop_condition ?? "–")], ...Object.entries(row.metrics ?? {}).map(([k, v]) => [k, fmt(v, 4)] as [string, string])]} />
          {example && example.attempts > 1 && (
            <div className="flex flex-wrap items-center gap-1 text-[11px] text-muted">
              attempts:
              {example.row_indexes.map((i) => (
                <button key={i} type="button" className={`btn !py-0.5 ${i === displayedIndex ? "btn-active" : ""}`} onClick={() => onOpen(i)} aria-pressed={i === displayedIndex}>#{i}</button>
              ))}
            </div>
          )}
          {row.error ? <ErrorNote error={String(row.error)} /> : null}
          <Transcript title="Prompt" messages={row.prompt as never} />
          <Transcript title="Completion" messages={row.completion as never} />
          <TextBlock title="Reference answer" text={row.answer === undefined || row.answer === null ? null : typeof row.answer === "string" ? row.answer : JSON.stringify(row.answer, null, 2)} />
          {row.info !== undefined && <TextBlock title="Info" text={JSON.stringify(row.info, null, 2)} />}
        </div>
      )}
    </Drawer>
  );
}
