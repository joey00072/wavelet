import { runUrl, usePoll } from "../api/client";
import type { QueueReport, RunEvent, RunSummary, Timeline, TimelineStep } from "../api/types";
import { ChartCard } from "../charts/ChartCard";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { StatTile } from "../charts/StatTile";
import { StepTimeline } from "../charts/StepTimeline";
import { Tag } from "../components/Badge";
import { DataTable } from "../components/DataTable";
import { Disclosure } from "../components/Disclosure";
import { Empty, ErrorNote, KeyValue, Section } from "../components/KeyValue";
import { fmt, fmtAge, fmtBytes, fmtDateTime, fmtInt, fmtSeconds, shortId } from "../lib/format";
import type { LineSeries } from "../lib/series";

export function PipelineView({ apiBase, runId, summary }: { apiBase: string; runId: string; summary: RunSummary | null }) {
  const live = summary?.status === "running";
  const queue = usePoll<QueueReport>(`${runUrl(apiBase, runId, "/queue")}?limit=500`, live ? 4000 : 0);
  const timeline = usePoll<Timeline>(`${runUrl(apiBase, runId, "/timeline")}?limit=2000`, live ? 4000 : 0);
  const events = usePoll<RunEvent[]>(`${runUrl(apiBase, runId, "/events")}?limit=100`, live ? 8000 : 0);
  const q = queue.data;
  const s = q?.summary;
  const steps = timeline.data?.queue_steps ?? [];
  const policies = timeline.data?.policies ?? [];
  const latency: LineSeries[] = [
    { id: "wait", label: "published → claimed (trainer wait)", colorIndex: 0, points: steps.flatMap((st) => (st.publish_to_claim_seconds !== null ? [{ x: st.queue_step, y: st.publish_to_claim_seconds }] : [])) },
    { id: "train", label: "claimed → consumed (train)", colorIndex: 1, points: steps.flatMap((st) => (st.claim_to_consume_seconds !== null ? [{ x: st.queue_step, y: st.claim_to_consume_seconds }] : [])) },
    { id: "policy", label: "policy export → load", colorIndex: 2, points: policies.flatMap((p) => (p.export_to_load_seconds !== null ? [{ x: p.policy_step, y: p.export_to_load_seconds }] : [])) },
  ];
  const lag: LineSeries[] = [
    { id: "lag", label: "optimizer step − policy step used", colorIndex: 0, points: steps.flatMap((st) => (st.optimizer_step !== null && st.policy_step !== null ? [{ x: st.queue_step, y: st.optimizer_step - st.policy_step }] : [])) },
  ];
  const payload: LineSeries[] = [{ id: "bytes", label: "batch payload bytes", colorIndex: 0, points: steps.flatMap((st) => (st.payload_bytes ? [{ x: st.queue_step, y: st.payload_bytes }] : [])) }];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Pipeline</h1>
        <p className="mt-1 text-xs text-muted">{live ? "Live rollout queue, policy movement, and end-to-end batch lifecycle." : "Final recorded queue, policy movement, and batch lifecycle."}</p>
      </header>
      <ErrorNote error={queue.error ?? timeline.error} />
      <div className="grid grid-cols-2 gap-x-8 gap-y-6 md:grid-cols-4 2xl:grid-cols-8">
        <StatTile label={live ? "Ready for trainer" : "Ready at stop"} number={s?.ready_count} format={fmtInt} tone={(s?.ready_count ?? 0) > 2 ? "warning" : null} sub="stable, unclaimed" />
        <StatTile label="Claimed" number={s?.claimed_count} format={fmtInt} sub="trainer is loading/training" />
        <StatTile label="Consumed" number={s?.consumed_count} format={fmtInt} sub={`latest ${s?.latest_consumed_queue_step ?? "–"}`} />
        <StatTile label={live ? "Incomplete" : "Incomplete at stop"} number={s?.incomplete_count} format={fmtInt} sub={live ? "being written by inference" : "unfinished artifacts"} />
        <StatTile label="Stale" number={s?.stale_ready_count} format={fmtInt} tone={(s?.stale_ready_count ?? 0) > 0 ? "serious" : null} sub="beyond off-policy window" />
        <StatTile label="Abandoned claims" number={s?.abandoned_claim_count} format={fmtInt} tone={(s?.abandoned_claim_count ?? 0) > 0 ? "critical" : null} />
        <StatTile label="Policy exported" value={q?.policy?.latest_exported_step ?? "–"} sub={`${q?.policy?.steps.length ?? 0} versions on disk${q?.policy?.incomplete_steps.length ? ` · ${q.policy.incomplete_steps.length} incomplete` : ""}`} />
        <StatTile label="Batches per minute" value={`${fmt((q?.rates?.rollouts_published_per_second ?? 0) * 60, 1)} → ${fmt((q?.rates?.rollouts_consumed_per_second ?? 0) * 60, 1)}`} sub="published → consumed" />
      </div>

      <ChartCard className="section" title="Step lifecycle" subtitle="last 16 queue steps on a shared wall clock; horizontal gaps are idle time between batches" refetching={timeline.refetching}>
        <StepTimeline steps={steps} policies={policies} maxRows={16} />
      </ChartCard>

      <div className="section grid gap-x-10 gap-y-8 md:grid-cols-3">
        <ChartCard title="Latencies" subtitle="seconds per step" refetching={timeline.refetching} table={<SeriesTable series={latency} yFormat={(v) => fmtSeconds(v)} />}>
          <LineChart series={latency} height={180} yFormat={(v) => fmtSeconds(v)} />
        </ChartCard>
        <ChartCard title="Policy lag at publish" subtitle={`limit ${summary?.batch?.max_off_policy_steps ?? "–"} (max_off_policy_steps)`} refetching={timeline.refetching} table={<SeriesTable series={lag} />}>
          <LineChart series={lag} height={180} yFormat={(v) => String(Math.round(v))} references={summary?.batch ? [{ y: summary.batch.max_off_policy_steps, label: "max_off_policy_steps" }] : []} />
        </ChartCard>
        <ChartCard title="Batch payload" refetching={timeline.refetching} table={<SeriesTable series={payload} yFormat={(v) => fmtBytes(v)} />}>
          <LineChart series={payload} height={180} yFormat={(v) => fmtBytes(v)} />
        </ChartCard>
      </div>

      <Disclosure id="pipeline.queue" title="Queue items" summary={`${q?.items.length ?? 0} items on disk`} className="section">
        <div className="card">
          <DataTable
            rows={(q?.items ?? []).slice().sort((a, b) => b.queue_step - a.queue_step).slice(0, 200)}
            rowKey={(item) => String(item.queue_step)}
            dense
            maxHeight={420}
            empty="No queue items"
            columns={[
              { key: "queue_step", label: "Queue step", align: "right", render: (i) => i.queue_step },
              { key: "status", label: "Status", render: (i) => <Tag tone={i.status === "ready" ? "good" : i.status === "stale" || i.status === "abandoned_claim" ? "serious" : i.status === "incomplete" ? "warning" : "neutral"}>{i.status}</Tag> },
              { key: "opt", label: "Optimizer", align: "right", render: (i) => String((i.manifest as { optimizer_step?: number } | null)?.optimizer_step ?? "–") },
              { key: "policy", label: "Policy", align: "right", render: (i) => String((i.manifest as { policy_step?: number } | null)?.policy_step ?? "–") },
              { key: "lag", label: "Lag now", align: "right", title: "latest exported policy − batch policy", render: (i) => { const p = (i.manifest as { policy_step?: number } | null)?.policy_step; return p === undefined || p === null || q?.policy?.latest_exported_step == null ? "–" : q.policy.latest_exported_step - p; } },
              { key: "rows", label: "Rows", align: "right", render: (i) => String((i.manifest as { rows?: number } | null)?.rows ?? "–") },
              { key: "reward", label: "Reward", align: "right", render: (i) => fmt((i.manifest as { reward_mean?: number } | null)?.reward_mean ?? null, 3) },
              { key: "tokens", label: "Tokens", align: "right", render: (i) => fmtInt((i.manifest as { tokens?: number } | null)?.tokens ?? null) },
              { key: "created", label: "Created", render: (i) => fmtDateTime((i.manifest as { created_at?: string } | null)?.created_at) },
              { key: "claimed", label: "Claimed", render: (i) => fmtDateTime((i.claim as { claimed_at?: string } | null)?.claimed_at) },
              { key: "consumed", label: "Consumed", render: (i) => fmtDateTime((i.consumed as { consumed_at?: string } | null)?.consumed_at) },
              { key: "consumer", label: "Consumer", render: (i) => shortId((i.claim as { consumer_id?: string } | null)?.consumer_id ?? null, 28) },
              { key: "age", label: "Age", align: "right", render: (i) => fmtSeconds(i.age_seconds) },
              { key: "errors", label: "Parse errors", render: (i) => (i.parse_errors.length ? <Tag tone="critical">{i.parse_errors.length}</Tag> : <span className="text-muted">0</span>) },
            ]}
          />
        </div>
      </Disclosure>

      <Disclosure id="pipeline.policy" title="Policy versions and run events" summary={`latest exported ${q?.policy?.latest_exported_step ?? "–"} · ${events.data?.length ?? 0} run events`} className="section">
      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Section title="Policy versions">
          <div>
            <KeyValue columns={3} items={[["Latest exported", String(q?.policy?.latest_exported_step ?? "–")], ["Stable versions", fmtInt(q?.policy?.steps.length ?? 0)], ["Incomplete", q?.policy?.incomplete_steps.join(", ") || "none"]]} />
            <div className="mt-3 max-h-64 overflow-auto">
              <table className="w-full">
                <thead><tr><th className="th">Policy</th><th className="th">Exported</th><th className="th">Loaded by inference</th><th className="th text-right">Export → load</th><th className="th text-right">Load</th><th className="th text-right">Bytes</th></tr></thead>
                <tbody>
                  {[...policies].reverse().slice(0, 100).map((p) => (
                    <tr key={p.policy_step} className="border-b border-edge last:border-0">
                      <td className="td tabular">{p.policy_step}</td>
                      <td className="td">{fmtDateTime(p.exported_at)}</td>
                      <td className="td">{p.loaded_at ? fmtDateTime(p.loaded_at) : <span className="text-warn">not loaded</span>}</td>
                      <td className="td tabular text-right">{fmtSeconds(p.export_to_load_seconds)}</td>
                      <td className="td tabular text-right">{fmtSeconds(p.load_seconds)}</td>
                      <td className="td tabular text-right">{fmtBytes(p.payload_bytes)}</td>
                    </tr>
                  ))}
                  {policies.length === 0 && <tr><td className="td py-4 text-center text-muted" colSpan={6}>No policy events</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
        </Section>
        <Section title="Run events">
          <div className="max-h-[22rem] overflow-auto">
            {events.data && events.data.length === 0 && <Empty title="No run events" />}
            <ul className="space-y-1">
              {[...(events.data ?? [])].reverse().map((e, i) => (
                <li key={`${e.timestamp}-${i}`} className="flex items-start gap-3 text-xs">
                  <span className="tabular w-32 shrink-0 text-muted">{fmtDateTime(e.timestamp)}</span>
                  <span className="font-medium text-ink">{e.event}</span>
                  <span className="text-muted">{e.step !== null ? `step ${e.step}` : ""}</span>
                  <span className="truncate text-ink2" title={JSON.stringify(e.payload)}>{Object.keys(e.payload).length ? JSON.stringify(e.payload) : ""}</span>
                </li>
              ))}
            </ul>
          </div>
        </Section>
      </div>
      </Disclosure>
      {timeline.data && timeline.data.dropped_events > 0 && <p className="text-[11px] text-serious">Lifecycle history is incomplete: {timeline.data.dropped_events} older event line(s) fell outside the 50,000-row reader window.</p>}
      {timeline.data && timeline.data.parse_errors > 0 && <p className="text-[11px] text-serious">{timeline.data.parse_errors} lifecycle event line(s) failed to parse.</p>}
      <p className="text-[11px] text-muted">Queue state is read from manifests, claim and consumed records, and events/queue.jsonl. Last refresh {fmtAge(queue.updatedAt)}.</p>
    </div>
  );
}

export type { TimelineStep };
