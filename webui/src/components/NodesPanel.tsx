import { useMemo } from "react";

import type { RankStats } from "../api/types";
import { ChartCard } from "../charts/ChartCard";
import { LineChart, SeriesTable } from "../charts/LineChart";
import { fmt, fmtAge, fmtInt } from "../lib/format";
import { seriesToLines } from "../lib/series";
import { useMetricKeys, useNodes, useSeries } from "../views/useRunData";
import { Column, DataTable } from "./DataTable";
import { Empty, Section } from "./KeyValue";

const RANK_COLUMNS: Column<RankStats>[] = [
  { key: "rank", label: "rank", align: "right", render: (r) => fmtInt(r.rank) },
  { key: "node", label: "node", render: (r) => r.node },
  { key: "device", label: "device", render: (r) => r.device },
  { key: "tokens_per_second", label: "tok/s", align: "right", render: (r) => fmtInt(r.tokens_per_second) },
  { key: "tokens", label: "tokens", align: "right", render: (r) => fmtInt(r.tokens) },
  { key: "seconds", label: "step s", align: "right", render: (r) => fmt(r.seconds, 2) },
  { key: "memory_allocated_gib", label: "alloc GiB", align: "right", render: (r) => fmt(r.memory_allocated_gib, 1) },
  { key: "peak_memory_gib", label: "peak GiB", align: "right", render: (r) => fmt(r.peak_memory_gib, 1) },
];

/**
 * Per-node trainer throughput over time plus the per-rank table from the
 * latest heartbeat. History comes from `node/<name>/*` trainer metrics; the
 * rank table is overwritten each step, so it costs nothing on disk.
 */
export function NodesPanel({ apiBase, runId }: { apiBase: string; runId: string }) {
  const nodes = useNodes(apiBase, runId);
  const keys = useMetricKeys(apiBase, runId);
  const nodeKeys = useMemo(() => (keys.data?.trainer ?? []).map((k) => k.key).filter((k) => /^node\/[^/]+\/(tokens_per_second|peak_memory_gib)$/.test(k)), [keys.data]);
  const history = useSeries(apiBase, runId, "trainer", nodeKeys, 0, 5000);
  const nodeNames = useMemo(() => [...new Set(nodeKeys.map((k) => k.split("/")[1]))].sort(), [nodeKeys]);
  const labels = Object.fromEntries(nodeNames.flatMap((n) => [[`node/${n}/tokens_per_second`, n], [`node/${n}/peak_memory_gib`, n]]));
  const throughput = seriesToLines(history.data, nodeNames.map((n) => `node/${n}/tokens_per_second`), { labels });
  const memory = seriesToLines(history.data, nodeNames.map((n) => `node/${n}/peak_memory_gib`), { labels });
  const ranks = nodes.data?.ranks ?? [];
  const total = ranks.reduce((sum, r) => sum + (r.tokens_per_second ?? 0), 0);
  const slowest = ranks.length ? Math.min(...ranks.map((r) => r.tokens_per_second)) : null;
  const fastest = ranks.length ? Math.max(...ranks.map((r) => r.tokens_per_second)) : null;

  if (nodeNames.length === 0 && ranks.length === 0) {
    return (
      <Section title="Nodes" className="section">
        <Empty title="No per-node telemetry" hint="Trainer ranks report node/<host>/* metrics and a per-rank heartbeat table once the first optimizer step completes." />
      </Section>
    );
  }

  return (
    <Section title="Nodes" className="section" actions={<span className="text-[11px] text-muted">{nodeNames.length} node(s) · {ranks.length} rank(s) · heartbeat {fmtAge(nodes.data?.timestamp)}</span>}>
      <div className="grid gap-x-10 gap-y-8 md:grid-cols-2">
        <ChartCard title="Tokens per second per node" subtitle={history.data?.downsampled ? `bucket-averaged, ${history.data.bucket} steps per point` : undefined} refetching={history.refetching} table={<SeriesTable series={throughput} />}>
          <LineChart series={throughput} height={170} yFormat={(v) => fmtInt(v)} />
        </ChartCard>
        <ChartCard title="Peak memory per node (GiB)" refetching={history.refetching} table={<SeriesTable series={memory} />}>
          <LineChart series={memory} height={170} yFormat={(v) => fmt(v, 1)} />
        </ChartCard>
      </div>
      {ranks.length > 0 && (
        <div className="mt-6">
          <div className="mb-2 flex items-center justify-between text-[11px] text-muted">
            <span>Ranks at step {nodes.data?.step ?? "–"}</span>
            <span>total {fmtInt(total)} tok/s · slowest {fmtInt(slowest)} · fastest {fmtInt(fastest)}</span>
          </div>
          <DataTable rows={ranks} columns={RANK_COLUMNS} rowKey={(r) => String(r.rank)} dense maxHeight={320} empty="No ranks reported" />
        </div>
      )}
    </Section>
  );
}
