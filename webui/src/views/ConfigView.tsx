import { useState } from "react";

import { runUrl, usePoll } from "../api/client";
import type { Json } from "../api/types";
import { Field, Toolbar } from "../components/Controls";
import { JsonTree } from "../components/JsonTree";
import { Empty, ErrorNote } from "../components/KeyValue";

export function ConfigView({ apiBase, runId, otherRuns }: { apiBase: string; runId: string; otherRuns: string[] }) {
  const config = usePoll<Json>(runUrl(apiBase, runId, "/config"), 0);
  const [filter, setFilter] = useState("");
  const [other, setOther] = useState("");
  const otherConfig = usePoll<Json>(other ? runUrl(apiBase, other, "/config") : null, 0);
  const configMessage = typeof config.data?._dashboard_error === "string" ? config.data._dashboard_error : null;
  const otherMessage = typeof otherConfig.data?._dashboard_error === "string" ? otherConfig.data._dashboard_error : null;
  return (
    <div className="space-y-3">
      <Toolbar>
        <h1 className="text-sm font-semibold">Resolved config</h1>
        <input className="input w-full sm:w-64" aria-label="Filter configuration" placeholder="Filter keys or values" value={filter} onChange={(e) => setFilter(e.target.value)} />
        <Field label="diff against">
          <select className="select" value={other} onChange={(e) => setOther(e.target.value)}>
            <option value="">none</option>
            {otherRuns.filter((id) => id !== runId).map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </Field>
        {other && <span className="text-[11px] text-muted">green = only in this run · amber = differs · red = only in the comparison run</span>}
      </Toolbar>
      <ErrorNote error={config.error ?? configMessage ?? otherConfig.error ?? (otherMessage ? `Comparison config: ${otherMessage}` : null)} />
      <div className="section">
        {configMessage ? <Empty title="Config unavailable" hint={configMessage} /> : config.data ? <JsonTree value={config.data} other={other && !otherMessage ? otherConfig.data ?? undefined : undefined} filter={filter} /> : <div className="text-xs text-muted">Loading…</div>}
      </div>
      {!configMessage && <p className="text-[11px] text-muted">Secrets are redacted server-side. The config shown is the resolved orchestrator config written at launch, falling back to the root rl.yaml.</p>}
    </div>
  );
}
