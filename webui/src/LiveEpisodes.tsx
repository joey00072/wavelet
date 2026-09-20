import { useState } from "react";
import { runUrl, usePoll } from "./api/client";
import "./live-episodes.css";

type Episode = {
  id: string;
  env: string;
  kind: string;
  status: string;
  phase: string;
  policy_step: number | null;
  revision?: string;
  events?: Array<{ at: string; phase: string; prompt?: unknown; messages?: unknown; response?: unknown; error?: unknown }>;
};

export function LiveEpisodes({ apiBase, runId, interval }: { apiBase: string; runId: string; interval: number }) {
  const episodes = usePoll<{ episodes: Episode[]; total: number }>(
    runUrl(apiBase, runId, "/episodes"), interval ? 1000 : 0, { resourceKey: runId },
  );
  const [selection, setSelection] = useState<{ runId: string; id: string } | null>(null);
  const selectedId = selection?.runId === runId ? selection.id : null;
  const selected = episodes.data?.episodes.find((item) => item.id === selectedId);
  const detail = usePoll<Episode>(selectedId ? runUrl(apiBase, runId, `/episodes/${selectedId}?revision=${selected?.revision ?? ""}`) : null,
    0, { resourceKey: `${runId}:${selectedId}` });
  if (!episodes.data?.episodes.length && !episodes.error) return null;
  return <section className="live-episodes" aria-label="Live episodes">
    <h3>Live episodes and recent activity</h3>
    {episodes.error && <p role="alert">{episodes.error}</p>}
    <div className="live-episode-list">
      {episodes.data?.episodes.map((episode) => <button key={episode.id} type="button"
        aria-pressed={selectedId === episode.id} onClick={() => setSelection({ runId, id: episode.id })}>
        <strong>{episode.env}</strong> · {episode.kind} · policy {episode.policy_step ?? "–"} · {episode.status} · {episode.phase}
      </button>)}
    </div>
    {detail.error && <p role="alert">{detail.error}</p>}
    {detail.data && <div aria-label="Live transcript">
      <h4>{detail.data.env} · {detail.data.status}</h4>
      {detail.data.events?.map((event, index) => <details key={`${event.at}:${index}`} open>
        <summary>{event.phase} · {new Date(event.at).toLocaleTimeString()}</summary>
        {event.prompt !== undefined && <><span>Model request</span><pre>{JSON.stringify(event.prompt, null, 2)}</pre></>}
        {event.messages !== undefined && <pre>{JSON.stringify(event.messages, null, 2)}</pre>}
        {event.response !== undefined && <><span>Response</span><pre>{JSON.stringify(event.response, null, 2)}</pre></>}
        {event.error !== undefined && <pre>{JSON.stringify(event.error, null, 2)}</pre>}
      </details>)}
    </div>}
  </section>;
}
