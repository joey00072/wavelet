# Wavelet Dashboard

A compact, read-only dashboard for live and completed Wavelet runs. The UI uses
one run picker and five tabs:

- **Metrics** shows the essential reward, loss, throughput, policy, queue, and
  rank signals. Switch to **All** to search and plot any numeric metric written
  by the trainer, orchestrator, evaluator, W&B, or Trackio. A single observation
  appears as a point, so the first reward or baseline evaluation is visible
  before a second observation arrives. Hover or use arrow keys for exact values;
  drag to zoom and double-click or press Escape to reset. Choose step or elapsed
  time, compare raw and smoothed curves, expand a chart, or export its plotted
  values as CSV. Overlay another Wavelet run with matching metric names using
  Compare run. Downsampled histories show their min/max envelope and identify
  bucket means in the readout. Offscreen charts defer SVG rendering. Metric filters accept
  regular expressions, custom metric selections persist per run, and untouched
  overview selections include metrics logged after startup.
- **Traces** browses retained rollout rows in latest-batch or explicit-step mode.
  Pages contain 50 metadata-only rows; transcripts are fetched only on selection. Long
  conversations render 12 messages at a time as you scroll, and large tool
  outputs expand on demand. Tool calls remain visible even without text content.
  Outcome filters select errors, non-errors, or truncated rows. Copy link keeps
  the selected run, step, and row; selecting an episode pins the batch so live
  updates cannot silently replace it.
- **Report** shows aggregate evaluation history and retained per-example rows.
- **Logs** merges or splits selected component logs and supports level and
  regular-expression filtering.
- **Config** searches and copies the redacted resolved configuration.

Queue state, policy progress, model placement, and run health stay visible in
the sidebar and metric summary. The sidebar contains run selection, navigation,
live refresh, and a light/dark theme toggle. The selected theme persists locally.

## Run

```bash
cd webui
bun install
bun run build
cd ..
uv run wavelet dashboard --runs-root outputs --port 8766
```

Open `http://127.0.0.1:8766/`. Immediate children of each `--runs-root` that
look like run directories are discovered automatically. Explicit run paths can
be supplied as positional arguments:

```bash
uv run wavelet dashboard outputs/my_run /mnt/other/run --runs-root outputs
```

The server never claims, consumes, deletes, or rewrites run state. Source
checkouts serve `webui/dist`; wheels embed the same build under the package.

The URL keeps the selected run and tab in `?run=<id>&tab=<tab>`. API discovery
tries `?api=http://host:port`, the last working address, the Vite development
default on port 8766, and finally the same origin. Use the API input shown on a
connection failure to point the page at another host.

## Live state server

The optional orchestrator state server exposes the same `/api/runs/...` routes,
so the UI can connect directly to a live run:

```yaml
orchestrator:
  state_server:
    enabled: true
    host: 0.0.0.0
    port: 8765
```

## Develop and test

Generate deterministic data without using a GPU:

```bash
uv run wavelet synth-run --output outputs/demo_run --steps 40
uv run wavelet dashboard --runs-root outputs
```

Run the browser suite against temporary synthetic runs:

```bash
cd webui
bun run build
bun run test:e2e
```

The frontend intentionally has no component framework, icon package, CSS
pipeline, or chart dependency. The SVG charts and five views live together in
`src/App.tsx`; visual rules live in `src/main.css`.

## API

The UI consumes the read-only routes under `/api`:

| Route | Purpose |
| --- | --- |
| `/api/health`, `/api/runs`, `/api/current` | Discovery and liveness |
| `/api/runs/{id}/summary` | Run, queue, policy, and latest-metric summary |
| `/api/runs/{id}/metrics/keys` | Every available numeric signal |
| `/api/runs/{id}/series?source=&keys=&points=` | Bounded chart data |
| `/api/runs/{id}/nodes` | Rank, node, and inference-replica telemetry |
| `/api/runs/{id}/rollouts` | Rollout batch list |
| `/api/runs/{id}/rollouts/rows` | Filtered rollout samples |
| `/api/runs/{id}/evals` | Evaluation history and retained sets |
| `/api/runs/{id}/evals/{step}/{env}/rows` | Evaluation samples |
| `/api/runs/{id}/logs`, `/logs/{name}` | Role logs |
| `/api/runs/{id}/config` | Redacted resolved config |

The server keeps metric reads incremental and downsamples chart responses, so a
long run does not produce an unbounded browser payload.


## Loading and data fidelity

The run picker requests compact metadata every 15 seconds; the selected run
refreshes every three seconds. Polling pauses in hidden browser tabs and resumes
when visible. Rollout/evaluation list and detail reads use server worker threads,
so a cold JSONL scan does not occupy the API event loop. The compact row cache
retains byte offsets: subsequent detail requests seek to one selected JSONL row
instead of parsing all preceding rows. Files that change invalidate that index.
The reader also discovers numbered resolved attempts when a manually submitted
run has no `configs/latest` symlink.

Only retained queue payloads are browseable. Training rows may represent branches
of one episode; do not infer distinct episode counts from the table. The dashboard
does not synthesize token overlays, physical branch graphs, or timed replay from
rows without the required recorded annotations.

Reward overview charts distinguish all episodes from the effective training subset and
show each recorded denominator. Chart descriptions identify the population and
source; step axes distinguish optimizer updates from rollout queue indices.
Older runs retain their original problem-average reward series. Missing cohort
metrics are not reconstructed or shown as zero. See
[`docs/evaluation_and_live_traces.md`](../docs/evaluation_and_live_traces.md) for
population and step semantics.

Metric cards keep plots aligned with a fixed header and compact step-axis label.
Use the ⓘ button for the metric's source, reward population, and denominator
semantics; explanations do not displace the plot. Hover or focus a card to
show CSV export and expand controls. Rollout queue and policy freshness metrics
have their own section. Charts and trace details continue to load lazily.

The default overview groups charts into Training, Evaluation, Stability,
Inference, Performance, and Queue blocks. Training shows `effective/agent/reward`
and `all/agent/reward` separately. The latter uses episode-weighted rewards;
historical trainer rewards supply it when the newer explicit all-episode metric
is absent. Missing effective rewards are marked **Not recorded**, never inferred
from the all-episode mean. Metric identifiers and population definitions remain
available through ⓘ. Empty evaluation/inference blocks indicate missing telemetry;
available per-replica inference signals appear automatically.
