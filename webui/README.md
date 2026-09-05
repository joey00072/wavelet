# Wavelet Dashboard

Browser dashboard for observing Wavelet RL runs end to end: training signals,
rollout generation, individual rollouts, fixed-policy evaluation, the
rollout/policy pipeline, and infrastructure. It reads only the artifacts a run
writes under its output directory, so it works for live runs and for completed
runs on disk, and it can compare several runs side by side.

## Run

Build the UI once, then serve one or more run directories with the Python
dashboard command:

```bash
cd webui && bun install && bun run build && cd ..
uv run wavelet dashboard --runs-root outputs --port 8766
```

Open `http://127.0.0.1:8766/`. The page lands on the **current/recent run**: the run
with a fresh `running` heartbeat, else a `running` run whose heartbeat went
stale, else the run that wrote artifacts most recently. Hosts train one run at
a time, so this is the run you care about; every discovered run stays one click
away under **Runs** and can be opened or compared at any time. The `current` alias
in URLs (`#/run/current/overview`, `/api/runs/current/...`) always follows
whichever run is active, so bookmarks survive run changes.

Every immediate child of `--runs-root` that looks like a run directory is
discovered; pass run directories explicitly instead, or in addition:

```bash
uv run wavelet dashboard outputs/my_run /mnt/other/run --runs-root outputs
```

The server never claims, consumes, deletes, or rewrites queue state. A source
checkout serves the built UI from `webui/dist`; distribution wheels embed the
same assets under the Python package. Building a wheel runs the frontend build
and therefore requires Bun. `--static-dir` can override either location.

For UI development, run the Vite dev server against the same API:

```bash
cd webui && bun run dev --host 0.0.0.0
```

The UI finds its API automatically. On load it probes, in order, the
`?api=http://host:port` query parameter, the address remembered from a previous
visit, the Vite dev default at port `8766` on the current browser hostname, and
finally the server that served the page, and uses the first one whose
`/api/health` answers. A stale
address in the URL or in local storage is therefore harmless: opening the
dashboard's own URL always lands on the current run. `?api=same` forces the
page's own server; the sidebar field overrides everything for that browser.

### Live orchestrator state server

A running `wavelet rl` process with `orchestrator.state_server.enabled: true`
exposes the same `/api/runs/...` routes for its own run beside the legacy
`/state`, `/queues`, `/metrics`, and `/rollouts/inspect` endpoints, so the UI
can point at either server:

```yaml
orchestrator:
  state_server:
    enabled: true
    host: 0.0.0.0
    port: 8765
```

### Synthetic runs without a GPU

`uv run wavelet synth-run --output outputs/demo_run --steps 40` writes a
deterministic run directory with the full artifact layout (metrics, eval sets,
stable rollout batches, queue events, policies, logs). Use it to develop the UI
or to reproduce a rendering issue on a machine without GPUs.

Run the browser regression suite without a GPU or existing outputs:

```bash
cd webui
bun run test:e2e
```

The suite creates temporary synthetic runs, starts a dashboard server, and
checks every view in desktop Chromium and a narrow mobile viewport.

## Views

- **Runs**: the current or recent run pinned on top, then every other discovered
  run with status, step progress, reward trend, latest eval, model,
  environment, and algorithm. Tick runs to compare. Viewing an older run shows
  a banner with a link back to the current one.
- **Overview**: status strip, headline tiles (trailing five-step train reward
  mean with early/late delta, eval avg@k, loss, grad norm, entropy, truncation,
  solve-none rate, throughput), reward before and after filtering, fixed-policy eval with the
  step-0 baseline, loss, entropy, trainer/inference mismatch, group outcomes,
  off-policy distance, completion length, truncation, and rollout fate.
  Health checks flag failed runs, stale heartbeats, config validation errors,
  stale or abandoned queue batches, queue event parse errors, a trainer that is
  behind, lag beyond `max_off_policy_steps`, high truncation, low admission,
  mostly unsolved groups, verifier errors, entropy collapse, heavy DPPO
  masking, logprob mismatch, non-finite gradient norms, eval failures, low
  disk, and saturated KV cache. Each finding links to the view that explains
  it.
- **Trainer** and **Generation**: metric explorers over every logged trainer
  or orchestrator key. Pick a preset or tick keys in the collapsed group list;
  the selection lives in the URL. Charts sit in a flat grid with the latest
  value in each header. One settings popover holds the global smoothing
  slider, x-axis, log scale, step window, column count, and overlay layout.
- **Inspector**: browse stable rollout batches. Pin a batch or follow the
  newest from the clickable horizontal batch timeline at the top of the page. Sort by
  reward, advantage, tokens, logprob mean, truncation, group,
  environment, or policy step; filter by environment, reward range, advantage
  sign, truncation, stop condition, group, or free text without blanking the
  batch while new results load. The default group view keeps each GRPO group’s
  shared prompt, reward signal, outcome, and member rollouts together; expand a
  group to compare every completion, reward, advantage, length, and stop reason
  before opening a rollout. Row detail shows the prompt and completion
  transcript, metadata, token and logprob summaries, and jumps between
  rollouts of the same group.
- **Evals**: shown when the run records evaluations. Per environment: avg@k,
  pass@k, pass^k over policy steps with the step-0 baseline, length and
  truncation and failure rates, the evaluation history table, and the
  per-example browser over eval rollout sets kept on disk. Compare two eval
  steps to see which examples were newly solved or regressed.
- **Pipeline**: queue counts, per-step lifecycle bars (published → claimed →
  consumed, policy export → load) on a shared wall clock, latency series,
  policy lag at publish against the configured window, batch payload sizes,
  the queue item table, policy versions, and run events.
- **Infra**: trainer heartbeat, GPU memory, tokens per second, MFU, disk,
  step-time breakdown, vLLM replica KV cache, requests, preemptions, and
  generated tokens per second, concurrency, orchestrator timing, process
  facts, and role log tails with grep and follow. The **Nodes** panel charts
  tokens per second per node over time and lists every trainer rank with its
  node, device, token rate, and memory from the latest heartbeat.
- **Config**: the redacted resolved config as a tree, filterable, with an
  optional diff against another run.
- **Compare**: overlay any trainer, generation, or eval metric across runs with
  one line per run, plus a latest-eval table.

Charts follow W&B line-plot conventions. Each panel has a quiet title with the
latest value, a hover toolbar (settings, table, expand), and opens full-size on
click. The gear holds the per-panel settings, remembered per chart:

- Smoothing: time-weighted EMA (default type), EMA, running average, or
  gaussian, with a 0 to 0.99 slider. The raw series stays visible as a faint
  line behind the smoothed one.
- Y range: fit data, from zero, or custom min and max. Bounded or non-negative
  families (rewards, rates, losses, lengths, times) default to from zero so the
  scale is not relative to the visible window.
- Log scale on x, on y, or both. Log axes always label their ticks, using 1-2-5
  mantissas across decades and linear ticks inside a single decade; axis labels
  share one precision so neighbouring ticks stay distinguishable.
- Drag across a chart to zoom the x range and double-click to reset.

Pages open on their primary signals; secondary detail sits behind collapsible
sections that remember their state.

### External trackers

When a run also logs to Weights & Biases or Trackio, the explorers offer those
histories as extra sources beside the local JSONL files. W&B is read through
the public API using the run id the monitor writes to `wandb_run_id.txt`, so
`monitor.wandb.enabled` must be true and W&B credentials must be available to
the dashboard process (`WANDB_API_KEY` or `~/.netrc`). Trackio is read from its
SQLite store under `TRACKIO_DIR` (default `~/.cache/huggingface/trackio`); the
run is matched by a `trackio_run.json` sidecar (`{"project", "run"}`) in the
run directory, else by the W&B run name, else by the directory name. Fetches
run in the background with a two-minute refresh and never block the local
sources. `GET /api/runs/{id}/external` reports each provider as unavailable,
loading, ready, or error with the reason.

## API

All routes are read-only `GET`s under `/api`:

| Route | Returns |
| --- | --- |
| `/api/health` | Liveness and the number of discovered runs |
| `/api/runs` | Summaries for every discovered run, current run first with `is_current: true` |
| `/api/current` | Id of the current run (or `null`) and the run count |
| `/api/runs/current/...` | Any run route below, resolved to the current run |
| `/api/runs/{id}/summary` | Status, heartbeat, config facts, latest metric rows, queue and policy summary, logs |
| `/api/runs/{id}/metrics/keys` | Numeric metric keys per source (`trainer`, `orchestrator`, `eval`) |
| `/api/runs/{id}/series?source=&keys=a,b&limit=&start=&end=&after=&points=` | Columnar step/timestamp/value series. `start`/`end` select a step window, `after` returns only rows past a step for incremental polling, and `points` caps the response by bucket-averaging with a `min`/`max` envelope |
| `/api/runs/{id}/nodes` | Latest per-node throughput and memory, the per-rank table from the heartbeat, and per-replica vLLM load and token rates |
| `/api/runs/{id}/evals` | Eval metrics grouped by environment plus eval rollout sets on disk |
| `/api/runs/{id}/evals/{step}/{env}/rows?sort=&order=&offset=&limit=&…` | Sorted, filtered eval rows with per-example aggregates |
| `/api/runs/{id}/evals/{step}/{env}/rows/{index}` | One raw eval row |
| `/api/runs/{id}/rollouts` | Rollout batches with manifest and queue status |
| `/api/runs/{id}/rollouts/rows?step=&sort=&order=&offset=&limit=&…` | Sorted, filtered compact rollout rows, stats, histograms, and group aggregates |
| `/api/runs/{id}/rollouts/{step}/rows/{index}` | One raw rollout row with token arrays summarized |
| `/api/runs/{id}/queue` | Queue report with items, policy snapshot, and rates |
| `/api/runs/{id}/timeline` | Per-step publish/claim/consume and per-policy export/load times |
| `/api/runs/{id}/events`, `/samples`, `/config`, `/logs`, `/logs/{name}?lines=` | Run events, trainer samples, redacted config, log listing and tails |

### Memory and disk

The server writes nothing under a run. Metric files are held as one numeric
column per key (8 bytes per value) and refreshed by reading only appended
bytes; event and sample files keep the newest 50,000 rows; compact rollout rows
are cached for at most 16 batch files; idle run readers are evicted beyond
eight runs (live runs are never evicted). Chart requests should pass `points`
so a 100,000-step run returns the same payload size as a 1,000-step run, and
`after` to fetch only new rows while following a live run. Zoom into a step
window with `start` and `end` to get the raw rows back.

Rollout filters: `env`, `group_key`, `example_id`, `min_reward`, `max_reward`,
`truncated`, `stop_condition`, `advantage` (`positive`, `negative`, `zero`,
`nonzero`), `has_error`, `search`. Large token arrays are never returned in
row listings; the detail route summarizes them (length, true count, min, max,
mean, sum).

Rollout and eval row filtering scans at most the first 50,000 physical JSONL
rows per file to keep memory and response times bounded. Responses include
`scanned` and `scan_limited`; the UI warns when matching results may exist past
that scan window.
