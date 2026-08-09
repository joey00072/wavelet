# Wavelet RL WebUI

Lightweight dashboard for the Wavelet orchestrator state server.

## Run

```bash
bun install
bun run dev --host 0.0.0.0
```

Open the Vite URL from your workstation. If the state server is on another
machine, pass the API URL in the page query:

```text
http://<webui-host>:5173/?api=http://<state-server-host>:8765
```

The API base is also editable in the page header and is stored in local storage.

## Views

- `Overview`: run health, trainer progress, queue throughput, policy state,
  latest trainer metrics, and the resolved algorithm mix. Mixed runs show each
  source's algorithm, teacher, configured or observed share, reward, trainable
  rate, and reference-logprob coverage.
- `Rollouts`: rollout inspection, recent rollout buffers, saved snapshots, and
  per-lane queue detail. Inspected samples identify their source and active RL,
  CE, or REF-KL loss streams.

The rollout view polls in the background but does not require the reader to
follow the newest sample. Use `Pause Reading` to freeze the current inspection,
select any buffered batch to read it, or `Save Snapshot` to keep a specific
inspection in the page while training continues.

## Backend

The RL state server must be enabled in the RL config:

```yaml
orchestrator:
  state_server:
    enabled: true
    host: 0.0.0.0
    port: 8765
```

The server exposes read-only endpoints and allows browser CORS by default.

## Algorithm API

`GET /algorithms` returns the resolved default algorithm and every effective
training source without importing user-owned custom algorithm files. Teacher
API-key environment variable names and values are omitted. The same payload is
embedded as `algorithms` in `GET /state`, allowing the UI to poll run and
algorithm state atomically.

The response includes the single student model and adapter count, unique
teacher count, active loss components, source weights, teacher endpoint replica
counts, and the latest source-local observations from
`orchestrator_metrics.jsonl`. Observation values are `null` until the first
rollout batch is published.

## Rollout Inspection API

The WebUI uses a read-only endpoint that inspects stable rollout batch files
without claiming, consuming, deleting, or rewriting queue state:

```text
GET /rollouts/inspect
GET /rollouts/inspect?step=12&random_count=3&seed=42&max_scan_rows=5000
```

When `step` is omitted, the latest stable rollout batch is inspected. The
response includes bounded reward and advantage stats, random compact samples,
and min, max, and near-mean reward examples. Large training-only arrays such as
token ids, masks, and logprobs are omitted from samples. Compact samples retain
only loss-component names and whether aligned reference logprobs were present.
