# Metrics

Wavelet has one local metrics contract for training, rollout orchestration, and
evaluation. Every producer appends flat rows to the run directory's
`metrics.jsonl` and `metrics.csv` files. A row always includes:

- `timestamp`: UTC ISO-8601 write time.
- `step`: the producer's current step.
- `subsystem`: `trainer`, `orchestrator`, or `eval`.

The remaining fields are numeric measurements using the existing Wavelet names,
such as `loss`, `reward/all/mean`, or `eval/alphabet/pass@1`. Writes from the
separate processes are serialized with `metrics.lock` so a shared run directory
does not produce interleaved JSON or mismatched CSV headers. Rows from older
runs without `subsystem` are interpreted as trainer rows.

Metric persistence follows `monitor.enabled`, `monitor.write_metrics_jsonl`, and
`monitor.write_metrics_csv`. Local files do not require WandB; existing optional
WandB reporting remains unchanged.

## JSON API

Enable `orchestrator.state_server` and query its existing metrics endpoint:

```text
GET /metrics?format=json&limit=20
GET /metrics?format=json&subsystem=trainer&limit=100
GET /metrics?format=json&subsystem=orchestrator&limit=20
GET /metrics?format=json&subsystem=eval&limit=20
```

`format=json` is the default and can be omitted. Without `subsystem`, the API
returns the latest rows across all producers in journal order. Filtering happens
before the limit is applied.

## Prometheus and Grafana

The same endpoint exposes the latest finite numeric value for every metric:

```text
GET /metrics?format=prometheus
GET /metrics?format=prometheus&subsystem=trainer
```

Configure Prometheus to scrape the state server with the query parameter:

```yaml
scrape_configs:
  - job_name: wavelet
    metrics_path: /metrics
    params:
      format: [prometheus]
    static_configs:
      - targets: ["127.0.0.1:8765"]
```

Every measurement uses the `wavelet_metric` gauge with `subsystem` and `name`
labels. For example:

```promql
wavelet_metric{subsystem="trainer", name="loss"}
wavelet_metric{subsystem="orchestrator", name="reward/all/mean"}
wavelet_metric{subsystem="eval", name="eval/alphabet/pass@1"}
```

Grafana can use that Prometheus data source directly. The endpoint also exposes
`wavelet_metric_timestamp_seconds{subsystem="..."}` so dashboards and alerts can
detect a stalled producer.

Older `orchestrator_metrics.jsonl`, `orchestrator_metrics.csv`, and
`eval_metrics.jsonl` files are no longer written. They may remain in resumed run
directories, and the algorithm state API retains a read-only fallback for old
orchestrator observations.
