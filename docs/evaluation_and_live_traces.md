# Resumable evaluation and live episodes

Evaluation writes one atomic episode file per example/repetition under
`evals/step-<step>/journals/<environment>/`. Set `eval.resume: true` to reuse
successful episodes after interruption. The default is false: an existing
journal requires an explicit resume or a fresh output directory. Failed
responses are retried. Cancellation drains pending requests before releasing
the single-writer journal lock.

The journal checks the environment and effective environment arguments, model,
policy step, endpoints, and sampling settings. Example content is part of the
cache key; adding examples or increasing group size only generates missing
entries. Corrupt committed files fail explicitly; incomplete temporary files
are ignored. Resumed throughput uses cumulative attempt time.

Set `eval.model_revision` to an immutable model/checkpoint identifier when
serving externally managed weights. Endpoint identity alone cannot detect
weights replaced behind the same URL. Never resume against changed weights
under the same revision.

`eval.live_traces` and `orchestrator.live_traces` default to true. Actual verifier
hooks record setup, model requests, environment responses, scoring and cleanup
under `traces/live/`. Group-scored rollouts have individual child episodes
linked to a group episode. Prompts appear while a request is pending; terminal
states distinguish success, errors and cancellation. The dashboard's Traces
view polls small metadata files and fetches the selected episode when its
revision changes.

Snapshots are limited to 256 KiB and 128 events, with bounded depth and item
counts. Publication failures do not fail rollouts. Terminal cleanup retains
256 episodes per bounded scan; active episodes remain available. On the local
host, dead writers are displayed as abandoned using process identity. Shared
filesystem readers on another host cannot determine whether a writer has died.
The reader rejects symlinks and nonregular files and confines access to the run
directory. Traces may contain task data; keep run directories appropriately
restricted.

Async verifier hooks and episode lifecycle writes run filesystem publication
and retention work in worker threads. They await completion without blocking
the rollout event loop. Cancellation waits for an already-started write before
publishing terminal state, so it cannot race a stale snapshot into the same file.
Synchronous callers retain synchronous publication. On shared filesystems,
`eval.live_traces: false` and `orchestrator.live_traces: false` provide a useful
performance control; completed rollout artifacts and metrics remain available.

### Reward populations and step labels

The dashboard distinguishes episode-weighted rewards from filtered training
rewards. New orchestrator logs publish `reward/episodes/all/{mean,count}` and
`reward/episodes/effective/{mean,count}`. Counts are the denominators of those
means, using rollout ownership metadata so continuation branches do not count
again. The effective subset requires training tokens and excludes filtered,
dummy, and errored rows. Missing rewards do not enter either denominator; an
empty subset has count zero and no mean. These describe the published batch,
not every attempted or rejected candidate in the generation pipeline.

`generation/reward/mean` covers all scored candidates, including groups rejected
before publication. Historical orchestrator `reward/all/mean` is an average of
per-problem means; trainer `reward/all/mean` is episode-weighted and includes
filtered episodes. They coincide for equally sized complete groups, but need
not coincide for unequal groups. A filtered reward is not an overall solve rate
and should not be used by itself to claim learning improvement.

Charts retain the recorded step values. The orchestrator starts with rollout
queue step 0; the trainer records optimizer step 1 after consuming its first
batch. Chunked runs can have multiple queue steps per optimizer update, so the
UI does not shift queue indices by one. Historical runs keep their original
series and explicit labels; new cohort metrics are not fabricated retroactively.
