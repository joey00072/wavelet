# Functionality Register

This register is the preservation contract for Wavelet's consolidation work.
Run commands from the repository root with the same seed and a clean output
directory. GPU-dependent rows are opt-in CI or release checks; CPU-safe checks
run in the ordinary test suite.

| Surface | Preservation check |
|---|---|
| All example and legacy configs | `uv run pytest tests/test_example_configs.py tests/test_rl_config.py tests/test_sft_config.py` |
| CLI and debug command shapes | `uv run pytest tests/test_cli.py tests/test_preflight.py` |
| Filesystem queue lifecycle and artifacts | `uv run pytest tests/test_queue.py tests/test_queue_lifecycle.py tests/test_queue_inspect.py` |
| Rollout scheduling, freshness, and chunking | `uv run pytest tests/test_scheduler.py tests/test_verifiers_rollouts.py` |
| Weighted multi-environment scheduling and resume cursors | `uv run pytest tests/test_rl_config.py tests/test_verifiers_rollouts.py tests/test_preflight.py tests/test_orchestrator_metrics.py` |
| Curriculum sampling, admission gates, and state restore | `uv run pytest tests/test_curriculum.py tests/test_verifiers_rollouts.py tests/test_queue.py` |
| Rewards and advantage algorithms | `uv run pytest tests/test_reward.py tests/test_algorithms.py` |
| OPD, OPSD, and online SFT distillation | `uv run pytest tests/test_distillation.py tests/test_algorithms.py tests/test_rl_loss.py` |
| Agent trajectory token provenance | `uv run pytest tests/test_agent_trajectory.py` |
| RL records, packing, and tokenization | `uv run pytest tests/test_rl_dataset.py tests/test_tokenization_alignment.py` |
| Component RL/CE/ref-KL losses and trainer stepping | `uv run pytest tests/test_rl_loss.py tests/test_rl_trainer.py` |
| End-to-end reverse-text SFT and checkpoint resume | `uv run pytest tests/integration/test_reverse_text_sft.py -q` |
| LoRA, policy metadata, and policy loading | `uv run pytest tests/test_trainer_lora.py tests/test_policy_metadata.py tests/test_rl_inference_policy.py` |
| HTTP, offline, and native inference | `uv run pytest tests/test_http_inference.py tests/test_native_inference_server.py tests/test_vllm_batching.py` |
| Diagnostics, metrics, monitoring, state server | `uv run pytest tests/test_monitor.py tests/test_monitoring.py tests/test_inference_diagnostics.py tests/test_orchestrator_diagnostics.py tests/test_trainer_diagnostics.py tests/test_orchestrator_metrics.py tests/test_state_server.py` |
| Launcher modes and placement | `uv run pytest tests/test_launcher.py tests/test_rl_launcher.py` |
| MoE router metrics and expert parallelism | `uv run pytest tests/test_moe.py tests/integration/test_expert_parallel.py -q` |
| Context parallel loss scaling and sampling-mask sharding | `uv run pytest tests/test_context_parallel.py` |
| vLLM sampling-mask replay in RL | `uv run pytest tests/test_vllm_batching.py tests/test_rl_records.py tests/test_rl_trainer.py` |
| Adaptive inference concurrency | `uv run pytest tests/test_adaptive_concurrency.py` |
| Checkpointing, FSDP2 checkpoints, and checkpoint conversion | `uv run pytest tests/test_checkpointing.py tests/test_convert_checkpoint.py tests/integration/test_fsdp2_checkpoint.py -q` |
| Trace export and benchmark harness | `uv run pytest tests/test_convert_traces.py tests/test_benchmark.py` |
| Reverse-text SFT | `uv run python -m wavelet sft @ examples/reverse_text/sft.yaml` |
| Reverse-text integrated RL | `uv run python -m wavelet rl @ examples/reverse_text/rl.yaml` |
| Process-mode roles | `uv run python -m wavelet debug preflight @ examples/reverse_text/rl.yaml --json` followed by the resolved `rl-inference`, `rl-trainer`, and `rl-orchestrator` commands |
| Alphabet-sort LoRA RL | `uv run python -m wavelet rl @ examples/alphabet_sort/rl.yaml` |
| OpenAI-compatible vLLM server | `uv run python -m wavelet inference-server @ examples/alphabet_sort/rl.yaml` and run the configured inference probe |
| Native inference server | `uv run python -m wavelet native-inference-server @ examples/alphabet_sort/rl.yaml` and verify status, memory, policy, and inference routes |
| Offline inference engine | Set `server_backend: offline` in a temporary copy of an example config and run `rl-inference` |
| Filesystem policy transport | Run reverse-text RL and verify adjacent policy metadata plus adapter load |
| NCCL policy transport | Run the dedicated two-GPU configuration and `uv run pytest tests/test_vllm_weight_update.py` |
| Colocate and sleep choreography | Run the colocate and colocate-sleep smoke configs under `examples/qwen30b_math/` after checking GPU availability |
| Ray launcher backend | Set `launcher.backend: ray` in a temporary process-mode config and run preflight plus the resolved launcher |
| Dashboard API and artifact readers | `uv run pytest tests/test_dashboard.py` |
| Per-node trainer telemetry and heartbeat rank table | `uv run pytest tests/test_telemetry.py tests/test_monitoring.py` |
| Web UI compatibility | `uv run wavelet synth-run --output outputs/demo_run`, `uv run wavelet dashboard --runs-root outputs`, then verify the current-run landing, older-runs list, overview, inspector, evals, pipeline, infra, config, and compare views; repeat against a live state server for a smoke run |

For training parity, record baseline evaluation and final evaluation at fixed
policy steps, failed-rollout counts, queue lifecycle counts, and policy lag.
Training-batch reward alone is not a parity signal.

## Canonical Implementation Owners

Compatibility modules preserve established imports, but new code should use
these implementation owners:

| Contract | Canonical owner |
|---|---|
| SFT and RL data pipelines | `wavelet.data.sft`, `wavelet.data.rl` |
| Rollout scheduling and source selection | `wavelet.orchestrator.scheduler`, `wavelet.orchestrator.sources` |
| Verifier clients, environments, and evaluation | `wavelet.orchestrator.envs` |
| Queue artifacts, lifecycle, metrics, and inspection | `wavelet.transport.queue` |
| Filesystem and NCCL policy transfer | `wavelet.transport.policy` |
| Model loading, LoRA, and QLoRA | `wavelet.trainer.model` |
| Distributed world and device meshes | `wavelet.trainer.distributed` |
| Shared monitoring and JSONL readers | `wavelet.monitor` |
| Read-only run artifact readers and the `/api/runs` dashboard router | `wavelet.dashboard.artifacts`, `wavelet.dashboard.server` |
