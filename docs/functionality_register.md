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
| Rewards and advantage algorithms | `uv run pytest tests/test_reward.py tests/test_algorithms.py` |
| Agent trajectory token provenance | `uv run pytest tests/test_agent_trajectory.py` |
| RL records, packing, and tokenization | `uv run pytest tests/test_rl_dataset.py tests/test_tokenization_alignment.py` |
| Component RL/CE/ref-KL losses and trainer stepping | `uv run pytest tests/test_rl_loss.py tests/test_rl_trainer.py` |
| LoRA, policy metadata, and policy loading | `uv run pytest tests/test_trainer_lora.py tests/test_policy_metadata.py tests/test_rl_inference_policy.py` |
| HTTP, offline, and native inference | `uv run pytest tests/test_http_inference.py tests/test_native_inference_server.py tests/test_vllm_batching.py` |
| Diagnostics, metrics, monitoring, state server | `uv run pytest tests/test_monitor.py tests/test_monitoring.py tests/test_inference_diagnostics.py tests/test_orchestrator_diagnostics.py tests/test_trainer_diagnostics.py tests/test_orchestrator_metrics.py tests/test_state_server.py` |
| Launcher modes and placement | `uv run pytest tests/test_launcher.py tests/test_rl_launcher.py` |
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
| Web UI compatibility | Start the state server for a smoke run and verify the overview, queues, policies, metrics, and trace views |

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
