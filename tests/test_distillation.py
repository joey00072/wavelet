import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl import RLExample
from wavelet.inference import server
from wavelet.inference.server import _scored_prompt_logprobs
from wavelet.orchestrator import envs


def _config(**values) -> RLConfig:
    values.setdefault(
        "orchestrator",
        {
            "custom_rollout_function": (
                "wavelet.orchestrator.verifiers:generate_rollouts"
            ),
            "verifier_env_id": "test-env",
        },
    )
    return RLConfig(**values)


def _record() -> RLExample:
    return RLExample(
        prompt=[],
        completion=[],
        advantage=None,
        reward=1.0,
        input_ids=[10, 11],
        target_ids=[11, 12],
        loss_mask=[False, True],
        inference_logprobs=[-0.5],
        temperatures=[1.0],
        ref_kl_weight=1.0,
        metadata={"verifier_example": {"demonstration": "expert answer"}},
    )


def test_opd_prefill_scoring_selects_only_trainable_target_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[int]] = []

    def score(_client, token_ids: list[int]) -> list[float]:
        seen.append(token_ids)
        return [0.0, -1.0, -2.0]

    monkeypatch.setattr(envs.PrefillScoringClient, "score", score)
    config = _config(
        algo={"type": "opd"},
        teacher={"model": "teacher", "base_url": "http://teacher:8000/v1"},
    )

    annotated = envs.annotate_distillation_records([_record()], config)

    assert seen == [[10, 11, 12]]
    assert annotated[0].teacher_logprobs == [-2.0]


def test_opsd_scores_policy_tokens_after_demo_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[int]] = []
    monkeypatch.setattr(
        envs,
        "_opsd_prefix_token_ids",
        lambda _record, _config, **_kwargs: [1, 2],
    )
    monkeypatch.setattr(
        "wavelet.trainer.model.setup_tokenizer",
        lambda _config: object(),
    )

    def score(_client, token_ids: list[int]) -> list[float]:
        seen.append(token_ids)
        return [0.0, -1.0, -2.0, -3.0, -4.0]

    monkeypatch.setattr(envs.PrefillScoringClient, "score", score)
    config = _config(algo={"type": "opsd"})

    annotated = envs.annotate_distillation_records(
        [_record()],
        config,
        policy_model_name="policy",
    )

    assert seen == [[1, 2, 10, 11, 12]]
    assert annotated[0].teacher_logprobs == [-4.0]


def test_sft_distillation_uses_teacher_as_rollout_source() -> None:
    config = _config(
        algo={"type": "sft"},
        teacher={"model": "teacher", "base_url": "http://teacher:8000"},
    )

    assert envs._verifier_base_urls(config) == ["http://teacher:8000/v1"]
    assert envs._verifier_model(config) == "teacher"
    assert envs._verifier_extra_env_kwargs(config)["timeout_seconds"] == 120.0


def test_sft_teacher_client_does_not_inherit_student_data_parallel_ranks() -> None:
    seen: list[dict[str, object]] = []

    class VF:
        @staticmethod
        def ClientConfig(**kwargs):
            seen.append(kwargs)
            return kwargs

    config = _config(
        algo={"type": "sft"},
        teacher={
            "model": "teacher",
            "base_url": "http://teacher:8000",
            "api_key_var": "TEACHER_API_KEY",
        },
        inference={"vllm": {"data_parallel_size": 4}},
    )

    clients = envs._verifier_clients(VF, config)

    assert len(clients) == 1
    assert seen[0]["api_base_url"] == "http://teacher:8000/v1"
    assert seen[0]["api_key_var"] == "TEACHER_API_KEY"
    assert seen[0]["extra_headers"] == {}


def test_server_flattens_prompt_logprobs_for_requested_tokens() -> None:
    rows = [
        None,
        {11: SimpleNamespace(logprob=-1.25)},
        {12: {"logprob": -2.5}},
    ]

    assert _scored_prompt_logprobs(rows, [10, 11, 12]) == [0.0, -1.25, -2.5]


def test_server_rejects_misaligned_prompt_logprobs() -> None:
    with pytest.raises(ValueError, match="do not align"):
        _scored_prompt_logprobs([None], [10, 11])


def test_server_score_route_uses_temperature_one_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Engine:
        def generate(self, prompt, params, request_id, *, lora_request):
            captured.update(
                prompt=prompt,
                params=params,
                request_id=request_id,
                lora_request=lora_request,
            )

            async def outputs():
                yield SimpleNamespace(
                    prompt_logprobs=[
                        None,
                        {11: SimpleNamespace(logprob=-1.0)},
                    ]
                )

            return outputs()

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                policy_adapter_name=None,
                policy_adapter_path=None,
            )
        )
    )
    monkeypatch.setattr(server, "_CONFIG", RLConfig(model={"name": "student"}))
    monkeypatch.setattr(server, "_engine_client", lambda _request: Engine())

    response = asyncio.run(
        server.score_prompt(
            {"model": "student", "token_ids": [10, 11]},
            request,
        )
    )

    assert response["prompt_logprobs"] == [0.0, -1.0]
    assert captured["prompt"] == {"prompt_token_ids": [10, 11]}
    assert captured["params"].temperature == pytest.approx(1.0)


def test_sft_algorithm_routes_converted_rollout_to_ce() -> None:
    output = {
        "example_id": 1,
        "reward": 1.0,
        "sampling_args": {"temperature": 1.0},
        "trajectory": [
            {
                "prompt": [],
                "completion": [],
                "tokens": {
                    "prompt_ids": [10],
                    "prompt_mask": [False],
                    "completion_ids": [11],
                    "completion_mask": [True],
                    "completion_logprobs": [-0.5],
                },
            }
        ],
    }
    config = _config(
        algo={"type": "sft"},
        teacher={"model": "teacher", "base_url": "http://teacher:8000"},
    )

    envs._assign_group_advantages([output], algorithm_config=config.algo)
    records = envs._records_from_output(output)

    assert len(records) == 1
    assert records[0].advantage is None
    assert records[0].ce_weight == pytest.approx(1.0)
    assert records[0].ref_kl_weight is None


@pytest.mark.parametrize("loss_component", ["ce", "ref_kl"])
def test_distillation_groups_do_not_require_scalar_advantages(
    loss_component: str,
) -> None:
    output = {
        "trajectory": [
            {
                "tokens": {
                    "completion_mask": [True],
                }
            }
        ]
    }

    assert envs._is_usable_training_group(
        [output],
        expected_rollouts=1,
        filter_zero_advantage=True,
        advantage_epsilon=1.0e-6,
        loss_component=loss_component,
    )


def test_opsd_requires_demonstration_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "wavelet.trainer.model.setup_tokenizer",
        lambda _config: object(),
    )

    with pytest.raises(ValueError, match="demonstration"):
        envs.annotate_distillation_records(
            [replace(_record(), metadata={})],
            _config(algo={"type": "opsd"}),
            policy_model_name="policy",
        )
