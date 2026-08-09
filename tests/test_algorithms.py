from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from wavelet.configs.rl_config import (
    AlgorithmScope,
    CustomAlgorithmConfig,
    FrozenModelConfig,
    GRPOAlgorithmConfig,
    MaxRLAlgorithmConfig,
    OPDAlgorithmConfig,
    PassthroughAlgorithmConfig,
    RewardAlgorithmConfig,
    RLAlgorithmConfig,
    RLConfig,
)
from wavelet.data.rl_dataset import RLExample
from wavelet.orchestrator.algorithms import (
    BaseAlgorithm,
    GRPOAlgorithm,
    MaxRLAlgorithm,
    OPDAlgorithm,
    PassthroughAlgorithm,
    RewardAlgorithm,
    algorithm_config_for_source,
    algorithm_epsilon,
    algorithm_scope,
    build_algorithm,
    register_algorithm,
    score_algorithm_records,
    score_records_by_source,
    uses_group_advantages,
)
from wavelet.orchestrator.rollouts import RLOrchestrator

CUSTOM_ALGORITHM_FILE = Path(__file__).parent / "fixtures" / "custom_algorithm.py"


def _example(*, reward: float | None) -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": "prompt"}],
        completion=[{"role": "assistant", "content": "completion"}],
        advantage=None,
        reward=reward,
        source="test",
    )


def _custom_config(
    algorithm: str,
    *,
    scope: AlgorithmScope = "group",
    kwargs: dict[str, object] | None = None,
    epsilon: float = 1e-6,
) -> CustomAlgorithmConfig:
    return CustomAlgorithmConfig(
        file=CUSTOM_ALGORITHM_FILE,
        algorithm=algorithm,
        scope=scope,
        kwargs=kwargs or {},
        epsilon=epsilon,
    )


def test_grpo_algorithm_scores_group_without_mutating_records() -> None:
    records = [_example(reward=1.0), _example(reward=0.0)]

    scored = GRPOAlgorithm().score_group(records)

    assert [record.advantage for record in scored] == pytest.approx([0.5, -0.5])
    assert [record.advantage for record in records] == [None, None]
    assert all(original is not updated for original, updated in zip(records, scored))


def test_grpo_algorithm_preserves_input_order() -> None:
    records = [
        replace(_example(reward=0.0), source="first"),
        replace(_example(reward=2.0), source="second"),
        replace(_example(reward=1.0), source="third"),
    ]

    scored = GRPOAlgorithm().score_group(records)

    assert [record.source for record in scored] == ["first", "second", "third"]
    assert [record.advantage for record in scored] == pytest.approx([-1.0, 1.0, 0.0])


def test_grpo_algorithm_requires_every_reward() -> None:
    records = [_example(reward=1.0), _example(reward=None)]

    with pytest.raises(
        ValueError,
        match=r"GRPO requires a reward for every rollout; missing at index\(es\): 1",
    ):
        GRPOAlgorithm().score_group(records)


def test_max_rl_algorithm_mean_normalizes_rewards() -> None:
    records = [_example(reward=1.0), _example(reward=0.0)]

    scored = MaxRLAlgorithm().score_group(records)

    assert [record.advantage for record in scored] == pytest.approx([1.0, -1.0])


def test_max_rl_algorithm_zeroes_groups_without_success() -> None:
    records = [_example(reward=0.0), _example(reward=0.0)]

    scored = MaxRLAlgorithm().score_group(records)

    assert [record.advantage for record in scored] == pytest.approx([0.0, 0.0])


class _FakeReferenceScorer:
    def __init__(self) -> None:
        self.token_ids: list[int] | None = None

    def score(self, token_ids: list[int]) -> list[float]:
        self.token_ids = token_ids
        return [0.0, -0.1, -0.2, -0.3]


def test_opd_scores_shifted_tokens_and_selects_trainable_logprobs() -> None:
    config = OPDAlgorithmConfig(
        teacher=FrozenModelConfig(
            name="teacher",
            base_url="http://teacher:8000/v1",
        )
    )
    scorer = _FakeReferenceScorer()
    algorithm = OPDAlgorithm(config, scorer=scorer)
    record = replace(
        _example(reward=None),
        input_ids=[10, 11, 12],
        target_ids=[11, 12, 13],
        loss_mask=[False, True, True],
    )

    scored = algorithm.score_rollout(record)

    assert scorer.token_ids == [10, 11, 12, 13]
    assert scored.ref_logprobs == pytest.approx([-0.2, -0.3])
    assert scored.advantage is None
    assert scored.rl_weights == 0.0
    assert scored.ref_kl_weights == 1.0


def test_opd_rejects_unshifted_token_streams() -> None:
    config = OPDAlgorithmConfig(
        teacher=FrozenModelConfig(name="teacher", base_url="http://teacher:8000")
    )
    record = replace(
        _example(reward=None),
        input_ids=[10, 99],
        target_ids=[11, 12],
        loss_mask=[True, True],
    )

    with pytest.raises(ValueError, match="causal shifted"):
        OPDAlgorithm(config, scorer=_FakeReferenceScorer()).score_rollout(record)


@pytest.mark.parametrize(
    ("config", "expected_type", "expected_scope"),
    [
        (PassthroughAlgorithmConfig(), PassthroughAlgorithm, "rollout"),
        (RewardAlgorithmConfig(), RewardAlgorithm, "rollout"),
        (GRPOAlgorithmConfig(), GRPOAlgorithm, "group"),
        (MaxRLAlgorithmConfig(), MaxRLAlgorithm, "group"),
    ],
)
def test_build_algorithm_dispatches_named_config(
    config: RLAlgorithmConfig,
    expected_type: type[BaseAlgorithm],
    expected_scope: AlgorithmScope,
) -> None:
    algorithm = build_algorithm(config)

    assert isinstance(algorithm, expected_type)
    assert algorithm_scope(config) == expected_scope


def test_custom_algorithm_loads_external_class_with_kwargs() -> None:
    config = _custom_config(
        "multiplier",
        kwargs={"multiplier": 3.0},
        epsilon=1e-4,
    )

    algorithm = build_algorithm(config)
    scored = algorithm.score_group([_example(reward=2.0)])

    assert scored[0].advantage == pytest.approx(6.0)
    assert uses_group_advantages(config) is True
    assert algorithm_epsilon(config) == pytest.approx(1e-4)


def test_native_orchestrator_uses_external_algorithm() -> None:
    orchestrator = RLOrchestrator(
        RLConfig(
            algo={
                "file": CUSTOM_ALGORITHM_FILE,
                "algorithm": "multiplier",
                "scope": "group",
                "kwargs": {"multiplier": 2.0},
            }
        )
    )
    records = [replace(_example(reward=1.5), metadata={"group_key": "a"})]

    scored = orchestrator._assign_advantages(records)  # noqa: SLF001

    assert scored[0].advantage == pytest.approx(3.0)


def test_file_algorithm_config_infers_custom_type() -> None:
    config = RLConfig(
        algo={
            "file": CUSTOM_ALGORITHM_FILE,
            "algorithm": "multiplier",
            "scope": "group",
            "kwargs": {"multiplier": 1.0},
        }
    )

    assert isinstance(config.algo, CustomAlgorithmConfig)
    assert config.algo.type == "custom"


def test_source_local_algorithm_overrides_run_default() -> None:
    config = RLConfig(
        algo={"type": "grpo"},
        orchestrator={
            "train_sources": [
                {"name": "direct-reward", "algo": {"type": "reward"}},
            ]
        },
    )
    records = [
        replace(_example(reward=1.0), source="grpo"),
        replace(_example(reward=0.0), source="grpo"),
        replace(_example(reward=3.0), source="direct-reward"),
    ]

    scored = score_records_by_source(config, records)

    assert [record.advantage for record in scored] == pytest.approx([0.5, -0.5, 3.0])
    assert algorithm_config_for_source(config, "grpo").type == "grpo"
    assert algorithm_config_for_source(config, "direct-reward").type == "reward"


def test_opd_config_is_available_as_source_override() -> None:
    config = RLConfig(
        orchestrator={
            "train_sources": [
                {
                    "name": "math",
                    "algo": {
                        "type": "opd",
                        "teacher": {
                            "name": "math-teacher",
                            "base_url": "http://teacher:8001/v1",
                        },
                    },
                }
            ]
        }
    )

    algorithm_config = algorithm_config_for_source(config, "math")

    assert isinstance(algorithm_config, OPDAlgorithmConfig)
    assert algorithm_config.teacher.name == "math-teacher"


def test_train_source_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        RLConfig(
            orchestrator={
                "train_sources": [
                    {"name": "math"},
                    {"name": "math"},
                ]
            }
        )


def test_custom_algorithm_rejects_missing_file(tmp_path: Path) -> None:
    config = CustomAlgorithmConfig(
        file=tmp_path / "missing.py",
        algorithm="Algorithm",
        scope="group",
    )

    with pytest.raises(ValueError, match="file does not exist"):
        build_algorithm(config)


def test_custom_algorithm_resolves_relative_file_from_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(CUSTOM_ALGORITHM_FILE.parent)
    config = CustomAlgorithmConfig(
        file=CUSTOM_ALGORITHM_FILE.name,
        algorithm="UndecoratedAlgorithm",
        scope="group",
    )

    algorithm = build_algorithm(config)

    assert isinstance(algorithm, BaseAlgorithm)


def test_custom_algorithm_requires_both_hooks() -> None:
    config = _custom_config("missing_group_hook", scope="rollout")

    with pytest.raises(TypeError, match="score_group"):
        build_algorithm(config)


def test_custom_algorithm_supports_undecorated_class_name() -> None:
    algorithm = build_algorithm(_custom_config("UndecoratedAlgorithm"))

    assert algorithm.score_group([_example(reward=1.0)])[0].advantage is None


def test_custom_algorithm_supports_registered_factory() -> None:
    config = _custom_config(
        "offset_factory",
        scope="rollout",
        kwargs={"offset": 2.0},
    )

    scored = score_algorithm_records(
        build_algorithm(config),
        [_example(reward=1.0)],
        scope=config.scope,
    )

    assert scored[0].advantage == pytest.approx(3.0)


def test_custom_file_algorithm_selects_trainer_loss_component() -> None:
    config = _custom_config("ce_actions", scope="rollout")

    scored = score_algorithm_records(
        build_algorithm(config),
        [_example(reward=None)],
        scope=config.scope,
    )

    assert scored[0].rl_weights == 0.0
    assert scored[0].ce_weights == 1.0
    assert scored[0].ref_kl_weights == 0.0


def test_custom_algorithm_rejects_duplicate_registration() -> None:
    with pytest.raises(ValueError, match="registered more than once"):
        build_algorithm(_custom_config("duplicate"))


def test_custom_algorithm_rejects_non_callable_name() -> None:
    with pytest.raises(TypeError, match="is not callable"):
        build_algorithm(_custom_config("NOT_CALLABLE"))


def test_custom_rollout_hook_must_return_example() -> None:
    config = _custom_config("invalid_rollout_return", scope="rollout")

    with pytest.raises(TypeError, match="score_rollout must return RLExample"):
        score_algorithm_records(
            build_algorithm(config),
            [_example(reward=1.0)],
            scope=config.scope,
        )


def test_custom_group_hook_must_preserve_group_length() -> None:
    config = _custom_config("short_group_return")

    with pytest.raises(ValueError, match="one RLExample for every input record"):
        score_algorithm_records(
            build_algorithm(config),
            [_example(reward=1.0), _example(reward=0.0)],
            scope=config.scope,
        )


@pytest.mark.parametrize(
    ("scope", "expected_advantage"),
    [("rollout", 1.0), ("group", 1.0), ("both", 2.0)],
)
def test_custom_scope_controls_hook_execution(
    scope: AlgorithmScope,
    expected_advantage: float,
) -> None:
    config = _custom_config("both_hooks", scope=scope)

    scored = score_algorithm_records(
        build_algorithm(config),
        [_example(reward=1.0)],
        scope=config.scope,
    )

    assert scored[0].advantage == pytest.approx(expected_advantage)


def test_group_scoring_preserves_interleaved_record_order() -> None:
    records = [
        replace(_example(reward=1.0), metadata={"group_key": "a"}),
        replace(_example(reward=10.0), metadata={"group_key": "b"}),
        replace(_example(reward=0.0), metadata={"group_key": "a"}),
        replace(_example(reward=0.0), metadata={"group_key": "b"}),
    ]

    scored = score_algorithm_records(
        GRPOAlgorithm(),
        records,
        scope="group",
        group_key=lambda record: str((record.metadata or {})["group_key"]),
    )

    assert [record.advantage for record in scored] == pytest.approx(
        [0.5, 5.0, -0.5, -5.0]
    )


def test_group_scope_preserves_fully_pre_scored_records() -> None:
    records = [replace(_example(reward=None), advantage=3.0)]

    scored = score_algorithm_records(
        GRPOAlgorithm(),
        records,
        scope="group",
    )

    assert scored == records


def test_external_algorithm_action_loss_type_selects_ce_stream() -> None:
    class CEAlgorithm(BaseAlgorithm):
        action_loss_type = "ce"

    scored = score_algorithm_records(
        CEAlgorithm(),
        [_example(reward=None)],
        scope="rollout",
    )

    assert scored[0].rl_weights == 0.0
    assert scored[0].ce_weights == 1.0
    assert scored[0].ref_kl_weights == 0.0


def test_external_algorithm_rejects_unknown_action_loss_type() -> None:
    class InvalidAlgorithm(BaseAlgorithm):
        action_loss_type = "unknown"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="action_loss_type"):
        score_algorithm_records(
            InvalidAlgorithm(),
            [_example(reward=1.0)],
            scope="rollout",
        )


def test_external_algorithm_runs_setup_and_close_lifecycle() -> None:
    class LifecycleAlgorithm(BaseAlgorithm):
        def __init__(self) -> None:
            self.events: list[str] = []

        def setup(self) -> None:
            self.events.append("setup")

        def score_rollout(self, record: RLExample) -> RLExample:
            self.events.append("score")
            return record

        def close(self) -> None:
            self.events.append("close")

    algorithm = LifecycleAlgorithm()

    score_algorithm_records(algorithm, [_example(reward=1.0)], scope="rollout")

    assert algorithm.events == ["setup", "score", "close"]


@pytest.mark.parametrize("name", ["", "  ", " surrounded "])
def test_registration_name_must_be_clean(name: str) -> None:
    with pytest.raises(ValueError, match="registration name"):
        register_algorithm(name)
