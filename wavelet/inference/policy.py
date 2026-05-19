from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

from wavelet.configs.rl_config import RLConfig
from wavelet.data.rl_dataset import RLExample


def create_policy_inference_engine(config: RLConfig) -> PolicyInferenceEngine:
    if config.inference.mode == "vllm_http":
        from wavelet.inference.http import HTTPPolicyInferenceEngine

        return HTTPPolicyInferenceEngine(config)
    if config.inference.mode == "passthrough":
        return PassthroughPolicyInferenceEngine(config)
    raise ValueError(
        f"Unsupported inference mode: {config.inference.mode}. "
        "Use inference.mode='vllm_http' for model serving."
    )


def token_ids(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if hasattr(value, "input_ids"):
        input_ids = getattr(value, "input_ids")
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [int(item) for item in input_ids[0]]
            return [int(item) for item in input_ids]
    raise TypeError(f"Unsupported tokenized value: {type(value)!r}")


class PolicyInferenceEngine(ABC):
    def __init__(self, config: RLConfig) -> None:
        self.config = config
        self.policy_step: int | None = None

    @abstractmethod
    def setup(self) -> None:
        """Prepare resources needed for rollout inference."""

    @abstractmethod
    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        """Load a trainer-exported policy snapshot."""

    @abstractmethod
    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        """Generate or annotate rollout records."""

    def close(self) -> None:
        """Release inference resources."""

    def sleep(self) -> None:
        """Release inference-side GPU memory when supported."""

    def wake(self, *, tags: list[str] | None = None) -> None:
        """Restore inference-side GPU memory when supported."""
        del tags


class RLInference:
    def __init__(self, config: RLConfig) -> None:
        self.config = config

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        if not self.config.inference.enabled:
            return records
        if self.config.inference.mode == "passthrough":
            return self.annotate_passthrough(records)
        raise ValueError(
            "Standalone RLInference only supports inference.mode='passthrough'. "
            "Use create_policy_inference_engine() for vLLM rollouts."
        )

    def annotate_passthrough(self, records: list[RLExample]) -> list[RLExample]:
        annotated: list[RLExample] = []
        for record in records:
            if record.temperatures is None:
                temperatures: float | list[float] | None = (
                    self.config.inference.default_temperature
                )
            else:
                temperatures = record.temperatures
            annotated.append(replace(record, temperatures=temperatures))
        return annotated

    def prompt_token_ids(self, tokenizer, record: RLExample) -> list[int]:
        kwargs: dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if record.tools is not None:
            kwargs["tools"] = record.tools
        if record.chat_template_kwargs is not None:
            kwargs.update(record.chat_template_kwargs)
        return token_ids(tokenizer.apply_chat_template(record.prompt, **kwargs))


class PassthroughPolicyInferenceEngine(PolicyInferenceEngine):
    def setup(self) -> None:
        return None

    def load_policy(self, policy_dir: Path, *, step: int) -> None:
        del policy_dir
        self.policy_step = step

    def annotate(self, records: list[RLExample]) -> list[RLExample]:
        return RLInference(self.config).annotate_passthrough(records)
