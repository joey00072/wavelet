from __future__ import annotations

import asyncio

import pytest

from wavelet.data.rl import RLExample
from wavelet.orchestrator.sources import CustomSource, NativeSource, VerifierSource


SOURCE_TYPES = (NativeSource, VerifierSource, CustomSource)


def _example() -> RLExample:
    return RLExample(
        prompt=[{"role": "user", "content": "prompt"}],
        completion=[],
        advantage=None,
        reward=None,
    )


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_rollout_sources_preserve_sync_group_contract(source_type) -> None:
    example = _example()
    calls: list[tuple[RLExample, int, dict[str, object]]] = []

    def rollout(
        value: RLExample, n: int, sampling: dict[str, object]
    ) -> list[RLExample]:
        calls.append((value, n, sampling))
        return [value] * n

    source = source_type(rollout)
    sampling = {"temperature": 0.7}

    result = asyncio.run(source.rollout_group(example, 2, sampling))

    assert result == [example, example]
    assert calls == [(example, 2, sampling)]


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_rollout_sources_preserve_async_group_contract(source_type) -> None:
    example = _example()

    async def rollout(
        value: RLExample, n: int, sampling: dict[str, object]
    ) -> list[RLExample]:
        assert n == 1
        assert sampling == {"seed": 3}
        return [value]

    source = source_type(rollout)

    assert asyncio.run(source.rollout_group(example, 1, {"seed": 3})) == [example]


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_rollout_sources_propagate_failures(source_type) -> None:
    def fail(
        _example: RLExample, _n: int, _sampling: dict[str, object]
    ) -> list[RLExample]:
        raise RuntimeError("rollout failed")

    source = source_type(fail)

    with pytest.raises(RuntimeError, match="rollout failed"):
        asyncio.run(source.rollout_group(_example(), 1, {}))
