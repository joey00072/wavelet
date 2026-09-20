from wavelet.configs.config import RLDataConfig
from wavelet.data.rl import RLExample, deserialize_rl_record, serialize_rl_record


def test_rollout_record_round_trip_preserves_training_fields():
    config = RLDataConfig(source="fake")
    record = RLExample(
        prompt=[{"role": "user", "content": "question"}],
        completion=[{"role": "assistant", "content": "answer"}],
        target_completion=[{"role": "assistant", "content": "target"}],
        advantage=0.5,
        reward=1.0,
        input_ids=[1, 2],
        target_ids=[2, 3],
        loss_mask=[False, True],
        inference_logprobs=[-0.2],
        teacher_logprobs=[-0.1],
        temperatures=[1.0],
        ce_weight=[0.25],
        ref_kl_weight=[0.75],
        sampling_mask=[[2, 3]],
        tools=[{"type": "function"}],
        chat_template_kwargs={"reasoning": True},
        metadata={"group_key": "group"},
        source="environment",
    )

    payload = serialize_rl_record(record, config, task="reward", example_id="id")
    restored = deserialize_rl_record(payload, config)

    assert restored == record


def test_binary_payload_matches_json_training_and_preserves_metadata(tmp_path):
    import json

    import msgpack

    from wavelet.data.rl import count_rollout_rows, load_rl_records, prepare_rl_sample

    config = RLDataConfig(source="local", metadata_column="details")
    record = RLExample(
        prompt=[{"role": "user", "content": "question"}],
        completion=[{"role": "assistant", "content": "answer"}],
        advantage=[0.5],
        reward=1.0,
        input_ids=[1, 2],
        target_ids=[2, 3],
        loss_mask=[False, True],
        inference_logprobs=[-0.2],
        teacher_logprobs=[-0.1],
        temperatures=[0.7],
        ce_weight=[0.25],
        ref_kl_weight=[0.75],
        sampling_mask=[[2, 3]],
        metadata={
            "verifier_example": {"task": "large task"},
            "_wavelet_rollout_count": 2,
            "policy_step": 7,
        },
    )
    full = serialize_rl_record(record, config, task="reward", example_id="id")
    compact = serialize_rl_record(
        record, config, task="reward", example_id="id", for_training=True
    )
    paths = [tmp_path / "full.jsonl", tmp_path / "train.msgpack"]
    paths[0].write_text(json.dumps(full) + "\n")
    paths[1].write_bytes(msgpack.packb([compact], use_bin_type=True))
    loaded = [
        load_rl_records(config.model_copy(update={"path": path}))[0] for path in paths
    ]
    assert "verifier_example" in loaded[0].metadata
    assert "verifier_example" not in loaded[1].metadata
    assert record.metadata == full["details"]
    assert count_rollout_rows(paths[1]) == 1
    assert prepare_rl_sample(loaded[0], None, config, 8) == prepare_rl_sample(
        loaded[1], None, config, 8
    )


def test_binary_payload_rejects_truncation_and_wrong_root(tmp_path):
    import msgpack
    import pytest

    from wavelet.data.rl import count_rollout_rows, load_rl_records

    path = tmp_path / "bad.msgpack"
    config = RLDataConfig(source="local", path=path)
    path.write_bytes(msgpack.packb({"not": "an array"}))
    with pytest.raises(ValueError, match="MessagePack list"):
        load_rl_records(config)
    path.write_bytes(msgpack.packb([{"prompt": [], "completion": []}])[:-1])
    with pytest.raises(ValueError):
        load_rl_records(config)
    path.write_bytes(msgpack.packb([]))
    with pytest.raises(ValueError, match="no rows"):
        count_rollout_rows(path)
