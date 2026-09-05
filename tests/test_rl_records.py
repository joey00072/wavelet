from wavelet.configs.rl_config import RLDataConfig
from wavelet.data.rl_records import deserialize_rl_record, serialize_rl_record
from wavelet.data.rl_types import RLExample


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
