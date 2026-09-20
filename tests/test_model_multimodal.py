import torch
from torch import nn

from wavelet.configs.config import ModelConfig
from wavelet.trainer.model import (
    freeze_vision_encoder,
    multimodal_forward_kwargs,
    setup_processor,
)


def test_freeze_vision_encoder_dotted_path():
    model = nn.Module()
    model.model = nn.Module()
    model.model.visual = nn.Linear(2, 2)
    assert freeze_vision_encoder(model, "model.visual") == 6
    assert all(not p.requires_grad for p in model.model.visual.parameters())


def test_multimodal_kwargs_excludes_training_fields():
    batch = {
        "input_ids": torch.ones(1, 2),
        "pixel_values": torch.ones(1, 3, 2, 2),
        "mm_token_type_ids": torch.zeros(1, 2),
    }
    assert set(multimodal_forward_kwargs(batch)) == {
        "pixel_values",
        "mm_token_type_ids",
    }


def test_setup_processor_text_model_returns_none(monkeypatch):
    monkeypatch.setattr(
        "wavelet.trainer.model.AutoProcessor.from_pretrained", lambda *a, **k: object()
    )
    assert setup_processor(ModelConfig(name="text")) is None


def _tiny_vlm():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    v = Qwen3VLVisionConfig(
        hidden_size=16,
        num_heads=2,
        depth=1,
        intermediate_size=32,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    t = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [1, 1, 2],
        },
    )
    c = Qwen3VLConfig(
        text_config=t.to_dict(),
        vision_config=v.to_dict(),
        image_token_id=4,
        video_token_id=5,
        vision_start_token_id=6,
        vision_end_token_id=7,
    )
    c._attn_implementation = "eager"
    m = Qwen3VLForConditionalGeneration(c)
    p = Qwen2VLImageProcessor(
        patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=64, max_pixels=64
    )
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<eos>": 2,
        "user": 3,
        "<|image_pad|>": 4,
        "<|video_pad|>": 5,
        "<|vision_start|>": 6,
        "<|vision_end|>": 7,
        "assistant": 8,
        "red": 9,
        "describe": 10,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
        additional_special_tokens=[
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ],
    )
    template = "{% for message in messages %}{{message['role']}} {% if message['content'] is string %}{{message['content']}}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{content['text']}}{% endif %}{% endfor %}{% endif %} <eos> {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tok.chat_template = template
    vp = Qwen3VLVideoProcessor(
        patch_size=2, temporal_patch_size=2, merge_size=2, min_pixels=64, max_pixels=64
    )
    proc = Qwen3VLProcessor(
        image_processor=p, tokenizer=tok, video_processor=vp, chat_template=template
    )
    return m, proc, tok


def test_real_vlm_processor_training_and_checkpoint(tmp_path):
    from PIL import Image

    from wavelet.configs.config import LossMaskConfig
    from wavelet.data.sft import Example, build_sample, collate_batch
    from wavelet.trainer.model import setup_model

    model, processor, tokenizer = _tiny_vlm()
    record = Example(
        prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (8, 8), "red")},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        completion=[{"role": "assistant", "content": "red"}],
    )
    sample = build_sample(
        record,
        tokenizer,
        seq_len=64,
        loss_mask_config=LossMaskConfig(),
        processor=processor,
    )
    assert sample is not None
    assert sample["input_ids"].count(4) == 4
    batch = collate_batch([sample], pad_token_id=1)
    model.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    loaded = setup_model(
        ModelConfig(
            name=str(tmp_path),
            vlm={},
            torch_dtype="float32",
            attn_implementation="eager",
            activation_checkpointing=None,
        )
    )
    loaded_processor = setup_processor(ModelConfig(name=str(tmp_path), vlm={}))
    assert loaded_processor is not None
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        **multimodal_forward_kwargs(batch),
    }
    expected = model(**kwargs).logits
    logits = loaded(**kwargs).logits
    torch.testing.assert_close(logits, expected)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), batch["labels"].flatten()
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert loaded.lm_head.weight.grad is not None
    assert all(p.grad is None for p in loaded.model.visual.parameters())
    import pytest

    with pytest.raises(ValueError, match="cannot be truncated"):
        build_sample(
            record,
            tokenizer,
            seq_len=2,
            loss_mask_config=LossMaskConfig(),
            processor=processor,
        )


def test_real_vlm_sampled_rl_tokens_and_backward(tmp_path):
    from PIL import Image

    from wavelet.configs.config import LossMaskConfig, RLConfig, RLDataConfig
    from wavelet.data.rl import RLExample, collate_rl_batch, prepare_rl_sample
    from wavelet.data.sft import Example, build_sample
    from wavelet.trainer.distributed import World
    from wavelet.trainer.rl import RLTrainer

    model, processor, tokenizer = _tiny_vlm()
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (8, 8), "red")},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    completion = [{"role": "assistant", "content": "red"}]
    source = build_sample(
        Example(prompt=prompt, completion=completion),
        tokenizer,
        seq_len=64,
        loss_mask_config=LossMaskConfig(),
        processor=processor,
    )
    record = RLExample(
        prompt=prompt,
        completion=completion,
        advantage=1.0,
        reward=1.0,
        input_ids=source["input_ids"],
        target_ids=source["target_ids"],
        loss_mask=source["loss_mask"],
    )
    sample = prepare_rl_sample(
        record, tokenizer, RLDataConfig(seq_len=64), 64, processor
    )
    batch = collate_rl_batch([sample], pad_token_id=1)
    trainer = RLTrainer(
        RLConfig(output_dir=tmp_path, model={"vlm": {}, "torch_dtype": "float32"})
    )
    trainer.model = model
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    batch = trainer._prepare_batch(batch)
    logprobs = trainer._model_logprobs(batch, batch["attention_mask"])
    loss = -logprobs[batch["loss_mask"]].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    import pytest

    record.input_ids[0] = 31
    with pytest.raises(ValueError, match="do not match sampled"):
        prepare_rl_sample(record, tokenizer, RLDataConfig(seq_len=64), 64, processor)


def test_vlm_trainer_export_and_lora_freeze(tmp_path):
    from peft import PeftModel

    from wavelet.configs.config import SFTConfig
    from wavelet.trainer.distributed import World
    from wavelet.trainer.model import setup_model
    from wavelet.trainer.trainer import SFTTrainer

    model, processor, tokenizer = _tiny_vlm()
    base = tmp_path / "base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    config = SFTConfig(
        output_dir=tmp_path / "run",
        model={
            "name": str(base),
            "vlm": {},
            "torch_dtype": "float32",
            "attn_implementation": "eager",
            "activation_checkpointing": None,
        },
        lora={"rank": 2, "target_modules": ["qkv", "proj", "q_proj"]},
    )
    trainer = SFTTrainer(config)
    trainer.world = World(
        rank=0,
        local_rank=0,
        world_size=1,
        local_world_size=1,
        device=torch.device("cpu"),
    )
    trainer.tokenizer = tokenizer
    trainer.processor = processor
    trainer._setup_model()
    assert isinstance(trainer.model, PeftModel)
    vision = trainer.model.get_base_model().model.visual
    assert all(not p.requires_grad for p in vision.parameters())
    assert any(
        "lora" in name and p.requires_grad
        for name, p in trainer.model.named_parameters()
    )
    trainer._save_model()
    adapter = config.output_dir / "adapter"
    assert setup_processor(ModelConfig(name=str(adapter), vlm={})) is not None
    reloaded = setup_model(config.model.model_copy(update={"adapter_path": adapter}))
    assert all(
        not p.requires_grad for p in reloaded.get_base_model().model.visual.parameters()
    )


def test_video_temporal_metadata_is_concatenated_and_forwarded():
    from wavelet.data.multimodal import collate_multimodal_fields

    samples = [
        {
            "input_ids": [1, 2],
            "mm_kwargs": {"second_per_grid_ts": [0.5], "mm_token_type_ids": [2, 2]},
        },
        {
            "input_ids": [1, 2, 3],
            "mm_kwargs": {
                "second_per_grid_ts": [1.0, 2.0],
                "mm_token_type_ids": [2, 2, 2],
            },
        },
    ]
    batch = collate_multimodal_fields(samples, 3)
    assert batch["second_per_grid_ts"].tolist() == [0.5, 1.0, 2.0]
    assert batch["mm_token_type_ids"].tolist() == [[2, 2, 0], [2, 2, 2]]
    assert "second_per_grid_ts" in multimodal_forward_kwargs(batch)


def test_online_image_capture_survives_json_queue_and_branches(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    from PIL import Image

    from wavelet.configs.config import RLDataConfig
    from wavelet.data.rl import (
        collate_rl_batch,
        prepare_rl_sample,
        rl_examples_from_payload,
        rl_examples_to_payload,
    )
    from wavelet.data.sft import _ProcessorTokenizer
    from wavelet.orchestrator.envs import _records_from_output
    from wavelet.orchestrator.multimodal import install_multimodal_rollout_hooks

    model, processor, tokenizer = _tiny_vlm()
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(path)

    class Env:
        serial = 0

        async def get_model_response(self, state, prompt, **kwargs):
            self.serial += 1
            ids = _ProcessorTokenizer(processor, tokenizer).apply_chat_template(
                prompt, add_generation_prompt=True
            )
            return SimpleNamespace(
                id=str(self.serial),
                message=SimpleNamespace(tokens=SimpleNamespace(prompt_ids=ids)),
            )

        async def add_trajectory_step(self, state, trajectory_step):
            state["trajectory"].append(trajectory_step)

    env = Env()
    install_multimodal_rollout_hooks(env, processor)
    state = {"trajectory": []}

    async def turn(prompt):
        response = await env.get_model_response(state, prompt)
        step = {
            "prompt": list(prompt),
            "completion": [{"role": "assistant", "content": "red"}],
            "response": response,
            "extras": {},
            "tokens": {
                "prompt_ids": response.message.tokens.prompt_ids,
                "prompt_mask": [0] * len(response.message.tokens.prompt_ids),
                "completion_ids": [9, 2],
                "completion_mask": [1, 1],
                "completion_logprobs": [-1.0, -1.0],
            },
        }
        await env.add_trajectory_step(state, step)
        return step

    def prompt(text):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": str(path)}},
                    {"type": "text", "text": text},
                ],
            }
        ]

    async def rollout():
        first = await turn(prompt("describe"))
        Image.new("RGB", (8, 8), "blue").save(path)
        await turn(
            first["prompt"]
            + first["completion"]
            + [{"role": "user", "content": "describe"}]
        )
        await turn(prompt("describe"))

    asyncio.run(rollout())
    for step in state["trajectory"]:
        step.pop("response")
    records = _records_from_output(
        {
            "trajectory": state["trajectory"],
            "reward": 1.0,
            "advantage": 1.0,
            "example_id": 1,
        }
    )
    assert len(records) == 2
    assert records[0].mm_kwargs["pixel_values"] != records[1].mm_kwargs["pixel_values"]
    rows = rl_examples_from_payload(
        json.loads(json.dumps(rl_examples_to_payload(records)))
    )
    samples = [
        prepare_rl_sample(row, tokenizer, RLDataConfig(seq_len=64), 64, processor)
        for row in rows
    ]
    batch = collate_rl_batch(samples, pad_token_id=1)
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        **multimodal_forward_kwargs(batch),
    ).logits
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), batch["labels"].flatten()
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_online_snapshot_rejects_nonregular_media_and_oversized_png(
    tmp_path, monkeypatch
):
    import os

    import pytest
    from PIL import Image

    from wavelet.orchestrator import multimodal

    fifo = tmp_path / "image.pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular files"):
        multimodal._load_image(str(fifo))
    image = Image.new("RGB", (8, 8), "red")
    monkeypatch.setattr(multimodal, "_load_image", lambda _: image)
    monkeypatch.setattr(multimodal, "_MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="Canonical PNG"):
        multimodal._snapshot_prompt(
            [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "image"}}],
                }
            ],
            None,
            {},
        )
