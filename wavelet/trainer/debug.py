from __future__ import annotations

import json
import string
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer

DEBUG_MODEL_NAME = "debug/tiny-random"
DEBUG_ROLE_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
]
DEBUG_LORA_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]


def _debug_vocab_tokens() -> list[str]:
    special_tokens = ["<pad>", "<eos>", "<unk>", *DEBUG_ROLE_TOKENS]
    char_tokens = list(dict.fromkeys(string.printable + " "))
    return special_tokens + char_tokens


class DebugTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *, model_max_length: int = 256) -> None:
        tokens = _debug_vocab_tokens()
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self._ids_to_tokens = {index: token for token, index in self._vocab.items()}
        super().__init__(
            pad_token="<pad>",
            eos_token="<eos>",
            unk_token="<unk>",
            additional_special_tokens=DEBUG_ROLE_TOKENS,
            model_max_length=model_max_length,
        )
        self.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}{{ eos_token }}"
            "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}{{ eos_token }}"
            "{% elif message['role'] == 'assistant' %}<|assistant|>\n{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}"
            "{% elif message['role'] == 'tool' %}<|tool|>\n{{ message['content'] }}{{ eos_token }}"
            "{% else %}<|user|>\n{{ message['content'] }}{{ eos_token }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
        )
        self.padding_side = "left"

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is None:
            return list(token_ids_0)
        return [*token_ids_0, *token_ids_1]

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str]:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        filename = "debug_vocab.json"
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        vocab_path = path / filename
        vocab_path.write_text(json.dumps(self._vocab, indent=2), encoding="utf-8")
        return (str(vocab_path),)


def build_debug_tokenizer(*, model_max_length: int) -> DebugTokenizer:
    return DebugTokenizer(model_max_length=model_max_length)


def build_debug_model(*, max_seq_length: int | None) -> GPT2LMHeadModel:
    context_length = max(max_seq_length or 128, 64)
    vocab_size = len(_debug_vocab_tokens())
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=context_length,
        n_ctx=context_length,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(config)
