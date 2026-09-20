"""Capture image inputs at generation time and preserve sampled-token provenance."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import stat
import time
from functools import wraps
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageOps

from wavelet.data.sft import _ProcessorTokenizer

_MARK = "_wavelet_multimodal_capture"


_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_PIXELS = 16 * 1024 * 1024


def _load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        deadline = time.monotonic() + 30.0
        raw = bytearray()
        with httpx.stream(
            "GET", source, timeout=10.0, follow_redirects=True
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise TimeoutError("Image download exceeded 30 seconds.")
                if len(raw) + len(chunk) > _MAX_IMAGE_BYTES:
                    raise ValueError("Image input exceeds 32 MiB.")
                raw.extend(chunk)
    elif source.startswith("data:image/"):
        encoded = source.split(",", 1)[1]
        if len(encoded) > 4 * ((_MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("Image input exceeds 32 MiB.")
        raw = base64.b64decode(encoded, validate=True)
    else:
        descriptor = os.open(Path(source), os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("Local image inputs must be regular files.")
            raw = handle.read(_MAX_IMAGE_BYTES + 1)
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError("Image input exceeds 32 MiB.")
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > _MAX_IMAGE_PIXELS:
            raise ValueError("Image input exceeds 16 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")


def _snapshot_prompt(
    prompt: list[Any], processor: Any, template_kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    frozen = []
    rendered = []
    for message in prompt:
        payload = (
            message.model_dump() if hasattr(message, "model_dump") else dict(message)
        )
        content = payload.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                part = dict(part)
                if part.get("type") == "image_url":
                    image = _load_image(part["image_url"]["url"])
                    with io.BytesIO() as buffer:
                        image.save(buffer, format="PNG")
                        if buffer.tell() > _MAX_IMAGE_BYTES:
                            raise ValueError("Canonical PNG image exceeds 32 MiB.")
                        url = "data:image/png;base64," + base64.b64encode(
                            buffer.getvalue()
                        ).decode("ascii")
                    part = {"type": "image_url", "image_url": {"url": url}}
                elif part.get("type") != "text":
                    raise ValueError(
                        "Online multimodal capture currently supports image_url and text parts."
                    )
                parts.append(part)
            payload["content"] = parts
        rendered.append(payload)
        frozen.append(
            type(message).model_validate(payload)
            if hasattr(message, "model_validate")
            else payload
        )
    adapter = _ProcessorTokenizer(processor, processor.tokenizer)
    ids = adapter.apply_chat_template(
        rendered, add_generation_prompt=True, **template_kwargs
    )
    fields = {
        key: value.detach().cpu().tolist() if hasattr(value, "detach") else value
        for key, value in adapter.encoded.items()
        if key not in {"input_ids", "attention_mask"}
    }
    return frozen, {"prompt_ids": ids, "mm_kwargs": fields, "prompt": rendered}


def install_multimodal_rollout_hooks(env: Any, processor: Any) -> None:
    """Capture per response, then attach its tensors to the matching trajectory step."""
    if getattr(env, "env_client", None) is not None:
        raise ValueError(
            "Multimodal capture requires a local verifier environment, not server mode."
        )
    if getattr(env, _MARK, False):
        return
    original_response = getattr(env, "get_model_response", None)
    original_step = getattr(env, "add_trajectory_step", None)
    if not callable(original_response) or not callable(original_step):
        raise ValueError(  # noqa: TRY004 - missing runtime hook is a rollout contract error
            "Multimodal rollouts require get_model_response and add_trajectory_step hooks."
        )

    @wraps(original_response)
    async def response(
        state: Any,
        prompt: Any,
        client: Any = None,
        model: Any = None,
        tool_defs: Any = None,
        sampling_args: Any = None,
    ) -> Any:
        if not isinstance(prompt, list):
            raise ValueError(  # noqa: TRY004 - invalid verifier message shape
                "Multimodal rollouts require structured chat messages."
            )
        tools = tool_defs if tool_defs is not None else state.get("tool_defs")
        if tools:
            raise ValueError("Online multimodal tool schemas are not yet supported.")
        effective_sampling = sampling_args or state.get("sampling_args") or {}
        extra_body = effective_sampling.get("extra_body") or {}
        template_kwargs = (
            extra_body.get("chat_template_kwargs")
            or effective_sampling.get("chat_template_kwargs")
            or {}
        )
        frozen, capture = await asyncio.to_thread(
            _snapshot_prompt, prompt, processor, template_kwargs
        )
        prompt[:] = frozen
        result = await original_response(
            state,
            frozen,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
        )
        tokens = getattr(getattr(result, "message", None), "tokens", None)
        if tokens is None or list(tokens.prompt_ids) != capture["prompt_ids"]:
            raise ValueError(
                "Inference prompt IDs do not match the multimodal processor capture."
            )
        state.setdefault("_wavelet_mm_captures", {})[result.id] = capture
        return result

    @wraps(original_step)
    async def step(state: Any, trajectory_step: Any) -> Any:
        result = trajectory_step.get("response")
        response_id = (
            result.get("id")
            if isinstance(result, dict)
            else getattr(result, "id", None)
        )
        capture = state.get("_wavelet_mm_captures", {}).pop(response_id, None)
        if capture is None:
            raise ValueError(
                "Multimodal trajectory step has no generation-time capture."
            )
        if (trajectory_step.get("tokens") or {}).get("prompt_ids") != capture[
            "prompt_ids"
        ]:
            raise ValueError(
                "Trajectory prompt IDs changed after multimodal generation."
            )
        trajectory_step.setdefault("extras", {})["wavelet_multimodal"] = capture
        return await original_step(state, trajectory_step)

    env.get_model_response = response
    env.add_trajectory_step = step
    setattr(env, _MARK, True)
