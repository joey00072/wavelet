from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _request_json(
    url: str,
    *,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Inference request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach inference server at {url}: {exc.reason}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("Inference server returned a non-object JSON response.")
    return result


def _resolve_model(base_url: str, *, api_key: str | None, timeout: float) -> str:
    response = _request_json(
        f"{base_url}/models",
        api_key=api_key,
        payload=None,
        timeout=timeout,
    )
    models = response.get("data")
    if not isinstance(models, list) or not models:
        raise RuntimeError("Inference server returned no models from /v1/models.")
    model = models[0]
    if not isinstance(model, dict) or not isinstance(model.get("id"), str):
        raise RuntimeError("Inference server returned an invalid model listing.")
    return model["id"]


def _completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat completion response did not contain a choice.")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("Chat completion response did not contain a message.")
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps(tool_calls, ensure_ascii=False, indent=2)
    return ""


def _chat_once(
    base_url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    temperature: float,
    max_completion_tokens: int | None,
    reasoning_effort: str | None,
    timeout: float,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    response = _request_json(
        f"{base_url}/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )
    return _completion_text(response)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with a running OpenAI-compatible Wavelet server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", help="Served model id; defaults to /v1/models.")
    parser.add_argument("--api-key-var", default="OPENAI_API_KEY")
    parser.add_argument("--system", help="Optional system message.")
    parser.add_argument("--prompt", help="Send one prompt instead of interactive chat.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"]
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")
    api_key = os.environ.get(args.api_key_var)
    model = args.model or _resolve_model(
        base_url,
        api_key=api_key,
        timeout=args.timeout,
    )
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def send(prompt: str) -> None:
        messages.append({"role": "user", "content": prompt})
        content = _chat_once(
            base_url,
            model=model,
            messages=messages,
            api_key=api_key,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
            timeout=args.timeout,
        )
        messages.append({"role": "assistant", "content": content})
        print(content)

    if args.prompt is not None:
        send(args.prompt)
        return 0

    print(f"Connected to {model}. Enter /exit to quit.")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip().lower() in {"/exit", "/quit"}:
            return 0
        if prompt.strip():
            send(prompt)


if __name__ == "__main__":
    sys.exit(main())
