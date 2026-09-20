"""Check a Qwen3 SWE renderer using the native environment's tokenizer."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    from pydantic import TypeAdapter
    from renderers import RendererConfig, ToolCallParseStatus, create_renderer
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--renderer", default='{"name":"auto"}')
    args = parser.parse_args()
    config = TypeAdapter(RendererConfig).validate_json(args.renderer)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    renderer = create_renderer(tokenizer, config)
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Inspect the repository."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    expected = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_dict=False
    )
    actual = renderer.render_ids(messages, tools=tools, add_generation_prompt=True)
    completion = tokenizer.encode(
        '<tool_call>\n{"name":"bash","arguments":{"command":"pwd"}}'
        "\n</tool_call><|im_end|>",
        add_special_tokens=False,
    )
    parsed = renderer.parse_response(completion, tools=tools)
    calls = parsed.tool_calls
    tool_ok = (
        len(calls) == 1
        and calls[0].status == ToolCallParseStatus.OK
        and calls[0].name == "bash"
        and calls[0].arguments == {"command": "pwd"}
    )
    history = messages + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "pwd"}},
                }
            ],
        },
        {"role": "tool", "content": "/testbed", "tool_call_id": "call_0"},
    ]
    expected_history = tokenizer.apply_chat_template(
        history, tools=tools, add_generation_prompt=True, return_dict=False
    )
    actual_history = renderer.render_ids(
        history, tools=tools, add_generation_prompt=True
    )
    result = {
        "model": args.model,
        "renderer": type(renderer).__name__,
        "initial_prompt_matches_model_template": actual == expected,
        "model_prompt_tokens": len(expected),
        "renderer_prompt_tokens": len(actual),
        "model_prompt_suffix": tokenizer.decode(expected[-12:]),
        "renderer_prompt_suffix": tokenizer.decode(actual[-12:]),
        "qwen3_tool_call_parses": tool_ok,
        "tool_history_matches_model_template": actual_history == expected_history,
    }
    print(json.dumps(result, indent=2))
    if actual != expected or actual_history != expected_history or not tool_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
