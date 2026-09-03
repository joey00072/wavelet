from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from wavelet.utils.serialization import load_yaml

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _parse_value(raw: str) -> Any:
    for parser in (json.loads,):
        try:
            return parser(raw)
        except json.JSONDecodeError:
            continue

    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    return raw


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = mapping
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise TypeError(f"Cannot override nested key '{dotted_key}'.")
        cursor = child
    cursor[parts[-1]] = value


def load_config(config_type: type[ConfigT], argv: list[str]) -> ConfigT:
    raw: dict[str, Any] = {}
    tokens = list(argv)

    if "@" in tokens:
        index = tokens.index("@")
        if index + 1 >= len(tokens):
            raise SystemExit("Expected a path after '@'.")
        config_path = Path(tokens[index + 1])
        raw = load_yaml(config_path)
        del tokens[index : index + 2]
    elif tokens and tokens[0].startswith("@"):
        config_path = Path(tokens[0][1:])
        raw = load_yaml(config_path)
        del tokens[0]

    if len(tokens) % 2 != 0:
        raise SystemExit("Overrides must be provided as --key value pairs.")

    for key_token, value_token in zip(tokens[::2], tokens[1::2], strict=True):
        if not key_token.startswith("--"):
            raise SystemExit(f"Invalid override '{key_token}'. Expected --section.key.")
        _set_nested(raw, key_token[2:], _parse_value(value_token))

    return config_type.model_validate(raw)
