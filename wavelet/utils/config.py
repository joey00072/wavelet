from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin

from pydantic import BaseModel

from wavelet.utils.serialization import load_yaml

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
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


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _normalize_key(raw: str) -> str:
    return ".".join(part.replace("-", "_") for part in raw.split("."))


def _model_types(annotation: Any) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    models: list[type[BaseModel]] = []
    for argument in get_args(annotation):
        for model in _model_types(argument):
            if model not in models:
                models.append(model)
    return models


def _field_annotations(config_type: type[BaseModel], dotted_key: str) -> list[Any]:
    models = [config_type]
    parts = dotted_key.split(".")
    annotations: list[Any] = []
    for index, part in enumerate(parts):
        annotations = []
        nested_models: list[type[BaseModel]] = []
        for model in models:
            field = model.model_fields.get(part)
            if field is None:
                continue
            annotations.append(field.annotation)
            nested_models.extend(_model_types(field.annotation))
        if index < len(parts) - 1:
            models = nested_models
    return annotations


def _includes_bool(annotation: Any) -> bool:
    return annotation is bool or any(
        _includes_bool(arg) for arg in get_args(annotation)
    )


def _is_bool_field(config_type: type[BaseModel], dotted_key: str) -> bool:
    annotations = _field_annotations(config_type, dotted_key)
    return bool(annotations) and all(_includes_bool(item) for item in annotations)


def _type_label(annotation: Any) -> str:
    if annotation is bool:
        return "BOOL"
    if annotation is str:
        return "STR"
    if annotation is int:
        return "INT"
    if annotation is float:
        return "FLOAT"
    if annotation is Path:
        return "PATH"
    origin = get_origin(annotation)
    if origin is Literal:
        return "{" + ",".join(str(item) for item in get_args(annotation)) + "}"
    if origin is not None:
        name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
        return name.upper()
    return getattr(annotation, "__name__", "VALUE").upper()


def _help_rows(
    model: type[BaseModel],
    *,
    prefix: str = "",
    ancestors: tuple[type[BaseModel], ...] = (),
) -> list[tuple[str, str, str | None]]:
    rows: list[tuple[str, str, str | None]] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}.{name}" if prefix else name
        flag = "--" + key.replace("_", "-")
        nested = [
            item for item in _model_types(field.annotation) if item not in ancestors
        ]
        if nested:
            for nested_model in nested:
                rows.extend(
                    _help_rows(
                        nested_model,
                        prefix=key,
                        ancestors=(*ancestors, model),
                    )
                )
            continue
        label = _type_label(field.annotation)
        if _includes_bool(field.annotation):
            label = "BOOL (also --no-...)"
        description = field.description
        if not field.is_required() and isinstance(
            field.default, (str, int, float, bool, type(None), Path)
        ):
            default = str(field.default)
            description = f"{description or ''} [default: {default}]".strip()
        rows.append((flag, label, description))
    return rows


def format_config_help(config_type: type[BaseModel]) -> str:
    rows_by_flag: dict[str, tuple[str, str, str | None]] = {}
    for row in _help_rows(config_type):
        rows_by_flag.setdefault(row[0], row)
    rows = list(rows_by_flag.values())
    width = max((len(flag) for flag, _, _ in rows), default=0)
    lines = [
        f"{config_type.__name__} configuration",
        "Usage: [@ CONFIG ...] [--field VALUE] [--flag | --no-flag]",
        "",
        "Options:",
    ]
    for flag, label, description in rows:
        line = f"  {flag:<{width}}  {label}"
        if description:
            line += f"  {description}"
        lines.append(line)
    return "\n".join(lines)


def load_config(config_type: type[ConfigT], argv: list[str]) -> ConfigT:
    if "--help" in argv or "-h" in argv:
        print(format_config_help(config_type))
        raise SystemExit(0)

    raw: dict[str, Any] = {}
    override_tokens: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "@":
            if index + 1 >= len(argv):
                raise SystemExit("Expected a path after '@'.")
            raw = _deep_merge(raw, load_yaml(Path(argv[index + 1])))
            index += 2
            continue
        if token.startswith("@") and len(token) > 1:
            raw = _deep_merge(raw, load_yaml(Path(token[1:])))
            index += 1
            continue
        override_tokens.append(token)
        index += 1

    index = 0
    while index < len(override_tokens):
        token = override_tokens[index]
        if not token.startswith("--"):
            raise SystemExit(f"Invalid override '{token}'. Expected --section.key.")
        raw_key = token[2:]
        negated = raw_key.startswith("no-")
        key = _normalize_key(raw_key[3:] if negated else raw_key)
        if not key:
            raise SystemExit("Configuration override names cannot be empty.")
        if negated:
            value: Any = False
            index += 1
        elif _is_bool_field(config_type, key) and (
            index + 1 >= len(override_tokens)
            or override_tokens[index + 1].startswith("--")
        ):
            value = True
            index += 1
        else:
            if index + 1 >= len(override_tokens):
                raise SystemExit(f"Expected a value after '{token}'.")
            value = _parse_value(override_tokens[index + 1])
            index += 2
        _set_nested(raw, key, value)

    return config_type.model_validate(raw)
