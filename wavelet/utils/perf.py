from __future__ import annotations

import os


def perf_enabled() -> bool:
    return os.environ.get("WAVELET_PERF_LOG", "").lower() in {"1", "true", "yes", "on"}


def emit_perf(event: str, *, force: bool = False, **fields: object) -> None:
    if not force and not perf_enabled():
        return
    parts = [f"WAVELET_PERF {event}"]
    parts.extend(f"{key}={_format_value(value)}" for key, value in fields.items())
    print(" ".join(parts), flush=True)


def _format_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value
