"""Compatibility fix shared by both SWE runners, without changing task code."""

from __future__ import annotations


def install() -> None:
    """Require PEP 723 sync support instead of accepting any preinstalled uv."""
    from verifiers.v1.runtimes import base

    old = "command -v uv >/dev/null 2>&1"
    capable = "{ uv sync --help 2>/dev/null | grep -q -- --script; }"
    if capable in base._ENSURE_UV:
        return
    if old not in base._ENSURE_UV:
        raise RuntimeError(
            "Native uv bootstrap changed; review the SWE compatibility fix."
        )
    base._ENSURE_UV = base._ENSURE_UV.replace(old, capable)


def main() -> None:
    install()
    from prime_rl.entrypoints.env_server import main as serve

    serve()


if __name__ in {"__main__", "__mp_main__"}:
    install()

if __name__ == "__main__":
    main()
