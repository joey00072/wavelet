from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wavelet.dashboard.server import RunRegistry, build_dashboard_app

PACKAGE_STATIC_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "static"
SOURCE_STATIC_DIR = Path(__file__).resolve().parents[2] / "webui" / "dist"


def default_static_dir() -> Path:
    """Prefer UI assets embedded in a wheel, then a source-tree build."""
    if (PACKAGE_STATIC_DIR / "index.html").is_file():
        return PACKAGE_STATIC_DIR
    return SOURCE_STATIC_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavelet dashboard",
        description=(
            "Serve a read-only dashboard over one or more RL run directories. "
            "Runs may be live or completed; nothing under them is modified."
        ),
    )
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Run output directories to serve explicitly.",
    )
    parser.add_argument(
        "--runs-root",
        action="append",
        default=[],
        type=Path,
        help="Directory whose immediate children are run directories (repeatable).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=default_static_dir(),
        help="Built web UI directory to serve at '/'. Missing directories are skipped.",
    )
    parser.add_argument("--log-level", default="warning")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if not args.runs and not args.runs_root:
        args.runs_root = [Path("outputs")]
    registry = RunRegistry(roots=args.runs_root, runs=args.runs)
    discovered = registry.discover()
    app = build_dashboard_app(registry, static_dir=args.static_dir)

    import uvicorn

    print(
        f"Wavelet dashboard on http://{args.host}:{args.port} "
        f"({len(discovered)} run(s) discovered)"
    )
    if not (args.static_dir / "index.html").is_file():
        print(
            f"No built web UI at '{args.static_dir}'. Run `bun run build` in webui/ "
            "or open the Vite dev server with ?api=http://host:port."
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
