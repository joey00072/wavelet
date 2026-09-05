"""Build and embed the dashboard frontend in distribution wheels."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Compile the browser dashboard before Hatch assembles a wheel."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return

        bun = shutil.which("bun")
        if bun is None:
            raise RuntimeError(
                "Building a Wavelet wheel requires Bun so the dashboard UI can be "
                "included. Install Bun and run `uv build` again."
            )

        webui = Path(self.root) / "webui"
        subprocess.run([bun, "install", "--frozen-lockfile"], cwd=webui, check=True)
        subprocess.run([bun, "run", "build"], cwd=webui, check=True)

        static_dir = webui / "dist"
        if not (static_dir / "index.html").is_file():
            raise RuntimeError("Dashboard build did not produce webui/dist/index.html")
        build_data.setdefault("force_include", {})[str(static_dir)] = (
            "wavelet/dashboard/static"
        )
