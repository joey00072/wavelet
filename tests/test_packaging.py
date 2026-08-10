from __future__ import annotations

import subprocess
from pathlib import Path


def test_orchestrator_reference_module_is_not_vcs_ignored() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "--quiet",
            "wavelet/orchestrator/reference.py",
        ],
        cwd=repository_root,
        check=False,
    )

    assert result.returncode == 1
