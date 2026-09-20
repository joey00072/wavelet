from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_torchrun_cp_token_normalization_matches_single_process() -> None:
    """Run a real two-rank reduction with unequal supervised-token counts."""
    if __import__("torch").cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required for the torchrun CP regression")
    worker = Path(__file__).with_name("context_parallel_gpu_worker.py")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            str(worker),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "cp ring sdpa gradient parity ok" in result.stdout
