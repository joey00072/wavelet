import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

from wavelet.dashboard.server import RunRegistry, build_dashboard_app
from wavelet.dashboard.synth import write_synthetic_run


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(prefix="wavelet-dashboard-e2e-") as temporary:
        runs_root = Path(temporary)
        write_synthetic_run(
            runs_root / "synthetic-a",
            steps=6,
            groups=4,
            rollouts_per_group=4,
            seed=11,
            started_at=datetime(2026, 9, 5, 12, 0, tzinfo=UTC),
        )
        write_synthetic_run(
            runs_root / "synthetic-b",
            steps=8,
            groups=4,
            rollouts_per_group=4,
            seed=17,
            started_at=datetime(2026, 9, 5, 13, 0, tzinfo=UTC),
        )
        app = build_dashboard_app(
            RunRegistry(roots=[runs_root]),
            static_dir=repo_root / "webui" / "dist",
        )
        port = int(os.environ.get("WAVELET_E2E_PORT", "8767"))
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
