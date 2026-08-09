"""Compatibility entrypoint for the consolidated rollout scheduler."""

import sys

from wavelet.orchestrator import scheduler as _scheduler

sys.modules[__name__] = _scheduler


if __name__ == "__main__":
    raise SystemExit(_scheduler.main())
