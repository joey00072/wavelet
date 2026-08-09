"""Compatibility entrypoint for SFT training consolidated in trainer.py."""

import sys

from wavelet.trainer import trainer as _trainer

sys.modules[__name__] = _trainer


if __name__ == "__main__":
    raise SystemExit(_trainer.main())
