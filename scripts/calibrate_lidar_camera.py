#!/usr/bin/env python3
"""Backwards-compatible entry point for FAST-Calib2 Python calibration."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fast_calib2.pipeline import main


if __name__ == "__main__":
    main()