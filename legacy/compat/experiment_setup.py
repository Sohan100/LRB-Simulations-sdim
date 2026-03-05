"""Backward-compatible import shim for ``src/lrb/experiment_setup.py``."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lrb.experiment_setup import *  # noqa: F401,F403

if __name__ == "__main__":
    runpy.run_module("lrb.experiment_setup", run_name="__main__")
