"""Backward-compatible import shim for ``src/lrb/circuit_generator.py``."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lrb.circuit_generator import *  # noqa: F401,F403

