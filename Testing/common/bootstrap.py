"""Add Testing/common to sys.path for scripts under T1–T6."""

from __future__ import annotations

import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
TESTING = COMMON.parent
ROOT = TESTING.parent

if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
