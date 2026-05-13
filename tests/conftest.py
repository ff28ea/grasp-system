"""Pytest configuration: make the grasp_system package importable."""
from __future__ import annotations

import sys
from pathlib import Path

# Tests live in ``<repo>/tests``; the package sits one level up as
# ``<repo>/grasp_system``. Insert the repo root on sys.path so
# ``import grasp_system`` works regardless of where pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
