"""Test startup path fixes.

This module is auto-imported by Python at startup (via `site`) when it is
available on `sys.path`. Keeping it under `tests/` ensures test files run
directly from this folder can still import project-level modules.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
root_str = str(ROOT_DIR)

if root_str not in sys.path:
    sys.path.insert(0, root_str)
