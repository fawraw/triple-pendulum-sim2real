"""Shared pytest configuration. Adds the project root to sys.path so tests
can import `sim`, `training`, and `scripts` packages without installing the
project."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
