"""Make the `app` and `eval` packages importable when running pytest from anywhere."""
import sys
from pathlib import Path

# backend/ — parent of tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
