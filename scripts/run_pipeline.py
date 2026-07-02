"""Convenience entry point for the production-oriented pipeline layout."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import __file__ as _  # noqa: F401

if __name__ == "__main__":
    print("Pipeline package available at src/pipeline")
    print("Use: python scripts/run_pipeline.py")
