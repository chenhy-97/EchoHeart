"""Video model training and validation."""

from pathlib import Path
import sys


_RUNTIME_DIRECTORY = str(Path(__file__).resolve().parent)
if _RUNTIME_DIRECTORY not in sys.path:
    sys.path.insert(0, _RUNTIME_DIRECTORY)
