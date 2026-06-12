"""CLI entrypoints. Importing this package puts ``src/`` on ``sys.path`` so the
scripts can ``import mhqa`` whether run as ``python -m scripts.train`` or directly.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
