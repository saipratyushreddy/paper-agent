"""Pytest config — adds project root to sys.path so `import src` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
