import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports during tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
