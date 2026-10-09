"""Native Windows/macOS deployment CLI. Run from any working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.deployment.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
