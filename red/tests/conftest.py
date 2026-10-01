"""Use this worktree's project modules with the existing isolated interpreter."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cage-challenge-4"))
