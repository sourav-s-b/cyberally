"""Pytest path bootstrap for blue/tests_blue.

Inserts the repo root (for absolute ``blue.*`` package imports) and the
pinned EPyMARL source tree. Replaces the per-file sys.path hacks and the
old reliance on cwd=blue/. Run the suite from the repo root:
  .venv/bin/python -m pytest blue/tests_blue -q
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "third_party", "epymarl", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
