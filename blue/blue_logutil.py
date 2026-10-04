"""Shared run logging: elapsed/ETA/throughput console lines + JSON rows.

Standing convention: every training/eval command runs as
  ... 2>&1 | tee /tmp/<run>.log | tail -N
so console lines are the progression UI and the returned ``hist`` list
(dumped to <out>/train_log.jsonl by each trainer) is the curve source.
Stdlib only.
"""

import time


def fmt_dur(s):
    """87 -> '1m27s', 3700 -> '1h01m'."""
    s = max(0, int(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """Rate-based ETA over countable units (epochs, iters, cells)."""

    def __init__(self, total):
        self.total = max(1, int(total))
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def line(self, done, extra=""):
        """'[done/total] t=.. eta=.. [extra]' with rate from wall clock."""
        done = max(1, int(done))
        el = self.elapsed()
        rate = done / max(el, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        unit = f"[{done}/{self.total}]"
        pace = f"t={fmt_dur(el)} eta={fmt_dur(eta)}"
        return f"{unit} {pace} {extra}".rstrip()
