"""Explicit ranking resolution for new policies; frozen v1 stays unchanged."""
import math


def stable_pick(scored, tolerance=1e-6):
    """Largest hostname among scores within tolerance of the maximum.

    This deliberately changes near-tie policy semantics. It is a reproducibility
    intervention, not a guarantee of identical cross-platform trajectories.
    """
    if not math.isfinite(tolerance) or tolerance < 0 or not scored:
        raise ValueError('invalid ranking tolerance or empty candidates')
    if any(not math.isfinite(s) for s, _ in scored):
        raise ValueError('nonfinite ranking score')
    best = max(s for s, _ in scored)
    return max(h for s, h in scored if best-s <= tolerance)
