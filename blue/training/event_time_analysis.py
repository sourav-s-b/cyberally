"""Paired development analysis for the Qwen headroom experiment.

One simulator seed is one resampling unit. Qwen inference can vary even at
the same initial seed; this conditional interval is not a repeat-inference
interval, and the historical actor has only one training seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from blue.training.event_time_rl import FROZEN_QWEN


ARMS = ("lancer", "lancer_guard", "actor_guard")
CONTRASTS = (("lancer_guard", "lancer"),
             ("actor_guard", "lancer_guard"),
             ("actor_guard", "lancer"))


def analyze(blob, bootstrap_seed=20261024, n_boot=10000):
    if blob.get("model") != FROZEN_QWEN:
        raise ValueError("Qwen model/digest/version drift")
    seeds = blob.get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("missing or duplicate simulator seeds")
    if set(seeds) != set(range(7641, 7665)) or blob.get("steps") != 400:
        raise ValueError("headroom seed/horizon contract drift")
    arrays = {}
    per_arm = {}
    for arm in ARMS:
        rows = blob.get("arms", {}).get(arm, {})
        if set(rows) != {str(seed) for seed in seeds}:
            raise ValueError(f"incomplete {arm} arm")
        vals = np.asarray([rows[str(seed)]["return"] for seed in seeds],
                          dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("nonfinite native return")
        arrays[arm] = vals
        per_arm[arm] = {
            "mean_return": float(vals.mean()),
            "sd_return": float(vals.std(ddof=1)),
            "min_return": float(vals.min()),
            "mean_red_plan_requests": float(np.mean([
                rows[str(seed)]["red_plan_requests"] for seed in seeds])),
            "mean_remediation": float(np.mean([
                rows[str(seed)]["remediation"] for seed in seeds])),
            "mean_confirmed_hosts": float(np.mean([
                rows[str(seed)]["confirmed_hosts"] for seed in seeds])),
        }
    rng = np.random.default_rng(bootstrap_seed)
    sample = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
    contrasts = {}
    for left, right in CONTRASTS:
        d = arrays[left] - arrays[right]
        boots = d[sample].mean(axis=1)
        contrasts[f"{left}_minus_{right}"] = {
            "mean_paired_delta": float(d.mean()),
            "ci95_episode_bootstrap": [float(x) for x in
                                       np.quantile(boots, [.025, .975])],
            "seeds_improved": int((d > 0).sum()),
            "min_paired_delta": float(d.min()),
        }
    return {"status": "development_only", "n_seeds": len(seeds),
            "seeds": seeds, "arms": per_arm, "contrasts": contrasts,
            "bootstrap_seed": bootstrap_seed, "bootstrap_samples": n_boot,
            "caveats": [
                "Qwen output can vary across repeats of one simulator seed",
                "historical actor has one training seed",
                "same-guard contrast is actor_guard minus lancer_guard",
                "Llama returns and variance are a different attacker regime",
            ]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    source = Path(args.input)
    result = analyze(json.loads(source.read_text()))
    result["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["contrasts"], indent=2))


if __name__ == "__main__":
    main()
