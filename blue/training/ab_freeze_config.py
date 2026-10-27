"""Frozen configuration for the risk-input ablation.

Written BEFORE any run so the comparison cannot be quietly reshaped after
seeing results. Records:
  - the exact bytes of every input artifact (risk model, Phase-3 scorer);
  - the code version;
  - training budget and environment settings;
  - every seed list, with the audit that justifies the fresh dev block;
  - the success criteria and the two questions the run must answer.

The Lancer baseline is cached under exactly this configuration: any change
to the seed list, step count or scenario settings invalidates the cache.

Usage:
    .venv-train/bin/python -m blue.training.ab_freeze_config            # write
    .venv-train/bin/python -m blue.training.ab_freeze_config --verify   # check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT = os.path.join(REPO, "blue", "results", "ablation_risk")

# --- frozen experiment definition -----------------------------------------
ARMS = ("real", "zero")
TRAIN_SEEDS = [0, 1, 2]          # TRAINING randomness, n=3
TRAIN_EPISODE_SEEDS = [7706, 7707, 7708, 7709]   # unchanged, matched arms
EVAL_SEEDS = list(range(8221, 8253))            # 32 fresh dev seeds

BUDGET = {"iters": 4, "eps_per_iter": 4, "steps": 400, "hidden": 64,
          "lr": 3e-4, "bonus": 1.0, "temp": 0.5, "train_min_proba": 0.0}

RESERVED_BLOCK = [7809, 8200]

SUCCESS_CRITERIA = {
    "primary_question_1": "RL+risk vs Lancer: does the RL agent improve "
                          "defense at all?",
    "primary_question_2": "RL+risk vs RL+norisk: does the risk input "
                          "contribute to that improvement?",
    "reporting": "report ALL 6 runs. No best-of selection, no dropping "
                 "unlucky runs, no changing the scoring metric afterwards.",
    "uncertainty_rule": "The independent unit is the TRAINING seed (n=3). "
                        "Evaluation seeds are shared across training seeds, "
                        "so 3x32 is NOT 96 independent observations. The "
                        "across-training-seed CI uses df=2 (t=4.303). "
                        "Per-training-seed eval-seed CIs are reported as a "
                        "within-unit spread only, never as the headline.",
    "success_threshold": "no pre-registered numeric gate is claimed; the "
                         "existing +5 dev gate is NOT applied here because "
                         "n=3 training seeds cannot resolve it. Report the "
                         "estimate and its interval, including if it "
                         "straddles zero.",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return None


def build():
    art = {}
    for rel in ("blue/results/risk_model_v2.pkl",
                "blue/results/scorer_mlp.npz"):
        p = os.path.join(REPO, rel)
        art[rel] = ({"sha256": sha256(p), "bytes": os.path.getsize(p)}
                    if os.path.exists(p) else {"MISSING": True})
    env = {}
    try:
        from blue.training import mappo_guide as mg
        env = {k: v for k, v in mg.ENV_KW.items()}
    except Exception as e:
        env = {"error": str(e)}
    return {
        "experiment": "risk-input ablation: RL with real risk score vs "
                      "RL with the risk input held at zero",
        "arms": list(ARMS),
        "train_seeds": TRAIN_SEEDS,
        "train_episode_seeds": TRAIN_EPISODE_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "eval_seed_count": len(EVAL_SEEDS),
        "budget": BUDGET,
        "env_kw": env,
        "artifacts": art,
        "git_commit": git_rev(),
        "git_dirty_at_freeze": git_dirty(),
        "reserved_block_untouched": RESERVED_BLOCK,
        "success_criteria": SUCCESS_CRITERIA,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "frozen_config.json"))
    args = ap.parse_args()

    cfg = build()
    if args.verify:
        if not os.path.exists(args.out):
            print("MISSING frozen config:", args.out)
            return 1
        with open(args.out) as f:
            old = json.load(f)
        bad = []
        for k in ("eval_seeds", "train_seeds", "train_episode_seeds",
                  "budget", "arms", "artifacts", "env_kw"):
            if json.dumps(old.get(k), sort_keys=True) != \
                    json.dumps(cfg.get(k), sort_keys=True):
                bad.append(k)
        if bad:
            print("FROZEN CONFIG DRIFT in:", ", ".join(bad))
            print("the cached Lancer baseline is no longer valid")
            return 1
        print("frozen config intact (commit %s)" % cfg["git_commit"][:8])
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=1)
    print(json.dumps(cfg, indent=1))
    print("\nwrote", args.out)
    if cfg["git_dirty_at_freeze"]:
        print("\nWARNING: working tree was DIRTY at freeze time; the code "
              "version above does not fully describe what ran. Commit "
              "before trusting these results.")
    missing = [k for k, v in cfg["artifacts"].items() if v.get("MISSING")]
    if missing:
        print("\nWARNING: missing artifacts:", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())