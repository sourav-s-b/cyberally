"""Audit the frozen four-run Qwen event-time RL pilot, without return ranking.

Completion, source/model parity, fresh free-choice updates, and guard coverage
are the gate. Returns and first-episode request digests are reported only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from blue.training.event_time_rl import FROZEN_QWEN


EXPECTED_SCORER = "ec4026a3c81c2746253f306fd7add00a07f42fe13533b856fb6b4d01b8402fcd"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(base):
    base = Path(base)
    failures, runs, provenance = [], {}, None
    for arm, gae in (("mc", 1.0), ("gae", 0.9)):
        for rng in (0, 1):
            label = f"pilot_{arm}_s{rng}"
            run_dir = base / label
            path = run_dir / "manifest.json"
            if not path.exists():
                failures.append(f"missing manifest: {label}")
                continue
            m = json.loads(path.read_text())
            cfg = m.get("config", {})
            expected = {"red_agent": "llm", "model": "qwen2.5:3b",
                        "seed": rng, "train_seeds": [7706, 7707, 7708, 7709],
                        "steps": 400, "iters": 2, "eps_per_iter": 2,
                        "gae_lambda": gae, "bonus": 1.0,
                        "explore": 0.5, "lr": 0.003,
                        "critic_lr": 0.0003, "temp": 0.5}
            for key, value in expected.items():
                if cfg.get(key) != value:
                    failures.append(f"{label}: config {key} drift")
            if m.get("status") != "complete":
                failures.append(f"{label}: incomplete")
            if m.get("model_info") != FROZEN_QWEN:
                failures.append(f"{label}: Qwen digest/version drift")
            if m.get("scorer_sha256") != EXPECTED_SCORER:
                failures.append(f"{label}: scorer artifact drift")
            if provenance is None:
                provenance = m.get("source_files")
            elif m.get("source_files") != provenance:
                failures.append(f"{label}: executable source-file drift")
            history = m.get("history", [])
            if len(history) != 2:
                failures.append(f"{label}: expected two complete iterations")
            checks = []
            for i, h in enumerate(history):
                eps = h.get("episodes", [])
                expected_seeds = [7706 + 2 * i, 7707 + 2 * i]
                if ([e.get("seed") for e in eps] != expected_seeds or
                        h.get("iteration") != i + 1):
                    failures.append(f"{label}: episode/iteration schedule drift")
                for e in eps:
                    if (e.get("ticks") != 399 or
                            e.get("n_actor", 0) < 1 or
                            e.get("coverage", 0) < 0.999 or
                            e.get("singleton_forced", -1) < 0):
                        failures.append(f"{label}: episode validity/coverage")
                kl = h.get("kl_exact")
                grad = h.get("actor_gradient_norm_mean")
                if (h.get("policy_steps", 0) < 1 or
                        h.get("decision_rows", 0) < 1 or
                        kl is None or not np.isfinite(kl) or not 0 <= kl <= .02 or
                        grad is None or not np.isfinite(grad) or grad <= 0):
                    failures.append(f"{label}: missing/nonfinite/oversized update")
                for field, sha_field in (("actor_path", "actor_sha256"),
                                         ("critic_path", "critic_sha256")):
                    artifact = run_dir / h.get(field, "missing")
                    if not artifact.is_file() or digest(artifact) != h.get(sha_field):
                        failures.append(f"{label}: {field} checkpoint hash drift")
                checks.append({"iteration": i + 1,
                               "returns": [e.get("return") for e in eps],
                               "coverage": [e.get("coverage") for e in eps],
                               "free_decisions": [e.get("n_actor") for e in eps],
                               "singleton_forced": [e.get("singleton_forced") for e in eps],
                               "kl_exact": kl, "actor_gradient_norm_mean": grad,
                               "policy_steps": h.get("policy_steps")})
            runs[label] = {"manifest_sha256": digest(path),
                           "source_commit": m.get("source_commit"),
                           "iterations": checks,
                           "first_episode_request_sha256":
                           history[0]["episodes"][0]["requests"]["requested_actions_sha256"]
                           if history and history[0].get("episodes") else None}
    request_parity = {}
    for rng in (0, 1):
        mc = runs.get(f"pilot_mc_s{rng}", {}).get("first_episode_request_sha256")
        gae = runs.get(f"pilot_gae_s{rng}", {}).get("first_episode_request_sha256")
        request_parity[str(rng)] = None if mc is None or gae is None else mc == gae
    return {"status": "pass" if not failures else "fail",
            "failures": failures, "runs": runs,
            "first_episode_request_parity": request_parity,
            "source_files": provenance,
            "claim": "pilot mechanics only; returns do not rank RL arms"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="blue/results/event_time_rl")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = analyze(args.base)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "failures": result["failures"],
                      "request_parity": result["first_episode_request_parity"]},
                     indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
