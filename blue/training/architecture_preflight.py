"""Fail-closed checks for the frozen Kaggle architecture cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from blue.harness.scoring import MLScorer, sha256
from blue.training.architecture_study import episode
from blue.harness.policy import HarnessPolicy
from blue.training.harness_redesign import runtime


def preflight(folder: str | Path) -> None:
    folder = Path(folder)
    config = json.loads((folder / "study.json").read_text())
    cohort = {k: v for k, v in config.items() if k != "cohort_sha256"}
    expected_cohort_hash = hashlib.sha256(
        json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if config.get("cohort_sha256") != expected_cohort_hash:
        raise ValueError("frozen cohort hash mismatch")
    if sha256(folder / "scorer.pkl") != config["input_hashes"]["scorer.pkl"]:
        raise ValueError("risk scorer hash mismatch")
    if sha256(folder / "golden.npz") != config["input_hashes"]["golden.npz"]:
        raise ValueError("golden prediction fixture hash mismatch")

    scorer = MLScorer(folder / "scorer.pkl")
    bundle = scorer.bundle
    golden = np.load(folder / "golden.npz")
    features = golden["X"]
    risk = bundle["calibrator"].predict(
        bundle["risk_model"].predict_proba(features)[:, 1]
    )
    novelty = -bundle["anomaly_model"].score_samples(
        np.where(np.isnan(features), bundle["mean"], features)
    )
    np.testing.assert_allclose(risk, golden["risk"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(novelty, golden["novelty"], rtol=1e-6, atol=1e-7)

    actual_runtime = runtime()
    if actual_runtime != config["runtime_versions"]:
        raise ValueError(f"runtime mismatch: {actual_runtime}")
    if scorer.sha256 != config["input_hashes"]["scorer.pkl"]:
        raise ValueError("loaded scorer differs from frozen scorer")

    from blue.training.mappo_guide import ENV_KW
    env_kw = {**ENV_KW, "red_agent": config["red_agent"]}
    smoke = {"steps": 400, "env_kw": env_kw, "gamma": 0.99}
    result = episode(HarnessPolicy(max_age=config["max_age"]),
                     config["preflight"]["seed"], smoke)
    for key, expected in config["preflight"].items():
        if key == "seed":
            continue
        if result[key] != expected:
            raise ValueError(f"simulator preflight mismatch for {key}: "
                             f"{result[key]} != {expected}")
    print("PREFLIGHT PASSED: scorer predictions, runtime and Finite Red episode match",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    preflight(parser.parse_args().input)
