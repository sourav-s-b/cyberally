"""Fail closed on scorer/runtime drift before expensive remote rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from blue.harness.scoring import MLScorer
from blue.harness.policy import HarnessPolicy
from blue.training import mappo_guide as mg


def preflight(folder):
    folder = Path(folder)
    scorer = MLScorer(folder / 'scorer.pkl')
    b = scorer.bundle
    golden = np.load(folder / 'golden.npz')
    X = golden['X']
    p = b['calibrator'].predict(b['risk_model'].predict_proba(X)[:, 1])
    novelty = -b['anomaly_model'].score_samples(np.where(np.isnan(X), b['mean'], X))
    np.testing.assert_allclose(p, golden['risk'], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(novelty, golden['novelty'], rtol=1e-6, atol=1e-7)
    cfg = json.loads((folder / 'pilot.json').read_text())
    result = mg.run_team_episode(HarnessPolicy(max_age=cfg['max_age']), 7706, 40, **mg.ENV_KW)
    expected = cfg['preflight_episode']
    for k in ('return', 'ticks', 'coverage'):
        if result[k] != expected[k]:
            raise ValueError(f'simulator runtime mismatch {k}: {result[k]} != {expected[k]}')
    print('PREFLIGHT PASSED: golden ML predictions and simulator episode match', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    preflight(ap.parse_args().input)
