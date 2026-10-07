"""Audit all matched pilot cells without treating episodes as model replicas."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(cells):
    required = ['lancer', 'guard'] + [f'{a}_s{s}' for s in (0, 1)
                                     for a in ('zero', 'risk', 'both')]
    if set(cells) != set(required):
        raise ValueError('missing or unexpected pilot arms')
    seeds = sorted(cells['lancer'], key=int)
    if not seeds or any(set(cells[k]) != set(seeds) for k in required):
        raise ValueError('unmatched evaluation episodes')
    if any(7809 <= int(s) <= 8200 for s in seeds):
        raise ValueError('protected final seeds')
    if not all(np.isfinite(v) for k in required for v in cells[k].values()):
        raise ValueError('nonfinite evaluation return')
    means = {k: float(np.mean([cells[k][s] for s in seeds])) for k in required}
    runs = []
    for rng in (0, 1):
        for arm in ('zero', 'risk', 'both'):
            tag = f'{arm}_s{rng}'
            diffs = {b: {s: cells[tag][s] - cells[b][s] for s in seeds}
                     for b in ('lancer', 'guard')}
            runs.append({'arm': arm, 'training_seed': rng, 'mean_return': means[tag],
                         'mean_vs_lancer': means[tag]-means['lancer'],
                         'mean_vs_guard': means[tag]-means['guard'],
                         'worst_return': min(cells[tag].values()),
                         'episode_differences': diffs})
    pairs = []
    for rng in (0, 1):
        pairs.append({'training_seed': rng,
                      'risk_minus_zero': means[f'risk_s{rng}']-means[f'zero_s{rng}'],
                      'both_minus_risk': means[f'both_s{rng}']-means[f'risk_s{rng}'],
                      'monotone_episode_count': sum(cells[f'both_s{rng}'][s] >=
                          cells[f'risk_s{rng}'][s] >= cells[f'zero_s{rng}'][s] for s in seeds)})
    return {'evaluation_seeds': list(map(int, seeds)), 'baselines': {b: means[b] for b in ('lancer', 'guard')},
            'guard_minus_lancer': means['guard']-means['lancer'], 'runs': runs,
            'matched_training_pairs': pairs, 'independent_training_seeds': 2,
            'interpretation': 'Descriptive technical pilot on reused development episodes. No confidence interval, ordering chance test, or superiority claim. Shared episodes are blocks, not independent trained models.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cells', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = audit(json.loads(Path(a.cells).read_text()))
    Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
