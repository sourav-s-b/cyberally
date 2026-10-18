"""Resume the frozen high-risk audit after an interrupted local process."""
import json
import os
from pathlib import Path

import torch

from blue.harness.scoring import sha256
from blue.training import harness_high_risk_audit as audit


def main():
    torch.set_num_threads(1)
    cfg = json.loads((audit.OUT / 'frozen_config.json').read_text())
    if cfg['runner_sha256'] != sha256(audit.__file__):
        raise AssertionError('original replay implementation changed')
    if cfg['scorer_sha256'] != sha256(audit.SCORER):
        raise AssertionError('scorer changed')
    if cfg['episodes'] != list(audit.EPISODES) or cfg['models'] != list(audit.MODELS):
        raise AssertionError('frozen cells changed')
    for model, digest in cfg['actor_sha256'].items():
        if sha256(audit.ROOT / model / 'actor.th') != digest:
            raise AssertionError(f'{model} changed')
    frozen = json.loads((audit.ROOT / 'cells.json').read_text())
    existing = json.loads((audit.OUT / 'progress.json').read_text())['cells']
    for key, row in existing.items():
        if not row['frozen_replay_validated'] or row['return'] != frozen[key]['return']:
            raise AssertionError(f'{key} was not validated')
        if row['requested_action_digest'] != frozen[key]['requests']:
            raise AssertionError(f'{key} request digest drift')
    cells = dict(existing)
    for model in audit.MODELS:
        for episode in audit.EPISODES:
            key = f'{model}_{episode}_0'
            if key in cells:
                continue
            for suffix in ('.blue.jsonl', '.privileged.jsonl'):
                partial = audit.OUT / (key + suffix)
                if partial.exists():
                    preserved = partial.with_name(partial.name + '.interrupted')
                    if preserved.exists():
                        raise FileExistsError(preserved)
                    os.replace(partial, preserved)
            cells[key] = audit.replay(model, episode, frozen)
            audit.save(audit.OUT / 'progress.json', {'config': cfg, 'cells': cells})
    audit.save(audit.OUT / 'resume.json', {
        'reason': 'interactive process interruption',
        'original_runner_sha256': cfg['runner_sha256'],
        'resume_runner_sha256': sha256(__file__),
        'validated_completed_before_resume': sorted(existing),
    })
    audit.save(audit.OUT / 'report.json', {'config': cfg, 'cells': cells})


if __name__ == '__main__':
    main()
