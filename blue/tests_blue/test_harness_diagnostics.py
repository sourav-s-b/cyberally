import copy

import numpy as np
import pytest

from blue.harness.diagnostics import RequestDigest, input_sensitivity
from blue.training.harness_audit import audit


def test_request_hash_accounts_for_host_order_busy_ticks():
    def digest(events):
        d = RequestDigest()
        for event in events:
            d.add(*event)
        return d.result()
    events = [(0, 'a', 'Analyse', 'h1'), (1, 'a', 'Sleep', None)]
    assert digest(events) == digest(events)
    assert digest(events) != digest(list(reversed(events)))
    assert digest(events) != digest([(0, 'a', 'Analyse', 'h2'), events[1]])
    assert digest(events)['requested_actions_count'] == 2


def test_sensitivity_changes_logits_without_mutating_inputs_or_rng():
    th = pytest.importorskip('torch')
    actor = th.nn.Linear(4, 1, bias=False)
    class ScalarActor(th.nn.Module):
        def forward(self, x):
            return actor(x).squeeze(-1)
    with th.no_grad(): actor.weight[:] = th.tensor([[0., 0., 1., 1.]])
    features = th.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.]])
    before = features.clone(); rng = th.get_rng_state().clone()
    result = input_sensitivity(ScalarActor(), features, th.tensor([0., .1]), .25)
    assert result['ml_logit_delta_max'] > 0
    assert result['novelty_top_index_changed'] == 1
    assert th.equal(before, features) and th.equal(rng, th.get_rng_state())


def test_audit_requires_matched_episodes_and_preserves_every_model():
    cells = {k: {'8221': -10., '8222': -20.} for k in
             ['lancer', 'guard', 'zero_s0', 'zero_s1', 'risk_s0', 'risk_s1', 'both_s0', 'both_s1']}
    cells['guard']['8222'] = -22
    result = audit(cells)
    assert result['guard_minus_lancer'] == -1
    assert len(result['runs']) == 6
    assert result['independent_training_seeds'] == 2
    broken = copy.deepcopy(cells); del broken['risk_s1']['8221']
    with pytest.raises(ValueError, match='unmatched'): audit(broken)


def test_detector_summary_weights_episodes_not_rows():
    from blue.training.harness_ml import episode_summary
    episodes = {}
    for seed, n, delta in ((1, 10, .2), (2, 10000, -.1), (3, 50, .2)):
        base = dict(n=n, pr_auc=.5, roc_auc=.5, brier=.2, log_loss=.3)
        hgb = {**base, 'pr_auc': .5+delta}
        episodes[str(seed)] = {'hgb': hgb, 'logistic': base}
    result = episode_summary(episodes)
    assert result['n_episodes'] == 3
    assert result['paired_hgb_minus_logistic']['pr_auc']['mean'] == pytest.approx(.1)
    assert result['paired_hgb_minus_logistic']['pr_auc']['n_episodes'] == 3
