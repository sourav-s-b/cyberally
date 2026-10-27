"""Behavioral checks for the anchored categorical pilot."""
import copy

import numpy as np
import pytest


def test_reference_distribution_tie_and_singleton():
    torch = pytest.importorskip("torch")
    from blue.training.harness_anchor import reference_log_probs
    scored = [(1.0, "a"), (1.0, "z"), (0.9, "b")]
    q = reference_log_probs(scored).exp()
    assert torch.isclose(q.sum(), torch.tensor(1.0))
    assert q[1] == pytest.approx(0.9)
    assert (q > 0).all()
    assert reference_log_probs([(0.1, "solo")]).exp().item() == pytest.approx(1.0)


def test_ppo_rollback_restores_actor_critic_and_optimizer():
    torch = pytest.importorskip("torch")
    from blue.training.harness_anchor import ppo_update
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(3, 1)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.zeros_(actor.bias)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=0.1)
    prior = np.log(np.array([0.9, 0.1], dtype=np.float32))
    rows = []
    for i in range(32):
        choice = i % 2
        rows.append({"feats": np.array([[1., i % 3], [i % 4, 1.]], dtype=np.float32),
                     "joint": np.array([1., i % 2, i % 5], dtype=np.float32),
                     "mask": [1., 1.], "prior_log": prior,
                     "old_log_probs": prior, "choice": choice,
                     "logp": float(prior[choice]), "ret": float(10 if choice else -10)})
    before_actor = copy.deepcopy(actor.state_dict())
    before_critic = copy.deepcopy(critic.state_dict())
    before_opt = copy.deepcopy(opt.state_dict())
    stats = ppo_update(actor, critic, opt, rows, max_update_kl=1e-12,
                       epochs=1, minibatch=32)
    assert stats["rejected_epochs"] == 1
    assert stats["accepted_epochs"] == 0
    assert all(torch.equal(v, before_actor[k]) for k, v in actor.state_dict().items())
    assert all(torch.equal(v, before_critic[k]) for k, v in critic.state_dict().items())
    assert opt.state_dict() == before_opt
