import numpy as np

import blue.training.log_transitions as lt  # noqa: E402


def _env_kw():
    from blue.training.mappo_guide import ENV_KW
    kw = dict(ENV_KW)
    kw["red_agent"] = "finite"
    return kw


def test_log_episode_contract_gates_and_determinism():
    r = lt.log_episode("guard", 7706, 8, _env_kw())
    assert r["obs"].shape == (7, 5, 867)
    assert r["actions"].shape == (7, 5)
    assert r["rewards"].shape == (7,)
    for t in range(r["actions"].shape[0]):
        for a in range(5):
            assert r["masks"][t, a, r["actions"][t, a]] == 1
    assert r["done"].sum() == 1 and r["done"][-1] == 1
    # native horizon: steps=8 ends at tick 7 (one-step-early, as the
    # mappo guide-window tests also observed)
    r2 = lt.log_episode("guard", 7706, 8, _env_kw())
    assert np.array_equal(r["obs"], r2["obs"])
    assert r["rewards"].tolist() == r2["rewards"].tolist()


def test_random_arm_policy_seed_changes_actions():
    import blue.core.wrapper as w
    seqs = {}
    for pol_seed in (111, 222):
        pol = lt.build_arm("random", api_seed=pol_seed)
        env = w.CC4MARLEnv(seed=7706, steps=10, **_env_kw())
        env.reset(seed=7706)
        pol.reset()
        seqs[pol_seed] = [int(pol.select(env, "blue_agent_0"))
                          for _ in range(8)]
        env.close()
    assert seqs[111] != seqs[222]
