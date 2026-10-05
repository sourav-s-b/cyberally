"""Response-rule variants: defaults preserve behavior exactly."""

import pytest  # noqa: F401 (kept for venv symmetry with sibling tests)


def _run(policy, seed, steps=60):
    from blue.training import mappo_guide as mg
    return mg.run_team_episode(policy, seed, steps, **mg.ENV_KW)


def _lancer(**kw):
    from blue.policies.ordered import LancerValues, OrderedPolicy
    return OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5),
                         guard=False, **kw)


def test_variant_flags_off_reproduce_lancer_exactly():
    for seed in (7629, 7701):
        a = _run(_lancer(), seed)
        b = _run(_lancer(no_restore_escalation=False, no_verify=False),
                 seed)
        assert a["return"] == b["return"]
        assert a["coverage"] == b["coverage"]


def test_variants_execute_and_report():
    for kw in ({"no_restore_escalation": True}, {"no_verify": True}):
        r = _run(_lancer(**kw), 7629)
        assert r["ticks"] > 0
