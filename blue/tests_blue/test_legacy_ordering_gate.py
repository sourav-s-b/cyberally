"""The legacy ordering entry point must not silently publish invalid metrics.

`blue/analysis/ordering.py` computed coverage/age/delay from its own
privileged tick loop, which disagrees with the wrapper in four ways (see
`blue/analysis/metrics.py`). Its `return` is exact, which is why the module
stays for reproducing history, so the invalid columns are withheld behind an
explicit opt-in instead of being deleted or silently emitted.
"""
import inspect

import pytest

from blue.analysis.ordering import (LEGACY_INVALID_METRICS,
                                    LEGACY_INVALID_REASON, VARIANTS,
                                    run_ordering_episode)

ENV_KW = {"temporal_features": ("ages", "belief"),
          "include_root_session": True, "red_agent": "discovery"}


def _episode(**kwargs):
    from blue.policies.ordered import CursorSweep, OrderedPolicy
    policy = OrderedPolicy(order_fn=CursorSweep(), guard=True)
    return run_ordering_episode(policy, seed=7629, steps=40, **ENV_KW,
                                **kwargs)


@pytest.fixture(scope="module")
def episodes():
    return _episode(), _episode(emit_invalid_metrics=True)


def test_invalid_columns_are_withheld_by_default(episodes):
    withheld, _ = episodes
    for key in LEGACY_INVALID_METRICS:
        assert withheld[key] is None, f"{key} must not be silently populated"
    assert withheld["legacy_invalid_metrics"] == LEGACY_INVALID_REASON


def test_valid_fields_do_not_depend_on_the_opt_in(episodes):
    withheld, opted_in = episodes
    for key in ("return", "steps", "n_compromised", "n_remediations"):
        assert withheld[key] == opted_in[key], \
            f"{key} must be identical with and without the opt-in"


def test_opt_in_reproduces_the_legacy_columns(episodes):
    _, opted_in = episodes
    assert "legacy_invalid_metrics" not in opted_in
    for key in ("max_age", "n_undetected"):
        assert isinstance(opted_in[key], int)
    assert 0.0 <= opted_in["coverage"] <= 1.0


def test_cli_exposes_the_opt_in_flag_and_warns_without_it():
    src = inspect.getsource(__import__("blue.analysis.ordering",
                                       fromlist=["main"]).main)
    assert '"--emit-invalid-metrics"' in src
    assert "KNOWN INVALID" in src
    assert "parity" in VARIANTS
