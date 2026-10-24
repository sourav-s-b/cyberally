"""Same-guard attribution preflight without invoking the LLM planner."""

from blue.core.wrapper import CC4MARLEnv
from blue.policies.ordered import argmax_pick
from blue.training.llmred_eval import run_arm


def _env():
    return CC4MARLEnv(
        steps=40, red_agent="finite", temporal_features=("ages", "belief"),
        include_root_session=True,
    )


def test_strict_guard_noop_hook_has_identical_requests_and_return():
    baseline = run_arm(_env(), 7629, lambda: None, guard=True, steps=40)
    no_op_hook = run_arm(
        _env(), 7629,
        lambda: (lambda env, agent, cands, scored: argmax_pick(scored)),
        guard=True, steps=40,
    )
    assert baseline["ticks"] == no_op_hook["ticks"]
    assert baseline["actions_sha256"] == no_op_hook["actions_sha256"]
    assert baseline["return"] == no_op_hook["return"]
