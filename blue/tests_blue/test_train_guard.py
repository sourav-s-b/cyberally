"""Training must run in the same guard regime as evaluation.

`risk_actor.py` hardcoded `guard=False` in the training policy while
honouring `--sweep-guard` only in `cmd_eval`. The consequence was that every
checkpoint the project owns was fitted in the unguarded regime the LLM-Red
evaluation identified as invalid (the learner may re-pick the same host every
tick), and then scored under a guard it had never trained against.

These tests fail if the training path stops consulting `--sweep-guard`, if
the flag is accepted but not recorded in the manifest, or if the guard stops
actually restricting the candidate list the learner chooses from.
"""
import argparse
import inspect

import pytest

torch = pytest.importorskip("torch")

from blue.training import risk_actor as ra  # noqa: E402


def _cli_parser():
    """The training CLI, rebuilt without running main()."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--sweep-guard", choices=("off", "strict"), default="off")
    return ap


def test_training_policy_is_not_hardcoded_unguarded():
    src = inspect.getsource(ra.cmd_train)
    assert "guard=False" not in src, (
        "cmd_train must not hardcode an unguarded sweep; the guard is part "
        "of the policy being trained, not an evaluation-time add-on"
    )
    assert "guard=train_guard" in src
    assert 'args.sweep_guard == "strict"' in src


def test_manifest_records_the_training_regime():
    """A reader must never have to infer which regime produced the weights."""
    src = inspect.getsource(ra.cmd_train)
    assert '"trained_with_guard": train_guard' in src
    assert '"sweep_guard": args.sweep_guard' in src


def test_default_is_still_off_so_existing_commands_are_unchanged():
    assert _cli_parser().parse_args(["--mode", "train"]).sweep_guard == "off"
    assert _cli_parser().parse_args(
        ["--mode", "train", "--sweep-guard", "strict"]).sweep_guard == "strict"


class _Tracker(object):
    def __init__(self, stamps):
        self.last_analysis = stamps


class _Env(object):
    def __init__(self, agent_hosts, analysed):
        self.hostnames = {"blue_agent_0": list(agent_hosts)}
        self.trackers = {"blue_agent_0": _Tracker(analysed)}


def test_guard_restricts_the_candidate_list_the_learner_sees():
    """Not decorative: a visited host must drop out under the guard."""
    from blue.policies.ordered import SweepScheduler

    hosts = ["public_access_zone_subnet_user_host_%d" % i for i in range(3)]
    cands = list(hosts)

    open_ = SweepScheduler(guard=False)
    guarded = SweepScheduler(guard=True)

    # Host 0 has a completed analysis, so the sweep counts it as visited.
    env = _Env(hosts, {hosts[0]: 3, hosts[1]: None, hosts[2]: None})

    seen = {}
    for name, sched in (("open", open_), ("guard", guarded)):
        seen[name] = sched.pick(env, "blue_agent_0", cands,
                                score_fn=lambda e, a, h: (
                                    10.0 if h == hosts[0] else 1.0))
    # Both prefer host 0 on score. Only the guard is expected to refuse it.
    assert seen["open"] == hosts[0]
    assert seen["guard"] != hosts[0]
    assert seen["guard"] in hosts[1:]


def test_attacker_is_an_explicit_choice_not_a_hardcoded_constant():
    """The training attacker must be selectable and recorded.

    `mappo_guide.ENV_KW` pins `red_agent="discovery"`, the regime the LLM-Red
    evaluation showed cannot punish a broken investigation policy. Training
    silently in it would optimise against a world where coverage and ML
    inputs barely matter.
    """
    from blue.core.wrapper import RED_AGENTS

    kw = ra.env_kwargs("finite")
    assert kw["red_agent"] == "finite"
    assert RED_AGENTS["finite"] is not None
    # The rest of the observation contract is untouched.
    assert kw["temporal_features"] == ("ages", "belief")
    assert kw["include_root_session"] is True
    # Defaults must not silently change for existing commands.
    assert ra.env_kwargs("discovery")["red_agent"] == "discovery"
    with pytest.raises(ValueError, match="unknown red_agent"):
        ra.env_kwargs("nonexistent")


def test_manifests_record_the_attacker():
    train_src = inspect.getsource(ra.cmd_train)
    eval_src = inspect.getsource(ra.cmd_eval)
    assert '"red_agent": args.red_agent' in train_src
    assert '"red_agent": args.red_agent' in eval_src
    # and both call sites actually use it
    assert train_src.count("env_kwargs(args.red_agent)") == 1
    assert eval_src.count("env_kwargs(args.red_agent)") == 1
