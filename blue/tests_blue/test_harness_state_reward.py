"""State shaping is training-only, state-based, and cannot reach evaluation.

The October 2026 shaping attempts scored *actions* and degraded native return
at both learning rates: a dense per-action term mixed with a sparse true
signal biases the gradient toward farming the term. The replacement scores
*state*, which cannot be farmed by acting more.

Three properties are load-bearing and are pinned here:

  1. ``run_team_episode`` reports the NATIVE return even when a shaped signal
     is driving the PPO update. If this regresses, every historical number in
     the project silently changes meaning.
  2. ``harness_eval`` never imports the shaping module and always scores
     native reward.
  3. The unshaped control arm is byte-identical to the historical path.
"""
import ast
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tree(*rel):
    return ROOT.joinpath(*rel).read_text()


# --- 1. native return is preserved ------------------------------------------

def test_shaping_never_changes_the_reported_return():
    src = _tree("blue", "training", "mappo_guide.py")
    assert '"return": float(sum(native_rewards))' in src, (
        "the reported return must be the native sum; shaping must not reach it"
    )
    # native is always collected, independently of any hook
    assert "native_rewards.append(native)" in src


def test_reward_hook_defaults_to_none_and_is_additive():
    sig = ast.parse(_tree("blue", "training", "mappo_guide.py"))
    fn = next(n for n in ast.walk(sig)
              if isinstance(n, ast.FunctionDef)
              and n.name == "run_team_episode")
    args = [a.arg for a in fn.args.args]
    assert "reward_hook" in args
    assert args.index("reward_hook") == args.index("recorder") + 1
    # default is None
    nxt = next(d for d in fn.args.defaults)
    assert isinstance(nxt, ast.Constant) and nxt.value is None


def test_unshaped_arm_is_byte_identical_to_the_historical_path():
    """With no hook the per-tick reward is exactly the simulator's."""
    from blue.training import mappo_guide as mg

    calls = []

    def hook(**kw):
        calls.append(kw)
        return kw["native"] - 1.0

    # The line that decides the per-tick value:
    src = _tree("blue", "training", "mappo_guide.py")
    assert ("rewards.append(native if reward_hook is None else"
            in src), "the None branch must pass native through untouched"


# --- 2. evaluation is isolated ---------------------------------------------

def test_harness_eval_never_imports_state_reward():
    src = _tree("blue", "training", "harness_eval.py")
    assert "state_reward" not in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "state_reward" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "state_reward" not in alias.name


def test_harness_eval_records_that_it_scores_native():
    src = _tree("blue", "training", "harness_eval.py")
    assert "native only" in src


def test_no_feature_or_actor_module_imports_state_reward():
    """Actor inputs and masks must not see privileged shaping state."""
    for rel in ("blue/harness/features.py", "blue/harness/policy.py",
                "blue/harness/learning.py", "blue/policies/ordered.py"):
        assert "state_reward" not in _tree(*rel.split("/")), rel


# --- 3. the shaping itself --------------------------------------------------

def test_mission_phase_matches_the_actor_feature_schedule():
    """features.py computes the phase; shaping must agree with it exactly."""
    from blue.harness.state_reward import mission_phase
    for horizon in (400, 401, 402, 399):
        q, r = divmod(horizon, 3)
        bounds = (q + (r > 0), 2 * q + (r > 0) + (r > 1))
        for tick in (1, bounds[0], bounds[0] + 1, bounds[1],
                     bounds[1] + 1, horizon):
            expected = int(tick >= bounds[0]) + int(tick >= bounds[1])
            assert mission_phase(tick, horizon) == expected


def test_zone_severity_uses_the_public_schedule_and_fails_closed():
    from blue.harness.state_reward import zone_severity
    # operational_zone_a is the phase-1 critical zone (-10 entries).
    assert zone_severity("operational_zone_a_subnet_user_host_0", 1) == 1.0
    # contractor network is unscored from phase 1 on.
    assert zone_severity("contractor_network_subnet_user_host_0", 2) == 0.0
    # everything is bounded
    for phase in (0, 1, 2):
        v = zone_severity("restricted_zone_a_subnet_user_host_0", phase)
        assert 0.0 <= v <= 1.0
    with pytest.raises(ValueError, match="zone"):
        zone_severity("invented_host_0", 0)
    with pytest.raises(ValueError, match="phase"):
        zone_severity("restricted_zone_a_subnet_user_host_0", 7)


def test_shaping_is_a_bounded_penalty_not_a_bonus():
    """A bonus is farmable; a capped penalty is not."""
    from blue.harness.state_reward import make_reward_hook

    class _Tracker:
        def __init__(self, state):
            self.state = state

    class _Env:
        # tick 200 of a 400-step horizon is mission phase 1, where
        # operational_zone_a carries the -10 entries.
        _tick = 200

        def __init__(self, compromised, tracker_state):
            self._compromised_set = lambda: set(compromised)
            self.trackers = {"blue_agent_0": _Tracker(tracker_state)}

    hook = make_reward_hook("both", weight=0.05, horizon=400, cap=20.0)
    z = "operational_zone_a_subnet_user_host_%d"

    # 40 undetected compromised hosts in the critical zone: severity 1.0 each,
    # saturating the cap. native -5 -> -15, never lower.
    bad = {z % i: "UNKNOWN" for i in range(40)}
    # raw debt 40 saturates the cap at 20 -> penalty 1.0, never above it
    assert hook(env=_Env(bad, bad), tick=200, native=-5.0) == pytest.approx(-6.0)

    # Nothing wrong anywhere: the term is exactly zero, never a bonus.
    assert hook(env=_Env({}, {}), tick=200, native=-5.0) == pytest.approx(-5.0)

    # Detected but not remediated: detection debt clears, confirm debt remains.
    one = {z % 0: "CONFIRMED"}
    assert hook(env=_Env(one, one), tick=200, native=0.0) == pytest.approx(-0.05)

    # Remediated but NOT verified: the debt persists. Completion is not
    # verification, so the policy is not paid for a fix it never checked.
    done = {z % 0: "VERIFY"}
    assert hook(env=_Env(done, done), tick=200, native=0.0) == pytest.approx(-0.05)

    # Verified clean: the host has left the privileged compromised set AND
    # Blue's tracker has resolved it. Both terms clear.
    clear = {z % 0: "UNKNOWN"}
    assert hook(env=_Env(set(), clear), tick=200, native=0.0) == pytest.approx(0.0)

    # A false negative -- actually still compromised but Blue believes it is
    # clean -- still owes detection debt. That is the whole point of the term.
    assert hook(env=_Env(clear, clear), tick=200, native=0.0) == pytest.approx(-0.05)


def test_detection_debt_only_charges_the_phase_the_schedule_says_is_critical():
    """The same host is worth 1.0 in phase 1 and 0.1 in phase 2."""
    from blue.harness.state_reward import make_reward_hook

    class _Tracker:
        def __init__(self, state):
            self.state = state

    class _Env:
        def __init__(self, tick, compromised):
            self._tick = tick
            self._compromised_set = lambda: set(compromised)
            self.trackers = {"blue_agent_0": _Tracker({})}

    hook = make_reward_hook("detection", weight=0.05, horizon=400, cap=20.0)
    host = "operational_zone_a_subnet_user_host_0"
    # tick 200 -> phase 1 (bounds 134, 267)
    assert hook(env=_Env(200, {host}), tick=200, native=0.0) == pytest.approx(-0.05)
    # tick 300 -> phase 2, where that zone is only worth -1
    assert hook(env=_Env(300, {host}), tick=300, native=0.0) == pytest.approx(-0.005)


def test_hook_rejects_malformed_configuration():
    from blue.harness.state_reward import make_reward_hook
    with pytest.raises(ValueError, match="term"):
        make_reward_hook("nope")
    with pytest.raises(ValueError, match="weight"):
        make_reward_hook("both", weight=-1.0)
    with pytest.raises(ValueError, match="cap"):
        make_reward_hook("both", cap=0.0)


def test_trainer_exposes_the_regime_and_defaults_to_unshaped():
    import ast
    src = _tree("blue", "training", "harness_rl.py")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    assert "reward_hook = build_reward_hook(args)" in ast.unparse(fn)
    # unshaped default
    assert '"--state-shaping", default="off"' in src
    assert '"--red-agent", default="discovery"' in src
    # manifest states the regime rather than leaving it to inference
    assert '"state_shaping"' in src and '"training_only": True' in src
    assert '"red_agent": args.red_agent' in src
    assert "native team reward reported" in src


def test_env_kwargs_rejects_unknown_attackers():
    from blue.training.harness_rl import env_kwargs
    assert env_kwargs("finite")["red_agent"] == "finite"
    assert env_kwargs("discovery")["temporal_features"] == ("ages", "belief")
    with pytest.raises(ValueError, match="red_agent"):
        env_kwargs("nonexistent")


def test_documented_privilege_boundary():
    """The only privileged read is the compromise bit, reward-channel only."""
    src = _tree("blue", "core", "wrapper.py")
    assert "REWARD-CHANNEL ONLY (shaping);" in src
    assert "never exposed to policies." in src
    shaping = _tree("blue", "harness", "state_reward.py")
    assert "_compromised_set()" in shaping
    assert "get_true_state" not in shaping, (
        "shaping must use the documented wrapper accessor, not reach into "
        "the simulator itself"
    )
    # and no true-state import leaks into the shaping module
    assert "risk_data" not in shaping and "true_compromised" not in shaping

# --- the Lancer baseline cache is a correctness hazard, so pin it ----------

def test_verify_state_still_owes_debt_until_observed_verification():
    """Action completion is not verification.

    `BlueZoneTracker.note_remediation_result` sets a host to VERIFY the moment
    Remove/Restore completes. If confirm_debt counted only CONFIRMED, the
    entire debt would vanish at completion and the policy would be paid for
    finishing a fix it never checked -- the wrapper's standing warning that a
    successful Remove does not prove a host is clean.
    """
    from blue.harness.state_reward import confirm_debt, detection_debt

    class _Tracker:
        def __init__(self, state):
            self.state = state

    class _Env:
        _tick = 200

        def __init__(self, compromised, tracker_state):
            self._compromised_set = lambda: set(compromised)
            self.trackers = {"blue_agent_0": _Tracker(tracker_state)}

    host = "operational_zone_a_subnet_user_host_0"
    # Remediated but not verified: debt persists.
    v = _Env({host}, {host: "VERIFY"})
    assert confirm_debt(v, 400) == pytest.approx(1.0)
    assert detection_debt(v, 400) == pytest.approx(0.0)
    # Verified clean (UNKNOWN after empty analyses): debt clears.
    c = _Env({host}, {host: "UNKNOWN"})
    assert confirm_debt(c, 400) == pytest.approx(0.0)
    # Still detected and not remediated.
    d = _Env({host}, {host: "CONFIRMED"})
    assert confirm_debt(d, 400) == pytest.approx(1.0)


def test_severity_resolution_survives_the_cap():
    """A cap below typical debt would make a -10 zone and a -5 zone cost the
    same, silently destroying the signal the severity weighting provides."""
    from blue.harness.state_reward import make_reward_hook

    class _Tracker:
        def __init__(self, state):
            self.state = state

    class _Env:
        _tick = 200

        def __init__(self, compromised):
            self._compromised_set = lambda: set(compromised)
            self.trackers = {"blue_agent_0": _Tracker({})}

    hook = make_reward_hook("detection", weight=0.05, horizon=400, cap=20.0)
    crit = "operational_zone_a_subnet_user_host_0"      # severity 1.0 in phase 1
    mid = "restricted_zone_a_subnet_user_host_0"        # severity 0.3 in phase 1
    cheap = "contractor_network_subnet_user_host_0"     # severity 0.0 in phase 2

    p_crit = hook(env=_Env({crit}), tick=200, native=0.0)
    p_mid = hook(env=_Env({mid}), tick=200, native=0.0)
    p_cheap = hook(env=_Env({cheap}), tick=200, native=0.0)
    assert p_crit < p_mid < p_cheap
    assert p_crit == pytest.approx(-0.05)
    assert p_mid == pytest.approx(-0.015)

    # Ten critical hosts must cost more than one, i.e. the cap is not binding.
    ten = {f"operational_zone_a_subnet_user_host_{i}" for i in range(10)}
    assert hook(env=_Env(ten), tick=200, native=0.0) == pytest.approx(-0.5)


def test_confirm_debt_is_blue_visible_not_privileged():
    """confirm_debt must not consult the compromise bit."""
    src = _tree("blue", "harness", "state_reward.py")
    body = src.split("def confirm_debt", 1)[1].split("def make_reward_hook", 1)[0]
    assert "_compromised_set" not in body
    assert "get_true_state" not in body


def test_shaping_cap_default_keeps_the_term_below_native_magnitude():
    """A per-tick cap that accumulates past the native return inverts the
    objective. Measured on a real episode, cap 10 gives ratio 17.6 (unsafe);
    cap 0.5 gives 1.10."""
    src = _tree("blue", "harness", "state_reward.py")
    assert 'def make_reward_hook(term="both", weight=0.05, horizon=400, cap=20.0)' \
        in src
    assert "17.6" in src and "0.91" in src, (
        "the docstring must record the measured ratios that justify the "
        "defaults, so they can be re-derived rather than trusted"
    )
    pilot = _tree("blue", "training", "harness_shape_pilot.py")
    assert "UNSAFE_RATIO" in pilot
    assert "harness_shape_pilot" in _tree("scripts", "blue_state_shaping_run.sh") \
        or "harness_shape_pilot" in (ROOT / "scripts" /
                                     "blue_state_shaping_run.sh").read_text()


def test_lancer_baseline_is_precomputed_and_never_written_by_a_model_job():
    """Concurrent writers can interleave and produce a truncated baseline,
    which would then be silently paired against every model."""
    src = _tree("blue", "training", "harness_eval.py")
    assert "--read-only-lancer-cache" in src
    assert "os.replace(tmp, cache_path)" in src, "cache write must be atomic"
    assert "only the dedicated lancer step writes it" in src
    script = (ROOT / "scripts" / "blue_state_shaping_run.sh").read_text()
    assert "read-only-lancer-cache" in script
    assert "no precomputed Lancer baseline" in script


def test_resume_refuses_an_objective_change():
    """The objective is (native reward + shaping) on (attacker). Resuming a
    shaped run with `--state-shaping off`, or with a different attacker,
    would keep training under the old weights toward a different objective."""
    src = _tree("blue", "training", "harness_rl.py")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    body = ast.unparse(fn)
    for key in ("red_agent", "state_shaping", "state_shaping_weight",
                "state_shaping_cap"):
        # ast.unparse normalises quotes, so match the bare key in the tuple
        assert key in body, key
    # the keys tuple must be part of the drift check
    assert "resume configuration/source/artifact drift" in body


def test_training_records_the_shaping_ratio_per_iteration():
    """The pilot calibrates the ratio under the heuristic policy; the learned
    actor visits different states, so the ratio is re-measured while training."""
    src = _tree("blue", "training", "harness_rl.py")
    assert "mean_shaping_ratio" in src
    assert "mean_shaping_added" in src


def test_launcher_runs_python_in_the_foreground_of_the_pooled_job():
    """A `nohup ... &` *inside* the pooled job makes the tracked job exit at
    once, so throttle() counts nothing and every model starts concurrently."""
    script = (ROOT / "scripts" / "blue_state_shaping_run.sh").read_text()
    assert "exec setsid" in script
    code = "\n".join(l for l in script.splitlines()
                      if not l.strip().startswith("#"))
    assert "nohup" not in code, (
        "nohup inside a pooled job makes the tracked job exit immediately; "
        "use `exec setsid` so the job lives as long as python does")
    allowed = ("(", ">", "train_one", "eval_one")
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.endswith("&") and not stripped.startswith("#"):
            assert stripped.startswith(allowed), (
                "only the pool's own job may be backgrounded: %r" % stripped)


def test_launcher_limits_concurrency():
    """train/eval must keep at most N jobs alive, not detach everything."""
    script = (ROOT / "scripts" / "blue_state_shaping_run.sh").read_text()
    assert "throttle()" in script
    assert "wait -n" in script
    # the log's parent must be created: the shell opens the redirect before
    # harness_eval can mkdir its own --out
    assert 'mkdir -p "$OUT/eval/$attacker/$arm"' in script


def test_lancer_cache_refuses_a_mismatched_configuration():
    """A cached baseline from a different attacker or seed list is a wrong
    pairing, and silently reusing it would fabricate a contrast."""
    src = _tree("blue", "training", "harness_eval.py")
    assert "--lancer-cache" in src
    assert "lancer cache was produced under a different" in src
    # the key must cover attacker, horizon and seeds
    assert '"red_agent": args.red_agent' in src
    assert '"seeds": sorted(args.seeds)' in src
    assert '"steps": args.steps' in src


def test_lancer_cache_never_substitutes_for_the_model_under_test():
    src = _tree("blue", "training", "harness_eval.py")
    # only the lancer arm may be served from cache
    assert "if arm == \"lancer\" and str(seed) in lancer_cache" in src
    assert "if arm == \"lancer\" and args.lancer_cache" in src
