"""
test_deterministic_evaluation.py
--------------------------------
Deterministic episode evaluation of the custom OWASP Juice Shop simulation.

Measures whether Blue's defensive actions improve the simulated security outcome
across standardized scenarios over an identical episode horizon (8 steps):

  Scenario A: Baseline — clean environment, no incident.
  Scenario B: Incident without Blue response — simulated incident occurs,
              Blue remains passive (Monitor only); ongoing penalties accumulate.
  Scenario C: Incident with Blue direct response — simulated incident occurs,
              Blue executes DETECT, BLOCK, and RESTORE.
  Scenario D: Incident with autonomous HybridBluePolicy response — simulated incident
              occurs, Blue policy autonomously selects Analyse and Remove.

Reports:
  - Incident occurrence
  - Detection success
  - Attacker containment / blocking
  - Service restoration success
  - Final incident and compromise flags
  - Service health (web, API, DB) before and after response
  - Total Red reward
  - Total Blue reward
  - Step-by-step reward breakdown
  - Total ongoing compromise penalties incurred
  - Defensive net benefit (comparative improvement)
"""

from environment.owasp.env import OWASPEnvironment
from environment.owasp.actions import RedAction, BlueAction
from environment.owasp.blue_policy_adapter import OWASPBluePolicyAdapter


EPISODE_STEPS = 8


def evaluate_scenario_a():
    """Scenario A: Baseline (Clean environment, No incident)."""
    env = OWASPEnvironment(max_steps=EPISODE_STEPS)
    env.reset()

    # Step plan: Red performs discovery; Blue monitors; no exploits.
    red_actions = [
        RedAction.DISCOVER_APPLICATION,
        RedAction.DISCOVER_API,
        RedAction.NOOP,
        RedAction.NOOP,
        RedAction.NOOP,
        RedAction.NOOP,
        RedAction.NOOP,
        RedAction.NOOP,
    ]
    blue_actions = [BlueAction.MONITOR] * EPISODE_STEPS

    step_log = []
    total_red = 0.0
    total_blue = 0.0
    total_penalties = 0.0

    for step in range(EPISODE_STEPS):
        act_red = red_actions[step]
        act_blue = blue_actions[step]

        obs, rewards, term, trunc, info = env.step({
            "red": act_red,
            "blue": act_blue,
        })

        r_red = rewards["red"]
        r_blue = rewards["blue"]
        total_red += r_red
        total_blue += r_blue

        # Penalty check: -10 is deducted whenever state is compromised/incident active
        is_penalized = env.state.compromised or env.state.incident_active
        if is_penalized:
            total_penalties += 10.0

        step_log.append({
            "step": step + 1,
            "red_action": act_red.value,
            "blue_action": act_blue.value,
            "red_result": info["red_result"],
            "blue_result": info["blue_result"],
            "red_reward": r_red,
            "blue_reward": r_blue,
            "incident_active": env.state.incident_active,
            "compromised": env.state.compromised,
        })

    return {
        "scenario": "A. Baseline (Clean)",
        "steps": EPISODE_STEPS,
        "incident_occurred": False,
        "detection_success": env.state.attack_detected,
        "attacker_blocked": env.state.attacker_blocked,
        "restoration_success": False,  # Not needed
        "final_incident_active": env.state.incident_active,
        "final_compromised": env.state.compromised,
        "final_admin_access": env.state.admin_access,
        "initial_health": "UP/AVAIL/AVAIL",
        "final_health": f"{'UP' if env.state.web_server_up else 'DOWN'}/{'AVAIL' if env.state.api_available else 'DOWN'}/{'AVAIL' if env.state.database_available else 'DOWN'}",
        "total_red_reward": total_red,
        "total_blue_reward": total_blue,
        "total_penalties": total_penalties,
        "step_log": step_log,
    }


def evaluate_scenario_b():
    """Scenario B: Incident without Blue response (Unmitigated attack)."""
    env = OWASPEnvironment(max_steps=EPISODE_STEPS)
    env.reset()

    # Step plan: Red discovers, triggers SQL injection incident, escalates privilege.
    # Blue remains passive (MONITOR only).
    red_actions = [
        RedAction.DISCOVER_APPLICATION,  # Step 1: discover app
        RedAction.SQL_INJECTION,         # Step 2: trigger simulated incident
        RedAction.PRIVILEGE_ESCALATION,  # Step 3: escalate to admin
        RedAction.NOOP,                  # Steps 4-8: idle
        RedAction.NOOP,
        RedAction.NOOP,
        RedAction.NOOP,
        RedAction.NOOP,
    ]
    blue_actions = [BlueAction.MONITOR] * EPISODE_STEPS

    step_log = []
    total_red = 0.0
    total_blue = 0.0
    total_penalties = 0.0
    incident_occurred = False

    for step in range(EPISODE_STEPS):
        act_red = red_actions[step]
        act_blue = blue_actions[step]

        obs, rewards, term, trunc, info = env.step({
            "red": act_red,
            "blue": act_blue,
        })

        if env.state.incident_active or env.state.compromised:
            incident_occurred = True

        r_red = rewards["red"]
        r_blue = rewards["blue"]
        total_red += r_red
        total_blue += r_blue

        is_penalized = env.state.compromised or env.state.incident_active
        if is_penalized:
            total_penalties += 10.0

        step_log.append({
            "step": step + 1,
            "red_action": act_red.value,
            "blue_action": act_blue.value,
            "red_result": info["red_result"],
            "blue_result": info["blue_result"],
            "red_reward": r_red,
            "blue_reward": r_blue,
            "incident_active": env.state.incident_active,
            "compromised": env.state.compromised,
        })

    return {
        "scenario": "B. Incident (No Blue Response)",
        "steps": EPISODE_STEPS,
        "incident_occurred": incident_occurred,
        "detection_success": env.state.attack_detected,
        "attacker_blocked": env.state.attacker_blocked,
        "restoration_success": False,
        "final_incident_active": env.state.incident_active,
        "final_compromised": env.state.compromised,
        "final_admin_access": env.state.admin_access,
        "initial_health": "UP/AVAIL/AVAIL",
        "final_health": f"{'UP' if env.state.web_server_up else 'DOWN'}/{'AVAIL' if env.state.api_available else 'DOWN'}/{'AVAIL' if env.state.database_available else 'DOWN'}",
        "total_red_reward": total_red,
        "total_blue_reward": total_blue,
        "total_penalties": total_penalties,
        "step_log": step_log,
    }


def evaluate_scenario_c():
    """Scenario C: Incident with Blue direct response (DETECT, BLOCK, RESTORE)."""
    env = OWASPEnvironment(max_steps=EPISODE_STEPS)
    env.reset()

    # Step plan:
    # Step 1: Red discovers app; Blue monitors.
    # Step 2: Red triggers SQL injection incident; Blue monitors.
    # Step 3: Red attempts escalation; Blue DETECTS.
    # Step 4: Red attempts exploit; Blue BLOCKS.
    # Step 5: Red blocked; Blue RESTORES.
    # Steps 6-8: Red attempts actions (blocked); Blue monitors.
    red_actions = [
        RedAction.DISCOVER_APPLICATION,  # Step 1: discover app
        RedAction.SQL_INJECTION,         # Step 2: trigger simulated incident
        RedAction.PRIVILEGE_ESCALATION,  # Step 3: escalate attempt
        RedAction.SQL_INJECTION,         # Step 4: exploit attempt
        RedAction.SQL_INJECTION,         # Step 5: exploit attempt (blocked)
        RedAction.SQL_INJECTION,         # Step 6: exploit attempt (blocked)
        RedAction.PRIVILEGE_ESCALATION,  # Step 7: escalate attempt (blocked)
        RedAction.NOOP,                  # Step 8: idle
    ]
    blue_actions = [
        BlueAction.MONITOR,  # Step 1
        BlueAction.MONITOR,  # Step 2
        BlueAction.DETECT,   # Step 3: detect incident
        BlueAction.BLOCK,    # Step 4: block attacker
        BlueAction.RESTORE,  # Step 5: restore application
        BlueAction.MONITOR,  # Step 6
        BlueAction.MONITOR,  # Step 7
        BlueAction.MONITOR,  # Step 8
    ]

    step_log = []
    total_red = 0.0
    total_blue = 0.0
    total_penalties = 0.0
    incident_occurred = False
    restoration_success = False

    for step in range(EPISODE_STEPS):
        act_red = red_actions[step]
        act_blue = blue_actions[step]

        obs, rewards, term, trunc, info = env.step({
            "red": act_red,
            "blue": act_blue,
        })

        if env.state.incident_active or env.state.compromised:
            incident_occurred = True

        if info["blue_result"] == "service_restored":
            restoration_success = True

        r_red = rewards["red"]
        r_blue = rewards["blue"]
        total_red += r_red
        total_blue += r_blue

        is_penalized = env.state.compromised or env.state.incident_active
        if is_penalized:
            total_penalties += 10.0

        step_log.append({
            "step": step + 1,
            "red_action": act_red.value,
            "blue_action": act_blue.value,
            "red_result": info["red_result"],
            "blue_result": info["blue_result"],
            "red_reward": r_red,
            "blue_reward": r_blue,
            "incident_active": env.state.incident_active,
            "compromised": env.state.compromised,
        })

    return {
        "scenario": "C. Incident (Blue Response)",
        "steps": EPISODE_STEPS,
        "incident_occurred": incident_occurred,
        "detection_success": env.state.attack_detected,
        "attacker_blocked": env.state.attacker_blocked,
        "restoration_success": restoration_success,
        "final_incident_active": env.state.incident_active,
        "final_compromised": env.state.compromised,
        "final_admin_access": env.state.admin_access,
        "initial_health": "UP/AVAIL/AVAIL",
        "final_health": f"{'UP' if env.state.web_server_up else 'DOWN'}/{'AVAIL' if env.state.api_available else 'DOWN'}/{'AVAIL' if env.state.database_available else 'DOWN'}",
        "total_red_reward": total_red,
        "total_blue_reward": total_blue,
        "total_penalties": total_penalties,
        "step_log": step_log,
    }


def evaluate_scenario_d():
    """Scenario D: Incident with Autonomous HybridBluePolicy response."""
    env = OWASPEnvironment(max_steps=EPISODE_STEPS)
    env.reset()
    blue_adapter = OWASPBluePolicyAdapter(agent="blue_agent_0")

    red_actions = [
        RedAction.DISCOVER_APPLICATION,  # Step 1
        RedAction.SQL_INJECTION,         # Step 2
        RedAction.PRIVILEGE_ESCALATION,  # Step 3
        RedAction.SQL_INJECTION,         # Step 4 (blocked)
        RedAction.SQL_INJECTION,         # Step 5 (blocked)
        RedAction.NOOP,                  # Step 6
        RedAction.NOOP,                  # Step 7
        RedAction.NOOP,                  # Step 8
    ]

    step_log = []
    total_red = 0.0
    total_blue = 0.0
    total_penalties = 0.0
    incident_occurred = False
    restoration_success = False

    for step in range(EPISODE_STEPS):
        act_red = red_actions[step]

        # HybridBluePolicy autonomously selects action based on environment state
        action_idx, cage4_act, owasp_act = blue_adapter.select_action(env.state)

        # Count environment step
        env.state.step_count += 1
        red_result = env._execute_red_action(act_red)
        blue_result = blue_adapter.execute_action(env, cage4_act, owasp_act)

        if env.state.incident_active or env.state.compromised:
            incident_occurred = True

        if blue_result == "service_restored":
            restoration_success = True

        from environment.owasp.rewards import calculate_rewards
        rewards = calculate_rewards(env.state, red_result, blue_result)

        r_red = rewards["red"]
        r_blue = rewards["blue"]
        total_red += r_red
        total_blue += r_blue

        is_penalized = env.state.compromised or env.state.incident_active
        if is_penalized:
            total_penalties += 10.0

        step_log.append({
            "step": step + 1,
            "red_action": act_red.value,
            "blue_action": f"{type(cage4_act).__name__}->{owasp_act.value}",
            "red_result": red_result,
            "blue_result": blue_result,
            "red_reward": r_red,
            "blue_reward": r_blue,
            "incident_active": env.state.incident_active,
            "compromised": env.state.compromised,
        })

    return {
        "scenario": "D. Incident (Hybrid Policy)",
        "steps": EPISODE_STEPS,
        "incident_occurred": incident_occurred,
        "detection_success": env.state.attack_detected,
        "attacker_blocked": env.state.attacker_blocked,
        "restoration_success": restoration_success,
        "final_incident_active": env.state.incident_active,
        "final_compromised": env.state.compromised,
        "final_admin_access": env.state.admin_access,
        "initial_health": "UP/AVAIL/AVAIL",
        "final_health": f"{'UP' if env.state.web_server_up else 'DOWN'}/{'AVAIL' if env.state.api_available else 'DOWN'}/{'AVAIL' if env.state.database_available else 'DOWN'}",
        "total_red_reward": total_red,
        "total_blue_reward": total_blue,
        "total_penalties": total_penalties,
        "step_log": step_log,
    }


def print_comparison_table(results):
    print("\n" + "=" * 95)
    print("DETERMINISTIC EVALUATION SUMMARY ACROSS STANDARDIZED SCENARIOS (8 STEPS)")
    print("=" * 95)

    headers = [
        "Metric",
        "A. Baseline",
        "B. Incident (No Blue)",
        "C. Incident (Blue Resp)",
        "D. Hybrid Policy",
    ]
    col_w = [26, 16, 23, 23, 18]

    def fmt_row(row):
        return "".join(str(val).ljust(col_w[i]) for i, val in enumerate(row))

    print(fmt_row(headers))
    print("-" * 105)

    res_a, res_b, res_c, res_d = results

    rows = [
        ["Incident Occurred", res_a["incident_occurred"], res_b["incident_occurred"], res_c["incident_occurred"], res_d["incident_occurred"]],
        ["Detection Succeeded", res_a["detection_success"], res_b["detection_success"], res_c["detection_success"], res_d["detection_success"]],
        ["Attacker Blocked", res_a["attacker_blocked"], res_b["attacker_blocked"], res_c["attacker_blocked"], res_d["attacker_blocked"]],
        ["Restoration Succeeded", res_a["restoration_success"], res_b["restoration_success"], res_c["restoration_success"], res_d["restoration_success"]],
        ["Final Incident Active", res_a["final_incident_active"], res_b["final_incident_active"], res_c["final_incident_active"], res_d["final_incident_active"]],
        ["Final Compromised", res_a["final_compromised"], res_b["final_compromised"], res_c["final_compromised"], res_d["final_compromised"]],
        ["Final Admin Access", res_a["final_admin_access"], res_b["final_admin_access"], res_c["final_admin_access"], res_d["final_admin_access"]],
        ["Service Health", res_a["final_health"], res_b["final_health"], res_c["final_health"], res_d["final_health"]],
        ["Total Red Reward", f"{res_a['total_red_reward']:.1f}", f"{res_b['total_red_reward']:.1f}", f"{res_c['total_red_reward']:.1f}", f"{res_d['total_red_reward']:.1f}"],
        ["Total Blue Reward", f"{res_a['total_blue_reward']:.1f}", f"{res_b['total_blue_reward']:.1f}", f"{res_c['total_blue_reward']:.1f}", f"{res_d['total_blue_reward']:.1f}"],
        ["Ongoing Penalties", f"{res_a['total_penalties']:.1f}", f"{res_b['total_penalties']:.1f}", f"{res_c['total_penalties']:.1f}", f"{res_d['total_penalties']:.1f}"],
    ]

    for row in rows:
        print(fmt_row(row))

    print("-" * 105)

    benefit = res_c["total_blue_reward"] - res_b["total_blue_reward"]
    penalty_saved = res_b["total_penalties"] - res_c["total_penalties"]
    print(f"\nDEFENSIVE VALUE OF BLUE RESPONSE (Scenario C vs B):")
    print(f"  Blue Net Reward Improvement : {benefit:+.1f} points ({res_b['total_blue_reward']:.1f} -> {res_c['total_blue_reward']:.1f})")
    print(f"  Ongoing Penalties Avoided   : {penalty_saved:.1f} points saved by timely restoration")
    print(f"  Attacker Containment        : Further exploits blocked, unauthorized admin access removed")
    print("=" * 105)


def print_step_breakdown(res):
    print(f"\n--- STEP-BY-STEP LOG: {res['scenario']} ---")
    print(f"{'Step':<5} {'Red Action':<25} {'Blue Action':<25} {'Red Res':<18} {'Blue Res':<18} {'Red Rwd':<8} {'Blue Rwd':<8} {'Penalized'}")
    print("-" * 115)
    for row in res["step_log"]:
        pen = "YES (-10)" if (row["incident_active"] or row["compromised"]) else "NO"
        print(f"{row['step']:<5} {row['red_action']:<25} {row['blue_action']:<25} {row['red_result']:<18} {row['blue_result']:<18} {row['red_reward']:<8.1f} {row['blue_reward']:<8.1f} {pen}")


def main():
    print("=" * 80)
    print("DETERMINISTIC EVALUATION: OWASP JUICE SHOP INCIDENT & RESPONSE")
    print("=" * 80)

    res_a = evaluate_scenario_a()
    res_b = evaluate_scenario_b()
    res_c = evaluate_scenario_c()
    res_d = evaluate_scenario_d()

    print_step_breakdown(res_a)
    print_step_breakdown(res_b)
    print_step_breakdown(res_c)
    print_step_breakdown(res_d)

    print_comparison_table([res_a, res_b, res_c, res_d])

    # Assertions validating core comparative findings
    assert res_a["total_blue_reward"] == 0.0, "Baseline Blue reward must be 0.0"
    assert res_a["total_penalties"] == 0.0, "Baseline should have 0 ongoing penalties"

    assert res_b["incident_occurred"] is True, "Scenario B must have active incident"
    assert res_b["final_compromised"] is True, "Scenario B must remain compromised"
    assert res_b["final_admin_access"] is True, "Scenario B attacker must retain admin access"
    assert res_b["total_blue_reward"] <= -70.0, "Scenario B Blue must accumulate severe ongoing penalties"

    assert res_c["incident_occurred"] is True, "Scenario C must have incident occur"
    assert res_c["detection_success"] is True, "Scenario C Blue must detect incident"
    assert res_c["attacker_blocked"] is True, "Scenario C Blue must block attacker"
    assert res_c["restoration_success"] is True, "Scenario C Blue must restore application"
    assert res_c["final_compromised"] is False, "Scenario C application must be clean at end"
    assert res_c["final_admin_access"] is False, "Scenario C admin access must be cleared"
    assert res_c["total_blue_reward"] > res_b["total_blue_reward"], (
        f"Defensive response in C ({res_c['total_blue_reward']}) must outperform unmitigated incident in B ({res_b['total_blue_reward']})"
    )

    assert res_d["detection_success"] is True, "Hybrid policy must detect incident"
    assert res_d["attacker_blocked"] is True, "Hybrid policy must block attacker"
    assert res_d["restoration_success"] is True, "Hybrid policy must autonomously restore application"
    assert res_d["final_incident_active"] is False, "Hybrid policy must clear incident_active flag"
    assert res_d["final_compromised"] is False, "Hybrid policy must clear compromised flag"
    assert res_d["final_admin_access"] is False, "Hybrid policy restoration must revoke admin access"
    assert res_d["total_blue_reward"] == -17.0, f"Expected -17.0 total Blue reward, got {res_d['total_blue_reward']}"
    assert res_d["total_penalties"] == 30.0, f"Expected 30.0 total ongoing penalties, got {res_d['total_penalties']}"
    assert res_d["total_blue_reward"] > res_b["total_blue_reward"], (
        f"Autonomous defense in D ({res_d['total_blue_reward']}) must outperform unmitigated incident in B ({res_b['total_blue_reward']})"
    )

    print("\nALL DETERMINISTIC EVALUATION ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
