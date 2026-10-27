"""
test_owasp_incident_cycle.py
----------------------------
Deterministic end-to-end test suite for the simulated OWASP Juice Shop
incident and Blue-agent response lifecycle.

Verifies:
  1. Initial clean state and invalid defensive action rejection.
  2. Simulated incident transition (distinguished from native CAGE4 host compromise).
  3. Preconditions for exploits and incident creation.
  4. Blue detection transition via direct environment action.
  5. Blue detection and response via OWASPBluePolicyAdapter / HybridBluePolicy.
  6. Blue blocking transition and enforcement preventing subsequent Red attacks.
  7. Blue restoration, service health preservation, and penalty removal.
  8. Prevention of duplicate rewards on repeated defensive actions.
  9. Environment reset behavior.
"""

from environment.owasp.env import OWASPEnvironment
from environment.owasp.actions import RedAction, BlueAction
from environment.owasp.blue_policy_adapter import OWASPBluePolicyAdapter


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_initial_clean_state():
    """Verify initial state, baseline observations, and rejection of premature defense."""
    print_section("TEST 1: INITIAL CLEAN STATE & PREMATURE DEFENSE REJECTION")

    env = OWASPEnvironment()
    obs = env.reset()

    state = env.state

    # 1. Verify baseline flags
    assert state.web_server_up is True, "Web server should be up initially"
    assert state.api_available is True, "API should be available initially"
    assert state.database_available is True, "Database should be available initially"
    assert state.application_discovered is False, "App discovery should be False initially"
    assert state.api_discovered is False, "API discovery should be False initially"
    assert state.authenticated is False, "Authenticated should be False initially"
    assert state.user_access is False, "User access should be False initially"
    assert state.admin_access is False, "Admin access should be False initially"
    assert state.incident_active is False, "No incident should be active initially"
    assert state.incident_type == "none", "Incident type should be 'none' initially"
    assert state.compromised is False, "Compromised should be False initially"
    assert state.attack_detected is False, "Attack detected should be False initially"
    assert state.attacker_blocked is False, "Attacker blocked should be False initially"
    assert state.step_count == 0, "Step count should be 0 initially"
    assert len(state.events) == 0, "Events list should be empty initially"

    print("  [PASS] Initial state flags verified.")

    # 2. Verify role-scoped observations
    blue_obs = obs["blue"]
    assert blue_obs["incident_active"] is False, "Blue obs incident_active should be False"
    assert blue_obs["compromised"] is False, "Blue obs compromised should be False"
    assert blue_obs["attack_detected"] is False, "Blue obs attack_detected should be False"
    assert blue_obs["attacker_blocked"] is False, "Blue obs attacker_blocked should be False"
    assert "user_access" not in blue_obs, "Blue obs must not expose internal user_access"
    assert "authenticated" not in blue_obs, "Blue obs must not expose internal authenticated"
    assert "admin_access" not in blue_obs, "Blue obs must not expose internal admin_access"

    print("  [PASS] Initial Blue observation verified (role-scoped, no leaked access).")

    # 3. Premature DETECT should fail cleanly without rewarding Blue
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.DETECT,
    })
    assert info["blue_result"] == "no_attack_detected", (
        f"Expected 'no_attack_detected', got {info['blue_result']}"
    )
    assert rewards["blue"] == 0.0, f"Expected 0.0 reward for premature DETECT, got {rewards['blue']}"
    assert state.attack_detected is False, "attack_detected must remain False"
    print("  [PASS] Premature DETECT correctly returned 'no_attack_detected' with 0 reward.")

    # 4. Premature BLOCK should fail cleanly without rewarding Blue
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.BLOCK,
    })
    assert info["blue_result"] == "nothing_to_block", (
        f"Expected 'nothing_to_block', got {info['blue_result']}"
    )
    assert rewards["blue"] == 0.0, f"Expected 0.0 reward for premature BLOCK, got {rewards['blue']}"
    assert state.attacker_blocked is False, "attacker_blocked must remain False"
    print("  [PASS] Premature BLOCK correctly returned 'nothing_to_block' with 0 reward.")

    # 5. Premature RESTORE on clean state should return 'already_clean' with 0 reward
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.RESTORE,
    })
    assert info["blue_result"] == "already_clean", (
        f"Expected 'already_clean', got {info['blue_result']}"
    )
    assert rewards["blue"] == 0.0, f"Expected 0.0 reward for premature RESTORE, got {rewards['blue']}"
    print("  [PASS] Premature RESTORE correctly returned 'already_clean' with 0 reward.")


def test_complete_incident_response_cycle():
    """Verify the full deterministic incident, detection, blocking, and restoration cycle."""
    print_section("TEST 2: COMPLETE INCIDENT, DETECTION, BLOCKING & RESTORATION CYCLE")

    env = OWASPEnvironment()
    env.reset()
    state = env.state

    # -------------------------------------------------------------------------
    # Phase A: Reconnaissance & Preconditions
    # -------------------------------------------------------------------------
    print("\n--- Phase A: Reconnaissance & Preconditions ---")

    # Exploit before application discovery must be rejected
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.SQL_INJECTION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "application_not_discovered", (
        f"Expected 'application_not_discovered', got {info['red_result']}"
    )
    assert state.incident_active is False, "Incident must not activate prematurely"
    assert rewards["red"] == 0.0, "Red must not be rewarded for invalid action"
    print("  [PASS] Exploit before discovery rejected: 'application_not_discovered'.")

    # Red discovers application
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.DISCOVER_APPLICATION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "discovery_success"
    assert state.application_discovered is True
    assert rewards["red"] == 1.0, f"Expected +1.0 Red discovery reward, got {rewards['red']}"
    assert rewards["blue"] == 0.0, f"Expected 0.0 Blue reward, got {rewards['blue']}"
    print("  [PASS] Application discovery succeeded (+1.0 Red reward).")

    # -------------------------------------------------------------------------
    # Phase B: Simulated Incident Trigger (Clean -> Incident)
    # -------------------------------------------------------------------------
    print("\n--- Phase B: Simulated Incident Trigger (Clean -> Incident) ---")

    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.SQL_INJECTION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "exploit_success", f"Expected 'exploit_success', got {info['red_result']}"
    assert info["blue_result"] == "monitoring"

    # Verify state transitions: application-level incident active, distinct from OS compromise
    assert state.incident_active is True, "incident_active must be True"
    assert state.incident_type == "sql_injection", f"incident_type should be 'sql_injection', got {state.incident_type}"
    assert state.compromised is True, "compromised should be True (application-level compromise)"
    assert state.user_access is True, "user_access should be True"
    assert state.authenticated is True, "authenticated should be True"
    assert state.admin_access is False, "admin_access must remain False initially"
    assert state.attack_detected is False, "attack_detected must still be False before Blue DETECT"
    assert state.attacker_blocked is False, "attacker_blocked must still be False"

    # Verify rewards: Red gets +5.0, Blue gets -10.0 penalty
    assert rewards["red"] == 5.0, f"Expected +5.0 Red exploit reward, got {rewards['red']}"
    assert rewards["blue"] == -10.0, f"Expected -10.0 Blue compromise penalty, got {rewards['blue']}"

    # Verify observations: Blue sees incident_active indicator
    assert obs["blue"]["incident_active"] is True, "Blue obs must show incident_active=True"
    assert obs["blue"]["compromised"] is True, "Blue obs must show compromised=True"
    print("  [PASS] Simulated incident active: incident_type='sql_injection', Blue penalty=-10.0.")

    # -------------------------------------------------------------------------
    # Phase C: Blue Detection (Incident -> Detected)
    # -------------------------------------------------------------------------
    print("\n--- Phase C: Blue Detection (Incident -> Detected) ---")

    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.DETECT,
    })
    assert info["blue_result"] == "attack_detected", f"Expected 'attack_detected', got {info['blue_result']}"
    assert state.attack_detected is True, "attack_detected must transition to True"

    # Blue reward: +3.0 detection - 10.0 compromise penalty = -7.0
    assert rewards["blue"] == -7.0, f"Expected -7.0 net Blue reward (+3 - 10), got {rewards['blue']}"
    assert rewards["red"] == 0.0

    print("  [PASS] Detection transition succeeded (+3.0 detection, -7.0 net with penalty).")

    # Verify repeated DETECT does not award duplicate reward
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.DETECT,
    })
    assert info["blue_result"] == "already_detected", f"Expected 'already_detected', got {info['blue_result']}"
    assert rewards["blue"] == -10.0, f"Expected -10.0 (penalty only, no duplicate detection), got {rewards['blue']}"
    print("  [PASS] Duplicate DETECT prevented: returned 'already_detected', no duplicate reward.")

    # -------------------------------------------------------------------------
    # Phase D: Blue Blocking (Detected -> Blocked)
    # -------------------------------------------------------------------------
    print("\n--- Phase D: Blue Blocking (Detected -> Blocked) ---")

    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.BLOCK,
    })
    assert info["blue_result"] == "attack_blocked", f"Expected 'attack_blocked', got {info['blue_result']}"
    assert state.attacker_blocked is True, "attacker_blocked must transition to True"

    # Blue reward: +5.0 block - 10.0 compromise penalty = -5.0
    assert rewards["blue"] == -5.0, f"Expected -5.0 net Blue reward (+5 - 10), got {rewards['blue']}"
    print("  [PASS] Blocking transition succeeded (+5.0 block, -5.0 net with penalty).")

    # Verify repeated BLOCK does not award duplicate reward
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.BLOCK,
    })
    assert info["blue_result"] == "already_blocked", f"Expected 'already_blocked', got {info['blue_result']}"
    assert rewards["blue"] == -10.0, f"Expected -10.0 (penalty only, no duplicate block), got {rewards['blue']}"
    print("  [PASS] Duplicate BLOCK prevented: returned 'already_blocked', no duplicate reward.")

    # -------------------------------------------------------------------------
    # Phase E: Blocked Attacker Enforcement
    # -------------------------------------------------------------------------
    print("\n--- Phase E: Blocked Attacker Enforcement ---")

    # Blocked attacker attempts privilege escalation
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.PRIVILEGE_ESCALATION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "attacker_blocked", f"Expected 'attacker_blocked', got {info['red_result']}"
    assert state.admin_access is False, "admin_access must NOT be granted while blocked"
    assert rewards["red"] == 0.0, "Red must receive 0 reward while blocked"
    print("  [PASS] Red privilege escalation blocked; admin_access remains False.")

    # Blocked attacker attempts new exploit
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.SQL_INJECTION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "attacker_blocked", f"Expected 'attacker_blocked', got {info['red_result']}"
    assert rewards["red"] == 0.0
    print("  [PASS] Subsequent Red exploit blocked; attacker cannot act.")

    # -------------------------------------------------------------------------
    # Phase F: Remediation & Restoration (Blocked Incident -> Restored)
    # -------------------------------------------------------------------------
    print("\n--- Phase F: Remediation & Restoration (Blocked Incident -> Restored) ---")

    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.RESTORE,
    })
    assert info["blue_result"] == "service_restored", f"Expected 'service_restored', got {info['blue_result']}"

    # Verify incident state is fully restored
    assert state.incident_active is False, "incident_active must be cleared to False"
    assert state.incident_type == "none", "incident_type must be reset to 'none'"
    assert state.compromised is False, "compromised must be cleared to False"
    assert state.user_access is False, "user_access must be cleared to False"
    assert state.authenticated is False, "authenticated must be cleared to False"
    assert state.admin_access is False, "admin_access must remain False"

    # Attacker remains blocked (safeguard requirement)
    assert state.attacker_blocked is True, "attacker_blocked must NOT be cleared during restore"

    # Service health verified
    assert state.web_server_up is True, "web_server_up must remain True"
    assert state.api_available is True, "api_available must remain True"
    assert state.database_available is True, "database_available must remain True"

    # Blue reward: +5.0 restoration, penalty is lifted (-0.0) -> net +5.0!
    assert rewards["blue"] == 5.0, f"Expected +5.0 net Blue reward (penalty lifted), got {rewards['blue']}"
    print("  [PASS] Restoration succeeded: incident cleared, penalty lifted (+5.0 reward).")
    print("  [PASS] Attacker remains blocked after restoration.")

    # Verify repeated RESTORE returns 'already_clean' with 0 reward
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.NOOP,
        "blue": BlueAction.RESTORE,
    })
    assert info["blue_result"] == "already_clean", f"Expected 'already_clean', got {info['blue_result']}"
    assert rewards["blue"] == 0.0, f"Expected 0.0 reward for already_clean RESTORE, got {rewards['blue']}"
    print("  [PASS] Duplicate RESTORE prevented: returned 'already_clean' with 0 reward.")

    # Attacker is still blocked from any action
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.SQL_INJECTION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "attacker_blocked"
    print("  [PASS] Attacker continues to be blocked following restoration.")


def test_reset_behavior():
    """Verify that env.reset() returns the environment completely to initial clean state."""
    print_section("TEST 3: ENVIRONMENT RESET BEHAVIOR")

    env = OWASPEnvironment()
    env.reset()

    # Trigger incident and block
    env.step({"red": RedAction.DISCOVER_APPLICATION, "blue": BlueAction.MONITOR})
    env.step({"red": RedAction.SQL_INJECTION, "blue": BlueAction.DETECT})
    env.step({"red": RedAction.NOOP, "blue": BlueAction.BLOCK})

    assert env.state.attacker_blocked is True
    assert env.state.attack_detected is True
    assert env.state.step_count > 0
    assert len(env.state.events) > 0

    # Reset
    obs = env.reset()
    state = env.state

    assert state.incident_active is False, "Reset must clear incident_active"
    assert state.incident_type == "none", "Reset must clear incident_type"
    assert state.compromised is False, "Reset must clear compromised"
    assert state.attack_detected is False, "Reset must clear attack_detected"
    assert state.attacker_blocked is False, "Reset must clear attacker_blocked"
    assert state.step_count == 0, "Reset must reset step_count to 0"
    assert len(state.events) == 0, "Reset must clear events list"
    assert obs["blue"]["incident_active"] is False, "Blue obs must reflect clean reset"

    print("  [PASS] Reset cleanly restores initial state.")


def test_blue_adapter_hybrid_policy():
    """Verify autonomous incident detection and response via OWASPBluePolicyAdapter and HybridBluePolicy."""
    print_section("TEST 4: BLUE ADAPTER & HYBRID POLICY AUTONOMOUS RESPONSE")

    env = OWASPEnvironment()
    env.reset()
    blue_adapter = OWASPBluePolicyAdapter(agent="blue_agent_0")

    # Step 1: Discover application
    env.step({"red": RedAction.DISCOVER_APPLICATION, "blue": BlueAction.MONITOR})

    # Step 2: Trigger simulated incident
    env.step({"red": RedAction.SQL_INJECTION, "blue": BlueAction.MONITOR})
    assert env.state.incident_active is True
    assert env.state.compromised is True

    print("  [STATE] Simulated incident active on 10.0.0.10.")

    # Step 3: HybridBluePolicy should detect suspicious process and choose Analyse (DETECT)
    action_idx, cage4_act, owasp_act = blue_adapter.select_action(env.state)
    print(f"  [BLUE HYBRID] Selected: index={action_idx}, cage4={cage4_act}, owasp={owasp_act}")
    assert action_idx == 2, f"Expected action 2 (Analyse), got {action_idx}"
    assert owasp_act == BlueAction.DETECT, f"Expected BlueAction.DETECT, got {owasp_act}"

    res = blue_adapter.execute_action(env, cage4_act, owasp_act)
    assert res == "attack_detected", f"Expected 'attack_detected', got {res}"
    assert env.state.attack_detected is True, "State attack_detected must be True"
    print("  [PASS] HybridBluePolicy autonomously executed Analyse -> attack_detected.")

    # Step 4: With host confirmed in tracker, HybridBluePolicy should choose Remove (BLOCK)
    action_idx, cage4_act, owasp_act = blue_adapter.select_action(env.state)
    print(f"  [BLUE HYBRID] Selected: index={action_idx}, cage4={cage4_act}, owasp={owasp_act}")
    assert action_idx == 3, f"Expected action 3 (Remove), got {action_idx}"
    assert owasp_act == BlueAction.BLOCK, f"Expected BlueAction.BLOCK, got {owasp_act}"

    res = blue_adapter.execute_action(env, cage4_act, owasp_act)
    assert res == "attack_blocked", f"Expected 'attack_blocked', got {res}"
    assert env.state.attacker_blocked is True, "State attacker_blocked must be True"
    print("  [PASS] HybridBluePolicy autonomously executed Remove -> attack_blocked.")

    # Step 5: Verify attacker cannot execute further attacks in the environment
    obs, rewards, term, trunc, info = env.step({
        "red": RedAction.SQL_INJECTION,
        "blue": BlueAction.MONITOR,
    })
    assert info["red_result"] == "attacker_blocked"
    print("  [PASS] Verified attacker is blocked from subsequent exploits.")

    # Step 6: HybridBluePolicy verifies remediation with Analyse
    action_idx, cage4_act, owasp_act = blue_adapter.select_action(env.state)
    print(f"  [BLUE HYBRID] Selected: index={action_idx}, cage4={cage4_act}, owasp={owasp_act}")
    assert action_idx == 2, f"Expected action 2 (Analyse), got {action_idx}"
    res = blue_adapter.execute_action(env, cage4_act, owasp_act)
    assert res == "already_detected", f"Expected 'already_detected', got {res}"
    print("  [PASS] HybridBluePolicy verification scan detected remaining threat.")

    # Step 7: Re-detected threat triggers escalation to Restore
    action_idx, cage4_act, owasp_act = blue_adapter.select_action(env.state)
    print(f"  [BLUE HYBRID] Selected: index={action_idx}, cage4={cage4_act}, owasp={owasp_act}")
    assert action_idx == 4, f"Expected action 4 (Restore), got {action_idx}"
    assert owasp_act == BlueAction.RESTORE, f"Expected BlueAction.RESTORE, got {owasp_act}"
    res = blue_adapter.execute_action(env, cage4_act, owasp_act)
    assert res == "service_restored", f"Expected 'service_restored', got {res}"
    assert env.state.compromised is False, "Compromised flag must be cleared"
    assert env.state.incident_active is False, "Incident active flag must be cleared"
    assert env.state.attacker_blocked is True, "Attacker must remain blocked"
    print("  [PASS] HybridBluePolicy escalated to Restore -> service_restored.")

    # Step 8: Post-restoration verification scan confirms clean state
    action_idx, cage4_act, owasp_act = blue_adapter.select_action(env.state)
    assert action_idx == 2, f"Expected action 2 (Analyse), got {action_idx}"
    res = blue_adapter.execute_action(env, cage4_act, owasp_act)
    assert res == "no_attack_detected", f"Expected 'no_attack_detected', got {res}"
    print("  [PASS] Post-restoration verification confirmed clean state.")


def main():
    print("=" * 70)
    print("OWASP INCIDENT AND DEFENSE RESPONSE TEST SUITE")
    print("=" * 70)

    test_initial_clean_state()
    test_complete_incident_response_cycle()
    test_reset_behavior()
    test_blue_adapter_hybrid_policy()

    print("\n" + "=" * 70)
    print("ALL OWASP INCIDENT CYCLE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
