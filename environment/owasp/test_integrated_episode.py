
from red.llm_agent import LLMConfig, OllamaClient
from red.planned_agent import PlannedRedAgent, PlannerConfig

from environment.owasp.env import OWASPEnvironment
from environment.owasp.action_space import build_action_space
from environment.owasp.red_observation import build_red_observation
from environment.owasp.blue_policy_adapter import (
    OWASPBluePolicyAdapter,
)


def print_state(state):
    print("\nOWASP STATE")
    print("-" * 50)

    print(f"application_discovered = {state.application_discovered}")
    print(f"api_discovered         = {state.api_discovered}")
    print(f"user_access            = {state.user_access}")
    print(f"admin_access           = {state.admin_access}")
    print(f"compromised            = {state.compromised}")
    print(f"attack_detected        = {state.attack_detected}")
    print(f"attacker_blocked       = {state.attacker_blocked}")
    print(f"incident_active        = {state.incident_active}")


def main():
    print("=" * 70)
    print("INTEGRATED OWASP RED + BLUE EPISODE")
    print("=" * 70)

    # ---------------------------------------------------------
    # OWASP ENVIRONMENT
    # ---------------------------------------------------------

    environment = OWASPEnvironment(max_steps=20)
    environment.reset()
    state = environment.state

    # ---------------------------------------------------------
    # EXISTING OLLAMA RED AGENT
    # ---------------------------------------------------------

    config = LLMConfig(
        model="llama3.1:8b",
        timeout_seconds=60,
    )

    client = OllamaClient(config)

    red = PlannedRedAgent(
        name="red_agent",
        config=config,
        client=client,
        planner_config=PlannerConfig(
            interval=1,
            max_plans_per_agent=1,
        ),
    )

    # ---------------------------------------------------------
    # RED ACTION SPACE
    # ---------------------------------------------------------

    action_space = build_action_space("red_agent")

    # ---------------------------------------------------------
    # EXISTING BLUE POLICY
    # ---------------------------------------------------------

    blue = OWASPBluePolicyAdapter(agent="blue_agent_0")

    # ---------------------------------------------------------
    # INITIAL RED OBSERVATION
    # ---------------------------------------------------------

    red_observation = build_red_observation(state)

    print("\nINITIAL RED OBSERVATION")
    print(red_observation)

    restoration_success = False

    # ---------------------------------------------------------
    # EPISODE
    # Continue after blocking so Blue can verify and restore.
    # ---------------------------------------------------------

    for step in range(1, 11):
        print("\n")
        print("=" * 70)
        print(f"STEP {step}")
        print("=" * 70)

        # =====================================================
        # RED TURN
        # =====================================================

        if not state.attacker_blocked:
            print("\n[RED] Asking existing PlannedRedAgent...")

            red_action = red.get_action(
                red_observation,
                action_space,
            )

            print(f"[RED] Selected: {type(red_action).__name__}")

            red_result = environment.execute_red_action(red_action)

            print(f"[RED] Result: {red_result}")

            # Update Red observation
            red_observation = build_red_observation(
                state,
                success=red_result,
                action=red_action,
            )
        else:
            print("\n[RED] Attacker is blocked. Red turn skipped.")

        print_state(state)

        # =====================================================
        # BLUE TURN
        # =====================================================

        print("\n[BLUE] Asking existing HybridBluePolicy...")

        (
            blue_index,
            blue_cage4_action,
            blue_owasp_action,
        ) = blue.select_action(state)

        print(f"[BLUE] Action index: {blue_index}")
        print(f"[BLUE] CAGE4 action: {blue_cage4_action}")
        print(f"[BLUE] OWASP action: {blue_owasp_action}")

        blue_result = blue.execute_action(
            environment,
            blue_cage4_action,
            blue_owasp_action,
        )

        print(f"[BLUE] Result: {blue_result}")
        print_state(state)

        # =====================================================
        # RESTORATION CHECK
        # =====================================================

        restoration_success = (
            state.attacker_blocked
            and not state.incident_active
            and not state.compromised
            and not state.admin_access
        )

        if restoration_success:
            print("\n" + "=" * 70)
            print("INCIDENT RESTORED SUCCESSFULLY")
            print("Attacker remains blocked.")
            print("Unauthorized access has been cleared.")
            print("=" * 70)
            break

        if state.attacker_blocked:
            print(
                "\n[INFO] Attacker is blocked, but restoration "
                "is not complete. Continuing the episode."
            )

        if state.admin_access and not state.attacker_blocked:
            print("\n[WARNING] Red has achieved admin access.")
            print("[INFO] Continuing so Blue can respond.")

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EPISODE RESULT")
    print("=" * 70)

    print_state(state)

    print(f"\nrestoration_success = {restoration_success}")

    if not restoration_success:
        print(
            "\n[WARNING] Restoration was not confirmed within "
            "the episode limit."
        )

    print("\nEVENT LOG")
    print("-" * 50)

    for index, event in enumerate(state.events, start=1):
        print(f"{index}. {event}")

    print("\n")
    print("=" * 70)
    print("INTEGRATED EPISODE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
