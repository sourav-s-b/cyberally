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


def main():

    print("=" * 70)
    print("INTEGRATED OWASP RED + BLUE EPISODE")
    print("=" * 70)

    # ---------------------------------------------------------
    # OWASP ENVIRONMENT
    # ---------------------------------------------------------

    environment = OWASPEnvironment(
        max_steps=20
    )

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

    action_space = build_action_space(
        "red_agent"
    )

    # ---------------------------------------------------------
    # EXISTING BLUE POLICY
    # ---------------------------------------------------------

    blue = OWASPBluePolicyAdapter(
        agent="blue_agent_0"
    )

    # ---------------------------------------------------------
    # INITIAL RED OBSERVATION
    # ---------------------------------------------------------

    red_observation = build_red_observation(
        state
    )

    print("\nINITIAL RED OBSERVATION")
    print(red_observation)

    # ---------------------------------------------------------
    # EPISODE
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

            print(
                f"[RED] Selected: "
                f"{type(red_action).__name__}"
            )

            red_result = environment.execute_red_action(
                red_action
            )

            print(
                f"[RED] Result: "
                f"{red_result}"
            )

            # -------------------------------------------------
            # Update Red observation
            # -------------------------------------------------

            red_observation = build_red_observation(
                state,
                success=red_result,
                action=red_action,
            )

        else:

            print(
                "\n[RED] Attacker is blocked. "
                "Red turn skipped."
            )

        print_state(state)

        # =====================================================
        # BLUE TURN
        # =====================================================

        print("\n[BLUE] Asking existing HybridBluePolicy...")

        (
            blue_index,
            blue_cage4_action,
            blue_owasp_action,
        ) = blue.select_action(
            state
        )

        print(
            f"[BLUE] Action index: "
            f"{blue_index}"
        )

        print(
            f"[BLUE] CAGE4 action: "
            f"{blue_cage4_action}"
        )

        print(
            f"[BLUE] OWASP action: "
            f"{blue_owasp_action}"
        )

        blue_result = blue.execute_action(
            environment,
            blue_cage4_action,
            blue_owasp_action,
        )

        print(
            f"[BLUE] Result: "
            f"{blue_result}"
        )

        print_state(state)

        # =====================================================
        # TERMINATION
        # =====================================================

        if state.attacker_blocked:

            print(
                "\n" + "!" * 70
            )
            print(
                "ATTACKER HAS BEEN BLOCKED"
            )
            print(
                "!" * 70
            )

            break

        if (
            state.admin_access
            and not state.attacker_blocked
        ):

            print(
                "\n" + "!" * 70
            )
            print(
                "RED HAS ACHIEVED ADMIN ACCESS"
            )
            print(
                "BLUE DID NOT BLOCK THE ATTACKER YET"
            )
            print(
                "!" * 70
            )

            # IMPORTANT:
            # Do NOT terminate here.
            #
            # Blue must still get a chance to
            # detect and respond.

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EPISODE RESULT")
    print("=" * 70)

    print_state(state)

    print("\nEVENT LOG")
    print("-" * 50)

    for index, event in enumerate(
        state.events,
        start=1,
    ):
        print(
            f"{index}. {event}"
        )

    print("\n")
    print("=" * 70)
    print("INTEGRATED EPISODE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()