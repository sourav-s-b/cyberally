from environment.owasp.env import OWASPEnvironment
from environment.owasp.blue_policy_adapter import (
    OWASPBluePolicyAdapter,
)


def print_state(state):
    print("\nOWASP State:")
    print(f"  compromised      = {state.compromised}")
    print(f"  attack_detected  = {state.attack_detected}")
    print(f"  attacker_blocked = {state.attacker_blocked}")
    print(f"  user_access      = {state.user_access}")
    print(f"  admin_access     = {state.admin_access}")


def main():

    environment = OWASPEnvironment()

    blue = OWASPBluePolicyAdapter(
        agent="blue_agent_0"
    )

    print("=" * 60)
    print("OWASP BLUE MULTI-STEP TEST")
    print("=" * 60)

    environment.reset()

    state = environment.state

    # ---------------------------------------------------------
    # Simulate successful Red compromise
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 1 - SIMULATE RED COMPROMISE")
    print("-" * 60)

    state.application_discovered = True
    state.api_discovered = True
    state.authenticated = True
    state.user_access = True
    state.compromised = True

    state.add_event(
        "Red compromised the Juice Shop"
    )

    print_state(state)

    # ---------------------------------------------------------
    # BLUE STEP 1
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 2 - BLUE DETECTION")
    print("-" * 60)

    (
        action_index,
        cage4_action,
        owasp_action,
    ) = blue.select_action(state)

    print("\nSelected action index:")
    print(action_index)

    print("\nCAGE4 action:")
    print(cage4_action)

    print("\nOWASP action:")
    print(owasp_action)

    result = blue.execute_action(
        environment,
        cage4_action,
        owasp_action,
    )

    print("\nExecution result:")
    print(result)

    print_state(state)

    # ---------------------------------------------------------
    # BLUE STEP 2
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("STEP 3 - BLUE RESPONSE")
    print("-" * 60)

    (
        action_index,
        cage4_action,
        owasp_action,
    ) = blue.select_action(state)

    print("\nSelected action index:")
    print(action_index)

    print("\nCAGE4 action:")
    print(cage4_action)

    print("\nOWASP action:")
    print(owasp_action)

    result = blue.execute_action(
        environment,
        cage4_action,
        owasp_action,
    )

    print("\nExecution result:")
    print(result)

    print_state(state)

    # ---------------------------------------------------------
    # EVENTS
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("EVENT LOG")
    print("-" * 60)

    for i, event in enumerate(
        state.events,
        start=1,
    ):
        print(f"{i}. {event}")

    print("\n" + "=" * 60)
    print("MULTI-STEP BLUE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()