
import ipaddress
from pprint import pformat

from CybORG import CybORG
from CybORG.Simulator.Actions import (
    DiscoverRemoteSystems,
    AggressiveServiceDiscovery,
    Monitor,
)

from environment.owasp.cage4_scenario import OWASPScenarioGenerator


def print_controller_state(controller, step_name, requested_actions):
    print("\n" + "=" * 60)
    print(f"{step_name}: EXECUTION DETAILS")
    print("=" * 60)

    print("\n--- OBSERVATIONS ---")

    for agent_name, observation_set in controller.observation.items():
        print(f"\nAgent: {agent_name}")
        requested = requested_actions.get(agent_name)
        print("    Requested action:", requested)
        print("    Team reward breakdown:", controller.get_reward(agent_name))
        active_ids = sorted(
            session_id
            for session_id, session in controller.state.sessions.get(agent_name, {}).items()
            if session.active
        )
        print("    Active session IDs (simulator state):", active_ids)

        observations = getattr(
            observation_set, "observations", []
        )

        if not observations:
            print("    No observations recorded.")
            continue

        for index, observation in enumerate(observations, start=1):
            print(f"    Observation {index}:")

            data = getattr(observation, "data", None)
            print("        Observation:")
            print(pformat(data, indent=4, sort_dicts=False))

            if isinstance(data, dict):
                action_result = data.get("action")
                # Keep native TRUE/FALSE/UNKNOWN/IN_PROGRESS values intact.
                # A missing result must never be presented as success.
                print("        Success status:", data.get("success", "NOT REPORTED"))

                if action_result is not None:
                    print(
                        "        Action class:",
                        type(action_result).__name__,
                    )
                    error = getattr(action_result, "error", None)
                    # Normal actions inherit a callable logger named error;
                    # InvalidAction supplies an actual error message instead.
                    print("        Action error:", "None reported" if callable(error) or error is None else error)

    print("\n--- REWARDS ---")
    print(controller.reward)

    print("\n--- ACTIVE SESSIONS ---")
    for agent_name, sessions in controller.state.sessions.items():
        print(agent_name, sorted(sid for sid, session in sessions.items() if session.active))

    print("\n--- HOST SERVICES ---")
    for hostname, host in controller.state.hosts.items():
        print(f"Host: {hostname}")
        print("    Services:", host.services)

    print("\n--- RED ACTION SPACE ---")
    red_interface = controller.agent_interfaces.get("red_agent")

    if red_interface is not None:
        action_space = red_interface.action_space.get_action_space()
        print(action_space)

        print("\nRed session IDs recognised by action space:")
        print(action_space.get("session", {}))


def main():
    print("Initializing custom OWASP scenario...")

    cyborg = CybORG(
        scenario_generator=OWASPScenarioGenerator(),
        seed=42,
    )

    controller = cyborg.environment_controller

    print("\nInitial IP mapping:", controller.state.ip_addresses)

    # Display the initial Red action space before session registration.
    print("\n--- INITIAL RED ACTION SPACE ---")

    red_interface = controller.agent_interfaces.get("red_agent")

    if red_interface is not None:
        initial_action_space = (
            red_interface.action_space.get_action_space()
        )
        print(initial_action_space)

        print("\nInitially recognised Red sessions:")
        print(initial_action_space.get("session", {}))

    # Display the initial observations.
    print("\n--- INITIAL OBSERVATIONS ---")

    for agent_name, observation_set in controller.observation.items():
        print(f"\nAgent: {agent_name}")

        observations = getattr(
            observation_set, "observations", []
        )

        if not observations:
            print("    No initial observations recorded.")
            continue

        for index, observation in enumerate(observations, start=1):
            print(f"    Observation {index}:")
            print("        Data:", getattr(observation, "data", None))

    # Register sessions from initial observations before the first action.
    # This is a diagnostic initialization step to test whether the initial
    # observations correctly populate each agent's action space.
    print("\n--- REGISTERING INITIAL SESSIONS ---")

    for agent_name, observation_set in controller.observation.items():
        agent_interface = controller.agent_interfaces[agent_name]

        for observation in getattr(
            observation_set, "observations", []
        ):
            agent_interface.update(observation)

    print("\n--- RED ACTION SPACE AFTER SESSION REGISTRATION ---")

    red_action_space = (
        controller.agent_interfaces["red_agent"]
        .action_space.get_action_space()
    )

    print("Recognised Red sessions:")
    print(red_action_space.get("session", {}))

    # Step 1: Red discovers remote systems; Blue monitors.
    actions_step_1 = {
        "red_agent": DiscoverRemoteSystems(
            subnet=ipaddress.ip_network("10.0.0.0/24"),
            session=0,
            agent="red_agent",
        ),
        "blue_agent": Monitor(
            session=0,
            agent="blue_agent",
        ),
    }

    print("\n" + "=" * 60)
    print("STEP 1: DISCOVER REMOTE SYSTEMS + MONITOR")
    print("=" * 60)

    controller.step(actions_step_1)
    print_controller_state(controller, "STEP 1", actions_step_1)

    # Step 2: Red performs service discovery; Blue monitors again.
    actions_step_2 = {
        "red_agent": AggressiveServiceDiscovery(
            session=0,
            agent="red_agent",
            ip_address=ipaddress.ip_address("10.0.0.10"),
        ),
        "blue_agent": Monitor(
            session=0,
            agent="blue_agent",
        ),
    }

    print("\n" + "=" * 60)
    print("STEP 2: SERVICE DISCOVERY + MONITOR")
    print("=" * 60)

    controller.step(actions_step_2)
    print_controller_state(controller, "STEP 2", actions_step_2)

    print("\n" + "=" * 60)
    print("ACTION EXECUTION TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
