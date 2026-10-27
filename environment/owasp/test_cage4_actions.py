
from CybORG import CybORG
from CybORG.Shared.ActionSpace import SESSION_TYPES
from CybORG.Shared.Enums import SessionType

from environment.owasp.cage4_scenario import OWASPScenarioGenerator


def main():
    print("Initializing custom OWASP scenario...")

    cyborg = CybORG(
        scenario_generator=OWASPScenarioGenerator(),
        seed=42,
    )

    controller = cyborg.environment_controller

    print("\n" + "=" * 70)
    print("INITIAL STATE SESSIONS")
    print("=" * 70)

    state = controller.state

    for hostname, host in state.hosts.items():
        print(f"\nHost: {hostname}")

        sessions_by_agent = getattr(host, "sessions", {})

        for agent_name, sessions in sessions_by_agent.items():
            print(f"  Agent: {agent_name}")

            for session in sessions:
                print("    Session object:", session)
                print(
                    "    Session ID:",
                    getattr(session, "ident", None),
                )
                print(
                    "    Session type:",
                    getattr(session, "session_type", None),
                )
                print(
                    "    Session agent:",
                    getattr(session, "agent", None),
                )

    print("\n" + "=" * 70)
    print("SESSION TYPES ALLOWED BY CAGE4")
    print("=" * 70)

    for session_type in SESSION_TYPES:
        print(session_type)

    print("\n" + "=" * 70)
    print("ACTION SPACE SESSION REGISTRATION")
    print("=" * 70)

    for agent_name, interface in controller.agent_interfaces.items():
        action_space = interface.action_space

        print(f"\nAgent: {agent_name}")
        print("server_session:", action_space.server_session)
        print("client_session:", action_space.client_session)

        print("Observed initial session entries:")

        observation = cyborg.get_observation(agent_name)
        print(observation)

    print("\n" + "=" * 70)
    print("INITIAL STATE SESSION REGISTRY")
    print("=" * 70)

    print(getattr(state, "sessions", {}))


if __name__ == "__main__":
    main()
