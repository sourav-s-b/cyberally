
from environment.owasp.cage4_scenario import OWASPScenarioGenerator
from CybORG import CybORG


def main():
    generator = OWASPScenarioGenerator()
    cyborg = CybORG(
        scenario_generator=generator,
        seed=42,
    )

    print("\n=== CAGE4 OWASP Scenario Test ===")

    for agent_name in ["red_agent", "blue_agent"]:
        print(f"\n--- Agent: {agent_name} ---")

        try:
            observation = cyborg.get_observation(agent_name)
            print("Observation retrieved successfully.")
            print("Observation:", observation)

        except Exception as exc:
            print("Observation retrieval failed:")
            print(type(exc).__name__, exc)

        try:
            action_space = cyborg.get_action_space(agent_name)
            print("Action space retrieved successfully.")
            print("Action space:", action_space)

        except Exception as exc:
            print("Action-space retrieval failed:")
            print(type(exc).__name__, exc)


if __name__ == "__main__":
    main()
