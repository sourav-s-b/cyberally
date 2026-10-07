from red.llm_agent import LLMConfig, OllamaClient
from red.planned_agent import PlannedRedAgent, PlannerConfig

from environment.owasp.action_space import build_action_space
from environment.owasp.env import OWASPEnvironment
from environment.owasp.adapters.red_observation import build_red_observation

from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import ExploitRemoteService


def main():

    config = LLMConfig(
        model="llama3.1:8b",
        timeout_seconds=60,
    )

    client = OllamaClient(config)

    red_agent = PlannedRedAgent(
        name="red_agent",
        config=config,
        client=client,
        planner_config=PlannerConfig(
            interval=1,
            max_plans_per_agent=4,
        ),
    )

    env = OWASPEnvironment(
        max_steps=15
    )

    action_space = build_action_space(
        "red_agent"
    )

    env.reset()

    success = TernaryEnum.UNKNOWN
    completed_action = None

    observation = build_red_observation(
        env.state,
        success,
        completed_action,
    )

    print("\n=== OWASP RED EPISODE ===\n")

    for step in range(1, 16):

        print(f"\nSTEP {step}")
        print("=" * 60)

        # ---------------------------------------------------------
        # State BEFORE get_action()
        # ---------------------------------------------------------

        print("\nHOST STATES BEFORE get_action:")
        print(red_agent.host_states)

        # ---------------------------------------------------------
        # IMPORTANT DEBUG: inspect the observation being supplied
        # to PlannedRedAgent
        # ---------------------------------------------------------

        print("\nOBSERVATION SUCCESS:")
        print(observation.get("success"))

        completed = observation.get("action")

        print("\nOBSERVATION ACTION:")
        print(
            type(completed).__name__
            if completed is not None
            else None
        )

        print("\nOBSERVATION ACTION IS ExploitRemoteService:")
        print(
            isinstance(
                completed,
                ExploitRemoteService
            )
        )

        # ---------------------------------------------------------
        # If this is ExploitRemoteService, inspect its parameters
        # ---------------------------------------------------------

        if completed is not None:

            print("\nCOMPLETED ACTION PARAMETERS:")

            try:
                print(
                    completed.get_params()
                )
            except Exception as e:
                print(
                    "Could not get parameters:",
                    repr(e)
                )

        # ---------------------------------------------------------
        # Inspect the transition that CAGE4 SHOULD perform
        # ---------------------------------------------------------

        current_state = red_agent.host_states.get(
            "10.0.0.10",
            {}
        ).get(
            "state"
        )

        print("\nCURRENT CAGE4 HOST STATE:")
        print(current_state)

        if current_state is not None:

            try:

                print(
                    "\nEXPECTED SUCCESS TRANSITION:"
                )

                print(
                    red_agent.state_transitions_success[
                        current_state
                    ]
                )

                print(
                    "\nEXPLOIT INDEX:"
                )

                print(
                    red_agent.action_list.index(
                        ExploitRemoteService
                    )
                )

                exploit_index = red_agent.action_list.index(
                    ExploitRemoteService
                )

                print(
                    "\nEXPECTED RESULT FOR EXPLOIT:"
                )

                print(
                    red_agent.state_transitions_success[
                        current_state
                    ][exploit_index]
                )

            except Exception as e:

                print(
                    "Transition inspection failed:",
                    repr(e)
                )

        # ---------------------------------------------------------
        # Call existing PlannedRedAgent
        # ---------------------------------------------------------

        action = red_agent.get_action(
            observation,
            action_space,
        )

        # ---------------------------------------------------------
        # State AFTER get_action()
        # ---------------------------------------------------------

        print("\nHOST STATES AFTER get_action:")
        print(red_agent.host_states)

        # ---------------------------------------------------------
        # Selected action
        # ---------------------------------------------------------

        print(
            "\nSelected action:",
            type(action).__name__
        )

        print(
            "Parameters:",
            action.get_params()
        )

        # ---------------------------------------------------------
        # Execute action in OWASP environment
        # ---------------------------------------------------------

        success = env.execute_red_action(
            action
        )

        print(
            "\nResult:",
            success.name
        )

        # ---------------------------------------------------------
        # OWASP state
        # ---------------------------------------------------------

        print(
            "\nOWASP state:",
            {
                "application_discovered":
                    env.state.application_discovered,

                "api_discovered":
                    env.state.api_discovered,

                "user_access":
                    env.state.user_access,

                "admin_access":
                    env.state.admin_access,

                "compromised":
                    env.state.compromised,

                "attacker_blocked":
                    env.state.attacker_blocked,
            }
        )

        # ---------------------------------------------------------
        # Agent information
        # ---------------------------------------------------------

        print(
            "\nAgent goal:",
            red_agent.goal
        )

        print(
            "Agent focus:",
            red_agent.focus_ip
        )

        # ---------------------------------------------------------
        # Events
        # ---------------------------------------------------------

        print(
            "Events:",
            env.state.events
        )

        # ---------------------------------------------------------
        # Build observation for NEXT step
        # ---------------------------------------------------------

        observation = build_red_observation(
            env.state,
            success,
            action,
        )

        # ---------------------------------------------------------
        # Stop when admin access achieved
        # ---------------------------------------------------------

        if env.state.admin_access:

            print("\n" + "=" * 60)
            print("ADMIN ACCESS ACHIEVED")
            print("=" * 60)

            break

    print("\n=== FINAL OWASP STATE ===")
    print(env.state)


if __name__ == "__main__":
    main()