from red.llm_agent import LLMConfig, OllamaClient
from red.planned_agent import PlannedRedAgent, PlannerConfig

from environment.owasp.action_space import build_action_space


def main():
    config = LLMConfig(
        model="llama3.1:8b",
        timeout_seconds=60,
    )

    client = OllamaClient(config)

    agent = PlannedRedAgent(
        name="red_agent",
        config=config,
        client=client,
        planner_config=PlannerConfig(
            interval=1,
            max_plans_per_agent=1,
        ),
    )

    action_space = build_action_space("red_agent")

    observation = {
        "10.0.0.10": {
            "System info": {
                "Hostname": "juice-shop",
            },
            "Interface": [
                {
                    "ip_address": "10.0.0.10",
                    "Subnet": "10.0.0.0/24",
                }
            ],
        }
    }

    action = agent.get_action(observation, action_space)

    print()
    print("Selected action:")
    print(type(action).__name__)

    print()
    print("Parameters:")
    print(action.get_params())

    print()
    print("Agent host states:")
    print(agent.host_states)

    print()
    print("LLM goal:")
    print(agent.goal)

    print()
    print("LLM focus:")
    print(agent.focus_ip)

    print()
    print("Last decision:")
    print(agent.last_decision)


if __name__ == "__main__":
    main()