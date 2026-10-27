
import inspect
import ipaddress

from CybORG import CybORG
from CybORG.Simulator.Actions import (
    DiscoverRemoteSystems,
    AggressiveServiceDiscovery,
    Monitor,
)
from environment.owasp.cage4_scenario import OWASPScenarioGenerator


def main():
    cyborg = CybORG(
        scenario_generator=OWASPScenarioGenerator(),
        seed=42,
    )
    controller = cyborg.environment_controller

    print("\n" + "=" * 65)
    print("1. ACTION CONSTRUCTOR SIGNATURES")
    print("=" * 65)

    for action_class in (
        DiscoverRemoteSystems,
        AggressiveServiceDiscovery,
        Monitor,
    ):
        print(f"\n{action_class.__name__}")
        print("Signature:", inspect.signature(action_class))

    print("\n" + "=" * 65)
    print("2. AGENT INTERFACE DETAILS")
    print("=" * 65)

    for agent_name, interface in controller.agent_interfaces.items():
        print(f"\nAgent: {agent_name}")
        print("Interface class:", type(interface).__name__)

        for attr_name in dir(interface):
            if "action" in attr_name.lower():
                try:
                    value = getattr(interface, attr_name)
                    if not callable(value):
                        print(f"  {attr_name}: {value}")
                except Exception as exc:
                    print(f"  {attr_name}: <error: {exc}>")

    print("\n" + "=" * 65)
    print("3. RED ACTION SPACE")
    print("=" * 65)

    try:
        action_space = cyborg.get_action_space("red_agent")
        print(action_space)
        print("Type:", type(action_space).__name__)
    except Exception as exc:
        print("Could not retrieve Red action space:", repr(exc))

    print("\n" + "=" * 65)
    print("4. ACTION VALIDATION / STEP IMPLEMENTATION")
    print("=" * 65)

    for method_name in ("step", "get_action_space"):
        method = getattr(controller, method_name, None)
        if method is not None:
            try:
                print(f"\n--- {method_name} ---")
                print(inspect.getsource(method))
            except Exception as exc:
                print(f"Could not inspect {method_name}: {exc}")

    print("\n" + "=" * 65)
    print("5. RED SESSION")
    print("=" * 65)

    print(controller.state.sessions.get("red_agent", {}))

    print("\n" + "=" * 65)
    print("6. ACTION CONSTRUCTION")
    print("=" * 65)

    actions = [
        DiscoverRemoteSystems(
            subnet=ipaddress.ip_network("10.0.0.0/24"),
            session=0,
            agent="red_agent",
        ),
        AggressiveServiceDiscovery(
            session=0,
            agent="red_agent",
            ip_address=ipaddress.ip_address("10.0.0.10"),
        ),
    ]

    for action in actions:
        print(f"\nAction: {type(action).__name__}")
        print("Object:", action)
        print("Attributes:", getattr(action, "__dict__", {}))


if __name__ == "__main__":
    main()
