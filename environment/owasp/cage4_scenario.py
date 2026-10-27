
import ipaddress

from CybORG import CybORG
from CybORG.Shared.Enums import (
    SessionType,
    ProcessName,
    ProcessType,
    TransportProtocol,
)
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Scenario import Scenario, ScenarioAgent
from CybORG.Shared.Session import RedAbstractSession, Session
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Interface import Interface
from CybORG.Simulator.User import User
from CybORG.Simulator.Subnet import Subnet
from CybORG.Simulator.Process import Process
from CybORG.Simulator.Service import Service

from CybORG.Simulator.Actions import (
    DiscoverRemoteSystems,
    AggressiveServiceDiscovery,
    StealthServiceDiscovery,
    ExploitRemoteService,
    PrivilegeEscalate,
    Analyse,
    Remove,
    Restore,
    Monitor,
)

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import (
    EnterpriseScenarioGenerator,
)


class OWASPScenarioAgent(BaseAgent):
    """Minimal agent for testing native CAGE4 scenario initialization."""

    def get_action(self, observation, action_space):
        raise NotImplementedError(
            "An explicit CAGE4 action is required for this initialization test."
        )

    def train(self, results):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass


class OWASPScenarioGenerator(ScenarioGenerator):
    """Minimal CAGE4 scenario containing a simulated Juice Shop server."""

    def __init__(self):
        super().__init__()

        self.steps = 10
        self.enterprise_helpers = EnterpriseScenarioGenerator(
            steps=self.steps
        )

        # Reuse the message-length setting expected by this CAGE4 version.
        self.MESSAGE_LENGTH = self.enterprise_helpers.MESSAGE_LENGTH

    def create_scenario(self, np_random):
        self.enterprise_helpers.np_random = np_random

        network = ipaddress.ip_network("10.0.0.0/24")
        subnet_name = "owasp_subnet"
        hostname = "juice-shop"
        host_ip = ipaddress.ip_address("10.0.0.10")

        # Define the network subnet.
        subnet = Subnet(
            name=subnet_name,
            size=1,
            hosts=[hostname],
            cidr=network,
            ip_addresses=[host_ip],
        )

        # Define the server's network interface.
        interface = Interface(
            name="eth0",
            ip_address=host_ip,
            subnet=network,
            interface_type="wired",
            data_links=[],
        )

        # Define the simulated Juice Shop web service.
        web_pid = 3000

        web_service = Service(process=web_pid)

        web_process = Process(
            process_name=ProcessName.APACHE2,
            pid=web_pid,
            username="ubuntu",
            program_name="juice-shop-web",
            path="/opt/juice-shop",
            open_ports=[
                {
                    "local_port": 3000,
                    "local_address": str(host_ip),
                    "transport_protocol": TransportProtocol.TCP,
                }
            ],
            process_type=ProcessType.WEBSERVER,
        )

        # Define the simulated Juice Shop host.
        host = Host(
            np_random=np_random,
            system_info={
                "OSType": "LINUX",
                "OSDistribution": "UBUNTU",
                "OSVersion": "UNKNOWN",
                "Architecture": "x64",
            },
            hostname=hostname,
            users=[
                User(username="root", uid=0),
                User(username="ubuntu", uid=1000),
            ],
            interfaces=[interface],
            services={
                ProcessName.APACHE2: web_service,
            },
            processes=[web_process],
            respond_to_ping=True,
            host_type="server",
        )

        red_name = "red_agent"
        blue_name = "blue_agent"

        # Starting session for the Red agent.
        red_session = RedAbstractSession(
            ident=0,
            hostname=hostname,
            username="ubuntu",
            agent=red_name,
            pid=1001,
            session_type=SessionType.RED_ABSTRACT_SESSION,
        )

        # Starting session for the Blue agent.
        blue_session = Session(
            ident=0,
            hostname=hostname,
            username="ubuntu",
            agent=blue_name,
            pid=1002,
            session_type=SessionType.BLUE_DRONE_SESSION,
        )

        # Actions available to the Red agent.
        red_actions = [
            DiscoverRemoteSystems,
            AggressiveServiceDiscovery,
            StealthServiceDiscovery,
            ExploitRemoteService,
            PrivilegeEscalate,
        ]

        # Actions available to the Blue agent.
        blue_actions = [
            Monitor,
            Analyse,
            Remove,
            Restore,
        ]

        # Define the Red and Blue agents.
        agents = {
            red_name: ScenarioAgent(
                agent_name=red_name,
                team="Red",
                starting_sessions=[red_session],
                actions=red_actions,
                osint={
                    "Hosts": {
                        hostname: {
                            "Interfaces": "All"
                        }
                    }
                },
                allowed_subnets=[subnet_name],
                agent_type=OWASPScenarioAgent(red_name),
                active=True,
            ),
            blue_name: ScenarioAgent(
                agent_name=blue_name,
                team="Blue",
                starting_sessions=[blue_session],
                actions=blue_actions,
                osint={
                    "Hosts": {
                        hostname: {
                            "Interfaces": "All"
                        }
                    }
                },
                allowed_subnets=[subnet_name],
                agent_type=OWASPScenarioAgent(blue_name),
                active=True,
            ),
        }

        # Construct the CAGE4 scenario.
        scenario = Scenario(
            agents=agents,
            team_calcs=None,
            team_agents={
                "Red": [red_name],
                "Blue": [blue_name],
                "Green": [],
            },
            hosts={hostname: host},
            subnets={subnet_name: subnet},
            predeployed=False,
        )

        # Reuse reward calculators compatible with CAGE4.
        scenario.team_calc = (
            self.enterprise_helpers._generate_team_calcs()
        )

        return scenario


def main():
    generator = OWASPScenarioGenerator()

    cyborg = CybORG(
        scenario_generator=generator,
        seed=42,
    )

    controller = cyborg.environment_controller
    state = controller.state

    print("CybORG initialized successfully.")
    print("Agents:", list(controller.agent_interfaces.keys()))
    print("Hosts:", list(state.hosts.keys()))
    print("Subnets:", list(state.subnets.keys()))
    print("IP mapping:", state.ip_addresses)

    print("\nSimulated services:")
    for host_name, host in state.hosts.items():
        print(f"  {host_name}: {host.services}")

    print("\nSimulated processes and connections:")
    for host_name, host in state.hosts.items():
        print(f"  {host_name}:")
        for process in host.processes:
            print(f"    {process.name}: PID={process.pid}")
            print(f"    Connections: {process.get_state()}")

    print("\nStarting sessions:")
    for agent_name, sessions in state.sessions.items():
        print(f"  {agent_name}:")
        for session_id, session in sessions.items():
            print(
                f"    ID={session_id}, "
                f"class={type(session).__name__}, "
                f"hostname={session.hostname}, "
                f"session_type={session.session_type}, "
                f"active={session.active}"
            )


if __name__ == "__main__":
    main()
