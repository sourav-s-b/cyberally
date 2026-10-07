from ipaddress import IPv4Address, IPv4Network

from CybORG.Simulator.Actions import (
    DiscoverRemoteSystems,
    AggressiveServiceDiscovery,
    StealthServiceDiscovery,
    DiscoverDeception,
    ExploitRemoteService,
    PrivilegeEscalate,
    Impact,
    DegradeServices,
    Withdraw,
)


JUICE_SHOP_IP = IPv4Address("10.0.0.10")
JUICE_SHOP_HOSTNAME = "juice-shop"
JUICE_SHOP_SUBNET = IPv4Network("10.0.0.0/24")


def build_action_space(agent_name="red_agent"):
    """
    Build the CAGE4-style action space exposed to the
    existing PlannedRedAgent.

    This does not modify CAGE4 or the Red agent.
    """

    return {
        "action": {
            DiscoverRemoteSystems: True,
            AggressiveServiceDiscovery: True,
            StealthServiceDiscovery: True,
            DiscoverDeception: True,
            ExploitRemoteService: True,
            PrivilegeEscalate: True,
            Impact: True,
            DegradeServices: True,
            Withdraw: True,
        },

        "subnet": {
            JUICE_SHOP_SUBNET: True,
        },

        "ip_address": {
            JUICE_SHOP_IP: True,
        },

        "hostname": {
            JUICE_SHOP_HOSTNAME: True,
        },

        "session": {
            0: True,
        },

        "agent": {
            agent_name: True,
        },
    }