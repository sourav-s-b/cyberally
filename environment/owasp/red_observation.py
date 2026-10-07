from ipaddress import IPv4Address, IPv4Network

from CybORG.Shared.Enums import TernaryEnum


JUICE_SHOP_IP = IPv4Address("10.0.0.10")
JUICE_SHOP_SUBNET = IPv4Network("10.0.0.0/24")
JUICE_SHOP_HOSTNAME = "juice-shop"


def build_red_observation(
    state,
    success=TernaryEnum.UNKNOWN,
    action=None,
):
    """
    Convert the internal OWASP Juice Shop state into
    the CAGE4-style observation expected by the existing
    PlannedRedAgent.
    """

    observation = {
        "success": success,
        "action": action,

        str(JUICE_SHOP_IP): {
            "System info": {
                "Hostname": JUICE_SHOP_HOSTNAME,
            },

            "Interface": [
                {
                    "ip_address": JUICE_SHOP_IP,
                    "Subnet": JUICE_SHOP_SUBNET,
                }
            ],
        },
    }

    # The existing PlannedRedAgent checks for a Sessions
    # field when the host has user access. Without this,
    # its session-removal logic can incorrectly mark the
    # host as lost.
    if state.user_access:
        observation[str(JUICE_SHOP_IP)]["Sessions"] = [
            {
                "session_id": 0,
                "agent": "red_agent",
            }
        ]

    return observation