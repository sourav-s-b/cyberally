from enum import Enum


class RedAction(Enum):
    NOOP = "noop"

    DISCOVER_APPLICATION = "discover_application"
    DISCOVER_API = "discover_api"

    SQL_INJECTION = "sql_injection"
    BROKEN_AUTHENTICATION = "broken_authentication"
    IDOR = "idor"
    XSS = "xss"

    PRIVILEGE_ESCALATION = "privilege_escalation"


class BlueAction(Enum):
    NOOP = "noop"

    MONITOR = "monitor"
    DETECT = "detect"
    BLOCK = "block"
    RESTORE = "restore"