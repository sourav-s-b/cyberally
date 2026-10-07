VULNERABILITIES = {
    "sql_injection": {
        "description": "SQL injection vulnerability",
        "requires": ["api_discovered"],
        "impact": "user_access",
    },

    "broken_authentication": {
        "description": "Authentication bypass",
        "requires": ["application_discovered"],
        "impact": "user_access",
    },

    "idor": {
        "description": "Insecure direct object reference",
        "requires": ["api_discovered"],
        "impact": "user_access",
    },

    "xss": {
        "description": "Cross-site scripting vulnerability",
        "requires": ["application_discovered"],
        "impact": "user_access",
    },
}