"""Blue-facing environment whose default and only implicit attacker is LLM Red."""
from blue.core.wrapper import CC4MARLEnv
from red.llm_agent import LLMConfig, OllamaClient
from red.planned_agent import PlannerConfig, configured_planner_class


class CC4LLMEnv(CC4MARLEnv):
    """For Blue training/evaluation; baseline selection requires the old wrapper.

The planner is bounded per agent/episode. Request count is invocation-wide;
configure it to cover the intended rollout budget. Errors retain tactical
execution, never instantiate DiscoveryFSRed. No privileged labels enter actors.
    """
    def __init__(self, *, llm_config=None, planner_config=None, client=None, **kwargs):
        config = LLMConfig(**(llm_config or {"model": "llama3.1:8b", "timeout_seconds": 60,
                                           "max_requests": 72}))
        planning = PlannerConfig(**(planner_config or {}))
        if "red_agent" in kwargs or "red_agent_class" in kwargs:
            raise ValueError("CC4LLMEnv selects LLM Red; use base wrapper for baseline comparisons")
        if client is None:
            client = OllamaClient(config)
            self.llm_model = client.check_model()
        else:
            self.llm_model = {"model": "injected test client", "digest": None}
        self.llm_client = client
        super().__init__(red_agent="llm-planner",
            red_agent_class=configured_planner_class(config, client, planning), **kwargs)
