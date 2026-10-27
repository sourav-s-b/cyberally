"""CAGE4 Blue wrappers, foundation-v2 (not yet an EPyMARL training integration).

Fixed slots include all visible hosts, including routers and HQ's three subnets.
Action order remains Sleep, Monitor, then Analyse/Remove/Restore per sorted host.
Default masks enforce known simulator validity; evidence gating is opt-in.
Only per-agent observations/action spaces enter policy views. No true Red state.

v2 adds opt-in per-agent bounds (`per_agent_bounds=True`): slot counts derived
from each agent's own subnet assignment instead of one global 51-host bound.
Default mode is unchanged v1 behaviour. Stock EPyMARL shares one head across
agents and therefore requires the homogeneous (default) mode; per-agent widths
are reported for heuristics, logging, and future heterogeneous training.
"""
from copy import deepcopy
from collections.abc import Mapping
import numpy as np

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.AbstractActions import Monitor, Analyse, Remove, Restore
from blue_obs_features import host_to_vector, extract_subnets, VECTOR_LEN
from blue_action_masking import BlueZoneTracker

BLUE_AGENTS = [f"blue_agent_{i}" for i in range(5)]
ACTION_TEMPLATES = ("Analyse", "Remove", "Restore")
CLASSES = {"Analyse": Analyse, "Remove": Remove, "Restore": Restore}
# Scenario's largest Blue area is HQ: three subnets, each with a router.
DEFAULT_MAX_HOSTS = 3 * (EnterpriseScenarioGenerator.MAX_USER_HOSTS
                         + EnterpriseScenarioGenerator.MAX_SERVER_HOSTS + 1)
# Per-agent subnet counts mirror
# EnterpriseScenarioGenerator._generate_blue_agents: agents 0-3 own one zone
# each; agent 4 (HQ) owns public_access_zone + admin_network + office_network.
AGENT_SUBNET_COUNTS = (1, 1, 1, 1, 3)
PER_SUBNET_MAX_HOSTS = (EnterpriseScenarioGenerator.MAX_USER_HOSTS
                        + EnterpriseScenarioGenerator.MAX_SERVER_HOSTS + 1)  # 17
AGENT_MAX_HOSTS = tuple(n * PER_SUBNET_MAX_HOSTS
                        for n in AGENT_SUBNET_COUNTS)  # (17, 17, 17, 17, 51)
WRAPPER_VERSION = "foundation-v2"


class CC4MARLEnv:
    def __init__(self, seed=7629, max_hosts=DEFAULT_MAX_HOSTS, steps=400,
                 mask_mode="validity", common_reward=False,
                 reward_scalarisation="sum", per_agent_bounds=False):
        if not isinstance(max_hosts, int) or max_hosts < 1:
            raise ValueError("max_hosts must be a positive integer")
        if not isinstance(steps, int) or steps < 3:
            raise ValueError("steps must be >= 3 for the three mission phases")
        if mask_mode not in ("validity", "evidence"):
            raise ValueError("mask_mode must be validity or evidence")
        if reward_scalarisation not in ("sum", "mean"):
            raise ValueError("reward_scalarisation must be sum or mean")
        if per_agent_bounds and max_hosts != DEFAULT_MAX_HOSTS:
            raise ValueError("explicit max_hosts is incompatible with "
                             "per_agent_bounds (bounds come from the scenario)")
        # EPyMARL's EpisodeRunner always passes common_reward and
        # reward_scalarisation; accept and store them. `seed` stays an
        # attribute (applied on reset); the runner-style `seed()` setter
        # below writes to the same value for the *next* reset.
        self._seed = seed
        self._seed_pending = False
        self.common_reward = bool(common_reward)
        self.reward_scalarisation = reward_scalarisation
        self.per_agent_bounds = bool(per_agent_bounds)
        if self.per_agent_bounds:
            self.max_hosts_per_agent = list(AGENT_MAX_HOSTS)
        else:
            self.max_hosts_per_agent = [max_hosts] * len(BLUE_AGENTS)
        self.n_actions_per_agent = [2 + 3 * h for h in self.max_hosts_per_agent]
        self.obs_size_per_agent = [h * VECTOR_LEN for h in self.max_hosts_per_agent]
        # Scalar widths are the max across agents. In default (global) mode all
        # agents share them; in per-agent mode they are allocation upper bounds
        # for homogeneous consumers (stock EPyMARL requires default mode).
        self.max_hosts = max(self.max_hosts_per_agent)
        self.mask_mode = mask_mode
        self.n_agents = len(BLUE_AGENTS)
        self.episode_limit = steps
        self.n_actions = max(self.n_actions_per_agent)
        self.obs_size = max(self.obs_size_per_agent)
        sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent, red_agent_class=DiscoveryFSRed,
            steps=steps)
        self.cyborg = CybORG(scenario_generator=sg, seed=seed)
        self.env = self.cyborg.environment_controller
        self.hostnames, self.subnets, self.views, self.trackers = {}, {}, {}, {}
        self.observed_at = {}
        self._tick = 0
        self._awaiting = {}
        self._has_reset = False
        self._finished = True

    def reset(self, seed=None, options=None):
        if options:
            raise ValueError("reset options are not supported")
        if seed is not None:
            self._seed = seed
            self._seed_pending = True
        # First reset honors the constructor seed; an explicit seed()/reset(seed)
        # applies once; otherwise seed=None advances the RNG.
        reset_seed = (self._seed if seed is not None or not self._has_reset
                      or self._seed_pending else None)
        self._seed_pending = False
        self._finished = True
        self.cyborg.reset(seed=reset_seed)
        self._tick = 0
        self._awaiting = {}
        for i, agent in enumerate(BLUE_AGENTS):
            base = deepcopy(self.cyborg.get_observation(agent))
            hosts = sorted(h for h, value in base.items()
                           if isinstance(value, dict) and "System info" in value)
            bound = self.max_hosts_per_agent[i]
            if len(hosts) > bound:
                raise ValueError(f"{agent} has {len(hosts)} hosts; max_hosts="
                                 f"{bound} would truncate them")
            self.hostnames[agent] = hosts
            self.views[agent] = {h: base[h] for h in hosts}
            self.subnets[agent] = {h: extract_subnets(base[h]) for h in hosts}
            self.observed_at[agent] = {h: {k: 0 for k in base[h]} for h in hosts}
            self.trackers[agent] = BlueZoneTracker(hosts)
        self._has_reset = True
        self._finished = False
        return self.get_obs(), {}

    def _mask_agent(self, agent):
        mask = np.zeros(self.n_actions_per_agent[BLUE_AGENTS.index(agent)],
                        dtype=np.int64)
        mask[0] = 1
        if agent in self._awaiting:
            return mask
        space = self.env.get_action_space(agent)
        if not space["session"].get(0, False):
            return mask
        mask[1] = int(space["action"].get(Monitor, False))
        for i, host in enumerate(self.hostnames[agent]):
            if not space["hostname"].get(host, False):
                continue
            allowed = self.trackers[agent].mask_for(host, self.mask_mode)
            for offset, name in enumerate(ACTION_TEMPLATES):
                mask[2 + 3*i + offset] = int(
                    name in allowed and space["action"].get(CLASSES[name], False))
        return mask

    def _decode(self, agent, idx):
        if idx == 0:
            return Sleep(), None
        if idx == 1:
            return Monitor(session=0, agent=agent), None
        hi, action_type = divmod(idx - 2, 3)
        host = self.hostnames[agent][hi]
        return CLASSES[ACTION_TEMPLATES[action_type]](
            session=0, agent=agent, hostname=host), host

    def _obs_agent(self, agent):
        obs = np.zeros((self.max_hosts_per_agent[BLUE_AGENTS.index(agent)],
                        VECTOR_LEN), dtype=np.float32)
        for i, host in enumerate(self.hostnames[agent]):
            obs[i] = host_to_vector(self.views[agent][host], agent,
                                   self.subnets[agent][host])
        return obs.flatten()

    def get_host_presence(self, agent_id):
        mask = np.zeros(self.max_hosts_per_agent[agent_id], dtype=np.int64)
        mask[:len(self.hostnames[BLUE_AGENTS[agent_id]])] = 1
        return mask

    def get_obs(self):
        return [self._obs_agent(a) for a in BLUE_AGENTS]

    def get_obs_agent(self, agent_id):
        return self._obs_agent(BLUE_AGENTS[agent_id])

    def get_telemetry(self, agent_id):
        """Optional draft JSONL block from this agent's local retained Blue view.

        Does not change actor features, masks, simulator behavior or action state.
        """
        from blue_telemetry import telemetry_block
        agent = BLUE_AGENTS[agent_id]
        return telemetry_block(self.views[agent], self.observed_at[agent],
                               tick=self._tick, subnets_by_host=self.subnets[agent])

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        return np.concatenate(self.get_obs()).astype(np.float32)

    def get_state_size(self):
        return sum(self.obs_size_per_agent)

    def get_avail_actions(self):
        return [self._mask_agent(a) for a in BLUE_AGENTS]

    def get_avail_agent_actions(self, agent_id):
        return self._mask_agent(BLUE_AGENTS[agent_id])

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {"state_shape": self.get_state_size(), "obs_shape": self.obs_size,
                "n_actions": self.n_actions, "n_agents": self.n_agents,
                "episode_limit": self.episode_limit,
                "max_hosts_per_agent": list(self.max_hosts_per_agent),
                "n_actions_per_agent": list(self.n_actions_per_agent),
                "obs_size_per_agent": list(self.obs_size_per_agent),
                "per_agent_bounds": self.per_agent_bounds,
                "wrapper_version": WRAPPER_VERSION}

    def _consume(self, agent, data):
        tracker = self.trackers[agent]
        # Merge every visible host, including unsolicited Monitor observations.
        # Replace fields, never concatenate repeated snapshots/events. Timestamps
        # preserve freshness for later temporal features; absent fields stay unknown
        # or last-known, not fabricated zero measurements.
        for host in self.hostnames[agent]:
            detail = data.get(host, {})
            for field, value in detail.items():
                self.views[agent][host][field] = deepcopy(value)
                self.observed_at[agent][host][field] = self._tick
        pending = self._awaiting.get(agent)
        if pending is None:
            return
        host, name = pending
        success = str(data.get("success", ""))
        if success == "TRUE":
            if name == "Analyse":
                detail = data.get(host, {})
                # A completed empty scan supersedes the *latest file observation*,
                # but does not itself clear the tracker's prior positive evidence.
                self.views[agent][host]["Files"] = deepcopy(detail.get("Files", []))
                self.observed_at[agent][host]["Files"] = self._tick
                tracker.note_analyse_result(host, detail, self._tick)
            else:
                tracker.note_remediation_result(host, name, self._tick)
                # Historical artifacts are no longer current measurements. Do not
                # mark the host clean; VERIFY remains until subsequent observations.
                for field in ("Files", "Processes"):
                    if field not in data.get(host, {}):
                        self.views[agent][host].pop(field, None)
                        self.observed_at[agent][host].pop(field, None)
            del self._awaiting[agent]
        elif success == "FALSE":
            tracker.note_failure(host, name)
            del self._awaiting[agent]
        elif self._tick >= tracker.pending_until[host]:
            raise RuntimeError(f"{agent}: {name} on {host} overdue without result")

    def step(self, actions):
        if self._finished:
            raise RuntimeError("Call reset before stepping a new episode")
        if not isinstance(actions, Mapping):
            if hasattr(actions, "detach"):
                actions = actions.detach().cpu().numpy()
            values = np.asarray(actions)
            if values.shape != (self.n_agents,):
                raise ValueError(f"Expected {self.n_agents} scalar actions")
            actions = dict(zip(BLUE_AGENTS, values))
        if set(actions) - set(BLUE_AGENTS):
            raise ValueError("Unknown Blue agent in action mapping")
        decoded = {}
        for agent in BLUE_AGENTS:
            idx = actions.get(agent, 0)
            width = self.n_actions_per_agent[BLUE_AGENTS.index(agent)]
            if (not isinstance(idx, (int, np.integer)) or idx < 0
                    or idx >= width or not self._mask_agent(agent)[idx]):
                idx = 0
            action, host = self._decode(agent, int(idx))
            decoded[agent] = action
            if host is not None:
                name = type(action).__name__
                self.trackers[agent].note_action_issued(
                    host, name, action.duration, self._tick)
                self._awaiting[agent] = (host, name)
        self.env.step(decoded)
        self._tick += 1
        rewards = []
        for agent in BLUE_AGENTS:
            self._consume(agent, self.env.get_last_observation(agent).data)
            rewards.append(float(sum(self.env.get_reward(agent).values())))
        # Preserve the native finite scenario terminal (upstream ends at steps-1).
        # A wrapper-only cut-off is a truncation, not a fabricated natural terminal.
        terminated = bool(self.env.done)
        truncated = self._tick >= self.episode_limit and not terminated
        self._finished = terminated or truncated
        # EPyMARL reads info["episode_limit"] to tell horizon cut-offs (bootstrap)
        # from true terminals. The native end always coincides with our horizon.
        info = {"episode_limit": bool(
            truncated or (terminated and self._tick >= self.episode_limit - 1))}
        if self.common_reward:
            # Native Blue reward is already one shared team signal duplicated
            # per agent; normalize once (rewards[0]), never a 5x sum, per the
            # contracts invariant. sum/mean coincide on identical values.
            return self.get_obs(), float(rewards[0]), terminated, truncated, info
        return self.get_obs(), rewards, terminated, truncated, info

    def seed(self, seed=None):
        """Runner-style seed setter; takes effect on the next reset."""
        if seed is not None:
            self._seed = seed
            self._seed_pending = True

    def close(self):
        """No simulator resource to release; present for the runner lifecycle."""

    def render(self):
        """No visual rendering in this wrapper; present for the runner lifecycle."""

    def save_replay(self):
        raise NotImplementedError("CC4MARLEnv has no replay recording")

    def get_stats(self):
        return {}


class CC4BlueWrapper:
    """Single-agent development facade over the same joint-step implementation.

    Other Blue agents Sleep. This avoids separate, diverging bookkeeping logic.
    reset -> (obs, mask); step -> (obs, mask, scalar team reward, done).
    """
    def __init__(self, seed=7629, blue_id="blue_agent_0",
                 max_hosts=DEFAULT_MAX_HOSTS, steps=400, mask_mode="validity",
                 per_agent_bounds=False):
        if blue_id not in BLUE_AGENTS:
            raise ValueError("Unknown Blue agent")
        self.blue_id = blue_id
        self._agent_id = BLUE_AGENTS.index(blue_id)
        self._joint = CC4MARLEnv(seed, max_hosts, steps, mask_mode,
                                 per_agent_bounds=per_agent_bounds)
        self.cyborg = self._joint.cyborg
        self.env = self._joint.env
        self.max_hosts = self._joint.max_hosts_per_agent[self._agent_id]
        self.n_actions = self._joint.n_actions_per_agent[self._agent_id]
        self.obs_size = self._joint.obs_size_per_agent[self._agent_id]

    def reset(self, seed=None):
        self._joint.reset(seed=seed)
        return self.get_obs(), self.get_mask()

    @property
    def hostnames(self):
        return self._joint.hostnames[self.blue_id]

    @property
    def views(self):
        return self._joint.views[self.blue_id]

    @property
    def subnets(self):
        return self._joint.subnets[self.blue_id]

    @property
    def tracker(self):
        return self._joint.trackers[self.blue_id]

    @property
    def _awaiting(self):
        return self._joint._awaiting.get(self.blue_id)

    @property
    def _tick(self):
        return self._joint._tick

    def get_obs(self):
        return self._joint.get_obs_agent(self._agent_id)

    def get_mask(self):
        return self._joint.get_avail_agent_actions(self._agent_id)

    def get_obs_size(self):
        return self.obs_size

    def get_total_actions(self):
        return self.n_actions

    def get_state(self):
        return self.get_obs().copy()

    def step(self, idx):
        _, rewards, terminated, truncated, _ = self._joint.step({self.blue_id: idx})
        return self.get_obs(), self.get_mask(), rewards[self._agent_id], terminated or truncated

    def seed(self, seed=None):
        self._joint.seed(seed)

    def close(self):
        pass
