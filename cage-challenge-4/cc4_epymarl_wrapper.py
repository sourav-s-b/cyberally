"""EPyMARL-facing wrapper: one CAGE4 Blue agent as discrete obs + action mask.

Contract (matches EPyMARL MultiAgentEnv for a single agent):
- obs: flat float32 vector = concat of per-host v2 RAW feature vectors
  (10 dims each, fixed sorted host order, zero-padded to max_hosts).
  Raw dims BY DESIGN (review): MAPPO learns its own weighting of n_files /
  n_unknown_files / max_density. The collapsed scalar lives only in the
  masking module (detection_hit), never here.
- actions: fixed discrete 1-D space, size 2 + 3*max_hosts:
    [Sleep, Monitor, Analyse_h0, Remove_h0, Restore_h0, Analyse_h1, ...]
  Host slots beyond this seed's topology are permanently masked off, so
  n_actions is seed-independent (EPyMARL requires a fixed space).
- mask: binary vector from BlueZoneTracker (EPyMARL get_avail_* equivalent).
  Analyse masked while IN_PROGRESS (pending_until); Remove/Restore masked
  unless CONFIRMED; Sleep/Monitor always allowed.
- reward/done: straight from CybORG Results (Blue team reward).

LIMITATIONS (for handoff):
- get_state() returns own obs. Proper CTDE state = concat of all 5 Blue
  zones' views; needs a 5-agentulti wrapper.
- Single Blue agent only (blue_agent_0 default). Red = scripted DiscoveryFSRed.
- Session index assumed 0 (valid in all runs so far; wrapper falls back to
  Sleep if the decoded action is masked, so a bad decode can't corrupt).
"""

import numpy as np

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.AbstractActions import (
    Monitor, Analyse, Remove, Restore)

from blue_obs_features import host_to_vector, extract_subnets, VECTOR_LEN
from blue_action_masking import BlueZoneTracker

ACTION_TEMPLATES = ("Analyse", "Remove", "Restore")
CLASSES = {"Analyse": Analyse, "Remove": Remove, "Restore": Restore}


class CC4BlueWrapper:
    def __init__(self, seed=7629, blue_id="blue_agent_0", max_hosts=16,
                 steps=400):
        self.seed = seed
        self.blue_id = blue_id
        self.max_hosts = max_hosts
        self.n_actions = 2 + 3 * max_hosts
        self.obs_size = max_hosts * VECTOR_LEN
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=DiscoveryFSRed,
            steps=steps)
        self.cyborg = CybORG(scenario_generator=sg, seed=seed)
        self.env = self.cyborg.environment_controller
        self.hostnames = []
        self.subnets = {}
        self.views = {}
        self.tracker = None
        self._tick = 0
        self._awaiting = None  # (hostname, action_name) resolving IN_PROGRESS

    # ---- EPyMARL-shaped API ----
    def reset(self):
        base = self.cyborg.reset(agent=self.blue_id).observation
        self.hostnames = sorted(k for k in base
                                if k not in ("success", "action"))
        self.subnets = {h: extract_subnets(base[h]) for h in self.hostnames}
        self.views = {h: dict(base[h]) for h in self.hostnames}
        self.tracker = BlueZoneTracker(self.hostnames)
        self._tick = 0
        self._awaiting = None
        return self.get_obs(), self.get_mask()

    def get_obs(self):
        vecs = [host_to_vector(self.views[h], self.blue_id,
                               self.subnets.get(h))
                for h in self.hostnames]
        pad = self.max_hosts - len(vecs)
        if pad > 0:
            vecs += [[0.0] * VECTOR_LEN] * pad
        return np.array(vecs[:self.max_hosts], dtype=np.float32).flatten()

    def get_obs_size(self):
        return self.obs_size

    def get_total_actions(self):
        return self.n_actions

    def get_mask(self):
        """Binary avail mask over the discrete action space."""
        m = np.zeros(self.n_actions, dtype=np.int64)
        m[0] = 1  # Sleep
        m[1] = 1  # Monitor
        for i, h in enumerate(self.hostnames[:self.max_hosts]):
            allowed = self.tracker.mask_for(h)
            base = 2 + 3 * i
            if "Analyse" in allowed:
                m[base] = 1
            if "Remove" in allowed:
                m[base + 1] = 1
            if "Restore" in allowed:
                m[base + 2] = 1
        return m

    def get_state(self):
        # LIMITATION: proper CTDE state needs all 5 zones; own obs for now.
        return self.get_obs().copy()

    # ---- stepping ----
    def _decode(self, idx):
        if idx == 0:
            return Sleep(), None
        if idx == 1:
            return Monitor(session=0, agent=self.blue_id), None
        j = idx - 2
        hi, t = divmod(j, 3)
        if hi >= len(self.hostnames):
            return Sleep(), None
        host = self.hostnames[hi]
        cls = CLASSES[ACTION_TEMPLATES[t]]
        return cls(session=0, agent=self.blue_id, hostname=host), host

    def step(self, idx):
        """Decode -> joint env step -> update views/tracker. Returns
        (obs, mask, reward, done). Falls back to Sleep if idx is masked."""
        if idx < 0 or idx >= self.n_actions or self.get_mask()[idx] == 0:
            idx = 0  # masked/invalid pick can never corrupt the env
        action, host = self._decode(idx)
        res = self.cyborg.step(agent=self.blue_id, action=action)
        self._tick += 1
        name = type(action).__name__
        if name in ("Analyse", "Remove", "Restore") and host is not None:
            self.tracker.note_action_issued(host, name, action.duration,
                                            self._tick)
            self._awaiting = (host, name)
        self.tracker.note_step(self._tick)
        data = dict(res.observation)
        # Resolve a pending multi-step action: key on what is IN_PROGRESS
        # (self._awaiting), NOT on the current step's decoded action -- the
        # controller drops new actions while one resolves, so the current
        # decode is usually Sleep here. This is the IN_PROGRESS landmine.
        if self._awaiting is not None:
            succ = str(data.get("success", ""))
            h, aname = self._awaiting
            if succ == "TRUE":
                # Host key may be ABSENT on clean hosts (empty reveal) --
                # that is the resolve, not a missing observation.
                detail = data.get(h, {})
                for k, v in detail.items():
                    self.views[h][k] = v
                if aname == "Analyse":
                    self.tracker.note_analyse_result(h, detail, self._tick)
                else:  # Remove/Restore resolved: host remediated
                    self.tracker.pending_until.pop(h, None)
                    self.tracker.state[h] = "CLEAN"
                    self.tracker.empty_strikes[h] = 0
                self._awaiting = None
            elif succ == "FALSE":
                # Action died: release the mask, no strike, policy may retry.
                self.tracker.pending_until.pop(h, None)
                self._awaiting = None
            # IN_PROGRESS: keep waiting, mask stays closed.
        return self.get_obs(), self.get_mask(), res.reward, res.done


BLUE_AGENTS = [f"blue_agent_{i}" for i in range(5)]


class CC4MARLEnv:
    """5-blue-agent joint env with the EPyMARL MultiAgentEnv shape.

    One shared CybORG; per-agent zone views/trackers/masks (same logic as
    CC4BlueWrapper). Discrete fixed action space per agent (2+3*max_hosts).
    get_state() = concat of all 5 agents' obs -- the proper CTDE central
    state (the single-agent wrapper can only return its own obs).
    Red = scripted DiscoveryFSRed. Rewards = per-blue CybORG rewards.
    EPyMARL registration (next step): add a `cc4_fn(**kwargs)` returning
    this class to EPyMARL's envs REGISTRY, plus a yaml (mappo, common_reward
    as configured, episode_limit=steps).
    """

    def __init__(self, seed=7629, max_hosts=16, steps=400):
        self.seed = seed
        self.max_hosts = max_hosts
        self.n_agents = len(BLUE_AGENTS)
        self.episode_limit = steps
        self.n_actions = 2 + 3 * max_hosts
        self.obs_size = max_hosts * VECTOR_LEN
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=DiscoveryFSRed,
            steps=steps)
        self.cyborg = CybORG(scenario_generator=sg, seed=seed)
        self.env = self.cyborg.environment_controller
        self.hostnames = {}
        self.subnets = {}
        self.views = {}
        self.trackers = {}
        self._tick = 0
        self._awaiting = {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        for a in BLUE_AGENTS:
            base = self.cyborg.reset(agent=a).observation
            hosts = sorted(k for k in base if k not in ("success", "action"))
            self.hostnames[a] = hosts
            self.subnets[a] = {h: extract_subnets(base[h]) for h in hosts}
            self.views[a] = {h: dict(base[h]) for h in hosts}
            self.trackers[a] = BlueZoneTracker(hosts)
        self._tick = 0
        self._awaiting = {}
        return self.get_obs(), {}

    def _decode(self, agent, idx):
        hosts = self.hostnames[agent]
        if idx == 0:
            return Sleep(), None
        if idx == 1:
            return Monitor(session=0, agent=agent), None
        j = idx - 2
        hi, t = divmod(j, 3)
        if hi >= len(hosts):
            return Sleep(), None
        host = hosts[hi]
        return CLASSES[ACTION_TEMPLATES[t]](session=0, agent=agent,
                                            hostname=host), host

    def _mask_agent(self, agent):
        m = np.zeros(self.n_actions, dtype=np.int64)
        m[0] = 1
        m[1] = 1
        for i, h in enumerate(self.hostnames[agent][:self.max_hosts]):
            allowed = self.trackers[agent].mask_for(h)
            base = 2 + 3 * i
            if "Analyse" in allowed:
                m[base] = 1
            if "Remove" in allowed:
                m[base + 1] = 1
            if "Restore" in allowed:
                m[base + 2] = 1
        return m

    def _obs_agent(self, agent):
        vecs = [host_to_vector(self.views[agent][h], agent,
                               self.subnets[agent].get(h))
                for h in self.hostnames[agent]]
        pad = self.max_hosts - len(vecs)
        if pad > 0:
            vecs += [[0.0] * VECTOR_LEN] * pad
        return np.array(vecs[:self.max_hosts], dtype=np.float32).flatten()

    def get_obs(self):
        return [self._obs_agent(a) for a in BLUE_AGENTS]

    def get_obs_agent(self, agent_id):
        return self._obs_agent(BLUE_AGENTS[agent_id])

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        return np.concatenate(self.get_obs()).astype(np.float32)

    def get_state_size(self):
        return self.obs_size * self.n_agents

    def get_avail_actions(self):
        return [self._mask_agent(a) for a in BLUE_AGENTS]

    def get_avail_agent_actions(self, agent_id):
        return self._mask_agent(BLUE_AGENTS[agent_id])

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {"state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit}

    def step(self, actions):
        """actions: dict {agent_name: discrete_idx} (or list in agent order).
        Returns (obss, rewards, terminated, truncated, info). Masked/invalid
        picks fall back to Sleep."""
        if isinstance(actions, (list, tuple)):
            actions = dict(zip(BLUE_AGENTS, actions))
        acts = {}
        for a in BLUE_AGENTS:
            idx = actions.get(a, 0)
            if (not isinstance(idx, (int, np.integer)) or idx < 0
                    or idx >= self.n_actions
                    or self._mask_agent(a)[idx] == 0):
                idx = 0
            action, host = self._decode(a, int(idx))
            acts[a] = action
            name = type(action).__name__
            if name in ("Analyse", "Remove", "Restore") and host is not None:
                self.trackers[a].note_action_issued(
                    host, name, action.duration, self._tick)
                self._awaiting[a] = (host, name)
        self.env.step(acts)  # single joint controller step, all agents act
        self._tick += 1
        for a in BLUE_AGENTS:
            self.trackers[a].note_step(self._tick)
        obss, rewards = [], []
        for a in BLUE_AGENTS:
            data = self.env.get_last_observation(a).data
            data = dict(data) if not isinstance(data, dict) else data
            if self._awaiting.get(a) is not None:
                succ = str(data.get("success", ""))
                h, aname = self._awaiting[a]
                if succ == "TRUE":
                    detail = data.get(h, {})
                    for k, v in detail.items():
                        self.views[a][h][k] = v
                    if aname == "Analyse":
                        self.trackers[a].note_analyse_result(
                            h, detail, self._tick)
                    else:
                        self.trackers[a].pending_until.pop(h, None)
                        self.trackers[a].state[h] = "CLEAN"
                        self.trackers[a].empty_strikes[h] = 0
                    del self._awaiting[a]
                elif succ == "FALSE":
                    self.trackers[a].pending_until.pop(h, None)
                    del self._awaiting[a]
            obss.append(self._obs_agent(a))
            rewards.append(round(sum(self.env.get_reward(a).values()), 1))
        done = self.env.done or self._tick >= self.episode_limit
        return obss, rewards, done, False, {}
