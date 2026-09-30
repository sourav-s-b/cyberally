# Self-Evolving SOC — Team Split & Research Guide

Detailed version: who starts on what, what to research, and exactly where to look.

---

## The one decision that changes everything: build on CybORG / CAGE 4, don't build from scratch

Your base paper's "CC4" almost certainly refers to **CAGE Challenge 4**, run on **CybORG** (Cyber Operations Research Gym) — an open-source, actively maintained gym purpose-built for training Blue and Red cyber-defense agents. CAGE 4 is specifically the multi-agent version: a defence-industry enterprise network scenario where Blue agents defend 1–3 subnets of up to 16 hosts each (≈92–242 action space, ≈82–210 observation space), which lines up almost exactly with your 2–3 zone scope, and it ships with MAPPO baselines and a red agent already.

**Why this matters for your timeline:** if you build your own Docker network simulator from scratch, the Environment person alone could eat half your semester. If instead you fork CybORG/CAGE4 and adapt it, you get: a working network topology, a Blue action space, telemetry/observation format, and a reference Red agent — on day one. Then the actual work becomes *scoping it down to 2–3 zones, wiring in your ML anomaly features, swapping in an LLM Red agent, and building the retrain loop* — which is a much saner ask for four students on a deadline.

**Where to look:**
- `github.com/cage-challenge/cage-challenge-4` — the CC4 repo itself: environment, docs, install guide, baseline agents
- `github.com/cage-challenge/CybORG` — the underlying gym
- Read the CAGE 4 paper first (search "CAGE Challenge 4 scalable multi-agent reinforcement learning gym") — it tells you the actual reward structure, action set, and known failure modes, which saves you from re-deriving them
- Also worth 20 minutes: a 2025 paper titled *"Guidelines for Applying RL and MARL in Cybersecurity Applications"* — it surveys CybORG/CAGE 2/CAGE 4 and other environments and flags common pitfalls in this exact space

**First team task (do this together, before splitting):** clone CAGE4, get its example agents running end-to-end (even the dumb random baseline), and look at its `env.step()` / observation dict / action dict directly in a notebook. That one exercise answers most of your "how do the three pieces even talk to each other" questions for free, because CybORG already defines those interfaces.

---

## The Shared Schema — your common starting point

This is the piece that lets three people build in parallel without waiting on each other. Treat it as your team's internal "wire format" — a thin layer you all code against, that wraps whatever CAGE4/CybORG actually hands you underneath. You don't need CAGE4's exact internal field names memorized to start; you need everyone agreeing on *this* shape, then the Environment person's job includes translating CAGE4's real output into it. Copy this into your shared repo as `schema.md` or equivalent on day one, and treat any change to it as a team decision, not a solo one.

### 1. Zone & host naming (fix this first — everything else refers to it)

```
zone_id:  "restricted" | "operational" | "dmz"
host_id:  "<zone_id>_host_<n>"        e.g. "restricted_host_3"
```
Agree the actual list of zones and how many hosts per zone *before* anyone writes code that assumes a count.

### 2. Observation schema (what each Blue agent sees per step)

```json
{
  "agent_id": "blue_restricted",
  "zone_id": "restricted",
  "timestep": 142,
  "local_hosts": [
    {
      "host_id": "restricted_host_3",
      "status": "compromised" | "suspicious" | "clean" | "isolated",
      "anomaly_score": 0.83,
      "recent_events": ["login_fail", "process_spawn"]
    }
  ],
  "strategic_goal": "increase_monitoring" | "isolate_zone" | "normal",
  "peer_messages": []
}
```
- `anomaly_score` is the Isolation Forest / LOF output — this is the field that makes ML "an input feature, not a decision-maker." The Blue person needs it present (even as a placeholder/random value) before real training starts, so the Environment person should stub this field in week 1.
- `recent_events` — agree the fixed vocabulary of possible event strings as a team; an open-ended string field will break whatever encodes it into a feature vector.

### 3. Action schema (what an agent can submit per step)

```json
{
  "agent_id": "blue_restricted",
  "action_type": "isolate_host" | "kill_process" | "block_port" | "patch" | "restore_service" | "monitor" | "no_op",
  "target_host_id": "restricted_host_3",
  "params": {}
}
```
Red's action schema mirrors this shape with its own `action_type` vocabulary (`recon`, `exploit`, `escalate`, `pivot`, `impact`, etc.) — agree that vocabulary with whoever's building CAGE4's scripted Red baseline, since you'll likely inherit most of it directly.

### 4. Episode / telemetry log schema (what gets written to disk every step)

```json
{
  "episode_id": "ep_0042",
  "timestep": 142,
  "zone_id": "restricted",
  "event_type": "alert_raised" | "action_taken" | "host_state_change",
  "actor": "red" | "blue_restricted" | "system",
  "payload": {}
}
```
This is the file the evaluation person will parse to compute MTTD/MTTR/false-positive rate — get everyone writing to this format from week one, even in dummy form, rather than backfilling it before your final demo.

### 5. Red's "strategy pool" entry (the retrain-loop's core data structure)

```json
{
  "sequence_id": "seq_0007",
  "discovered_round": 3,
  "target_zone": "operational",
  "action_sequence": [
    {"action_type": "recon", "target_host_id": "operational_host_1"},
    {"action_type": "exploit", "target_host_id": "operational_host_1"}
  ],
  "outcome": "succeeded" | "contained",
  "blue_response_time": 6
}
```
This is what triggers a Blue retrain round: whenever a sequence's `outcome` is `"succeeded"`, it gets added to the pool the Blue agent next trains against.

### 6. Retrain / evaluation round record (what the loop+eval person logs each cycle)

```json
{
  "round": 3,
  "blue_policy_version": "v3",
  "strategy_pool_size": 12,
  "metrics": {
    "containment_reward": 0.71,
    "mttd": 4.2,
    "mttr": 9.8,
    "false_positive_rate": 0.06
  }
}
```
This is the exact data your headline result (metric vs. retrain-round-number) is plotted from — instrument it from round 1, not round 10.

### A note on getting this "right"

You will get some of these field names or shapes wrong on the first pass — that's fine and expected. What actually matters is that they're **written down and shared** before three people start coding independently, so a disagreement surfaces in a five-minute conversation now instead of a broken integration the week before your demo. Revisit this file together after your Week 0 CAGE4 exploration below, since CAGE4's real dicts may suggest better field names than this first draft — but always as a team edit, never a silent local change.

---

## Week 0 — Everyone, together, before splitting (aim: 2–3 days)

1. Get CAGE4/CybORG installed and running locally (shared repo, one install doc so nobody fights environment issues alone)
2. Read through the CAGE4 observation space, action space, and reward function in its docs/code — this becomes your team's shared vocabulary
3. Decide, in writing, in a shared doc:
   - Are you forking CAGE4's network topology directly, or trimming it to your own 2–3 zones?
   - What's the *contract* between Environment ↔ Blue and Environment ↔ Red (you'll mostly inherit this from CybORG, but note where you're diverging — e.g. adding your own anomaly-score field to the observation)
   - Tech stack: Python version, RL library, LLM API for Red, where logs get written, whether you use a shared GPU/cloud instance for training runs
   - A shared repo structure so nobody's work collides (`/env`, `/red`, `/blue`, `/loop_eval`, `/notebooks`)
4. Set a recurring 20–30 min sync (2–3x/week) — with three interdependent tracks, small misalignments compound fast if left to a weekly checkpoint

---

## Role 1 — Environment / Cyber-Range

**Start this week:**
- Fork/clone CAGE4, get it running, then trim its network to your team's 2–3 zones (Restricted, Operational, optionally DMZ)
- Map out concretely: what hosts/services live in each zone, what counts as "critical" (this feeds directly into the reward function Blue will use later)
- Extend the observation dict to carry an extra field for the ML anomaly score (even a placeholder/random value for now) — this unblocks the Blue person early
- Set up Docker so the whole thing is containerized and has controlled egress, matching your abstract's claim

**Research / read:**
- CybORG's own docs on its state model and action execution — you need to understand this cold, since you own the interface everyone else depends on
- The CAGE4 paper's description of partial observability and noisy monitoring (false positives/negatives in alerts) — this is a realism detail worth keeping, not simplifying away
- Later (not week 1): how to build a "digital twin" / shadow copy of the environment for the post-deployment shadow re-simulation feature — CybORG's own scenario-reset mechanics may already give you most of this for free

**Where to look:** CAGE4 GitHub repo + its issues tab (people ask exactly your questions there), CybORG's arXiv paper (search "CybORG: A Gym for the Development of Autonomous Cyber Agents"), Docker docs for network/egress rules if you're not already comfortable with `docker network` and firewall rules.

---

## Role 2 — Red Agent

**Start this week:**
- Get CAGE4's built-in scripted Red agent running against the environment first — this is your fallback/baseline and also the fastest way to understand the action space Red actually has (recon, exploit, escalate, impact, etc.)
- Sketch how an LLM sits on top of this: at each step, the LLM sees the current game state/history and picks (or reasons toward) the next Red action, rather than following the scripted decision tree
- Design the "strategy pool" log format early, even as a stub — every successful attack sequence should be logged in a schema the Blue/loop person can consume later (this is a shared-contract item, raise it in a sync)

**Research / read:**
- A 2025 paper, *"Large Language Models are Autonomous Cyber Defenders"* — despite the title, it documents building the **first LLM agent integrated into CybORG CAGE 4** (they built an adapter framework for it), which is close to exactly your task, just mirrored to the Red side — read their adapter design and the problems they hit (LLM inference latency vs. RL step speed is a real one)
- General LLM-agent patterns: tool-use/function-calling loops, ReAct-style reasoning traces (useful since you want Red to adapt tactics mid-attack, not follow a script)
- CAGE4's own scripted Red agent code — even though you're replacing its brain, its action repertoire and kill-chain structure is your starting menu of "things Red can do"

**Where to look:** the CAGE4 repo's red agent baseline, the LLM-cyber-defender paper above (search it directly, it's from 2025), your chosen LLM provider's tool-use/function-calling docs.

---

## Role 3 — Blue Agent (heaviest role — good candidate for 2 people if you can swing it)

**Start this week:**
- Get CAGE4's baseline MAPPO Blue agent training against the scripted Red — this alone proves your pipeline works end-to-end before you add any of your own complexity
- Separately, prototype the anomaly-scoring piece: train an Isolation Forest or LOF model on whatever telemetry format the Environment person is producing, output a risk score
- Once both work standalone, wire the anomaly score into the Blue agent's observation vector as one more input feature — explicitly *not* as a rule that makes decisions itself (this is the exact contradiction your report flagged between old Slide 3 and Slide 8, so get this right early)
- Add action masking so the agent can't select invalid actions (e.g. isolating an already-isolated host) — CAGE4 baselines usually already demonstrate this pattern, borrow it

**Research / read:**
- MAPPO under CTDE — the core idea: one centralized critic sees global state *during training only*; at deployment each agent acts on local observations alone. Read the original MAPPO paper's intuition section, you don't need to re-derive the math
- A practical MARL library so you're not writing MAPPO from scratch: **EPyMARL** (search `oxwhirl/epymarl` on GitHub — a clean, well-documented multi-agent PPO/MAPPO implementation with native PettingZoo support) or **MARLlib** (`Replicable-MARL/MARLlib`, built on Ray/RLlib, more powerful but heavier to learn). For a semester project, EPyMARL is the gentler on-ramp.
- scikit-learn docs for `IsolationForest` and `LocalOutlierFactor` — both are one-liners to instantiate, the real work is feature engineering from your telemetry
- Your own report already flags this finding, worth internalizing early: in CAGE4-style evaluations, carefully engineered heuristics have sometimes *outperformed* submitted MARL agents — the winning lesson wasn't more RL, it was valid-action handling and disciplined reward shaping. That should shape how much time you spend tuning MAPPO vs. getting the basics (action masking, reward shaping) right.

**Where to look:** EPyMARL GitHub + its README examples, CAGE4's baseline Blue agent code, scikit-learn's anomaly detection docs, the MAPPO paper (search "MAPPO Yu et al Surprising Effectiveness").

---

## Role 4 — Retrain Loop + Evaluation (your 4th person, or shared)

**Start this week:**
- Once Red logging and Blue training both produce *something*, even dummy versions, start on the harness that ties them: read from Red's strategy pool → trigger a Blue retrain/fine-tune round → log the result. Build this skeleton early even if it's mostly stubs — it's easier to grow than to retrofit
- Set up your evaluation tracking now, not later: containment reward, MTTD/MTTR, false-positive rate, plotted against retrain-round number. If you don't instrument this from week one, you'll be scrambling to reconstruct it before your final demo

**Research / read:**
- Elastic Weight Consolidation (EWC) — the original paper is Kirkpatrick et al., *"Overcoming Catastrophic Forgetting in Neural Networks"* (2017). You don't need the full derivation, just the core mechanic: penalize changes to weights that mattered for old tasks, so fine-tuning on a new attack pattern doesn't erase what Blue already learned
- Canary deployment patterns (this is a software-engineering concept, not RL-specific — any "canary release" writeup applies): deploy the new policy to one zone, compare statistically against the old one, promote or roll back
- Drift detection basics: the simplest version for your scope is just "policy confidence drops below a threshold" or "a human overrides the agent" — you don't need a from-scratch drift-detection algorithm here, keep it lightweight per your own report's scoping decision

**Where to look:** Kirkpatrick et al. EWC paper (widely available, search the title above), any standard MLOps writeup on canary rollouts (this concept is common outside cybersecurity too, e.g. in general ML deployment blogs), your own project report's Section 3.4.5 for the exact mechanism you've already scoped.

---

## Quick reference: who to loop in when

| If you hit... | Talk to |
|---|---|
| "I don't know what fields are in the observation dict" | Environment person |
| "Red's attack log format doesn't match what I need" | Red + Loop person |
| "Blue's reward isn't reflecting containment properly" | Blue + Environment person (reward often lives in env config) |
| "We need a number for the final report and don't have it" | Loop/eval person — raise this the moment it happens, not at the deadline |

---

## The single biggest time-saver

Don't start any of this from a blank file. Get CAGE4 running as a team in week 0, then diverge from that working baseline. Every hour spent building your own network simulator or your own MAPPO implementation from scratch is an hour not spent on your actual novelty — the closed retraining loop — which is the one thing a panel will actually be asking about.
