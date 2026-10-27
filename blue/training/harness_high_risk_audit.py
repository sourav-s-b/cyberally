"""Trace high-risk sweep opportunities in existing v2 development episodes.

This is a read-only mechanism audit. It never supplies diagnostic truth to Blue,
changes a sampled action, trains a model, or estimates a return advantage.
Each complete replay must reproduce its frozen native return and request hash.
"""
from __future__ import annotations

import copy
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from blue.core.baselines import decode_index
from blue.core.wrapper import BLUE_AGENTS, CC4MARLEnv
from blue.harness.redesign import FEATURE_NAMES
from blue.harness.scoring import sha256
from blue.training import harness_redesign as hr
from blue.training import mappo_guide as mg
from blue.training.harness_eval import penalty_events
from blue.training.harness_rl import check_seeds
from blue.training.harness_update_audit import ROOT, SCORER
from blue.training.risk_data import true_compromised

EPISODES = (8233, 8236, 8241, 8229)
MODELS = ('reference_s0', 'reference_s1')
RISK_THRESHOLD = 0.8
OUT = ROOT / 'high_risk_audit'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def visible(env, agent, host):
    tracker = env.trackers[agent]
    view = env.views[agent][host]
    return {
        'belief': tracker.state[host],
        'last_analysis': tracker.last_analysis[host],
        'last_remediation': tracker.last_remediation[host],
        'last_result': tracker.last_result[host],
        'observed_at': copy.deepcopy(env.observed_at[agent][host]),
        'observed_counts': {name: len(view.get(name) or [])
                            for name in ('Files', 'Processes', 'Connections')},
    }


class Capture:
    """Observe the same distribution, then delegate the sole RNG draw."""

    def __init__(self, hook):
        self.hook = hook
        self.decisions = []

    def reset(self):
        self.hook.reset()
        self.decisions = []

    def observe(self, env, agent):
        self.hook.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        with torch.no_grad():
            features, base, mask, teacher_index, probs = self.hook.distribution(
                env, agent, cands, scored)
            residual = self.hook.bonus * torch.tanh(
                self.hook.actor(features, base, mask))
        risk_col = FEATURE_NAMES.index('compromise_risk')
        risk = [float(features[i, risk_col]) for i in range(len(cands))]
        high = []
        for i, host in enumerate(cands):
            if risk[i] >= RISK_THRESHOLD:
                high.append({
                    'host': host, 'risk': risk[i],
                    'base_score': float(base[i]),
                    'learned_residual': float(residual[i]),
                    'sample_probability': float(probs[i]),
                    'blue_visible': visible(env, agent, host),
                })
        chosen = self.hook(env, agent, cands, scored)
        index = cands.index(chosen)
        self.decisions.append({
            'tick': int(env._tick), 'agent': agent,
            'teacher': cands[teacher_index], 'chosen': chosen,
            'teacher_risk': risk[teacher_index], 'chosen_risk': risk[index],
            'teacher_probability': float(probs[teacher_index]),
            'chosen_probability': float(probs[index]),
            'candidate_count': len(cands),
            'high_risk': high,
        })
        return chosen


def summarize(decisions, requests, completions, transitions, penalties):
    opportunities = [d for d in decisions if d['high_risk']]
    candidate_events = sum(len(d['high_risk']) for d in opportunities)
    chosen_high = sum(any(x['host'] == d['chosen'] for x in d['high_risk'])
                      for d in opportunities)
    teacher_high = sum(any(x['host'] == d['teacher'] for x in d['high_risk'])
                       for d in opportunities)
    first = {}
    for d in opportunities:
        for x in d['high_risk']:
            key = (d['agent'], x['host'])
            if key not in first:
                first[key] = (d, x)
    host_rows = []
    for (agent, host), (decision, x) in sorted(first.items()):
        tick = decision['tick']
        later = lambda rows, pred: next((r for r in rows if r['tick'] >= tick and pred(r)), None)
        next_chosen = later(decisions, lambda r: r['agent'] == agent and r['chosen'] == host)
        next_analysis = later(requests, lambda r: r['agent'] == agent and r['host'] == host
                              and r['action'] == 'Analyse')
        analysis_done = later(completions, lambda r: r['agent'] == agent and r['host'] == host
                              and r['action'] == 'Analyse')
        confirmation = later(transitions, lambda r: r['agent'] == agent and r['host'] == host
                              and r['to'] == 'CONFIRMED')
        repair = later(requests, lambda r: r['agent'] == agent and r['host'] == host
                       and r['action'] in ('Remove', 'Restore'))
        repair_done = later(completions, lambda r: r['agent'] == agent and r['host'] == host
                            and r['action'] in ('Remove', 'Restore'))
        host_rows.append({
            'agent': agent, 'host': host, 'first_high_risk_tick': tick,
            'first_risk': x['risk'], 'belief_at_first_opportunity': x['blue_visible']['belief'],
            'teacher_at_first_opportunity': decision['teacher'],
            'chosen_at_first_opportunity': decision['chosen'],
            'sample_probability_at_first_opportunity': x['sample_probability'],
            'first_later_chosen_tick': next_chosen['tick'] if next_chosen else None,
            'first_later_analysis_request_tick': next_analysis['tick'] if next_analysis else None,
            'first_later_analysis_completion_tick': analysis_done['tick'] if analysis_done else None,
            'first_later_analysis_result': analysis_done['result'] if analysis_done else None,
            'first_later_confirmation_tick': confirmation['tick'] if confirmation else None,
            'first_later_repair_request_tick': repair['tick'] if repair else None,
            'first_later_repair_completion_tick': repair_done['tick'] if repair_done else None,
            'penalty_after_first_opportunity_for_host': sum(
                p['penalty'] for p in penalties if p['tick'] >= tick and p.get('host') == host),
        })
    return {
        'sweep_decisions': len(decisions),
        'decisions_with_high_risk_candidate': len(opportunities),
        'high_risk_candidate_appearances': candidate_events,
        'high_risk_chosen_decisions': chosen_high,
        'teacher_high_risk_decisions': teacher_high,
        'high_risk_distinct_agent_hosts': len(first),
        'host_timelines': host_rows,
    }


def replay(model_name, episode, frozen):
    path = ROOT / model_name
    actor_hash = sha256(path / 'actor.th')
    policy, manifest = hr.load(path, SCORER, episode * 997)
    if manifest['config']['ml_inputs'] != 'risk':
        raise AssertionError('audit requires the real risk input')
    capture = Capture(policy.hook)
    policy.hook = capture
    env = CC4MARLEnv(seed=episode, steps=400, **mg.ENV_KW)
    env.reset(seed=episode)
    policy.reset()
    tag = f'{model_name}_{episode}_0'
    requests, completions, transitions, penalties = [], [], [], []
    total = 0.0
    try:
        with (OUT / f'{tag}.blue.jsonl').open('x') as blue_log, \
             (OUT / f'{tag}.privileged.jsonl').open('x') as truth_log:
            for _ in range(400):
                tick = int(env._tick)
                pending = dict(env._awaiting)
                beliefs = {a: dict(env.trackers[a].state) for a in BLUE_AGENTS}
                actions, agents = {}, {}
                for agent in BLUE_AGENTS:
                    action = int(policy.select(env, agent))
                    actions[agent] = action
                    name, host = decode_index(env, agent, action)
                    if (7809 <= episode <= 8200):
                        raise AssertionError('reserved final seed')
                    row = {'tick': tick, 'agent': agent, 'action': name, 'host': host,
                           'branch': policy.last_request['branch'],
                           'busy_before': agent in pending,
                           'pending_before': pending.get(agent)}
                    requests.append(row)
                    agents[agent] = {**row, 'blue_visible': visible(env, agent, host) if host else None}
                _, reward, terminated, truncated, _ = env.step(actions)
                reward = float(reward[0]); total += reward
                events = penalty_events(env)
                if not np.isclose(sum(e['penalty'] for e in events), reward):
                    raise AssertionError('native reward accounting mismatch')
                for event in events:
                    penalties.append({'tick': tick, **event})
                for agent in BLUE_AGENTS:
                    tracker = env.trackers[agent]
                    done = pending.get(agent) if agent not in env._awaiting else None
                    agents[agent]['pending_after'] = env._awaiting.get(agent)
                    agents[agent]['completion'] = None
                    if done:
                        host, name = done
                        result = {'tick': tick, 'agent': agent, 'action': name,
                                  'host': host, 'result': tracker.last_result.get(host),
                                  'belief_after': tracker.state.get(host),
                                  'last_analysis': tracker.last_analysis.get(host),
                                  'last_remediation': tracker.last_remediation.get(host)}
                        completions.append(result)
                        agents[agent]['completion'] = result
                    for host, state in tracker.state.items():
                        old = beliefs[agent].get(host)
                        if old != state:
                            transitions.append({'tick': tick, 'agent': agent, 'host': host,
                                                'from': old, 'to': state})
                blue_log.write(json.dumps({'tick': tick, 'agents': agents,
                                           'native_reward': reward}, default=str) + '\n')
                truth_log.write(json.dumps({
                    'tick': tick, 'red_session_labels_after': sorted(true_compromised(env)),
                    'penalty_events': events}, default=str) + '\n')
                if terminated or truncated:
                    break
    finally:
        env.close()
    digest = policy.request_digest.result()
    expected = frozen[tag]
    if total != expected['return'] or digest != expected['requests']:
        raise AssertionError(f'{tag}: instrumented replay changed native return or requested actions')
    if sha256(path / 'actor.th') != actor_hash:
        raise AssertionError('checkpoint changed')
    summary = summarize(capture.decisions, requests, completions, transitions, penalties)
    result = {'model': model_name, 'episode': episode, 'return': total,
              'policy_rng': episode * 997, 'actor_sha256': actor_hash,
              'requested_action_digest': digest, 'frozen_replay_validated': True,
              'all_native_rewards_accounted_for': True, **summary}
    save(OUT / f'{tag}.decisions.json', capture.decisions)
    save(OUT / f'{tag}.summary.json', result)
    print(json.dumps({k: result[k] for k in (
        'model', 'episode', 'return', 'sweep_decisions',
        'decisions_with_high_risk_candidate', 'high_risk_chosen_decisions',
        'high_risk_distinct_agent_hosts')}), flush=True)
    return result


def main():
    torch.set_num_threads(1)
    check_seeds(EPISODES)
    if any(7809 <= seed <= 8200 for seed in EPISODES):
        raise AssertionError('reserved final seed')
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f'preserve existing audit output: {OUT}')
    frozen = json.loads((ROOT / 'cells.json').read_text())
    config = {
        'scope': 'reused development episodes; diagnostic mechanism only',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runner_sha256': sha256(__file__),
        'scorer_sha256': sha256(SCORER),
        'actor_sha256': {m: sha256(ROOT / m / 'actor.th') for m in MODELS},
        'models': list(MODELS), 'episodes': list(EPISODES),
        'steps': 400, 'policy_rng': 'episode * 997',
        'high_risk_threshold': RISK_THRESHOLD,
        'claim': 'observed opportunities and consequences; no causal or held-out performance claim',
    }
    save(OUT / 'frozen_config.json', config)
    cells = {}
    for model in MODELS:
        for episode in EPISODES:
            key = f'{model}_{episode}_0'
            cells[key] = replay(model, episode, frozen)
            save(OUT / 'progress.json', {'config': config, 'cells': cells})
    save(OUT / 'report.json', {'config': config, 'cells': cells})


if __name__ == '__main__':
    main()
