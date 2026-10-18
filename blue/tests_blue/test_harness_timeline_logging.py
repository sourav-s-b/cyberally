"""Inactive-agent diagnostics must not fabricate a successful observation."""
from types import SimpleNamespace
from blue.training.harness_decision_timeline import diagnostic_actions

def test_inactive_agent_without_observation_is_recorded_as_unavailable():
    action=SimpleNamespace(hostname='host',ip_address=None)
    controller=SimpleNamespace(action={'red_agent_0':[action]},
        observation={'red_agent_0':SimpleNamespace(observations=[])},
        state=SimpleNamespace(ip_addresses={}))
    rows=diagnostic_actions(SimpleNamespace(env=controller))
    assert len(rows)==1 and rows[0]['host']=='host'
    assert rows[0]['success']=='None'

def test_visible_action_diagnostic_uses_ip_resolution_and_actual_success():
    action=SimpleNamespace(ip_address='10.0.0.1')
    controller=SimpleNamespace(action={'green_agent_0':[action],'blue_agent_0':[action]},
        observation={'green_agent_0':SimpleNamespace(observations=[SimpleNamespace(data={'success':False})])},
        state=SimpleNamespace(ip_addresses={'10.0.0.1':'host'}))
    rows=diagnostic_actions(SimpleNamespace(env=controller))
    assert len(rows)==1 and rows[0]['host']=='host' and rows[0]['success']=='False'
