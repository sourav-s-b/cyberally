"""Schema parity, provenance, freshness and real Blue-view coverage."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from ipaddress import IPv4Address, IPv4Network
import json

import numpy as np
import pytest

from blue.core.obs_features import host_to_vector
from blue.core.telemetry import (AuthEvent, ConnEvent, FileEvent, ProcEvent,
                            host_to_records, record_from_dict, record_to_dict,
                            records_to_vector, telemetry_block, TELEMETRY_VERSION)


def fixture_view():
    return {
        "Interface": [{"Subnet": IPv4Network("10.0.0.0/24")}],
        "Processes": [{"PID": 12, "PPID": 3, "process_name": "ssh",
                       "Connections": [
                           {"remote_address": IPv4Address("10.0.0.4"), "remote_port": 22},
                           {"remote_address": IPv4Address("10.1.0.4"), "remote_port": 80}]}],
        "Files": [{"File Name": "cmd.sh", "Path": "/tmp", "Known File": "UNKNOWN",
                   "Density": 0.9}, {"Known File": "KNOWN", "Density": None}],
        "Sessions": [{"username": "root", "Type": "UNKNOWN", "agent": "blue_agent_0"},
                     {"username": "user", "Type": "shell", "agent": "other_agent"}]}


def test_field_mapping_json_roundtrip_and_feature_parity():
    view = fixture_view()
    original = deepcopy(view)
    rows = host_to_records("actual-host", view, tick=9,
                           observed_at={"Processes": 2, "Files": 8, "Sessions": 0})
    assert len(rows) == 7
    proc = next(r for r in rows if isinstance(r, ProcEvent))
    assert (proc.ts, proc.pid, proc.ppid, proc.comm) == (2, 12, 3, "ssh")
    assert [(r.external, r.external_known) for r in rows if isinstance(r, ConnEvent)] == [
        (False, True), (True, True)]
    file = next(r for r in rows if isinstance(r, FileEvent))
    assert (file.path, file.name, file.known, file.density, file.ts) == (
        "/tmp", "cmd.sh", False, 0.9, 8)
    decoded = [record_from_dict(row) for row in json.loads(
        json.dumps([record_to_dict(r) for r in rows], allow_nan=False))]
    assert decoded == rows
    assert records_to_vector(decoded, "blue_agent_0") == host_to_vector(view, "blue_agent_0")
    assert all(r.origin == "simulated" and r.host == "actual-host" for r in rows)
    assert view == original
    with pytest.raises(FrozenInstanceError):
        proc.origin = "real"


@pytest.mark.parametrize("view", [{}, {"Processes": None, "Files": [], "Sessions": None},
    {"Processes": [{"Connections": [{}, {"remote_address": "bad"}]}],
     "Files": [{}, {"Density": -0.2}], "Sessions": [{}, {"agent": None}]},
    {"Sessions": [{"Type": "UNKNOWN", "username": "root"}]}])
def test_sparse_views_preserve_legacy_defaults(view):
    rows = host_to_records("host", view, tick=0)
    assert records_to_vector(rows, "blue_agent_0") == host_to_vector(view, "blue_agent_0")
    assert all(not r.external_known for r in rows if isinstance(r, ConnEvent))
    assert [record_from_dict(record_to_dict(r)) for r in rows] == rows


@pytest.mark.parametrize("change", [{"origin": "real"}, {"origin": None},
    {"ts": "2026-10-01T00:00:00Z"}, {"ts": -1}, {"ts": True},
    {"pid": -2}, {"type": "Unexpected"}, {"extra": "no"}])
def test_invalid_json_is_rejected(change):
    row = record_to_dict(ProcEvent(0, "host", None, None, None))
    row.update(change)
    with pytest.raises((ValueError, TypeError)):
        record_from_dict(row)


def test_missing_origin_rejected():
    row = record_to_dict(AuthEvent(0, "host", None, None, None))
    del row["origin"]
    with pytest.raises(ValueError, match="origin"):
        record_from_dict(row)


def test_stale_empty_missing_and_future_fields():
    views = {"host": {"Processes": [], "Files": [{"Density": 0.1}]}}
    observed = {"host": {"Processes": 8, "Files": 2}}
    first = telemetry_block(views, observed, tick=8)
    second = telemetry_block(views, observed, tick=9)
    assert first == second  # retained rows are not retimestamped
    assert first["telemetry"][0]["ts"] == 2
    assert first["field_observed_at"]["host"] == {"Processes": 8, "Files": 2}
    assert "Sessions" not in first["field_observed_at"]["host"]
    with pytest.raises(KeyError):
        host_to_records("host", views["host"], tick=9, observed_at={})
    with pytest.raises(ValueError, match="future"):
        host_to_records("host", views["host"], tick=1, observed_at=observed["host"])
    with pytest.raises(ValueError, match="future"):
        telemetry_block({"host": {"Files": []}}, {"host": {"Files": 10}}, tick=9)
    with pytest.raises(ValueError, match="simulator tick"):
        telemetry_block({"host": {"Files": []}}, {"host": {"Files": "wall-clock"}}, tick=9)


def test_no_mixed_host_feature_aggregation():
    with pytest.raises(ValueError, match="single host"):
        records_to_vector([ProcEvent(0, "a", None, None, None),
                           ProcEvent(0, "b", None, None, None)], "blue_agent_0")


def test_live_wrapper_all_agents_reset_passive_and_analysis(monkeypatch):
    import blue.core.wrapper as wrapper
    from CybORG.Agents import SleepAgent
    monkeypatch.setattr(wrapper, "DiscoveryFSRed", SleepAgent)
    monkeypatch.setattr(wrapper, "EnterpriseGreenAgent", SleepAgent)
    env = wrapper.CC4MARLEnv(steps=30)
    env.reset(seed=7629)

    def check():
        for i, agent in enumerate(wrapper.BLUE_AGENTS):
            before = env.get_obs_agent(i).copy()
            mask = env.get_avail_agent_actions(i).copy()
            block = env.get_telemetry(i)
            assert block["telemetry_version"] == TELEMETRY_VERSION
            by_host = {h: [] for h in env.hostnames[agent]}
            for row in block["telemetry"]:
                parsed = record_from_dict(row)
                by_host[parsed.host].append(parsed)
                assert parsed.ts <= env._tick
            assert set(block["field_observed_at"]) == set(env.hostnames[agent])
            for host, rows in by_host.items():
                assert records_to_vector(rows, agent) == host_to_vector(
                    env.views[agent][host], agent, env.subnets[agent][host])
            np.testing.assert_array_equal(before, env.get_obs_agent(i))
            np.testing.assert_array_equal(mask, env.get_avail_agent_actions(i))

    check()
    agent = "blue_agent_0"
    host = env.hostnames[agent][1]
    env.env.state.hosts[host].events.process_creation.append(
        {"pid": 98765, "process_name": "observed-test-process"})
    env.step({agent: 2})  # Analyse first host; passive event is on another host
    check()
    first = [r for r in env.get_telemetry(0)["telemetry"]
             if r["type"] == "ProcEvent" and r["pid"] == 98765]
    assert first and first[0]["ts"] == 1
    env.step([0] * 5)
    check()
    assert first == [r for r in env.get_telemetry(0)["telemetry"]
                     if r["type"] == "ProcEvent" and r["pid"] == 98765]
    assert env.get_telemetry(0)["field_observed_at"][env.hostnames[agent][0]]["Files"] == 2
    env.reset(seed=7630)
    check()
    assert all(r["ts"] == 0 for r in env.get_telemetry(0)["telemetry"])
    env.close()


def test_real_collection_manifest_and_no_overwrite(tmp_path):
    from blue.core.collect_telemetry import collect
    import hashlib
    output = tmp_path / "collection"
    manifest = collect(output, [8123], steps=4, policy_name="round-robin")
    rows = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
    assert len(rows) == 20 == manifest["records"]  # reset plus 3 native ticks, 5 agents
    assert manifest["seeds"] == [8123] and manifest["origin"] == "simulated"
    assert manifest["telemetry_sha256"] == hashlib.sha256(
        (output / "telemetry.jsonl").read_bytes()).hexdigest()
    assert rows[0]["telemetry_time_unit"] == "sim_tick"
    assert all(record_from_dict(r).origin == "simulated" for row in rows for r in row["telemetry"])
    assert all("truth" not in row and "labels" not in row for row in rows)
    with pytest.raises(FileExistsError):
        collect(output, [8123], steps=4)
