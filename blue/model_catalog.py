"""Discover locally saved CAGE4 ``agent.th`` actors for the dashboard.

Dashboard model weights are bundled in ``blue/assets/dashboard-models.zip``
and unpacked into ignored results paths when the server starts. Metadata is
read from the tensor state dict, never by executing a checkpoint as Python.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "blue" / "results" / "models"
AGENT_ID_WIDTH = 5
SLOTS = 51
BASE_FEATURES = 10
TEMPORAL_GROUPS = ("ages", "belief", "freshness", "mission")
GROUP_WIDTHS = {"ages": 2, "belief": 5, "freshness": 2, "mission": 1}

FAMILY_INFO = {
    "mappo": ("MAPPO", "Exploratory online PPO checkpoint; results vary by run and several fine-tunes regressed."),
    "bc": ("Behavior cloning", "Supervised policy imitation; attention distillation reached teacher parity on one held-out screen, with no confirmed gain."),
    "iql": ("IQL / offline AWR", "Offline-learning experiments; held-out gates failed or remained inconclusive. Not an established improvement."),
    "rvs": ("RvS", "Return-conditioned behavior cloning; held-out result was near teacher parity, not a confirmed improvement."),
    "setactor": ("SetActor", "Permutation-equivariant SetActor experiments; the gate-ablation arm is distinct from the v2 sensitivity pilot."),
    "harness": ("Main32 harness PPO", "Residual PPO over the 92-feature investigation harness; risk/novelty evidence is descriptive, not a confirmed gain."),
    "hybrid": ("SetActor / attention hybrid", "Fixed-blend and learned-gate routers over SetActor and candidate attention; descriptive screens only."),
}

RESULT_ROOT = ROOT / "blue" / "results"
SHORTLIST_CHECKPOINTS = {
    "bc_fact_attn1_20261002/0",
    "iql_14a05/0",
    "rvs_14/0",
}

SETACTOR_ARTIFACTS = (
    {
        "id": "setactor:gpt_harness/v2_sensitivity/reference_s0",
        "run_label": "SetActor · v2 sensitivity · reference · seed 0",
        "run_dir": "gpt_harness/v2_sensitivity/reference_s0",
        "scorer": "gpt_harness/legacy_candidate/scorer.pkl",
        "status": (
            "Historical pilot only: this is the SetActor branch, not the later "
            "fixed 0.5 SetActor/attention blend. Mean return gain over the "
            "zero-residual control was +0.25 on four development episodes "
            "(training seed 0); no confirmed improvement."
        ),
    },
)

HARNESS_RESIDUAL_ARTIFACTS = (
    {
        "id": "harness:gpt_harness/main32_status/s0/pilot/both_s0",
        "run_label": "Main32 · risk + novelty PPO · seed 0",
        "run_dir": "gpt_harness/main32_status/s0/pilot/both_s0",
        "scorer": "gpt_harness/legacy_candidate/scorer.pkl",
        "status": (
            "Main32 risk + novelty PPO. On reused development episodes it "
            "averaged −74.71 versus Lancer −79.91, but the paired training-seed "
            "interval includes zero. Not a confirmed improvement."
        ),
    },
)

ARCHITECTURE_ARTIFACTS = (
    {
        "arch": "fixed",
        "family": "hybrid",
        "label": "Fixed 0.5 SetActor / attention blend",
        "status": (
            "Fixed 0.5 SetActor/attention blend from the gate-utility ablation. "
            "The original artifact was missing; this is a local retrain from the "
            "frozen gate-ablation schedule. Descriptive two-training-seed screen, "
            "not a robust improvement claim."
        ),
    },
    {
        "arch": "hybrid",
        "family": "hybrid",
        "label": "Learned SetActor / attention gate",
        "status": (
            "Learned state-conditioned SetActor/attention gate from the gate-utility "
            "ablation. The original artifact was missing; this is a local retrain "
            "from the frozen schedule. The gate behaved almost like a "
            "training-seed-dependent constant, so situation-specific routing was "
            "not demonstrated."
        ),
    },
    {
        "arch": "set",
        "family": "setactor",
        "label": "SetActor · gate-ablation arm",
        "status": (
            "SetActor arm from the October 25 gate-utility ablation. The original "
            "artifact was missing; this is a local retrain from the frozen schedule. "
            "It is not the v2 sensitivity reference pilot and not the fixed 0.5 blend."
        ),
    },
)


def _family(name: str) -> str:
    return next((key for key in FAMILY_INFO if name.startswith(key)), "unknown")


def _friendly_run_name(run_name: str, family_title: str) -> str:
    """Turn an internal run folder into a short, readable label.

    ``mappo_cc4_seed7_20261004T104617Z`` becomes
    ``MAPPO · CAGE4 · seed 7 · 2026-10-04`` so the picker never shows raw
    directory names or requires the user to know checkpoint paths.
    """
    bits = []
    for token in run_name.split("_")[1:]:
        lowered = token.lower()
        if lowered in ("cc4", "cage4"):
            bits.append("CAGE4")
        elif lowered.startswith("seed") and token[4:].isdigit():
            bits.append(f"seed {token[4:]}")
        elif len(token) >= 8 and token[:8].isdigit():
            date = token[:8]
            bits.append(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
        elif token:
            bits.append(token.replace("-", " "))
    if not bits:
        bits.append(run_name)
    return f"{family_title} · " + " · ".join(bits)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _setactor_items(torch) -> list[dict]:
    items = []
    for spec in SETACTOR_ARTIFACTS:
        run_dir = RESULT_ROOT / spec["run_dir"]
        actor_path = run_dir / "actor.th"
        manifest_path = run_dir / "manifest.json"
        scorer_path = RESULT_ROOT / spec["scorer"]
        if not actor_path.exists():
            continue
        actor_sha256 = _file_sha256(actor_path)
        try:
            manifest = json.loads(manifest_path.read_text())
            state = torch.load(actor_path, map_location="cpu",
                               weights_only=True)
            encoded = state.get("encoder.0.weight")
            body = state.get("body.0.weight")
            head = state.get("head.weight")
            if encoded is None or body is None or head is None:
                raise ValueError("missing SetActor weights")
            hidden = int(encoded.shape[0])
            feature_width = len(manifest.get("features", [])) + 1
            if encoded.shape != (hidden, feature_width):
                raise ValueError("SetActor feature width does not match manifest")
            if body.shape != (hidden, 3 * hidden) or head.shape != (1, hidden):
                raise ValueError("unrecognized SetActor state-dict layout")
            scorer_sha256 = _file_sha256(scorer_path)
            expected_actor = manifest.get("actor_sha256")
            expected_scorer = manifest.get("scorer_sha256")
            verified = (actor_sha256 == expected_actor
                        and scorer_sha256 == expected_scorer)
            config = manifest.get("config", {})
            env_kw = manifest.get("env_kw", {})
            compatible = verified and manifest.get("status") == "complete"
            status = spec["status"] if compatible else (
                "Artifact hash verification failed; this SetActor pilot is "
                "not loadable as a verified fixed option.")
            items.append({
                "id": spec["id"],
                "label": spec["run_label"],
                "run_label": spec["run_label"],
                "family": "setactor",
                "family_label": FAMILY_INFO["setactor"][0],
                "kind": "set_actor",
                "shortlist": False,
                "architecture": "Permutation-equivariant SetActor + mean/max context",
                "agent_type": "set_actor",
                "hidden_dim": hidden,
                "temporal_features": list(env_kw.get("temporal_features", [])),
                "include_root_session": bool(
                    env_kw.get("include_root_session", True)),
                "return_conditioned": False,
                "attention_layers": 0,
                "compatible": compatible,
                "status": status,
                "bytes": actor_path.stat().st_size,
                "actor_path": actor_path.relative_to(ROOT).as_posix(),
                "scorer_path": scorer_path.relative_to(ROOT).as_posix(),
                "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                "actor_sha256": actor_sha256,
                "scorer_sha256": scorer_sha256,
                "artifact_verified": verified,
                "manifest_status": manifest.get("status"),
                "manifest_version": manifest.get("version"),
                "source_commit": manifest.get("source_commit"),
                "policy": {
                    "bonus": config.get("bonus"),
                    "ml_inputs": config.get("ml_inputs"),
                    "temperature": config.get("temp"),
                    "explore": config.get("explore"),
                    "max_age": config.get("max_age"),
                },
            })
        except Exception as exc:
            items.append({
                "id": spec["id"],
                "label": spec["run_label"],
                "run_label": spec["run_label"],
                "family": "setactor",
                "family_label": FAMILY_INFO["setactor"][0],
                "kind": "set_actor",
                "shortlist": False,
                "architecture": "SetActor (unverified)",
                "agent_type": "set_actor",
                "hidden_dim": 64,
                "temporal_features": [],
                "include_root_session": True,
                "return_conditioned": False,
                "attention_layers": 0,
                "compatible": False,
                "status": str(exc),
                "bytes": actor_path.stat().st_size,
                "actor_path": actor_path.relative_to(ROOT).as_posix(),
                "scorer_path": scorer_path.relative_to(ROOT).as_posix(),
                "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                "actor_sha256": actor_sha256,
                "scorer_sha256": None,
                "artifact_verified": False,
                "manifest_status": None,
                "manifest_version": None,
                "source_commit": None,
                "policy": {},
            })
    return items


def _harness_residual_items(torch) -> list[dict]:
    items = []
    for spec in HARNESS_RESIDUAL_ARTIFACTS:
        run_dir = RESULT_ROOT / spec["run_dir"]
        actor_path = run_dir / "actor.th"
        manifest_path = run_dir / "manifest.json"
        scorer_path = RESULT_ROOT / spec["scorer"]
        if not actor_path.exists() or not manifest_path.exists():
            continue
        actor_sha256 = _file_sha256(actor_path)
        try:
            manifest = json.loads(manifest_path.read_text())
            state = torch.load(actor_path, map_location="cpu",
                               weights_only=True)
            body = state.get("body.0.weight")
            head = state.get("head.weight")
            if body is None or head is None:
                raise ValueError("missing main32 residual weights")
            hidden = int(body.shape[0])
            feature_width = len(manifest.get("features", []))
            if feature_width == 0 or body.shape != (hidden, feature_width):
                raise ValueError("main32 feature width does not match manifest")
            if head.shape != (1, hidden):
                raise ValueError("unrecognized main32 state-dict layout")
            scorer_sha256 = _file_sha256(scorer_path)
            verified = (
                actor_sha256 == manifest.get("actor_sha256")
                and scorer_sha256 == manifest.get("scorer_sha256")
                and manifest.get("status") == "complete"
            )
            config = manifest.get("config", {})
            env_kw = manifest.get("env_kw", {})
            items.append({
                "id": spec["id"],
                "label": spec["run_label"],
                "run_label": spec["run_label"],
                "family": "harness",
                "family_label": FAMILY_INFO["harness"][0],
                "kind": "harness_residual",
                "shortlist": True,
                "architecture": "Main32 residual PPO + mean/max candidate context",
                "agent_type": "harness_residual",
                "hidden_dim": hidden,
                "training_seed": config.get("seed", 0),
                "temporal_features": list(env_kw.get("temporal_features", [])),
                "include_root_session": bool(
                    env_kw.get("include_root_session", True)),
                "return_conditioned": False,
                "attention_layers": 0,
                "compatible": verified,
                "status": spec["status"] if verified else (
                    "Artifact hash or completion check failed; this main32 "
                    "checkpoint is not loadable as a verified option."),
                "bytes": actor_path.stat().st_size,
                "run_dir": run_dir.relative_to(ROOT).as_posix(),
                "actor_path": actor_path.relative_to(ROOT).as_posix(),
                "scorer_path": scorer_path.relative_to(ROOT).as_posix(),
                "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                "actor_sha256": actor_sha256,
                "scorer_sha256": scorer_sha256,
                "artifact_verified": verified,
                "manifest_status": manifest.get("status"),
                "manifest_version": manifest.get("version"),
                "source_commit": manifest.get("source_commit"),
                "policy": {
                    "bonus": config.get("bonus"),
                    "ml_inputs": config.get("ml_inputs"),
                    "temperature": config.get("temp"),
                    "explore": config.get("explore"),
                    "max_age": config.get("max_age"),
                },
            })
        except Exception as exc:
            items.append({
                "id": spec["id"],
                "label": spec["run_label"],
                "run_label": spec["run_label"],
                "family": "harness",
                "family_label": FAMILY_INFO["harness"][0],
                "kind": "harness_residual",
                "shortlist": True,
                "architecture": "Main32 residual PPO (unverified)",
                "agent_type": "harness_residual",
                "hidden_dim": 64,
                "training_seed": None,
                "temporal_features": [],
                "include_root_session": True,
                "return_conditioned": False,
                "attention_layers": 0,
                "compatible": False,
                "status": str(exc),
                "bytes": actor_path.stat().st_size,
                "run_dir": run_dir.relative_to(ROOT).as_posix(),
                "actor_path": actor_path.relative_to(ROOT).as_posix(),
                "scorer_path": scorer_path.relative_to(ROOT).as_posix(),
                "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                "actor_sha256": actor_sha256,
                "scorer_sha256": None,
                "artifact_verified": False,
                "manifest_status": None,
                "manifest_version": None,
                "source_commit": None,
                "policy": {},
            })
    return items


def _architecture_items(torch) -> list[dict]:
    items = []
    root = RESULT_ROOT / "architecture_gate_ablation_v1"
    scorer_path = RESULT_ROOT / "event_time_rl" / "scorer.pkl"
    for spec in ARCHITECTURE_ARTIFACTS:
        for seed in (0, 1):
            run_dir = root / f"{spec['arch']}_s{seed}"
            actor_path = run_dir / "actor.th"
            manifest_path = run_dir / "manifest.json"
            if not actor_path.exists() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "complete":
                    continue
                config = manifest.get("config", {})
                if config.get("arch") != spec["arch"]:
                    raise ValueError("gate-ablation architecture mismatch")
                if config.get("seed") != seed:
                    raise ValueError("gate-ablation seed mismatch")
                actor_sha256 = _file_sha256(actor_path)
                scorer_sha256 = _file_sha256(scorer_path)
                verified = (
                    actor_sha256 == manifest.get("actor_sha256")
                    and scorer_sha256 == manifest.get("scorer_sha256")
                )
                family = spec["family"]
                title = FAMILY_INFO[family][0]
                run_label = f"{spec['label']} · seed {seed}"
                items.append({
                    "id": f"architecture:{spec['arch']}_s{seed}",
                    "label": run_label,
                    "run_label": run_label,
                    "family": family,
                    "family_label": title,
                    "kind": "architecture",
                    "shortlist": True,
                    "arch": spec["arch"],
                    "architecture": (
                        "SetActor" if spec["arch"] == "set" else
                        "SetActor + candidate attention (fixed gate 0.5)"
                        if spec["arch"] == "fixed" else
                        "SetActor + candidate attention (learned gate)"
                    ),
                    "agent_type": "architecture",
                    "hidden_dim": int(config.get("hidden", 64)),
                    "training_seed": seed,
                    "temporal_features": ["ages", "belief"],
                    "include_root_session": True,
                    "return_conditioned": False,
                    "attention_layers": (
                        0 if spec["arch"] == "set" else 3
                    ),
                    "compatible": verified,
                    "status": spec["status"] if verified else (
                        "Artifact hash verification failed; this gate-ablation "
                        "checkpoint is not loadable as a verified option."),
                    "bytes": actor_path.stat().st_size,
                    "run_dir": run_dir.relative_to(ROOT).as_posix(),
                    "actor_path": actor_path.relative_to(ROOT).as_posix(),
                    "scorer_path": scorer_path.relative_to(ROOT).as_posix(),
                    "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                    "actor_sha256": actor_sha256,
                    "scorer_sha256": scorer_sha256,
                    "artifact_verified": verified,
                    "manifest_status": manifest.get("status"),
                    "manifest_version": manifest.get("version"),
                    "source_commit": manifest.get("source_commit"),
                    "policy": {
                        "bonus": config.get("bonus"),
                        "ml_inputs": config.get("ml_inputs"),
                        "temperature": config.get("temp"),
                        "explore": config.get("explore"),
                        "max_age": None,
                    },
                })
            except Exception:
                continue
    return items


def _temporal_layout(input_width: int, has_rtg: bool) -> tuple[list[str] | None, bool]:
    feature_width = input_width - AGENT_ID_WIDTH - int(has_rtg)
    if feature_width <= 0 or feature_width % SLOTS:
        return None, False
    per_host = feature_width // SLOTS
    added = per_host - BASE_FEATURES
    for count in range(len(TEMPORAL_GROUPS) + 1):
        for groups in combinations(TEMPORAL_GROUPS, count):
            if sum(GROUP_WIDTHS[group] for group in groups) == added:
                return list(groups), per_host >= BASE_FEATURES
    return None, False


@lru_cache(maxsize=1)
def catalog() -> list[dict]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to inspect and load saved Blue checkpoints. "
            "Start the dashboard with ./.venv-train/bin/python -m blue.dashboard_server."
        ) from exc

    items = (_setactor_items(torch) + _harness_residual_items(torch)
             + _architecture_items(torch))
    for checkpoint in sorted(MODEL_ROOT.glob("**/agent.th")):
        relative = checkpoint.parent.relative_to(MODEL_ROOT).as_posix()
        run_name = checkpoint.parent.parent.name
        family = _family(run_name)
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            first = state.get("fc1.weight")
            if first is None or first.ndim != 2:
                raise ValueError("missing fc1.weight")
            keys = set(state)
            has_rtg = first.shape[1] == 873
            if "slot_enc.weight" in keys:
                architecture = "Factorized recurrent"
                agent_type = "rnn_factorized"
            elif "fc2.weight" in keys:
                architecture = "Vanilla recurrent"
                agent_type = "rnn"
            else:
                raise ValueError("unrecognized actor state-dict layout")
            attention_layers = len({key.split(".")[2] for key in keys
                                    if key.startswith("slot_attn.layers.")
                                    and len(key.split(".")) > 3})
            if attention_layers:
                architecture += f" + {attention_layers}-layer slot attention"
            if has_rtg:
                architecture += " + return conditioning"
            groups, includes_root = _temporal_layout(first.shape[1], has_rtg)
            output_width = (state.get("fc2.weight").shape[0]
                            if "fc2.weight" in state else 155)
            compatible = groups is not None and output_width == 155
            if family == "unknown":
                compatible = False
            title, outcome = FAMILY_INFO.get(family, ("Unknown", "No recorded model family."))
            step = checkpoint.parent.name
            run_label = _friendly_run_name(run_name, title)
            label = f"{run_label} · checkpoint {step}"
            items.append({
                "id": relative,
                "label": label,
                "run_label": run_label,
                "checkpoint_step": step,
                "family": family,
                "family_label": title,
                "kind": "checkpoint",
                "shortlist": relative in SHORTLIST_CHECKPOINTS,
                "architecture": architecture,
                "agent_type": agent_type,
                "hidden_dim": int(first.shape[0]),
                "temporal_features": groups or [],
                "include_root_session": includes_root,
                "return_conditioned": has_rtg,
                "attention_layers": attention_layers,
                "compatible": compatible,
                "status": outcome,
                "bytes": checkpoint.stat().st_size,
            })
        except Exception as exc:
            step = checkpoint.parent.name
            fallback_title = FAMILY_INFO.get(family, ("Unknown", ""))[0]
            run_label = _friendly_run_name(run_name, fallback_title)
            items.append({
                "id": relative,
                "label": f"{run_label} · checkpoint {step}",
                "run_label": run_label,
                "checkpoint_step": step,
                "family": family,
                "family_label": fallback_title,
                "kind": "checkpoint",
                "shortlist": relative in SHORTLIST_CHECKPOINTS,
                "architecture": "Could not inspect checkpoint",
                "temporal_features": [],
                "include_root_session": True,
                "return_conditioned": False,
                "attention_layers": 0,
                "compatible": False,
                "status": str(exc),
                "bytes": checkpoint.stat().st_size,
            })
    return items


def get_model(model_id: str) -> dict:
    if not isinstance(model_id, str):
        raise ValueError("blue_policy must be a policy id")
    model = next((item for item in catalog() if item["id"] == model_id), None)
    if model is None:
        raise ValueError("unknown Blue checkpoint")
    if not model["compatible"]:
        raise ValueError("this checkpoint architecture or observation layout is not supported by the dashboard")
    return model

