"""Small, auditable finite-Red DAgger comparison against matched BC.

The student predicts Lancer's candidate ranking from Blue-visible features
only. Lancer scores are used as training labels and to mix the rollout policy;
they are never student inputs or logits. Remediation, verification, and the
age-80 coverage guard remain in HarnessPolicy outside the learned hook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from blue.harness.redesign import FEATURE_NAMES, EvidenceMemory, RedesignHook
from blue.harness.scoring import MLScorer, sha256
from blue.training.harness_rl import check_seeds, source_identity

MODEL_SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEEDS = tuple(range(7730, 7755))
VALIDATION_SEEDS = tuple(range(7755, 7769))
EVALUATION_SEEDS = tuple(range(7769, 7801))
BETAS = (1.0, 0.5, 0.25, 0.125, 0.0625)
CONTROLS = ("lancer", "guard", "sleep", "masked_random")
SOURCE_FILES = (
    "blue/training/harness_dagger.py",
    "blue/training/harness_rl.py",
    "blue/training/mappo_guide.py",
    "blue/harness/redesign.py",
    "blue/harness/policy.py",
    "blue/harness/features.py",
    "blue/harness/scoring.py",
    "blue/harness/guided.py",
    "blue/core/wrapper.py",
    "blue/core/baselines.py",
    "blue/policies/ordered.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(name) for name in SOURCE_FILES}


def plan(scorer_path):
    return {
        "purpose": "finite-Red behavior-learning pilot; parity target, no claim of beating Lancer",
        "source_commit": source_identity(),
        "source_files": source_hashes(),
        "scorer_sha256": sha256(scorer_path),
        "red_agent": "finite",
        "native_reward_only": True,
        "student_inputs": "Blue-visible redesigned candidate features; no Lancer score in input or logits",
        "teacher": "OrderedPolicy LancerValues(fruitless_decay=0.5), labels only on investigation-hook decisions",
        "outer_policy": "HarnessPolicy age-80 guard; scripted urgent remediation/verification unchanged",
        "model_seeds": list(MODEL_SEEDS),
        "training_episode_seeds_by_model_seed": {
            str(seed): list(TRAIN_SEEDS[5 * i:5 * i + 5])
            for i, seed in enumerate(MODEL_SEEDS)
        },
        "validation_seeds": list(VALIDATION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "episodes_per_model_per_arm": 5,
        "dagger_rounds": len(BETAS),
        "dagger_beta": list(BETAS),
        "episode_horizon": 400,
        "hidden": 64,
        "optimizer": {"name": "Adam", "lr": 0.0003, "batch_decisions": 128,
                      "max_epochs_per_aggregation": 40, "early_stop_patience": 5},
        "comparison": "DAgger versus behavior cloning; paired finite-Red evaluation against Lancer, age-80 guard, Sleep and masked-random",
        "primary_unit": "model training RNG seed; average paired evaluation difference over 32 episodes within each model seed",
        "decision_gate": "exploratory only; do not claim superiority unless the 95% interval across five model seeds is above zero",
    }


def validate_seed_plan():
    sets = [set(TRAIN_SEEDS), set(VALIDATION_SEEDS), set(EVALUATION_SEEDS)]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError("training, validation and evaluation seeds overlap")
    for seeds in (TRAIN_SEEDS, VALIDATION_SEEDS, EVALUATION_SEEDS):
        check_seeds(list(seeds))
    from blue.training.seed_audit import build_used, reserved_conflicts
    all_seeds = TRAIN_SEEDS + VALIDATION_SEEDS + EVALUATION_SEEDS
    used = build_used()
    self_path = str(Path(__file__).resolve().relative_to(Path.cwd()))
    collisions = [(s, sorted(src for src in used[s] if src != self_path))
                   for s in all_seeds if s in used
                   and any(src != self_path for src in used[s])]
    if collisions:
        raise ValueError(f"seed plan contains previously used seeds: {collisions}")
    reserved = reserved_conflicts(min(all_seeds), max(all_seeds))
    if reserved:
        raise ValueError(f"seed plan overlaps declared reservations: {reserved}")


def freeze(root, scorer_path):
    validate_seed_plan()
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"experiment directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "frozen_plan.json").write_text(json.dumps(plan(scorer_path), indent=2) + "\n")


def verify_frozen(root, scorer_path):
    frozen = json.loads((Path(root) / "frozen_plan.json").read_text())
    if frozen != json.loads(json.dumps(plan(scorer_path))):
        raise ValueError("frozen plan, source hashes, or scorer changed")
    return frozen


class DaggerActor(nn.Module):
    """Permutation-equivariant candidate scorer with no teacher-score input."""

    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh())
        self.body = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)

    def forward(self, features, mask):
        single = features.ndim == 2
        if single:
            features, mask = features[None], mask[None]
        z = self.encoder(features)
        valid = mask.bool().unsqueeze(-1)
        count = valid.sum(1).clamp(min=1)
        mean = (z * valid).sum(1) / count
        maximum = z.masked_fill(~valid, -torch.inf).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        ctx = torch.cat((z, mean[:, None].expand_as(z), maximum[:, None].expand_as(z)), -1)
        logits = self.head(self.body(ctx)).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), -torch.inf)
        return logits[0] if single else logits


class DaggerHook:
    """Collect Lancer labels and execute beta-Lancer / (1-beta)-student."""

    def __init__(self, scorer_path, actor=None, beta=1.0, seed=0,
                 collect=True, episode_meta=None):
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.scorer = MLScorer(scorer_path)
        # Reuse the accepted Blue-visible feature path, including risk-input
        # setting and evidence memory. Its actor is never called here.
        self.feature_provider = RedesignHook(
            None, self.scorer, bonus=0.25, ml_inputs="risk", temperature=0.5,
            explore=0.2, policy_seed=seed, execution="greedy-diagnostic")
        self.actor = actor
        self.beta = float(beta)
        self.rng = random.Random(seed)
        self.collect = bool(collect)
        self.episode_meta = dict(episode_meta or {})
        self.rows = []
        self.per_agent = {}
        self.decisions = 0
        self.agreement = 0
        self._episode_start = 0

    def reset(self):
        self.feature_provider.reset()
        self.per_agent = {}
        self.decisions = 0
        self.agreement = 0

    def mark_episode(self):
        self._episode_start = len(self.rows)

    def observe(self, env, agent):
        self.feature_provider.observe(env, agent)

    def __call__(self, env, agent, cands, scored):
        import torch
        from blue.policies.ordered import argmax_pick

        features = self.feature_provider.features(env, agent, cands)
        label_host = argmax_pick(scored)
        label = cands.index(label_host)
        if self.actor is None:
            logits = None
            student_pick = label_host
        else:
            with torch.no_grad():
                x = torch.from_numpy(features.astype(np.float32, copy=False))
                mask = torch.ones(len(cands), dtype=torch.bool)
                logits = self.actor(x, mask).cpu().numpy()
            student_pick = max(range(len(cands)), key=lambda i: (float(logits[i]), cands[i]))
            student_pick = cands[student_pick]
        self.decisions += 1
        self.agreement += int(student_pick == label_host)
        use_teacher = self.rng.random() < self.beta
        executed = label_host if use_teacher else student_pick
        if self.collect:
            self.rows.append({
                "features": features.astype(np.float32, copy=True),
                "label": int(label),
                "agent": agent,
                "candidates": list(cands),
                "episode_meta": dict(self.episode_meta),
            })
        self.per_agent[agent] = self.per_agent.get(agent, 0) + 1
        return executed

    def close_episode(self, rewards):
        # Native rewards are deliberately not used as imitation labels.
        return None


def pack_rows(path, rows):
    lengths = np.asarray([len(r["features"]) for r in rows], dtype=np.int32)
    feat_dim = len(FEATURE_NAMES)
    flat = (np.concatenate([r["features"] for r in rows], axis=0).astype(np.float32)
            if rows else np.zeros((0, feat_dim), dtype=np.float32))
    labels = np.asarray([r["label"] for r in rows], dtype=np.int32)
    metadata = [{k: v for k, v in r.items() if k not in ("features",)} for r in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, flat_features=flat, lengths=lengths, labels=labels,
                            metadata=np.asarray(json.dumps(metadata)))
    os.replace(tmp, path)


def unpack_rows(path):
    with np.load(path, allow_pickle=False) as data:
        flat = data["flat_features"]
        lengths = data["lengths"]
        labels = data["labels"]
        metadata = json.loads(str(data["metadata"].item()))
    rows = []
    offset = 0
    for n, label, meta in zip(lengths, labels, metadata):
        rows.append({"features": flat[offset:offset + n].copy(),
                     "label": int(label), **meta})
        offset += int(n)
    if offset != len(flat):
        raise ValueError("dataset offsets do not cover candidate features")
    return rows


def collate(rows):
    B, F = len(rows), len(FEATURE_NAMES)
    K = max(len(r["features"]) for r in rows)
    features = np.zeros((B, K, F), dtype=np.float32)
    masks = np.zeros((B, K), dtype=np.float32)
    labels = np.zeros(B, dtype=np.int64)
    for i, row in enumerate(rows):
        n = len(row["features"])
        features[i, :n] = row["features"]
        masks[i, :n] = 1.0
        labels[i] = row["label"]
        if not 0 <= row["label"] < n:
            raise ValueError("teacher label outside candidate set")
    return torch.from_numpy(features), torch.from_numpy(masks), torch.from_numpy(labels)


def validation_loss(actor, rows, batch_size=128):
    if not rows:
        raise ValueError("empty validation rows")
    actor.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x, mask, labels = collate(rows[start:start + batch_size])
            losses.append(float(nn.functional.cross_entropy(actor(x, mask), labels,
                                                            reduction="sum")))
    return sum(losses) / len(rows)


def fit(actor, train_rows, validation_rows, seed, max_epochs=40, patience=5,
        batch_size=128, lr=3e-4):
    if not train_rows or not validation_rows:
        raise ValueError("training and whole-episode validation data are required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        actor.train()
        order = rng.permutation(len(train_rows))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train_rows[i] for i in order[start:start + batch_size]]
            x, mask, labels = collate(batch)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(actor(x, mask), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite imitation loss")
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(batch)
        val_loss = validation_loss(actor, validation_rows, batch_size)
        history.append({"epoch": epoch, "train_cross_entropy": total / len(train_rows),
                        "validation_cross_entropy": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")
    actor.load_state_dict(best_state)
    actor.eval()
    return {"best_epoch": best_epoch, "best_validation_cross_entropy": best_loss,
            "epochs_run": len(history), "history": history}


def init_actor(seed, hidden=64):
    torch.manual_seed(seed)
    actor = DaggerActor(hidden)
    actor.eval()
    return actor


def collect_episode(rows, scorer_path, seed, actor=None, beta=1.0,
                    model_seed=0, round_index=0, arm="teacher"):
    from blue.harness.policy import HarnessPolicy
    from blue.training import mappo_guide as mg
    from blue.training.harness_redesign import env_kwargs

    hook = DaggerHook(
        scorer_path, actor=actor, beta=beta,
        seed=model_seed * 100_003 + seed * 17 + round_index,
        collect=True,
        episode_meta={"episode_seed": seed, "model_seed": model_seed,
                      "round": round_index, "beta": beta, "arm": arm})
    policy = HarnessPolicy(hook, max_age=80, record_requests=True)
    result = mg.run_team_episode(policy, seed, 400, hook=hook, recorder=hook,
                                 **env_kwargs("finite"))
    rows.extend(hook.rows)
    result.update(decisions=hook.decisions, agreement=hook.agreement,
                  beta=beta, model_seed=model_seed, round=round_index)
    return result


def collect_validation(root, scorer_path):
    output = Path(root) / "validation.npz"
    if output.exists():
        return unpack_rows(output)
    rows, episodes = [], []
    for seed in VALIDATION_SEEDS:
        episodes.append(collect_episode(rows, scorer_path, seed, actor=None,
                                        beta=1.0, model_seed=0,
                                        round_index=-1, arm="validation_lancer"))
        print(f"validation seed={seed} decisions={episodes[-1]['decisions']}", flush=True)
    pack_rows(output, rows)
    (Path(root) / "validation_report.json").write_text(json.dumps({
        "seeds": list(VALIDATION_SEEDS), "episodes": episodes,
        "decision_rows": len(rows), "dataset_sha256": digest(output),
    }, indent=2) + "\n")
    return rows


def save_model(model_dir, actor, manifest):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = model_dir / "actor.tmp"
    torch.save(actor.state_dict(), tmp)
    os.replace(tmp, model_dir / "actor.pt")
    manifest["actor_sha256"] = digest(model_dir / "actor.pt")
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_model(model_dir, expected_source, scorer_hash):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"model is not complete: {model_dir}")
    if (manifest.get("source_files") != expected_source
            or manifest.get("scorer_sha256") != scorer_hash
            or digest(model_dir / "actor.pt") != manifest.get("actor_sha256")):
        raise ValueError(f"model artifact contract mismatch: {model_dir}")
    actor = DaggerActor(manifest["hidden"])
    actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location="cpu", weights_only=True))
    actor.eval()
    return actor, manifest


def train_bc(root, scorer_path, validation_rows):
    root = Path(root)
    frozen = verify_frozen(root, scorer_path)
    source = frozen["source_files"]
    for model_seed in MODEL_SEEDS:
        model_dir = root / "models" / "bc" / f"s{model_seed}"
        if (model_dir / "manifest.json").exists():
            load_model(model_dir, source, frozen["scorer_sha256"])
            continue
        episode_seeds = frozen["training_episode_seeds_by_model_seed"][str(model_seed)]
        dataset = []
        episodes = []
        for j, seed in enumerate(episode_seeds):
            episodes.append(collect_episode(dataset, scorer_path, seed, actor=None,
                                            beta=1.0, model_seed=model_seed,
                                            round_index=j, arm="bc"))
            print(f"BC s{model_seed} seed={seed} decisions={episodes[-1]['decisions']}", flush=True)
        data_path = model_dir / "dataset.npz"
        pack_rows(data_path, dataset)
        actor = init_actor(19_000 + model_seed, frozen["hidden"])
        metrics = fit(
            actor, dataset, validation_rows, 20_000 + model_seed,
            batch_size=frozen["optimizer"]["batch_decisions"],
            max_epochs=frozen["optimizer"]["max_epochs_per_aggregation"],
            patience=frozen["optimizer"]["early_stop_patience"],
            lr=frozen["optimizer"]["lr"])
        manifest = {
            "arm": "bc", "status": "complete", "model_seed": model_seed,
            "config": frozen, "source_files": source,
            "scorer_sha256": frozen["scorer_sha256"],
            "training_episode_seeds": episode_seeds,
            "training_episodes": episodes,
            "decision_rows": len(dataset), "dataset_sha256": digest(data_path),
            "validation_dataset_sha256": digest(root / "validation.npz"),
            "fit": metrics, "hidden": frozen["hidden"],
        }
        save_model(model_dir, actor, manifest)


def train_dagger(root, scorer_path, validation_rows):
    root = Path(root)
    frozen = verify_frozen(root, scorer_path)
    source = frozen["source_files"]
    opt = frozen["optimizer"]
    for model_seed in MODEL_SEEDS:
        model_dir = root / "models" / "dagger" / f"s{model_seed}"
        state_path = model_dir / "progress.json"
        model_dir.mkdir(parents=True, exist_ok=True)
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "arm": "dagger", "model_seed": model_seed, "completed_rounds": 0,
            "rows": [], "rounds": [], "status": "started"}
        dataset_path = model_dir / "aggregate.npz"
        dataset = unpack_rows(dataset_path) if dataset_path.exists() else []
        start_round = int(state["completed_rounds"])
        for r in range(start_round, len(BETAS)):
            seed = frozen["training_episode_seeds_by_model_seed"][str(model_seed)][r]
            actor = None
            if r > 0:
                actor = DaggerActor(frozen["hidden"])
                prior = torch.load(model_dir / f"round{r-1}.pt", map_location="cpu", weights_only=True)
                actor.load_state_dict(prior)
                actor.eval()
            beta = BETAS[r]
            episode = collect_episode(dataset, scorer_path, seed, actor=actor,
                                      beta=beta, model_seed=model_seed,
                                      round_index=r, arm="dagger")
            pack_rows(dataset_path, dataset)
            fit_seed = 30_000 + model_seed * 100 + r
            fitted = init_actor(fit_seed, frozen["hidden"])
            metrics = fit(fitted, dataset, validation_rows, fit_seed + 1,
                          batch_size=opt["batch_decisions"],
                          max_epochs=opt["max_epochs_per_aggregation"],
                          patience=opt["early_stop_patience"], lr=opt["lr"])
            torch.save(fitted.state_dict(), model_dir / f"round{r}.pt")
            state["completed_rounds"] = r + 1
            state["rows"] = len(dataset)
            state["rounds"].append({"round": r, "beta": beta, "episode": episode,
                                    "fit": metrics, "fit_seed": fit_seed})
            state["status"] = "running" if r + 1 < len(BETAS) else "complete"
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"DAgger s{model_seed} round={r+1}/{len(BETAS)} beta={beta:.4g} "
                  f"rows={len(dataset)} val_ce={metrics['best_validation_cross_entropy']:.4f}",
                  flush=True)
        final_actor = DaggerActor(frozen["hidden"])
        final_actor.load_state_dict(torch.load(model_dir / f"round{len(BETAS)-1}.pt",
                                               map_location="cpu", weights_only=True))
        manifest = {
            "arm": "dagger", "status": "complete", "model_seed": model_seed,
            "config": frozen, "source_files": source,
            "scorer_sha256": frozen["scorer_sha256"],
            "training_episode_seeds": frozen["training_episode_seeds_by_model_seed"][str(model_seed)],
            "beta_schedule": list(BETAS), "rounds": state["rounds"],
            "decision_rows": len(dataset), "dataset_sha256": digest(dataset_path),
            "validation_dataset_sha256": digest(root / "validation.npz"),
            "hidden": frozen["hidden"],
        }
        save_model(model_dir, final_actor, manifest)


def evaluate(root, scorer_path):
    from blue.core.baselines import MaskedRandomBaseline, SleepBaseline
    from blue.harness.policy import HarnessPolicy
    from blue.policies.ordered import LancerValues, OrderedPolicy
    from blue.training import mappo_guide as mg
    from blue.training.harness_redesign import env_kwargs

    root = Path(root)
    frozen = verify_frozen(root, scorer_path)
    eval_root = root / "evaluation"
    eval_root.mkdir(exist_ok=True)
    controls_path = eval_root / "controls.json"
    if controls_path.exists():
        controls = json.loads(controls_path.read_text())
        if controls.get("seeds") != list(EVALUATION_SEEDS):
            raise ValueError("partial controls use a different seed list")
        controls.setdefault("cells", {name: {} for name in CONTROLS})
    else:
        controls = {"seeds": list(EVALUATION_SEEDS),
                    "cells": {name: {} for name in CONTROLS}}
    for seed in EVALUATION_SEEDS:
        if all(str(seed) in controls["cells"][name] for name in CONTROLS):
            continue
        if any(str(seed) in controls["cells"][name] for name in CONTROLS):
            raise ValueError(f"partial control row for evaluation seed {seed}")
        policies = {
            "lancer": OrderedPolicy(scorer=LancerValues(fruitless_decay=0.5)),
            "guard": HarnessPolicy(max_age=80),
            "sleep": SleepBaseline(),
            "masked_random": MaskedRandomBaseline(seed=seed),
        }
        for name, policy in policies.items():
            controls["cells"][name][str(seed)] = mg.run_team_episode(
                policy, seed, 400, **env_kwargs("finite"))
        controls_path.write_text(json.dumps(controls, indent=2) + "\n")
        print(f"controls seed={seed} lancer={controls['cells']['lancer'][str(seed)]['return']:+.0f}", flush=True)
    if controls["seeds"] != list(EVALUATION_SEEDS):
        raise ValueError("control seed mismatch")

    eval_rows = {}
    source = frozen["source_files"]
    for arm in ("bc", "dagger"):
        eval_rows[arm] = {}
        for model_seed in MODEL_SEEDS:
            model_dir = root / "models" / arm / f"s{model_seed}"
            actor, manifest = load_model(model_dir, source, frozen["scorer_sha256"])
            path = eval_root / arm / f"s{model_seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                partial = json.loads(path.read_text())
                if (partial.get("arm") != arm or partial.get("model_seed") != model_seed
                        or partial.get("seeds") != list(EVALUATION_SEEDS)
                        or partial.get("actor_sha256") != manifest["actor_sha256"]):
                    raise ValueError(f"evaluation resume identity mismatch: {path}")
                actor_rows = partial.get("cells", {})
            else:
                actor_rows = {}
            for seed in EVALUATION_SEEDS:
                if str(seed) in actor_rows:
                    continue
                hook = DaggerHook(scorer_path, actor=actor, beta=0.0,
                                  seed=model_seed * 1_000_003 + seed * 2,
                                  collect=False)
                policy = HarnessPolicy(hook, max_age=80, record_requests=True)
                result = mg.run_team_episode(policy, seed, 400, hook=hook, recorder=hook,
                                             **env_kwargs("finite"))
                result["lancer_agreement"] = hook.agreement / max(hook.decisions, 1)
                result["lancer_agreement_count"] = hook.agreement
                result["actor_decisions"] = hook.decisions
                actor_rows[str(seed)] = result
                path.write_text(json.dumps({"arm": arm, "model_seed": model_seed,
                                            "model_manifest_sha256": digest(model_dir / "manifest.json"),
                                            "actor_sha256": manifest["actor_sha256"],
                                            "seeds": list(EVALUATION_SEEDS),
                                            "cells": actor_rows}, indent=2) + "\n")
                print(f"eval {arm} s{model_seed} seed={seed} "
                      f"return={result['return']:+.0f} agreement={result['lancer_agreement']:.3f}",
                      flush=True)
            eval_rows[arm][str(model_seed)] = actor_rows
    analyze(root, frozen, controls, eval_rows)


def mean_interval(values):
    from scipy.stats import t
    vals = np.asarray(values, dtype=float)
    mean = float(vals.mean())
    if len(vals) < 2:
        return {"mean": mean, "low95": None, "high95": None}
    half = float(t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals)))
    return {"mean": mean, "low95": mean - half, "high95": mean + half}


def analyze(root, frozen, controls, eval_rows):
    report = {"claim": "exploratory matched-budget DAgger-vs-BC behavior-learning pilot",
              "primary_unit": frozen["primary_unit"], "evaluation_seeds": list(EVALUATION_SEEDS),
              "train_seeds_by_model": frozen["training_episode_seeds_by_model_seed"],
              "controls": {}, "arms": {}, "decision": "no superiority claim unless training-seed 95% interval is above zero"}
    expected = list(map(str, EVALUATION_SEEDS))
    for name in CONTROLS:
        if sorted(controls["cells"][name]) != expected:
            raise ValueError(f"incomplete control {name}")
        returns = [float(controls["cells"][name][s]["return"]) for s in expected]
        report["controls"][name] = {
            "mean": float(np.mean(returns)), "p10": float(np.quantile(returns, .1)),
            "min": float(np.min(returns)), "returns": returns,
        }
    lancer = {s: float(controls["cells"]["lancer"][s]["return"]) for s in expected}
    guard = {s: float(controls["cells"]["guard"][s]["return"]) for s in expected}
    for arm, by_model in eval_rows.items():
        per_model = {}
        for model_seed in MODEL_SEEDS:
            cells = by_model[str(model_seed)]
            if sorted(cells) != expected:
                raise ValueError(f"incomplete {arm} seed {model_seed}")
            returns = {s: float(cells[s]["return"]) for s in expected}
            lancer_delta = [returns[s] - lancer[s] for s in expected]
            guard_delta = [returns[s] - guard[s] for s in expected]
            per_model[str(model_seed)] = {
                "mean_return": float(np.mean(list(returns.values()))),
                "p10_return": float(np.quantile(list(returns.values()), .1)),
                "min_return": float(min(returns.values())),
                "mean_lancer_agreement": float(np.mean([cells[s]["lancer_agreement"] for s in expected])),
                "mean_actor_decisions": float(np.mean([cells[s]["actor_decisions"] for s in expected])),
                "paired_lancer_mean": float(np.mean(lancer_delta)),
                "paired_guard_mean": float(np.mean(guard_delta)),
                "returns": [returns[s] for s in expected],
                "paired_lancer": lancer_delta,
                "paired_guard": guard_delta,
            }
        lancer_training_seed_means = [per_model[str(m)]["paired_lancer_mean"] for m in MODEL_SEEDS]
        guard_training_seed_means = [per_model[str(m)]["paired_guard_mean"] for m in MODEL_SEEDS]
        agreement = [per_model[str(m)]["mean_lancer_agreement"] for m in MODEL_SEEDS]
        report["arms"][arm] = {
            "per_training_seed": per_model,
            "lancer_interval_over_training_seeds": mean_interval(lancer_training_seed_means),
            "guard_interval_over_training_seeds": mean_interval(guard_training_seed_means),
            "mean_lancer_agreement_over_training_seeds": float(np.mean(agreement)),
        }
    path = Path(root) / "analysis.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"analysis": str(path), "arms": {
        arm: {"lancer": stats["lancer_interval_over_training_seeds"],
              "guard": stats["guard_interval_over_training_seeds"],
              "agreement": stats["mean_lancer_agreement_over_training_seeds"]}
        for arm, stats in report["arms"].items()}}, indent=2))


def run(root, scorer_path):
    frozen = verify_frozen(root, scorer_path)
    torch.set_num_threads(1)
    validation_rows = collect_validation(root, scorer_path)
    train_bc(root, scorer_path, validation_rows)
    train_dagger(root, scorer_path, validation_rows)
    evaluate(root, scorer_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "run"))
    p.add_argument("--out", required=True)
    p.add_argument("--scorer", default="blue/results/gpt_harness/scorer.pkl")
    args = p.parse_args()
    if args.mode == "freeze":
        freeze(args.out, args.scorer)
    else:
        run(args.out, args.scorer)


if __name__ == "__main__":
    main()
