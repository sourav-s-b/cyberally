"""Trajectory-blocked validation splits + early stopping (Stream A / V1).

Why this exists: every extractor (BC, RvS, IQL) trained a fixed 30 epochs
on ALL log data with in-sample match % as the only metric. Per Codevilla
(action-MSE ~0.39 correlation with rollout success) and Mandlekar (best
val-loss ckpt 50-100% worse than best rollout), that metric cannot select
checkpoints. Per Bergmeir/Roberts, splits must be BLOCKED by trajectory:
random transition splits leak correlated ticks across the boundary.

- blocked_split: episode-level 80/20 split. All 5 agent-sequences of an
  episode stay together on one side (agents share the team reward and
  the same red rollout; splitting them leaks).
- EarlyStopper: Prechelt-style strips-of-rising-error on the val metric
  with best-weight restore (CPU-cloned state_dict, no GPU pinning).
- Nested use: call blocked_split twice (outer for reporting, inner for
  tuning) per Varma/Simon; the trainers expose --val-seed so the inner
  split is reproducible and the outer can vary it.
"""

import numpy as np


def blocked_split(n_eps, val_frac=0.2, seed=0):
    """Episode indices -> (train_eps, val_eps). Deterministic on seed."""
    rng = np.random.RandomState(seed)
    order = np.arange(n_eps)
    rng.shuffle(order)
    n_val = max(1, int(round(n_eps * val_frac)))
    val_eps = np.sort(order[:n_val])
    train_eps = np.sort(order[n_val:])
    return train_eps, val_eps


def seq_split(seq_episode, train_eps):
    """Sequence indices -> (train_idx, val_idx) from per-seq episode ids."""
    train_set = set(int(e) for e in train_eps)
    is_train = np.array([int(e) in train_set for e in seq_episode])
    idx = np.arange(len(seq_episode))
    return idx[is_train], idx[~is_train]


class EarlyStopper:
    """Stop after `patience` consecutive val-metric worsenings (lower is
    better); restores the best weights. patience=0 disables (fixed-epoch
    legacy behavior, reproducible)."""

    def __init__(self, patience=8, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.best_epoch = 0
        self.bad_strips = 0
        self.best_state = None
        self.stopped_epoch = None

    def update(self, epoch, val_metric, model):
        """Returns True when training should stop. Snapshots best weights
        to CPU (clone) so later epochs cannot overwrite them."""
        import torch as th
        if self.patience <= 0:
            return False
        improved = (self.best is None
                    or val_metric < self.best - self.min_delta)
        if improved:
            self.best = float(val_metric)
            self.best_epoch = int(epoch)
            self.bad_strips = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.bad_strips += 1
        if self.bad_strips >= self.patience:
            self.stopped_epoch = int(epoch)
            return True
        return False

    def restore(self, model):
        """Load best weights back. No-op if never improved/disabled."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    def summary(self):
        return {"best_val": self.best, "best_epoch": self.best_epoch,
                "stopped_epoch": self.stopped_epoch,
                "patience": self.patience}
