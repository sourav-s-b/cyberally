"""Read-only diagnostics; never alter a policy's inputs or sampled actions."""
import hashlib
import json


class RequestDigest:
    """Ordered full-episode requests, including busy ticks and host targets.

    This fingerprints requests, not simulator outcomes or completed actions.
    Equal hashes support request identity; equal returns/counts alone do not.
    """

    def __init__(self):
        self.digest = hashlib.sha256()
        self.count = 0

    def add(self, tick, agent, action, host):
        self.digest.update((json.dumps([int(tick), agent, action, host],
                                      separators=(",", ":")) + "\n").encode())
        self.count += 1

    def result(self):
        return {"requested_actions_sha256": self.digest.hexdigest(),
                "requested_actions_count": self.count,
                "semantics": "ordered tick/agent/action/target requests; not completion"}


def input_sensitivity(actor, features, base, bonus):
    """Compare logits on the SAME observed candidates after zeroing ML inputs.

    A local input intervention, not an estimated reward advantage or a rollout.
    No RNG is consumed. Top-index changes use fixed candidate ordering.
    """
    import torch as th
    with th.no_grad():
        real = th.tanh(actor(features))
        risk = features.clone(); risk[:, -1] = 0
        zero = features.clone(); zero[:, -2:] = 0
        risk_logits = base + bonus * th.tanh(actor(risk))
        zero_logits = base + bonus * th.tanh(actor(zero))
        real_logits = base + bonus * real
    return {"ml_logit_delta_max": float((real_logits-zero_logits).abs().max()),
            "novelty_logit_delta_max": float((real_logits-risk_logits).abs().max()),
            "ml_top_index_changed": int(real_logits.argmax() != zero_logits.argmax()),
            "novelty_top_index_changed": int(real_logits.argmax() != risk_logits.argmax())}
