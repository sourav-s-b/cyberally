import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from blue.harness.attention_actor import CandidateAttentionActor
from blue.harness.redesign import FEATURE_NAMES
from blue.harness.redesign import probabilities
from blue.harness.routed_hybrid_actor import SituationRouterActor


def _batch(batch=3, candidates=7):
    g = torch.Generator().manual_seed(17)
    x = torch.randn(batch, candidates, len(FEATURE_NAMES), generator=g)
    base = torch.randn(batch, candidates, generator=g)
    mask = torch.ones(batch, candidates, dtype=torch.bool)
    mask[0, -2:] = False
    if batch > 1:
        mask[1, -1:] = False
    return x, base, mask


def test_attention_actor_preserves_heuristic_at_initialization():
    x, base, mask = _batch()
    actor = CandidateAttentionActor(hidden=32, heads=4)
    residual = actor(x, base, mask)
    assert residual.shape == mask.shape
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(residual[~mask], torch.zeros_like(residual[~mask]))


def test_zero_residual_keeps_the_existing_teacher_mixture_distribution():
    x, base, mask = _batch()
    actor = CandidateAttentionActor(hidden=42, heads=3)
    teacher = torch.tensor([0, 1, 2])
    p = probabilities(actor, x, base, mask, teacher,
                      bonus=0.25, temperature=0.5, explore=0.2)
    logits = (base / 0.5).masked_fill(~mask, -torch.inf)
    expected = 0.2 * torch.softmax(logits, dim=-1)
    expected.scatter_add_(1, teacher[:, None], torch.full((3, 1), 0.8))
    torch.testing.assert_close(p, expected, atol=1e-7, rtol=1e-7)
    assert sum(p.numel() for p in actor.parameters()) == 19_027


def test_attention_actor_is_permutation_equivariant_and_masks_keys():
    x, base, mask = _batch(batch=1)
    actor = CandidateAttentionActor(hidden=32, heads=4)
    with torch.no_grad():
        actor.head.weight.normal_(0, 0.1)
        actor.head.bias.fill_(0.03)
    order = torch.tensor([4, 1, 6, 0, 3, 5, 2])
    with torch.no_grad():
        original = actor(x, base, mask)
        shuffled = actor(x[:, order], base[:, order], mask[:, order])
        torch.testing.assert_close(shuffled, original[:, order], atol=1e-6,
                                   rtol=1e-6)

        changed_padding = x.clone()
        changed_padding[:, ~mask[0], :] = 1e5
        padded = actor(changed_padding, base, mask)
        torch.testing.assert_close(padded[mask], original[mask], atol=1e-6,
                                   rtol=1e-6)


def test_attention_actor_has_finite_nonzero_gradients():
    x, base, mask = _batch(batch=1)
    actor = CandidateAttentionActor(hidden=32, heads=4)
    with torch.no_grad():
        actor.head.weight.normal_(0, 0.1)
    loss = actor(x, base, mask)[mask].square().mean()
    loss.backward()
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_attention_actor_rejects_empty_candidate_rows():
    x, base, mask = _batch(batch=1)
    mask[:] = False
    with pytest.raises(ValueError, match="at least one"):
        CandidateAttentionActor(hidden=32, heads=4)(x, base, mask)


def test_fixed_blend_is_exact_half_and_learned_gate_starts_at_half():
    x, base, mask = _batch(batch=1)
    actor = SituationRouterActor(set_hidden=32, attention_hidden=32,
                                  attention_heads=4, fixed_gate=0.5)
    with torch.no_grad():
        set_score = actor.set_branch(x, base, mask)
        attention_score = actor.attention_branch(x, base, mask)
        fixed = actor(x, base, mask)
        torch.testing.assert_close(fixed, 0.5 * set_score +
                                   0.5 * attention_score)
    actor.begin_gate_capture()
    actor(x, base, mask)
    assert actor.end_gate_capture() == [0.5]
    learned = SituationRouterActor(set_hidden=32, attention_hidden=32,
                                   attention_heads=4)
    learned.begin_gate_capture()
    learned(x, base, mask)
    assert learned.end_gate_capture() == [0.5]
