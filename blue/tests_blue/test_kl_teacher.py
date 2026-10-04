"""KL-to-teacher helper math (proposal 02 follow-up). Train venv (torch).

Tests the module-level teacher_kl() in isolation: identity, known value,
asymmetry, and mask exclusion. Learner wiring (frozen teacher construction,
loss addition, logging) is proven by the KL training run + its manifest.
"""
import pytest

torch = pytest.importorskip("torch", reason="learner imports torch")
pytest.importorskip("components.episode_buffer", reason="needs epymarl on path")

from learners.ppo_learner import teacher_kl  # noqa: E402


def test_identity_zero_and_known_value():
    pi = torch.tensor([[[[0.75, 0.25]]]])
    assert teacher_kl(pi, pi, torch.ones(1, 1, 1)).item() == pytest.approx(0.0)
    other = torch.tensor([[[[0.5, 0.5]]]])
    # KL([.5,.5] || [.75,.25]) = .5*ln(.5/.75)+.5*ln(.5/.25) ~= 0.1438
    assert teacher_kl(pi, other, torch.ones(1, 1, 1)).item() == pytest.approx(
        0.1438, abs=1e-3)


def test_asymmetric_and_nonnegative():
    a = torch.tensor([[[[0.9, 0.1]]]])
    b = torch.tensor([[[[0.5, 0.5]]]])
    m = torch.ones(1, 1, 1)
    assert teacher_kl(a, b, m).item() >= 0.0
    assert teacher_kl(a, b, m).item() != pytest.approx(
        teacher_kl(b, a, m).item())


def test_mask_excludes_invalid_triples():
    pi = torch.tensor([[[[0.9, 0.1]]], [[[0.0, 1.0]]]])
    teacher = torch.tensor([[[[0.9, 0.1]]], [[[1.0, 0.0]]]])
    mask = torch.tensor([[[1.0]], [[0.0]]])  # second triple masked out
    assert teacher_kl(pi, teacher, mask).item() == pytest.approx(0.0)
