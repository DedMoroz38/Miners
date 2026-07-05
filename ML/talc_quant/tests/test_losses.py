"""SegLoss: finite, differentiable, area term zero at perfect prediction."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.config import load_config  # noqa: E402
from talc_quant.constants import CLASS_TALC, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from talc_quant.losses import SegLoss, class_weights  # noqa: E402


def _loss():
    return SegLoss(load_config().loss)


def test_class_weights_boost_talc():
    w = class_weights(load_config().loss)
    assert w[CLASS_TALC] > w[0]


def test_finite_and_grad_flows():
    loss = _loss()
    logits = torch.randn(2, NUM_CLASSES, 32, 32, requires_grad=True)
    target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
    val = loss(logits, target)
    assert torch.isfinite(val)
    val.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_all_ignore_is_zero():
    loss = _loss()
    logits = torch.randn(1, NUM_CLASSES, 8, 8, requires_grad=True)
    target = torch.full((1, 8, 8), IGNORE_INDEX)
    assert float(loss(logits, target)) == 0.0


def test_area_term_vanishes_when_confident_correct():
    """A near-perfect prediction -> area component ~0. Compare total loss with a
    high vs low area weight on a confident correct prediction: they should match
    because |soft_area - gt_area| ~ 0."""
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :8, :8] = CLASS_TALC
    onehot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    logits = (onehot * 20.0)  # confident correct
    lo = SegLoss(load_config(overrides=["loss.area_w=0.0"]).loss)
    hi = SegLoss(load_config(overrides=["loss.area_w=10.0"]).loss)
    assert abs(float(lo(logits, target)) - float(hi(logits, target))) < 1e-3
