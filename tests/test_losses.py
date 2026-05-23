"""Tests for the segmentation losses."""

from __future__ import annotations

import torch

from terratrace.training.losses import (
    CombinedLoss,
    CombinedLossWeights,
    soft_dice_loss,
    weighted_bce_loss,
)


def test_dice_zero_when_prediction_matches_target_perfectly() -> None:
    # Construct logits so sigmoid(logits) is close to target.
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    logits = torch.where(target > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
    loss = soft_dice_loss(logits, target)
    assert loss.item() < 0.02  # near-zero after sigmoid saturation


def test_dice_high_when_prediction_opposite_of_target() -> None:
    target = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    # Predict the opposite class with high confidence.
    logits = torch.where(target > 0.5, torch.tensor(-10.0), torch.tensor(10.0))
    loss = soft_dice_loss(logits, target)
    assert loss.item() > 0.9


def test_weighted_bce_respects_validity_mask() -> None:
    target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    logits = torch.zeros_like(target)  # all predictions = 0.5
    full_loss = weighted_bce_loss(logits, target, None)
    weights = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])  # ignore the only positive pixel
    masked_loss = weighted_bce_loss(logits, target, weights)
    # Removing the positive pixel cuts the loss noticeably.
    assert masked_loss.item() < full_loss.item()


def test_combined_loss_is_finite_and_positive() -> None:
    target = (torch.rand(2, 16, 16) > 0.7).float()
    logits = torch.randn(2, 1, 16, 16)
    valid = torch.ones(2, 16, 16)
    loss = CombinedLoss(CombinedLossWeights(dice=0.5, bce=0.5, pos_weight=4.0))(
        logits, target, valid
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_combined_loss_decreases_with_better_predictions() -> None:
    """Training signal sanity check: a near-perfect logits batch should produce a much smaller
    loss than a random batch on the same targets."""
    target = (torch.rand(2, 16, 16) > 0.7).float()
    valid = torch.ones_like(target)
    bad_logits = torch.randn(2, 1, 16, 16)
    good_logits = (target * 8.0 - 4.0).unsqueeze(1)
    loss_fn = CombinedLoss()
    assert loss_fn(good_logits, target, valid).item() < loss_fn(bad_logits, target, valid).item()
