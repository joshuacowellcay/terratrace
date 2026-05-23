"""Loss functions for the forest-loss segmentation task.

Forest-loss is a small minority of pixels (often 1-10% per tile). Plain BCE will settle into
predicting "no loss" everywhere; the loss curve looks fine and IoU is ~0. We combine Dice
loss (which directly optimises the overlap statistic and handles imbalance because its
denominator scales with the foreground area) with weighted BCE (which keeps gradients
informative on confidently-correct pixels and regularises Dice's flat regions). The
positive-class weight is set to roughly the inverse class prevalence.

A per-pixel validity mask (``weights``) zeroes the loss on cloudy pixels, which have no
trustworthy label.

Inputs are raw logits. Sigmoid is applied internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Dice loss in [0, 1].

    Shapes:
        logits:  (B, 1, H, W) or (B, H, W)
        target:  (B, H, W) with values in {0, 1}
        weights: (B, H, W) or None. Per-pixel non-negative weight (0 = ignore).
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits)
    if weights is None:
        weights = torch.ones_like(probs)
    inter = (probs * target * weights).sum(dim=(1, 2))
    denom = (probs * weights).sum(dim=(1, 2)) + (target * weights).sum(dim=(1, 2))
    dice = (2.0 * inter + EPS) / (denom + EPS)
    return 1.0 - dice.mean()


def weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    pos_weight: float = 5.0,
) -> torch.Tensor:
    """Binary cross-entropy with positive-class weighting and per-pixel mask.

    ``pos_weight`` upweights the loss on positive (loss) pixels; set to the inverse of
    expected class prevalence (or close to it).
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    if weights is not None:
        per_pixel = per_pixel * weights
        denom = weights.sum().clamp_min(1.0)
        return per_pixel.sum() / denom
    return per_pixel.mean()


@dataclass(frozen=True)
class CombinedLossWeights:
    dice: float = 0.5
    bce: float = 0.5
    pos_weight: float = 5.0


class CombinedLoss(nn.Module):
    """Dice + weighted BCE, summed with configurable weights."""

    def __init__(self, weights: CombinedLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or CombinedLossWeights()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dice = soft_dice_loss(logits, target, valid)
        bce = weighted_bce_loss(logits, target, valid, pos_weight=self.weights.pos_weight)
        return self.weights.dice * dice + self.weights.bce * bce
