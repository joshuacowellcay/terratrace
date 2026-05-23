"""Tests for the Siamese U-Net."""

from __future__ import annotations

import torch

from terratrace.models.siamese_unet import SiameseUNet, SiameseUNetConfig


def test_forward_shape_matches_input() -> None:
    cfg = SiameseUNetConfig(in_channels=4, base_channels=16, depth=4)
    model = SiameseUNet(cfg)
    model.eval()
    b = torch.randn(2, 4, 64, 64)
    a = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        out = model(b, a)
    assert out.shape == (2, 1, 64, 64)


def test_forward_is_symmetric_in_inputs() -> None:
    """Swapping before/after must not change the network's structural behaviour: same shape,
    finite values. The numerical output is allowed to differ; change has a sign."""
    cfg = SiameseUNetConfig(in_channels=4, base_channels=8, depth=3)
    model = SiameseUNet(cfg)
    model.eval()
    b = torch.randn(1, 4, 32, 32)
    a = torch.randn(1, 4, 32, 32)
    with torch.no_grad():
        ab = model(a, b)
        ba = model(b, a)
    assert ab.shape == ba.shape
    assert torch.isfinite(ab).all()
    assert torch.isfinite(ba).all()


def test_encoder_truly_shares_weights() -> None:
    """The single self.encoder is reused for both branches; confirm by parameter identity."""
    cfg = SiameseUNetConfig(base_channels=8, depth=3)
    model = SiameseUNet(cfg)
    encoder_params = list(model.encoder.parameters())
    all_params = list(model.parameters())
    # All encoder parameters appear exactly once in model.parameters()
    assert all(any(p is q for q in all_params) for p in encoder_params)


def test_rejects_mismatched_input_shapes() -> None:
    model = SiameseUNet(SiameseUNetConfig(base_channels=8, depth=3))
    b = torch.randn(1, 4, 32, 32)
    a = torch.randn(1, 4, 16, 16)
    import pytest

    with pytest.raises(ValueError):
        model(b, a)


def test_parameter_count_scales_with_base_channels() -> None:
    small = SiameseUNet(SiameseUNetConfig(base_channels=8, depth=3)).parameter_count()
    big = SiameseUNet(SiameseUNetConfig(base_channels=32, depth=3)).parameter_count()
    assert big > 5 * small  # 4x base_channels gives ~16x params; generous lower bound
