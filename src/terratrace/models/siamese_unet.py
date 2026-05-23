"""Siamese U-Net change-detection model.

Two encoder branches **share weights** (the same ``nn.Module`` is called twice), so the model
sees the before and after images symmetrically — swapping the inputs would invert the sign of
the predicted change, never silently lose information. At each spatial scale the encoder
features are concatenated channel-wise and passed to the decoder as the skip connection. The
decoder upsamples back to input resolution and emits a per-pixel logit.

Why Siamese over an early-fusion U-Net (input = stacked 2C-channel pair)?
    * Symmetry: same backbone evaluates each image, so any bias the encoder learns applies
      identically to both timestamps.
    * Reusability: the same encoder can be plugged into a triplet-loss representation
      pre-training in a future iteration.
    * Cleaner separation: per-image embeddings are explicit and can be probed.

Why not subtract the features instead of concatenate?
    * Subtraction discards the raw appearance information and can mask cases where the model
      needs context (e.g. "this was forest, now it's cloud — don't call this loss"). Concat
      keeps both — the decoder learns whatever combination is most useful.

The encoder is a small VGG-ish stack of double-conv blocks, max-pooled between scales. No
external backbone (no ImageNet weights) so the model is self-contained and works identically
on the synthetic data path and the real Sentinel-2 path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SiameseUNetConfig:
    """Hyperparameters for :class:`SiameseUNet`."""

    in_channels: int = 4
    base_channels: int = 32  # raised in configs/full.yaml for GPU training
    depth: int = 4  # number of down/up stages (4 → 16x downsampling, fine for 128px tiles)

    def __post_init__(self) -> None:
        if self.depth < 2 or self.depth > 6:
            raise ValueError(f"depth must be in [2, 6], got {self.depth}")
        if self.base_channels < 4:
            raise ValueError(f"base_channels too small: {self.base_channels}")


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Double 3x3 conv + BN + ReLU, the U-Net workhorse."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SharedEncoder(nn.Module):
    """The encoder applied to each of (before, after). Returns features at every scale."""

    def __init__(self, in_channels: int, base_channels: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.stages = nn.ModuleList()
        prev_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.stages.append(_conv_block(prev_ch, out_ch))
            prev_ch = out_ch
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feats.append(x)
            if i < len(self.stages) - 1:
                x = self.pool(x)
        return feats  # len == depth, finest first


class DecoderBlock(nn.Module):
    """Upsample → align to skip → concat → double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Twin-encoder U-Net for change detection. Output: (B, 1, H, W) logits."""

    def __init__(self, config: SiameseUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SiameseUNetConfig()
        c = self.config
        self.encoder = SharedEncoder(
            in_channels=c.in_channels,
            base_channels=c.base_channels,
            depth=c.depth,
        )
        encoder_channels = [c.base_channels * (2**i) for i in range(c.depth)]
        # Bottleneck takes the concatenation of (before, after) at the deepest pool level.
        bottleneck_in = 2 * encoder_channels[-1]
        bottleneck_out = bottleneck_in
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _conv_block(bottleneck_in, bottleneck_out)

        # Decoder mirrors the encoder; each skip is the concatenation of the matching
        # (before, after) feature maps from the shared encoder.
        decoder_blocks: list[DecoderBlock] = []
        prev_ch = bottleneck_out
        for i in reversed(range(c.depth)):
            skip_ch = 2 * encoder_channels[i]
            out_ch = encoder_channels[i]
            decoder_blocks.append(DecoderBlock(prev_ch, skip_ch, out_ch))
            prev_ch = out_ch
        self.decoder = nn.ModuleList(decoder_blocks)

        self.head = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError(f"before/after shape mismatch: {before.shape} vs {after.shape}")
        feats_b = self.encoder(before)
        feats_a = self.encoder(after)
        skips = [torch.cat([fb, fa], dim=1) for fb, fa in zip(feats_b, feats_a, strict=True)]

        bot_in = torch.cat([self.pool(feats_b[-1]), self.pool(feats_a[-1])], dim=1)
        x = self.bottleneck(bot_in)

        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            x = block(x, skip)
        return self.head(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
