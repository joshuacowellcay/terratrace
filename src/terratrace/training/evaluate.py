"""Full evaluation: deep model and NDVI baseline on the held-out geographic test split.

Loads the trained checkpoint and runs it over the test split, collecting per-pixel
probabilities so we can report both thresholded metrics and a calibration curve. The NDVI
baseline threshold is calibrated on the val split (which the model also never saw) and
applied to test, for a fair comparison on the same data. IoU/precision/recall/F1 are
computed for the forest-loss class on both detectors, alongside accuracy (kept only for
completeness). The failure breakdown stratifies tiles by cloud fraction. Output: a JSON
file at ``results/metrics.json`` and figures under ``assets/``.

The headline IoU numbers the README quotes come from this function's output. Re-run
``scripts/evaluate.py`` to refresh; never edit the README's numbers by hand.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; never try to open a window during evaluation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import ChangeDetectionDataset
from ..data.manifest import DatasetManifest
from ..models.baseline import calibrate_threshold
from ..models.baseline import predict as ndvi_predict
from ..models.siamese_unet import SiameseUNet, SiameseUNetConfig
from .metrics import SegMetrics, compute_segmentation_metrics
from .train import select_device

log = logging.getLogger(__name__)


@dataclass
class CalibrationCurve:
    """Reliability-diagram data: predicted-probability bin to observed positive fraction."""

    bin_centers: list[float]
    observed_positive_fraction: list[float]
    bin_counts: list[int]


@dataclass
class FailureBreakdown:
    """IoU computed on subsets of the test split, to highlight where each detector struggles."""

    deep_clean_iou: float
    deep_cloudy_iou: float
    baseline_clean_iou: float
    baseline_cloudy_iou: float
    cloudy_threshold: float = 0.15  # cloud fraction above which a tile counts as "cloudy"


@dataclass
class EvaluationReport:
    deep_model: dict[str, float]  # metrics dict
    baseline: dict[str, float]
    baseline_threshold: float
    calibration: CalibrationCurve
    failure: FailureBreakdown
    n_test_tiles: int
    data_source: str
    region_name: str
    checkpoint_path: str
    metrics_path: str = ""
    figures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "deep_model": self.deep_model,
            "baseline": {**self.baseline, "calibrated_threshold": self.baseline_threshold},
            "calibration": asdict(self.calibration),
            "failure_breakdown": asdict(self.failure),
            "n_test_tiles": self.n_test_tiles,
            "data_source": self.data_source,
            "region_name": self.region_name,
            "checkpoint_path": self.checkpoint_path,
            "figures": self.figures,
        }


def load_model(checkpoint_path: Path, device: torch.device) -> SiameseUNet:
    """Reconstruct the model from a checkpoint that embeds its own config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SiameseUNetConfig(**ckpt["model_config"])
    model = SiameseUNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _collect_predictions(
    model: SiameseUNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over a loader. Returns (probs, preds, targets, weights, cloud_frac per tile)."""
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    cloud_frac_per_tile: list[float] = []
    with torch.no_grad():
        for batch in loader:
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            label = batch["label"].cpu().numpy()
            valid = batch["weights"].cpu().numpy()
            logits = model(before, after).squeeze(1)
            prob = torch.sigmoid(logits).cpu().numpy()

            probs.append(prob)
            targets.append(label)
            weights.append(valid)
            # cloud fraction per tile = fraction of pixels masked out
            for w in valid:
                cloud_frac_per_tile.append(float(1.0 - w.mean()))

    probs_arr = np.concatenate(probs, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    weights_arr = np.concatenate(weights, axis=0)
    preds_arr = (probs_arr > 0.5).astype(np.uint8)
    return probs_arr, preds_arr, targets_arr, weights_arr, np.array(cloud_frac_per_tile)


def _baseline_predictions_for_split(
    manifest: DatasetManifest, split: str, root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack (before, after, label, weights) tensors across all tiles in a split."""
    ds = ChangeDetectionDataset(manifest, split, root=root)
    befores, afters, labels, weights = [], [], [], []
    for i in range(len(ds)):
        sample = ds[i]
        # ChangeDetectionDataset returns (C, H, W); baseline.calibrate wants (H, W, C).
        befores.append(sample["before"].numpy().transpose(1, 2, 0))
        afters.append(sample["after"].numpy().transpose(1, 2, 0))
        labels.append(sample["label"].numpy().astype(np.uint8))
        weights.append(sample["weights"].numpy().astype(np.uint8))
    return (
        np.stack(befores),
        np.stack(afters),
        np.stack(labels),
        np.stack(weights),
    )


def _calibration_curve(
    probs: np.ndarray, targets: np.ndarray, weights: np.ndarray, *, n_bins: int = 10
) -> CalibrationCurve:
    """Bin pixels by predicted probability; report the observed positive rate per bin.

    A perfectly calibrated model lies on the diagonal y = x. Matters here because the
    inference service multiplies the predicted-loss mask by tile area to estimate hectares
    lost; systematically over- or under-confident probabilities give a wrong hectares figure.
    """
    mask = weights.ravel().astype(bool)
    p = probs.ravel()[mask]
    t = targets.ravel()[mask].astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        in_bin = (
            (p >= bins[i]) & (p < bins[i + 1])
            if i < n_bins - 1
            else (p >= bins[i]) & (p <= bins[i + 1])
        )
        n = int(in_bin.sum())
        center = float((bins[i] + bins[i + 1]) / 2.0)
        centers.append(center)
        observed.append(float(t[in_bin].mean()) if n > 0 else 0.0)
        counts.append(n)
    return CalibrationCurve(centers, observed, counts)


def _failure_breakdown(
    deep_preds: np.ndarray,
    baseline_preds: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    cloud_frac_per_tile: np.ndarray,
    *,
    cloudy_threshold: float = 0.15,
) -> FailureBreakdown:
    """Split test tiles by cloud fraction and report IoU on each subset for both detectors."""
    cloudy_tiles = cloud_frac_per_tile > cloudy_threshold
    clean_tiles = ~cloudy_tiles

    def _iou(tile_mask: np.ndarray, preds: np.ndarray) -> float:
        if not tile_mask.any():
            return float("nan")
        sub_preds = preds[tile_mask]
        sub_targets = targets[tile_mask]
        sub_weights = weights[tile_mask]
        return compute_segmentation_metrics(sub_preds, sub_targets, sub_weights).iou

    return FailureBreakdown(
        deep_clean_iou=_iou(clean_tiles, deep_preds),
        deep_cloudy_iou=_iou(cloudy_tiles, deep_preds),
        baseline_clean_iou=_iou(clean_tiles, baseline_preds),
        baseline_cloudy_iou=_iou(cloudy_tiles, baseline_preds),
        cloudy_threshold=cloudy_threshold,
    )


def _plot_calibration(curve: CalibrationCurve, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    valid_bins = [
        (c, o)
        for c, o, n in zip(
            curve.bin_centers,
            curve.observed_positive_fraction,
            curve.bin_counts,
            strict=True,
        )
        if n > 0
    ]
    if valid_bins:
        xs, ys = zip(*valid_bins, strict=True)
        ax.plot(xs, ys, marker="o", label="deep model")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive fraction")
    ax.set_title("reliability diagram (deep model)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_metrics_bar(deep: SegMetrics, baseline: SegMetrics, out_path: Path) -> None:
    metrics = ["iou", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, [getattr(baseline, m) for m in metrics], width, label="NDVI baseline")
    ax.bar(x + width / 2, [getattr(deep, m) for m in metrics], width, label="Siamese U-Net")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (forest-loss class)")
    ax.set_title("test-split metrics: model vs baseline")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate(
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    device: str = "auto",
    batch_size: int = 8,
    results_path: Path = Path("results/metrics.json"),
    assets_dir: Path = Path("assets"),
) -> EvaluationReport:
    """End-to-end evaluation. Writes JSON + figures and returns the report."""
    dev = select_device(device)
    manifest = DatasetManifest.read(manifest_path)
    root = manifest_path.parent

    # Calibrate NDVI baseline on the val split (the deep model also never saw val).
    log.info("Calibrating NDVI baseline threshold on val split.")
    val_b, val_a, val_l, val_w = _baseline_predictions_for_split(manifest, "val", root)
    calib = calibrate_threshold(val_b, val_a, val_l, valid_mask=val_w)
    log.info("Picked NDVI threshold = %.3f (val F1 = %.3f).", calib.best_threshold, calib.best_f1)

    log.info("Running Siamese U-Net on test split.")
    model = load_model(checkpoint_path, dev)
    test_ds = ChangeDetectionDataset(manifest, "test", root=root)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    deep_probs, deep_preds, targets, weights, cloud_frac_per_tile = _collect_predictions(
        model, test_loader, dev
    )

    log.info("Applying NDVI baseline to test split.")
    test_b, test_a, test_l, test_w = _baseline_predictions_for_split(manifest, "test", root)
    assert np.array_equal(test_l, targets), "test-split label ordering disagrees between loaders"
    baseline_preds = np.stack(
        [
            ndvi_predict(test_b[i], test_a[i], threshold=calib.best_threshold, valid_mask=test_w[i])
            for i in range(len(test_b))
        ]
    )

    deep_metrics = compute_segmentation_metrics(deep_preds, targets, weights)
    baseline_metrics = compute_segmentation_metrics(baseline_preds, targets, weights)
    log.info("Deep:     %s", deep_metrics.as_dict())
    log.info("Baseline: %s", baseline_metrics.as_dict())

    calibration = _calibration_curve(deep_probs, targets, weights)
    failure = _failure_breakdown(deep_preds, baseline_preds, targets, weights, cloud_frac_per_tile)

    assets_dir.mkdir(parents=True, exist_ok=True)
    calib_fig = assets_dir / "calibration.png"
    metrics_fig = assets_dir / "metrics_comparison.png"
    _plot_calibration(calibration, calib_fig)
    _plot_metrics_bar(deep_metrics, baseline_metrics, metrics_fig)

    report = EvaluationReport(
        deep_model=deep_metrics.as_dict(),
        baseline=baseline_metrics.as_dict(),
        baseline_threshold=calib.best_threshold,
        calibration=calibration,
        failure=failure,
        n_test_tiles=len(test_ds),
        data_source=manifest.source,
        region_name=manifest.region_name,
        checkpoint_path=str(checkpoint_path),
        figures={"calibration": str(calib_fig), "metrics_comparison": str(metrics_fig)},
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report.as_dict(), indent=2))
    report.metrics_path = str(results_path)
    log.info("Wrote metrics to %s and figures to %s.", results_path, assets_dir)
    return report
