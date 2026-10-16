"""Training loop for the Siamese U-Net change detector.

Single entry point: :func:`train`. Deterministic per-seed (numpy + torch + Python random),
logs every run to a local MLflow store at ``./mlruns``, and saves the best-val-IoU
checkpoint to ``{checkpoint_dir}/{checkpoint_name}``. ``device='auto'`` picks Apple MPS or
CUDA when available and falls back to CPU. The CPU path is the one that matters; defaults in
:mod:`terratrace.training.config` are chosen so the fast run completes on a single laptop
core in well under thirty minutes.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import ChangeDetectionDataset
from ..data.manifest import DatasetManifest
from ..models.siamese_unet import SiameseUNet
from .config import TrainConfig
from .losses import CombinedLoss, CombinedLossWeights
from .metrics import compute_segmentation_metrics

log = logging.getLogger(__name__)


@dataclass
class TrainOutcome:
    best_val_iou: float
    epochs_trained: int
    checkpoint_path: Path
    parameter_count: int
    device: str


def select_device(preference: str) -> torch.device:
    """Pick the device the user asked for, falling back if unavailable."""
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            log.warning("MPS requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seeds(seed: int) -> None:
    """Make this run reproducible. Best-effort; perfect determinism on GPU isn't free."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
        )
    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=0.9,
    )


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Run a full pass over the loader and return (mean loss, metrics dict)."""
    model.eval()
    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            before = batch["before"].to(device)
            after = batch["after"].to(device)
            label = batch["label"].to(device)
            valid = batch["weights"].to(device)

            logits = model(before, after)
            loss = loss_fn(logits, label, valid)
            losses.append(float(loss.item()))

            prob = torch.sigmoid(logits.squeeze(1))
            preds.append((prob > 0.5).cpu().numpy())
            targets.append(label.cpu().numpy())
            weights.append(valid.cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0)
    targets_arr = np.concatenate(targets, axis=0)
    weights_arr = np.concatenate(weights, axis=0)
    metrics = compute_segmentation_metrics(preds_arr, targets_arr, weights_arr)
    return float(np.mean(losses)) if losses else 0.0, metrics.as_dict()


def train(cfg: TrainConfig) -> TrainOutcome:
    """Run a full training session and return the outcome record."""
    set_global_seeds(cfg.seed)
    device = select_device(cfg.device)
    log.info("Using device: %s", device)

    manifest = DatasetManifest.read(cfg.manifest_path)
    log.info(
        "Loaded manifest with %d tiles (source=%s, region=%s).",
        len(manifest.tiles),
        manifest.source,
        manifest.region_name,
    )

    train_ds = ChangeDetectionDataset(
        manifest, "train", root=cfg.manifest_path.parent, augment=cfg.augment_train
    )
    val_ds = ChangeDetectionDataset(manifest, "val", root=cfg.manifest_path.parent)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = SiameseUNet(cfg.model).to(device)
    log.info("Model parameter count: %d", model.parameter_count())

    loss_fn = CombinedLoss(
        CombinedLossWeights(
            dice=cfg.loss.dice_weight,
            bce=cfg.loss.bce_weight,
            pos_weight=cfg.loss.pos_weight,
        )
    ).to(device)
    optimizer = _build_optimizer(model, cfg)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / cfg.checkpoint_name

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    best_val_iou = -1.0

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params(
            {
                "name": cfg.name,
                "seed": cfg.seed,
                "device": str(device),
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "optimizer": cfg.optimizer.name,
                "lr": cfg.optimizer.lr,
                "weight_decay": cfg.optimizer.weight_decay,
                "dice_weight": cfg.loss.dice_weight,
                "bce_weight": cfg.loss.bce_weight,
                "pos_weight": cfg.loss.pos_weight,
                "in_channels": cfg.model.in_channels,
                "base_channels": cfg.model.base_channels,
                "depth": cfg.model.depth,
                "parameter_count": model.parameter_count(),
                "n_train_tiles": len(train_ds),
                "n_val_tiles": len(val_ds),
                "data_source": manifest.source,
            }
        )

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running: list[float] = []
            for step, batch in enumerate(train_loader, start=1):
                before = batch["before"].to(device)
                after = batch["after"].to(device)
                label = batch["label"].to(device)
                valid = batch["weights"].to(device)

                logits = model(before, after)
                loss = loss_fn(logits, label, valid)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running.append(float(loss.item()))
                if step % cfg.log_every == 0:
                    log.debug(
                        "epoch=%d step=%d loss=%.4f",
                        epoch,
                        step,
                        float(np.mean(running[-cfg.log_every :])),
                    )

            train_loss = float(np.mean(running)) if running else float("nan")
            val_loss, val_metrics = _evaluate(model, val_loader, loss_fn, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            log.info(
                "epoch=%d/%d train_loss=%.4f val_loss=%.4f val_iou=%.4f val_f1=%.4f",
                epoch,
                cfg.epochs,
                train_loss,
                val_loss,
                val_metrics["iou"],
                val_metrics["f1"],
            )

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "in_channels": cfg.model.in_channels,
                            "base_channels": cfg.model.base_channels,
                            "depth": cfg.model.depth,
                        },
                        "epoch": epoch,
                        "val_iou": best_val_iou,
                        "data_source": manifest.source,
                    },
                    ckpt_path,
                )
                mlflow.log_metric("best_val_iou", best_val_iou, step=epoch)
                log.info("New best checkpoint saved to %s (val_iou=%.4f).", ckpt_path, best_val_iou)

        mlflow.log_artifact(str(ckpt_path))

    return TrainOutcome(
        best_val_iou=best_val_iou,
        epochs_trained=cfg.epochs,
        checkpoint_path=ckpt_path,
        parameter_count=model.parameter_count(),
        device=str(device),
    )
