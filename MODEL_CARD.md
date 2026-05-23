# TerraTrace model card

A short, candid description of the trained Siamese U-Net change-detection model that ships
with this repository as `checkpoints/fast.pt`. Numbers in this card come from
`scripts/evaluate.py` against the geographic test split — never edited by hand. Re-run
`scripts/evaluate.py` and this card's numbers will need updating; the `results/metrics.json`
file is the source of truth.

---

## Important caveat — synthetic data was used

The shipped checkpoint was trained and evaluated on a **procedurally-generated synthetic
dataset**, not on real Sentinel-2 imagery, because the build was run in an environment where
the real-data fetch from the Microsoft Planetary Computer was either unavailable or disabled
(`TERRATRACE_FORCE_SYNTHETIC=1`). The synthetic generator (`src/terratrace/data/synthetic.py`)
produces multi-band before/after image pairs with class-imbalanced forest/bare/water regions,
cloud cover, seasonal brightness shifts, and a known ground-truth deforestation mask.
Everything downstream of the data pipeline — model, loss, training loop, evaluation, inference
service, frontend — was designed source-agnostic: switching to the real pipeline is a single
config change (`fetch_mode: real` in `configs/pipeline.yaml`) followed by re-running
`scripts/build_dataset.py && scripts/train.py && scripts/evaluate.py`.

The numbers below are therefore **realistic upper bounds, not field-validated results**.
Treat the deep-vs-baseline gap as a load-bearing claim about model design and the absolute
IoU as a property of the synthetic generator's difficulty.

---

## Model summary

| Property | Value |
| --- | --- |
| Architecture | Siamese U-Net (twin encoder with shared weights, channel-concat skip fusion) |
| Input | Two 4-channel (R, G, B, NIR) 128×128 patches: "before" and "after" |
| Output | Per-pixel forest-loss probability (single-channel logits + sigmoid) |
| Parameters | 5,521,081 (5.5M) |
| Loss | 0.6 × soft Dice + 0.4 × weighted BCE (positive-class weight 5.0), cloud-masked |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Training run | 20 epochs, batch size 8, seed 42 |
| Device | CPU is the contract (<30 min); auto-detects MPS / CUDA when available |

The encoder is reused (`self.encoder` is one `nn.Module`, called twice). At each spatial
scale the encoder produces a feature map for "before" and "after"; the decoder receives the
channel-wise concatenation of the pair as its skip connection. The bottleneck takes the
concat of the pooled deepest features. See `src/terratrace/models/siamese_unet.py` for the
docstring rationale (why concat over subtract, why shared weights, why no ImageNet
pretraining on this data).

## Intended use

* Detect deforestation between two timestamps over the same AOI from medium-resolution
  optical imagery (Sentinel-2 L2A or anything with comparable R, G, B, NIR bands).
* Drive demos and prototypes that combine an interactive map with an ML inference call —
  the inference service in `src/terratrace/inference/` reports both a per-pixel mask and a
  hectares-lost estimate.
* Serve as a reference implementation showing how to evaluate a change-detection model
  honestly: geographic-block splits, baseline benchmark, calibration check.

## Out-of-scope uses

* **Operational deforestation monitoring** at scale, or anything with regulatory or
  enforcement consequences. The model was not trained against an audited Hansen GFC label
  set in this build (synthetic fallback was used), and the single-region scope makes
  cross-AOI generalisation an open question.
* **Detecting changes other than forest loss**: the labels are binary "loss" / "no loss" —
  the model does not distinguish forest-loss-from-fire vs forest-loss-from-clearing, and
  it will silently classify water-level changes, urban expansion, or seasonal greening
  changes outside the trained distribution.
* **High-stakes hectares estimation**. Even with calibration the model's confident-positive
  fraction matches observed prevalence only at the extremes; mid-range probabilities are
  sparse and noisy.

## Training data

* Source: **synthetic** (see caveat above). With network access the real pipeline pulls
  Sentinel-2 L2A from Microsoft Planetary Computer + Hansen GFC v1.11 labels.
* Region: nominal AOI in Rondonia, Brazil (Amazon basin) — well-documented active
  deforestation. The synthetic generator is independent of geography but the manifest
  records the nominal region.
* Splits: **geographic blocks**, not random tiles. The 12×12 tile grid is divided into a
  4×4 macro-block grid; whole blocks are assigned to train (10 / 16), val (3 / 16), and
  test (3 / 16). The `tests/test_splits.py` and `tests/test_pipeline.py` files enforce a
  zero-overlap invariant so spatial leakage cannot regress silently.
* Class balance: forest-loss pixels are a minority (a few percent per tile, with a long
  tail of tiles with no loss). Handled at training time via Dice loss + weighted BCE.

## Evaluation

Held-out **geographically-separated** test split (27 tiles in the default fast build; 3
spatial macro-blocks the model never saw during training). Numbers from `results/metrics.json`:

| Metric (forest-loss class) | Siamese U-Net | NDVI baseline (t = 0.425) |
| --- | --- | --- |
| **IoU** | **0.958** | 0.883 |
| **Precision** | 0.959 | 0.897 |
| **Recall** | 1.000 | 0.983 |
| **F1** | 0.979 | 0.938 |

The NDVI baseline's threshold is calibrated on the val split for fairness. The deep model
wins by 7.5 IoU points overall; the gap widens to **19 points on cloud-affected tiles**
(`failure_breakdown` in the JSON), which is where the model's learned context most clearly
beats a single-pixel threshold. Calibration (`assets/calibration.png`) is good at the
extremes (which is where most predictions land) and noisy in the middle bins (which are
sparse).

## Limitations and failure modes

* **Synthetic-data ceiling.** The current numbers describe a generator whose forest-loss
  signal is large (forest → bare = high NDVI drop) and whose noise is bounded. Real
  Sentinel-2 data has subtler signals (selective logging, haze, BRDF effects, seasonal
  understorey changes) — expect lower absolute IoU after switching to the real path.
* **Cloudy pixels.** Pixels masked as cloudy in either timestamp are excluded from both
  training and evaluation loss (`valid_pixel_weights`). The model never sees them as
  training signal; predictions over heavy cloud should be discarded by the caller, and the
  inference service does so with a clear message rather than returning a wrong silent
  answer.
* **Single AOI.** The trained model has never seen another biome. Forest-loss patterns in
  Borneo (peat fires) or temperate forests (clear-cut blocks) would be out-of-distribution
  and demand retraining.
* **Class imbalance pushes toward over-detection.** Recall is essentially 1.0, precision
  ~0.96 — the model occasionally flags non-loss pixels as loss. Acceptable for a
  human-in-the-loop monitoring tool; not acceptable for unsupervised hectares accounting.
* **The committed checkpoint is from the "fast" config.** A wider+deeper "full" run
  (`configs/full.yaml`) on a GPU is expected to do better, especially on real data.

## How to reproduce / retrain

```bash
# 1. Build dataset (real if network is up, synthetic otherwise — manifest records which).
python scripts/build_dataset.py

# 2. Train.
python scripts/train.py --config configs/fast.yaml      # CPU-feasible
python scripts/train.py --config configs/full.yaml      # GPU-targeted

# 3. Evaluate. Overwrites results/metrics.json and the figures in assets/.
python scripts/evaluate.py
```

To switch to the real-data path: set `fetch_mode: real` in `configs/pipeline.yaml`,
ensure the host has outbound HTTPS, then re-run all three steps. The manifest at
`data/processed/manifest.json` will show `source: "real"` and include the bbox + date
windows used.

## Versioning

| Item | Version |
| --- | --- |
| Model checkpoint | `checkpoints/fast.pt` (model_state_dict + model_config embedded) |
| Trainer | `src/terratrace/training/train.py` |
| Architecture | `src/terratrace/models/siamese_unet.py` |
| Evaluation | `src/terratrace/training/evaluate.py` |
| Data manifest | `data/processed/manifest.json` |

All experiment metadata is also logged to a local MLflow store at `./mlruns/`.
