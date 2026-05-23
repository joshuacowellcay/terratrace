# TerraTrace — handover notes

These notes are for you, the owner — not the recruiter audience. They cover what each
component does, the three or four design calls that took the most thought, how to switch from
the synthetic data path to real Sentinel-2 + Hansen GFC, how to record the demo GIF, how to
do the longer GPU training run, how to publish this as a standalone GitHub repo, how to host a
live demo, and a short set of interview questions with crisp answers.

---

## What is where

```
src/terratrace/
  data/        STAC fetch · synthetic fallback · cloud mask · tiling · geographic split
  models/      Siamese U-Net (twin encoder, shared weights) · NDVI baseline
  training/    losses (Dice + weighted BCE) · metrics · train loop · evaluate
  inference/   FastAPI · predict pipeline · file-backed tile cache
configs/       pipeline.yaml (real region + synthetic params) · fast.yaml · full.yaml
scripts/       build_dataset · train · evaluate · serve (one-command entry points)
tests/         pytest — offline-only, no GPU, no network
frontend/      Next.js 14 + MapLibre + Tailwind, calls /predict
data/sample/   six committed tiles (the offline demo path)
checkpoints/   fast.pt — what's served by default
results/       metrics.json — README pulls every headline number from here
assets/        figures + the demo.gif placeholder
```

## The four decisions that matter most

1. **Geographic-block splitting** instead of random tile shuffle. Most "easy" remote-sensing
   wins are leakage: adjacent tiles share clouds, sensor BRDF, even physical objects across
   their borders, so a random split lets the model memorise local context that won't transfer.
   `src/terratrace/data/splits.py` divides the tile grid into a 4×4 macro-block grid and
   assigns whole blocks (not individual tiles) to train/val/test, and `tests/test_splits.py`
   enforces a hard no-overlap invariant. This is the single most important honesty knob in
   the repo.

2. **Imbalance-aware loss.** Forest-loss pixels are a small minority per tile (often a few
   percent). Plain BCE settles into "predict 0 everywhere"; pixel accuracy looks fine; IoU
   is 0. The combined Dice + weighted BCE in `src/terratrace/training/losses.py` optimises
   the overlap statistic directly (Dice) and keeps gradients informative away from saturation
   (weighted BCE with `pos_weight=5`). Cloudy pixels are zero-weighted via the per-pixel
   validity mask so they never produce signal.

3. **NDVI baseline as the comparator.** Without a non-ML baseline the deep model's number
   is meaningless — IoU 0.5 might be impressive or terrible depending on prior. The
   NDVI-difference baseline in `src/terratrace/models/baseline.py` is what you'd build in an
   afternoon with `numpy` alone; its threshold is *calibrated on the val split* so we don't
   handicap it. The headline claim is the gap (+7.5 IoU, widening to +19 on cloudy tiles),
   not the absolute deep number.

4. **Synthetic-data fallback as a first-class path.** The real STAC fetcher and the
   synthetic generator both emit the same `Tile` type; everything downstream is
   source-agnostic. This means the build never silently half-fails: if Planetary Computer
   is unreachable, we fall back to a deterministic procedural generator and **disclose it in
   the manifest, the model card, and the README**. The price is some honesty in the
   demo metrics; the benefit is the build is reproducible on any laptop.

## Switching to the real-data path

```bash
# 1. Make sure you have outbound HTTPS to planetarycomputer.microsoft.com and storage.googleapis.com.
# 2. Edit configs/pipeline.yaml — change `fetch_mode: auto` to `fetch_mode: real`
#    (auto already prefers real and falls back; setting `real` makes it fail loudly instead).
# 3. Rebuild.
unset TERRATRACE_FORCE_SYNTHETIC
python scripts/build_dataset.py
python scripts/train.py --config configs/fast.yaml
python scripts/evaluate.py
```

The Hansen GFC granule is pulled directly from the
`storage.googleapis.com/earthenginepartners-hansen/` public bucket — no key required. Watch
the manifest's `source` field; it should flip from `"synthetic"` to `"real"`.

If you keep the synthetic build for the published portfolio version, update MODEL_CARD.md's
*Important caveat — synthetic data was used* note to reflect that decision rather than removing
it. The honesty is the point.

## Recording the demo GIF

The placeholder file at `assets/demo.gif.TODO` has the full capture script. The short version:

1. Two terminals: one running the backend (`python scripts/serve.py --port 8000`), one
   running the frontend (`cd frontend && npm run dev`).
2. Open `http://localhost:3000`, click two corners somewhere over Rondonia on the basemap.
3. Leave the default dates, click **Compute forest-loss**.
4. After the result appears, click through the overlay buttons (`before` → `after` → `mask`
   → `confidence`) so the GIF actually shows what the model is doing.
5. Capture with Kap, LICEcap, or
   `ffmpeg -i screen.mov -vf "fps=12,scale=720:-1" -loop 0 assets/demo.gif`.
6. Keep the GIF under ~5 MB so GitHub renders it inline. If it's larger, host externally
   and update the README link.
7. `git rm assets/demo.gif.TODO && git add assets/demo.gif && git commit -m "docs: add demo gif"`.

## Doing a longer / GPU training run

```bash
# Edit configs/full.yaml if you want different epoch / depth / lr settings (defaults are sane).
python scripts/train.py --config configs/full.yaml
python scripts/evaluate.py --checkpoint checkpoints/full.pt
```

`full.yaml` widens the model (base_channels 48 vs 24), goes deeper (depth 5 vs 4), and runs
80 epochs at batch size 16. On a single modern GPU it takes a few hours; on CPU it's not
recommended. `checkpoints/full.pt` is git-ignored by default — leave it that way to keep the
repo light; if you want to ship it, push to a release artifact rather than committing the
binary.

## Publishing this folder as its own GitHub repo

The repo is already self-contained — the `~/Desktop` → `~/Code` move done during the build
left no broken references — so publishing is just:

```bash
cd ~/Code/terratrace
gh repo create terratrace --public --source=. --remote=origin --description "Forest-loss change detection: Siamese U-Net + FastAPI + Next.js"
git push -u origin main
```

(or via the web UI if you prefer). Pin it on your GitHub profile.

After the first push, watch the Actions tab — `.github/workflows/ci.yml` will run lint +
tests on backend + frontend; it should go green.

## Deploying a live demo

* **Frontend → Vercel.** From the GitHub repo, "import" the `frontend/` directory as a Next.js
  project. Set `NEXT_PUBLIC_API_URL` to the public backend URL (see below). Vercel handles
  the rest; the standalone build output is already enabled in `next.config.js`.

* **Backend → Railway / Render / Fly.** Pick whichever has the longest free tier you don't
  mind running out. The repo includes `Dockerfile.backend` and `docker-compose.yml`; both
  Railway and Render can build straight from the Dockerfile. Render: New → Web Service →
  Docker → set start command to `uvicorn terratrace.inference.service:app --host 0.0.0.0 --port $PORT`
  and ensure the image bakes `checkpoints/fast.pt` in. Set `TERRATRACE_FORCE_SYNTHETIC=1`
  for the public demo if you don't want to deal with STAC failures during recorded usage.

* **CORS:** the service uses an open CORS policy in dev. For the production demo, tighten
  `allow_origins` in `src/terratrace/inference/service.py` to your Vercel domain before pushing.

## Things you may want to fix before publishing

* The macOS UF_HIDDEN flag occasionally lands on the editable install's `.pth` file in the
  local `.venv`, breaking `import terratrace`. Already worked around with `tests/conftest.py`,
  which adds `src/` to `sys.path` defensively. If you ever see "ModuleNotFoundError:
  No module named 'terratrace'" after a reinstall, run
  `chflags nohidden .venv/lib/python3.12/site-packages/__editable__.terratrace-0.1.0.pth`
  to undo it. This bug never travels to GitHub — the flag is local file-system metadata.
* The Map component uses MapLibre's free `demotiles.maplibre.org` style. It's fine for the
  demo but rate-limited; before shipping a public live demo, swap for a MapTiler / Carto /
  Stadia key (or a self-hosted vector tileset).
* If you switch to real data, the model's `base_channels: 24` may underfit. Bump to 32 in
  `configs/fast.yaml` or move to `configs/full.yaml`.

## Likely interview questions and crisp answers

* **"How do you know your test split isn't leaking?"** `tests/test_splits.py` asserts no
  tile coordinate appears in more than one split, and `tests/test_pipeline.py` runs the full
  build pipeline end-to-end and re-asserts the invariant. The split itself is
  geographic-block: the tile grid is partitioned into a 4×4 macro-block grid and whole blocks
  are assigned to one of train/val/test — adjacent tiles can never end up in different
  splits.

* **"Why an NDVI baseline?"** NDVI differencing is what you'd build in an afternoon with
  numpy alone. It's the right comparator because (a) it's well-understood and reproducible,
  (b) any deep-model win over it represents the value of learning context that single-pixel
  thresholds can't see, and (c) it lets us *headline a gap* (deep − baseline) rather than an
  absolute number that depends on dataset difficulty. The threshold is calibrated on val so
  we don't handicap the baseline.

* **"What does IoU 0.95 mean here?"** Of all pixels predicted as forest-loss plus all pixels
  that actually are forest-loss, ~95 % are in both sets. That's the standard binary-segmentation
  metric for the minority class. We headline it instead of accuracy because the class is
  imbalanced — a "predict zero everywhere" baseline would already score ~95 % accuracy with
  IoU 0. We also report precision and recall separately so the trade-off is visible.

* **"How would you detect silent failure on a new region?"** Three things in order of bang
  for buck: (1) watch the cloud-fraction-stratified IoU break down on a small labelled batch
  from the new region — if the gap to the original distribution widens, the model is
  out-of-distribution; (2) compare model predictions to the NDVI baseline on the new region
  and flag tiles where they disagree by more than X %; (3) calibration — check whether the
  predicted-probability bins still match the observed positive fraction. Each of these is
  cheaper than ground-truth re-labelling.

* **"Why a Siamese architecture rather than concatenating before/after at the input?"**
  Symmetry. A single encoder with weights shared between the two inputs sees both
  timestamps through the same lens, which makes the model's behaviour invariant to which one
  is labelled "before" — a useful property if you ever want to use the same encoder for
  triplet-loss representation pretraining. Concatenating at the input would also work for
  change detection in practice and is simpler; the trade-off is mostly about how cleanly you
  can plug the encoder into other tasks later.
