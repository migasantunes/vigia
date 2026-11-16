# Work plan

Eight phases. Each has a goal, concrete tasks, and an exit criterion that can be checked.
Do them in order: every phase produces something the next one measures against.

**Status legend:** done = in the repo and tested; next = start here; later = not started.

| Phase | What | Status |
|---|---|---|
| 0 | Data, terms and the split audit | next |
| 1 | Evaluation harness | done (scaffold), wire-up pending |
| 2 | Classical OpenCV baseline (M0) | later |
| 3 | Single-frame model (M1) | later |
| 4 | Spatiotemporal model (M2) | later |
| 5 | Generalisation: unseen stations, unseen continent | later |
| 6 | Alert service and replay demo | later |
| 7 | Write-up | later |

The project's claim, which every phase serves:

> On cameras it has never seen, a spatiotemporal model alerts X minutes sooner than a
> single-frame model at the same false-alarm rate.

If the data says otherwise, that is the result that gets written up.

---

## Phase 0: data, terms and the split audit

**Goal:** real frames on disk, a valid manifest, and the first finding.

Tasks

1. Read the FIgLib site's terms and citation requirements; record them in `DATA.md`.
   The paper calls the data publicly available but states no licence.
2. Download FIgLib. Inspect the real directory and file naming, document it in
   `DATA.md`, then implement `vigia.data.figlib.build_figlib_manifest` with tests built
   from real example names. Do not guess the convention.
3. Get the official fire lists (train 144 / val 64 / test 62) from the SmokeyNet
   authors' published split and load them with `assignment_from_lists`.
4. Decide the working resolution and build a resized frame cache (see compute budget).
5. EDA notebook: frames per fire, ignition offsets, image sizes, cameras per station,
   time of day, smoke box area as a fraction of the frame.
6. **Split audit:** run `leakage_report` on the official split for `camera_id` and
   `station_id`. How many test fires are filmed by a camera that also appears in train?
7. Re-check the reference numbers in `EVALUATION.md` §8 against the journal PDF.
8. Decide whether PYRONEAR-2025 can be used: it is distributed through a Google Drive
   link with no stated licence. If the terms are unclear, email the authors or leave it
   out (see `DECISIONS.md` D4).
9. Look for a source of long, continuous no-fire footage for false-alarm measurement.

**Exit:** `vigia manifest figlib` writes a manifest that passes `validate_manifest`;
`vigia audit-split` prints the camera/station overlap table; `DATA.md` has no
"unverified" items left for FIgLib.

## Phase 1: evaluation harness

**Goal:** trustworthy numbers before any model exists.

Already in the repo: alert policy, TTD, false alarms per camera-day with exact Poisson
intervals, operating curve, bootstrap, sanity baselines, group splits and leakage checks.

Tasks

1. Wire `vigia evaluate` to `vigia.eval.harness` (scores file + manifest -> report).
2. Run the three sanity baselines on the real FIgLib manifest; confirm the numbers are
   what `EVALUATION.md` §9 predicts.
3. Report writer: one Markdown table and one operating-curve plot per experiment.

**Exit:** `vigia evaluate` on the oracle scores prints detection rate 1.0, TTD 0, zero
false alarms on the real manifest.

## Phase 2: classical baseline (M0)

**Goal:** answer "do we even need deep learning?" with a number.

Tasks

1. Implement `vigia.classical.motion.MotionScorer`: MOG2 background subtraction,
   morphology, Farnebäck optical flow, keep blobs with slow upward motion and low
   contrast, score from surviving area.
2. Tune its handful of parameters on validation fires only.
3. Evaluate on P1 test. Save a gallery of what it fires on (clouds, vehicles, light).

**Exit:** M0 row in the results table with an operating curve; a short list of its
failure causes that motivates the learned models.

## Phase 3: single-frame model (M1)

**Goal:** a solid learned baseline, comparable to the paper's single-frame rows.

Tasks

1. Tile dataset and augmentations (`data.transforms`): haze, brightness/contrast,
   colour temperature, JPEG artefacts, horizontal flip.
2. `models.single_frame`: timm backbone on tiles, tile logits, pooled image score.
   Start with ResNet34 to match the paper, then one ViT backbone.
3. Training loop with MLflow logging; checkpoint chosen on a validation metric fixed
   in the config.
4. Pick the false-alarm budget for the headline on validation and freeze it.
5. Optional: CLIP zero-shot tile scorer as a no-training reference row.

**Exit:** P1 frame metrics within a few points of the paper's ResNet34 single-frame
row (79.4 accuracy, 78.9 F1), or a written explanation of the gap. Operating curve and
TTD at the frozen false-alarm budget.

## Phase 4: spatiotemporal model (M2)

**Goal:** the core contribution. Show what time adds.

Tasks

1. `models.temporal`: tile features over the last T frames, a temporal module per
   tile, optional spatial aggregation across tiles, tile and image heads.
2. Ablation grid, one change at a time:
   - context length T in {1, 2, 4}
   - temporal module: frame difference only, GRU, transformer
   - spatial aggregation on or off
   - tile loss on or off
3. Causality test (`check_causality`) on the trained scorer.
4. Compare with M1 at matched false-alarm rate using the paired bootstrap.

**Exit:** ablation table on P1 with confidence intervals; a clear statement of which
component buys what; reference comparison with SmokeyNet (83.49 accuracy, 82.59 F1).

## Phase 5: generalisation

**Goal:** the result nobody has published for this dataset.

Tasks

1. P2: leave-stations-out folds on FIgLib for M0, M1, M2. Report the drop from P1.
2. P3: implement the Pyro-SDIS adapter; evaluate FIgLib-trained models on French
   cameras without fine-tuning. Pyro-SDIS sequences are capped at 10 images and have no
   ignition time, so P3 reports detection metrics (precision, recall, F1) and the
   false-positive rate on its 1,634 no-smoke images, not TTD.
3. Failure analysis: which stations break the model, and what do they look like?
4. If time allows: does mixing a little target-domain data close the gap?

**Exit:** the P1 vs P2 vs P3 table, with the size of the generalisation gap stated in
one sentence.

## Phase 6: alert service and replay demo

**Goal:** it runs as a system, not a notebook.

Tasks

1. `serve.export`: ONNX export with a parity test against PyTorch outputs.
2. `serve.runtime.OnnxScorer` and `serve.service` (FastAPI), one `AlertPolicy` and
   frame buffer per camera.
3. `serve.replay`: play a held-out ignition through the service at accelerated speed.
4. Dockerfile; measure latency per frame on CPU.
5. Record the replay as a GIF for the README.

**Exit:** `docker run` + `vigia replay` shows an alert firing on a test fire, and the
alert minute equals what the offline harness computed for the same sequence.

## Phase 7: write-up

Tasks

1. README results section: headline table, operating curves, the replay GIF.
2. Model card: training data, intended use, known failure modes, what it must not be
   relied on for.
3. Failure gallery: false alarms and misses, grouped by cause.
4. If the results are good on Pyronear data, open a discussion with the Pyronear
   project about contributing.

**Exit:** someone who has never seen the project can read the README in five minutes
and say what was built, how well it works, and where it fails.

---

## Compute budget

| Item | Estimate | Note |
|---|---|---|
| FIgLib raw | tens of GB (24,800 multi-megapixel JPEGs) | check the real size in Phase 0 |
| Frame cache | resize to about 1040 × 1856, as in the paper | one-off CPU job |
| M1 training | hours on one consumer GPU | 45 tiles per frame; freeze early backbone stages if needed |
| M2 training | a few times M1 | cache tile features of a frozen backbone to make ablations cheap |

Estimates, not measurements: replace them with real numbers after the first run. A free
Kaggle or Colab GPU is enough if the local machine is not.

## Risks

| Risk | Mitigation |
|---|---|
| FIgLib terms forbid redistribution | Repo ships code and manifests' builders only, never images. Already the design. |
| Too little negative footage for a tight false-alarm rate | Report the Poisson interval and exposure; look for continuous footage in Phase 0. |
| M2 does not beat M1 | Still a result. The ablations and the P2 analysis carry the project. |
| TTD resolution is one minute | Say so; compare with paired bootstrap, not by eyeballing decimals. |
| PYRONEAR-2025 terms unclear | Optional. P3 runs on Pyro-SDIS, which is Apache-2.0. |
