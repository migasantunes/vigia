# Architecture

This document is the contract for the codebase. Module boundaries, data shapes and
metric definitions are fixed here; change this file first when one of them changes.

## 1. System in one picture

```
 camera frames ──► data adapters ──► manifest (one row per frame)
                                         │
                         ┌───────────────┼──────────────────┐
                         ▼               ▼                  ▼
                   splits (by fire,   tiling +         scorers
                   by station)        transforms       (classical | single-frame | temporal)
                                                            │  per-frame score in [0, 1]
                                                            ▼
                                                     alert policy  ◄── the SAME code in
                                                            │           evaluation and serving
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                      evaluation harness            alert service
                                  (TTD, false alarms/camera-day)   (ONNX + FastAPI, replay)
```

Two rules shape everything:

1. **A scorer only produces a number per frame.** Every model, classical or deep, is a
   `Scorer`. Turning scores into alerts is the alert policy's job, never the model's.
2. **One alert policy, used twice.** `vigia.eval.alerting` is imported by both the
   evaluation harness and the service, so the numbers in the README describe the
   behaviour of the deployed system and not a lookalike.

## 2. Dependency layers

| Layer | Modules | May import |
|---|---|---|
| core | `vigia.data.schema`, `vigia.data.splits`, `vigia.data.tiling`, `vigia.eval.*` | numpy, pandas, stdlib |
| vision | `vigia.classical.*`, `vigia.data.transforms`, dataset adapters that open images | core + OpenCV, Albumentations |
| train | `vigia.models.*`, `vigia.train.*` | core + vision + PyTorch, timm, MLflow |
| serve | `vigia.serve.*` | core + vision + ONNX Runtime, FastAPI (no PyTorch) |

Core must import and test with only `numpy`, `pandas`, `pyarrow`, `pyyaml` installed.
Heavy imports inside non-core modules happen at call time or behind the matching
extra (`pip install -e ".[train]"`), and their tests carry the matching pytest marker.

## 3. Repository layout

```
configs/            YAML configs (data, model, train, eval, serve)
docs/               PLAN, ARCHITECTURE, DATA, EVALUATION, DECISIONS
scripts/            one-off entry points (download, build manifest, cache frames)
src/vigia/
  cli.py            `vigia` command (subcommands map 1:1 to plan phases)
  data/
    schema.py       manifest columns, validation
    splits.py       group-aware splits + leakage checks
    tiling.py       tile grid + box-to-tile labels
    transforms.py   augmentation pipelines            (vision)
    figlib.py       FIgLib adapter                    (vision)
    pyrosdis.py     Pyro-SDIS adapter                 (vision)
  eval/
    metrics.py      frame metrics, TTD, false alarms, operating curve, bootstrap
    alerting.py     streaming alert policy
    harness.py      scores + manifest -> full report
  classical/
    motion.py       OpenCV motion baseline            (vision)
  models/
    base.py         Scorer protocol
    single_frame.py tile classifier                   (train)
    temporal.py     spatiotemporal model              (train)
  train/
    loop.py         training loop, logging            (train)
  serve/
    export.py       PyTorch -> ONNX                   (train)
    runtime.py      ONNX Runtime scorer               (serve)
    service.py      FastAPI app                       (serve)
    replay.py       replay a recorded ignition        (serve)
tests/              mirrors src/vigia
```

## 4. Data contract: the manifest

Every dataset is converted once into a **manifest**: a pandas DataFrame (stored as
Parquet) with one row per frame. Nothing downstream reads dataset-specific layouts.

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `dataset` | str | no | `figlib`, `pyrosdis`, ... |
| `sequence_id` | str | no | One fire event or one continuous clip. Unique within `dataset`. |
| `camera_id` | str | no | One physical camera/view. |
| `station_id` | str | no | Physical site hosting one or more cameras. Equals `camera_id` if unknown. |
| `frame_idx` | int | no | 0-based position inside the sequence, increasing with time. |
| `timestamp` | datetime64[ns, UTC] | no | Capture time. |
| `offset_s` | float | yes | Seconds relative to the labelled ignition; negative before. Null if the sequence has no ignition time. |
| `label` | int8 | no | 1 = smoke visible, 0 = no smoke. |
| `path` | str | no | Image path relative to the dataset root. |
| `width`, `height` | int | no | Pixel size of the stored image. |
| `boxes` | list[list[float]] | yes | Smoke boxes `[x0, y0, x1, y1]`, normalised to [0, 1]. Empty list = annotated, no smoke box. Null = not annotated. |

Invariants, enforced by `validate_manifest`:

- `(dataset, sequence_id, frame_idx)` is unique.
- Within a sequence, `timestamp` is strictly increasing with `frame_idx`, and
  `camera_id` and `station_id` are constant.
- `label` is 0 or 1. Box coordinates are within [0, 1] with `x0 < x1`, `y0 < y1`.
- `offset_s` is all-present or all-null within a sequence; where present it is strictly
  increasing.
- Ids are compared across datasets as `(dataset, id)` pairs, since ids are only unique
  within a dataset.

## 5. Splits

`vigia.data.splits` only ever assigns **whole groups** to a split.

- `split_by_group(manifest, group_key, fractions, seed)` keeps every row of a group
  (a fire, a camera, a station) in one split.
- `leave_groups_out_folds(manifest, group_key, n_folds, seed)` yields train/test folds
  with disjoint groups, for leave-stations-out evaluation.
- `assert_disjoint(manifest, assignment, keys)` raises if any value of any key appears
  in more than one split. Every experiment calls it and logs the result.
- `leakage_report(manifest, assignment, keys)` counts shared values per key between
  each pair of splits. Used to audit the published FIgLib split, which is disjoint by
  fire but not documented as disjoint by camera.

Three protocols are used (details in `EVALUATION.md`):

| Protocol | Group key | Question it answers |
|---|---|---|
| P1 official | published FIgLib fire lists | Are we comparable to SmokeyNet? |
| P2 leave-stations-out | `station_id` | Does it work on a tower it has never seen? |
| P3 cross-dataset | `dataset` | Does it survive a different continent and camera network? |

## 6. Tiling

Early smoke covers a tiny part of a multi-megapixel frame, so models see tiles.

- `tile_grid(height, width, tile, overlap)` returns an `(N, 4)` int array of
  `[y0, x0, y1, x1]`. Tiles are all exactly `tile` × `tile`, the grid covers the whole
  image, and the last row/column is shifted inwards (never padded) when the size does
  not divide evenly.
- `boxes_to_tile_labels(boxes, grid, height, width, min_fraction)` returns an `(N,)`
  0/1 array: a tile is positive when the union of smoke boxes covers strictly more than
  `min_fraction` of it (so the default 0 means any overlap).

## 7. Scorer protocol

```python
class Scorer(Protocol):
    name: str
    context_frames: int  # how many past frames it needs (1 = single-frame)

    def score_sequence(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """frames: T images (H, W, 3, uint8, RGB), in time order.
        Returns shape (T,), float in [0, 1]: the score for frame t may use frames <= t only."""
```

Causality is part of the contract: a score at time *t* must not depend on frames after
*t*. `tests` include a causality check that perturbs future frames and asserts earlier
scores do not change.

| Scorer | Module | What it is |
|---|---|---|
| M0 classical | `classical.motion` | Background subtraction + dense optical flow; score from area of slow, upward, low-contrast motion. No learning. |
| M1 single-frame | `models.single_frame` | timm backbone on tiles, tile logits, pooled image score. |
| M2 temporal | `models.temporal` | Tile features across the last T frames, temporal module per tile, spatial aggregation across tiles, tile + image heads. |
| (optional) CLIP zero-shot | `models.single_frame` | Text-prompt similarity per tile. Zero training. |

## 8. Alert policy

`vigia.eval.alerting.AlertPolicy` is a small streaming state machine per camera:

- `threshold`: score at or above which a frame is "hot".
- `k`, `n`: fire an alert when at least `k` of the last `n` frames are hot.
- `cooldown_s`: after an alert, suppress further alerts for this long.

The window counts frames, not seconds. When an alert fires the window is cleared, so the
next alert needs `k` fresh hot frames; frames seen during the cooldown do count.

`policy.step(timestamp_s, score) -> bool` returns True exactly on the frame an alert
fires. `run_policy(timestamps_s, scores, config)` is the batch form and must give the
same result as feeding the frames one at a time (tested).

## 9. Metrics

Defined in `EVALUATION.md`; implemented in `vigia.eval.metrics`. The headline pair is
**time-to-detection** (minutes from labelled ignition to first alert) and **false alarms
per camera-day**. Frame-level accuracy/precision/recall/F1 are reported only so results
can sit next to the SmokeyNet paper.

## 10. Serving

`serve.export` writes an ONNX file; `serve.runtime.OnnxScorer` implements `Scorer`
without PyTorch; `serve.service` keeps one `AlertPolicy` and one frame buffer per camera
and exposes:

- `POST /cameras/{camera_id}/frames` (image + timestamp) -> `{score, alert}`
- `GET /cameras/{camera_id}/alerts`
- `GET /healthz`

`serve.replay` feeds a recorded FIgLib sequence through the running service at
accelerated speed and prints the minute the alert fired. That is the demo.

## 11. Configuration and reproducibility

- Every run is driven by one YAML file in `configs/` plus `--set key=value` overrides; the resolved
  config, git commit, split assignment hash and seed are logged with the run (MLflow).
- Seeds are explicit arguments. No function reads global random state.
- Raw data lives outside git under `data/` (ignored). `scripts/` rebuild everything
  derived from it.
