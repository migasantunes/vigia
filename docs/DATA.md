# Data

No image is stored in this repository. Datasets are downloaded to `data/` (git-ignored)
and converted to manifests (see `ARCHITECTURE.md` §4).

Each fact below is tagged **verified** (read from the source named) or **to verify**
(must be confirmed in Phase 0 before anything depends on it).

## FIgLib (primary benchmark)

Fire Ignition Library, from the HPWREN camera network in Southern California.
Source for the verified facts: Dewangan et al., *FIgLib & SmokeyNet: Dataset and Deep
Learning Model for Real-Time Wildland Fire Smoke Detection*, Remote Sensing 14(4):1007,
2022 (arXiv:2112.08598).

| Fact | Status |
|---|---|
| 315 fire sequences, 24,800 images | verified (paper) |
| 101 cameras at 30 stations, June 2016 to July 2021 | verified (paper) |
| About 81 images per sequence: roughly 40 min before to 40 min after ignition, about 60 s apart; 114 fires miss a few images | verified (paper) |
| Image sizes 1536 × 2048 or 2048 × 3072 | verified (paper) |
| Image-level smoke/no-smoke labels; first smoke frame marked by a human with full temporal context | verified (paper) |
| 45 fires excluded by the authors (night, black-and-white, questionable smoke), leaving 270 | verified (paper) |
| Official split by fire: 144 train / 64 validation / 62 test | verified (paper) |
| Bounding boxes and contour masks exist for 144 fires | verified (paper) |
| Whether cameras/stations are disjoint across the official split | **to verify** (the paper does not say; this is the Phase 0 audit) |
| Download location: the HPWREN FIgLib site | verified (paper), page not yet read |
| Licence / terms of use / required acknowledgement | **to verify** (paper states none) |
| Directory and filename convention, how the ignition offset is encoded | **to verify** on a real download |
| Total download size | **to verify** |
| Where the official fire lists and the box/contour annotations are published | **to verify** (the paper points to the authors' GitLab) |

## Pyro-SDIS (cross-continent test, protocol P3)

Images from Pyronear cameras installed with French fire and rescue services.
Source: the dataset card of `pyronear/pyro-sdis` on Hugging Face.

| Fact | Status |
|---|---|
| Licence Apache-2.0 | verified (dataset card) |
| v2.0: 12,378 images, 10,744 with smoke, 1,634 without (false positives of the Pyronear system), 12,137 smoke instances | verified (dataset card) |
| Splits: 11,249 train / 1,129 validation | verified (dataset card) |
| Per-image fields: image, annotations (YOLO boxes, class 0 = smoke), image_name, partner, camera, date, split | verified (dataset card) |
| Smoke sequences capped at 10 images each | verified (dataset card) |
| How images group into sequences, and whether a sequence id can be rebuilt from camera + date | **to verify** |
| No ignition time, so no time-to-detection on this dataset | follows from the above; confirm when the adapter is written |

## PYRONEAR-2025 (optional)

Source: Lostanlen et al., *Constructing a Real-World Benchmark for Early Wildfire
Detection with the New PYRONEAR-2025 Dataset* (arXiv:2402.05349).

| Fact | Status |
|---|---|
| Image set: 4,228 images (4,041 with smoke). Video set: 1,049 videos, 640 wildfires, 80,727 frames | verified (paper) |
| Countries: France, Spain, Chile, United States | verified (paper) |
| Bounding-box annotations of smoke plumes | verified (paper) |
| Splits keep wildfire events disjoint | verified (paper) |
| Paper's sequential baseline (detector + ResNet + LSTM over 4 frames): F1 0.822 vs 0.790 single-frame; time to detect 1.17 vs 1.76 min | verified (paper) |
| Distribution: a Google Drive folder linked from the paper's code repository | verified (repository page) |
| Licence for the data | **to verify**: none stated; the paper says "research and non-profit purposes" |

The abstract says about 50,000 images while the paper's table gives 80,727 frames for
the video set. Use the table, and note the discrepancy if citing.

Because the terms are unstated, this dataset is **not** part of the core plan. See
`DECISIONS.md` D4.

## Rules

- Never commit images, frame caches or model weights.
- Cite each dataset as its authors ask; record the exact wording here once read.
- Any new dataset enters through an adapter that returns a manifest passing
  `validate_manifest`, plus a row in this file with verified/to-verify status.
