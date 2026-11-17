# Decisions

Short records of choices that shape the project. Add a new entry instead of editing an
old one when a decision changes.

## D1. The headline metric is time-to-detection at a fixed false-alarm rate

Frame accuracy rewards a model for being right on the easy frames forty minutes into a
fire. What matters operationally is how soon the first alert comes and how often the
system cries wolf. All model comparisons are made at matched false alarms per
camera-day. Frame metrics are kept only to sit next to published numbers.

## D2. Models produce scores; one shared policy produces alerts

Every model implements `Scorer` and returns a number per frame. Alerts come from
`vigia.eval.alerting`, which is imported by both the evaluation harness and the
service. This removes the usual gap between "the metric in the notebook" and "what the
deployed thing does".

## D3. Generalisation is tested by station, not by fire

The published FIgLib split is disjoint by fire. A model can still have seen the same
hillside from the same camera during training. The project adds a leave-stations-out
protocol and reports the gap. Station, not camera, is the default group because cameras
on one tower share weather, haze and much of the landscape.

## D4. FIgLib is the core; Pyro-SDIS is the cross-domain test; PYRONEAR-2025 is optional

FIgLib has full ignition sequences and published baselines, so it carries the
time-to-detection results. Pyro-SDIS is Apache-2.0 and comes from a different continent
and camera network, so it carries the cross-domain test. PYRONEAR-2025 has video and
would be the best second sequence dataset, but its data licence is not stated, so it
stays out until the terms are confirmed in writing.

## D5. No AGPL dependencies

Ultralytics YOLO is AGPL-3.0. To keep the repository free to license as the owner
chooses, backbones come from timm and torchvision, and any detector baseline uses
torchvision or Hugging Face Transformers.

## D6. The core installs without PyTorch

Data contracts, splits, metrics and the alert policy depend only on numpy and pandas.
CI runs in seconds, the evaluation code can be reviewed and trusted on its own, and the
service does not need PyTorch at all (it runs ONNX Runtime).

## D7. Tiles, not whole frames

Smoke is small: annotated plumes average under 2% of the frame in the PYRONEAR-2025
paper's datasets, and early smoke is smaller still (measure it on FIgLib in Phase 0). Downscaling
the whole frame to a classifier's input size erases it. Models see overlapping
224-pixel tiles and aggregate, as SmokeyNet does; the exact grid is a config value.

## D8. The code is licensed Apache-2.0

Chosen over MIT for two reasons: it carries an explicit patent grant, and it matches the
licence used by Pyronear's code and the Pyro-SDIS dataset, which keeps a future
contribution upstream simple. The licence covers this repository's code only. Datasets
keep their own terms (see `DATA.md`) and are never redistributed here.

## Open decisions (owner's call)

- **Experiment tracker.** MLflow is assumed (local, no account). Weights & Biases is a
  drop-in alternative if a hosted dashboard is preferred.
- **Working resolution.** Default is the paper's 1040 × 1856 crop. Revisit after the
  Phase 0 measurement of smoke size.
