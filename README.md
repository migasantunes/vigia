# vigia

Spotting wildfire smoke in the first minutes.

*Vigia* is Portuguese for a fire lookout. This project is a smoke detector for fixed
lookout cameras: it watches a stream of frames and raises an alert when a smoke column
appears, as early as possible and without firing on every cloud.

> **Status: scaffold.** The evaluation harness, data contracts and split tooling are
> implemented and tested. No model has been trained yet and there are no results. The
> build order is in [`docs/PLAN.md`](docs/PLAN.md).

## Why it is hard

- **The target is faint and small.** Early smoke is a translucent smudge on a tiny part
  of a multi-megapixel frame, and haze looks much the same.
- **One frame is often not enough.** Smoke gives itself away by how it rises and
  spreads over a few minutes.
- **False alarms end deployments.** A camera produces over a thousand frames a day; an
  operator who gets woken up for fog stops listening.
- **New towers break models.** Doing well on landscapes seen in training says little
  about a camera the model has never looked through.

## What is being built

| Piece | What it does |
|---|---|
| M0 classical baseline | OpenCV background subtraction and optical flow. No learning. |
| M1 single-frame model | PyTorch tile classifier with a pretrained CNN/ViT backbone. |
| M2 spatiotemporal model | Looks at the last few frames; the core of the project. |
| Evaluation harness | Time-to-detection and false alarms per camera-day, with confidence intervals. |
| Alert service | ONNX Runtime + FastAPI, with a replay of a real ignition. |

## How it is judged

Not by accuracy. By two numbers, reported together:

- **Time-to-detection:** minutes from ignition to the first alert.
- **False alarms per camera-day.**

Models are compared at the same false-alarm rate, on three protocols:

| Protocol | Test data | Question |
|---|---|---|
| P1 | FIgLib official test fires | Comparable to published results? |
| P2 | FIgLib stations held out of training | Does it work on an unseen tower? |
| P3 | Pyro-SDIS (France), no fine-tuning | Does it survive a new continent? |

Definitions are in [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Results

None yet. This section will hold the headline table, the operating curves and the
replay demo.

## Data

- **FIgLib**: 315 ignition sequences (24,800 images) from HPWREN cameras in Southern
  California. Primary benchmark.
- **Pyro-SDIS**: 12,378 images from Pyronear cameras in France, Apache-2.0.
  Cross-continent test.

No images are stored in this repository. Details and open questions are in
[`docs/DATA.md`](docs/DATA.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # core only: numpy, pandas, no PyTorch needed
make check                   # lint, type-check, tests
vigia version
```

Heavier stacks are opt-in: `.[vision]` (OpenCV), `.[train]` (PyTorch, timm, MLflow),
`.[serve]` (ONNX Runtime, FastAPI).

## Repository map

```
configs/      YAML configs for data, models, evaluation, experiments
docs/         PLAN, ARCHITECTURE, EVALUATION, DATA, DECISIONS
src/vigia/
  data/       manifest schema, group-aware splits, tiling, dataset adapters
  eval/       alert policy, metrics, harness
  classical/  OpenCV baseline
  models/     Scorer protocol, single-frame and temporal models
  train/      training loop
  serve/      ONNX export, runtime, FastAPI service, replay
tests/        mirrors src/vigia
```

## References

- Dewangan et al., *FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time
  Wildland Fire Smoke Detection*, Remote Sensing 14(4):1007, 2022. arXiv:2112.08598.
- Lostanlen et al., *Constructing a Real-World Benchmark for Early Wildfire Detection
  with the New PYRONEAR-2025 Dataset*. arXiv:2402.05349.
- Pyronear, *Pyro-SDIS* dataset, Hugging Face `pyronear/pyro-sdis`.
