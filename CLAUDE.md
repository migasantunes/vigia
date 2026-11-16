# vigia: working notes for Claude Code

Wildfire smoke detection for fixed lookout cameras. Owner: Miguel (MSc AI student).
This is a portfolio project he must be able to explain line by line in an interview.

## Read first

1. `docs/PLAN.md`: phases, tasks, exit criteria. Work on the phase marked **next**.
2. `docs/ARCHITECTURE.md`: module boundaries and data contracts. Binding.
3. `docs/EVALUATION.md`: metric definitions. Binding.
4. `docs/DATA.md`: what is verified about each dataset and what is not.
5. `docs/DECISIONS.md`: why things are the way they are.

## How to work here

- **One phase at a time.** Do not start a phase before the previous exit criterion is
  met. Update the status table in `docs/PLAN.md` when a phase closes.
- **Explain before building.** At the start of each phase, give Miguel a short theory
  and practice briefing on what is about to be built and why, then build.
- **Contracts first.** If a change needs a different interface or metric definition,
  edit the doc first, then the code and tests in the same commit.
- **Never invent dataset facts.** If `docs/DATA.md` says "to verify", verify it against
  the real files or source and update the doc before writing code that depends on it.
- **No results without a run.** Never write a number into the README or docs that did
  not come out of code in this repo. Placeholders stay labelled as placeholders.

## Rules that protect the results

- Thresholds, alert-policy settings and checkpoints are chosen on validation data and
  frozen before the test set is used. Never tune on test.
- Every experiment calls `assert_disjoint` on its split and logs `assignment_hash`.
- Every `Scorer` must pass `check_causality`: a score at time t uses frames <= t only.
- Models output scores. Alerts come only from `vigia.eval.alerting`, which the service
  also uses. Do not add alert logic anywhere else.
- Report uncertainty: bootstrap over fires, Poisson interval on false-alarm rates.

## Engineering rules

- Core (`vigia.data.schema|splits|tiling`, `vigia.eval`) imports only numpy, pandas and
  the standard library. Heavy imports (cv2, torch, timm, fastapi, onnxruntime) go
  inside functions or behind `TYPE_CHECKING`; `tests/test_imports.py` enforces this.
- Tests for code needing an extra carry the matching marker: `vision`, `train`,
  `serve`, `data`.
- No AGPL dependencies (no Ultralytics).
- Never commit images, caches, weights or anything under `data/`, `runs/`, `mlruns/`.
- Seeds are explicit arguments. No global random state.

## Commands

```bash
pip install -e ".[dev]"          # core + tooling
pip install -e ".[dev,vision,train,serve]"   # everything
make check                       # ruff + mypy + core tests: must pass before commit
make test-all                    # includes tests that need extras / data
vigia --help
```

## Where things stand

Implemented and tested: manifest schema and validation, group-aware splits and leakage
audit, tiling, alert policy, metrics, harness, config loader, CLI skeleton, `Scorer`
protocol and causality check.

Stubs raising `NotImplementedError` with their phase number: dataset adapters,
transforms, classical baseline, both models, training loop, serving.

Next action: Phase 0, task 1 (FIgLib terms), then the download and the adapter.
