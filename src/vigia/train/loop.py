"""Training loop and run logging (plan Phases 3 and 4). Needs the ``train`` extra."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def train(config_path: Path, overrides: Sequence[str] = ()) -> Path:
    """Train one model from an experiment config and return the selected checkpoint path.

    Steps:

    1. ``vigia.config.load_config(config_path, overrides)`` gives the resolved config;
       the model section selects ``single_frame`` or ``temporal``.
    2. Load the manifest, build the split assignment for the configured protocol, call
       ``vigia.data.splits.assert_disjoint`` on it, and compute
       ``vigia.data.splits.assignment_hash``. The test split is not loaded.
    3. Seed every generator (Python, numpy, torch, DataLoader workers) from the config's
       explicit ``seed``; nothing reads global random state.
    4. Train; after every epoch, score the validation sequences and compute the validation
       operating curve with ``vigia.eval`` (same alert policy as everywhere else).
    5. Select the checkpoint on the validation metric named in the config (e.g. penalised
       TTD at the configured false-alarm budget), never on test data.

    Every run logs to MLflow:

    - the resolved config (as an artifact) and ``vigia.config.resolved_config_hash``;
    - the git commit (and whether the working tree was dirty);
    - the seed;
    - the split assignment hash from ``vigia.data.splits.assignment_hash`` and the
      ``assert_disjoint`` result;
    - per-epoch train and validation losses (tile and image terms separately);
    - the validation operating curve of each epoch and of the selected checkpoint.

    Returns:
        Path of the selected checkpoint, under the run directory.
    """
    raise NotImplementedError("Phase 3/4: training loop (train)")
