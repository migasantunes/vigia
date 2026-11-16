"""Evaluation harness: manifest + per-frame scores -> sequence outcomes and summary.

Each sequence is the group ``(dataset, sequence_id)`` of the manifest (ARCHITECTURE.md
§4), ordered by ``frame_idx``. A fresh :class:`~vigia.eval.alerting.AlertPolicy` is run
per sequence, so no alert state leaks between sequences.

Choices made where the contract is silent:

- ``scores`` align with manifest rows **positionally** (row ``i`` of the manifest gets
  ``scores[i]``); a pandas Series index is ignored.
- Timestamps are converted to seconds **relative to the first frame of the sequence**.
  Every metric and the policy use only time differences, so this loses nothing and
  avoids depending on the datetime unit (ns vs us) or timezone. A numeric ``timestamp``
  column is accepted and read as seconds.
- Outcome ids are ``f"{dataset}/{sequence_id}"`` and come out sorted by
  ``(dataset, sequence_id)``.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import numpy.typing as npt
import pandas as pd

from vigia.eval.alerting import AlertConfig
from vigia.eval.metrics import (
    SUMMARY_FIELDS,
    ScoredSequence,
    SequenceOutcome,
    Summary,
    evaluate_scored,
    summarise,
)

__all__ = ["REQUIRED_COLUMNS", "evaluate", "sanity_baselines", "scored_sequences"]

REQUIRED_COLUMNS: tuple[str, ...] = ("dataset", "sequence_id", "frame_idx", "timestamp", "label")


def _seconds(timestamps: pd.Series) -> npt.NDArray[np.float64]:
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        delta = timestamps - timestamps.iloc[0]
        return np.asarray(delta.dt.total_seconds(), dtype=np.float64)
    if pd.api.types.is_numeric_dtype(timestamps):
        return np.asarray(timestamps, dtype=np.float64)
    raise TypeError(f"timestamp column must be datetime or numeric, got {timestamps.dtype}")


def scored_sequences(
    manifest: pd.DataFrame, scores: pd.Series | npt.ArrayLike
) -> list[ScoredSequence]:
    """Split a manifest and positionally aligned scores into time-ordered sequences."""
    missing = [c for c in REQUIRED_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing columns {missing}")
    values = (
        scores.to_numpy(dtype=np.float64)
        if isinstance(scores, pd.Series)
        else np.asarray(scores, dtype=np.float64)
    )
    if values.ndim != 1 or values.shape[0] != len(manifest):
        raise ValueError(
            f"scores must be 1-D with one value per manifest row: got shape {values.shape} "
            f"for {len(manifest)} rows"
        )
    frame = manifest.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True)
    frame["_score"] = values
    sequences = []
    for (dataset, sequence_id), group in frame.groupby(["dataset", "sequence_id"], sort=True):
        group = group.sort_values("frame_idx", kind="stable")
        sequences.append(
            ScoredSequence(
                sequence_id=f"{dataset}/{sequence_id}",
                timestamps_s=_seconds(group["timestamp"]),
                labels=group["label"].to_numpy(dtype=np.int_),
                scores=group["_score"].to_numpy(dtype=np.float64),
            )
        )
    return sequences


def evaluate(
    manifest: pd.DataFrame,
    scores: pd.Series | npt.ArrayLike,
    config: AlertConfig,
    max_gap_s: float = 180.0,
) -> tuple[Summary, list[SequenceOutcome]]:
    """Run the alert policy on every sequence of ``manifest`` and summarise the result."""
    outcomes = evaluate_scored(scored_sequences(manifest, scores), config, max_gap_s)
    return summarise(outcomes), outcomes


def sanity_baselines(
    manifest: pd.DataFrame, config: AlertConfig, max_gap_s: float = 180.0
) -> pd.DataFrame:
    """The three sanity rows of EVALUATION.md §9, under the given policy ``config``.

    never_alert: every score 0 (never hot because threshold > 0); always_alert: every
    score 1; oracle: score = label. Returns one row per baseline: ``baseline`` followed by
    every Summary field. Raises ValueError when ``config.threshold == 0``, where a score
    of 0 is hot and "never alert" cannot be expressed.
    """
    if config.threshold == 0.0:
        raise ValueError("never-alert baseline needs threshold > 0")
    labels = manifest["label"].to_numpy(dtype=np.float64)
    baselines = {
        "never_alert": np.zeros(len(manifest), dtype=np.float64),
        "always_alert": np.ones(len(manifest), dtype=np.float64),
        "oracle": labels,
    }
    rows = []
    for name, scores in baselines.items():
        summary, _ = evaluate(manifest, scores, config, max_gap_s)
        rows.append({"baseline": name, **asdict(summary)})
    return pd.DataFrame(rows, columns=["baseline", *SUMMARY_FIELDS])
