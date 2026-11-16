"""Evaluation: the alert policy, event-level metrics and the evaluation harness (core layer)."""

from vigia.eval.alerting import AlertConfig, AlertPolicy, run_policy
from vigia.eval.harness import evaluate, sanity_baselines
from vigia.eval.metrics import (
    FrameMetrics,
    ScoredSequence,
    SequenceOutcome,
    Summary,
    bootstrap_over_sequences,
    evaluate_sequence,
    frame_metrics,
    operating_curve,
    paired_bootstrap_difference,
    poisson_rate_interval,
    summarise,
)

__all__ = [
    "AlertConfig",
    "AlertPolicy",
    "FrameMetrics",
    "ScoredSequence",
    "SequenceOutcome",
    "Summary",
    "bootstrap_over_sequences",
    "evaluate",
    "evaluate_sequence",
    "frame_metrics",
    "operating_curve",
    "paired_bootstrap_difference",
    "poisson_rate_interval",
    "run_policy",
    "sanity_baselines",
    "summarise",
]
