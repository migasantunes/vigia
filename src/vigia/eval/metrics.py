"""Evaluation metrics, exactly as defined in docs/EVALUATION.md.

Frame-level metrics (§2) exist only for comparison with prior work. The headline
metrics are event-level: time-to-detection (§3) and false alarms per camera-day (§4),
computed from the alert events produced by :mod:`vigia.eval.alerting`.

Choices made where EVALUATION.md is silent (all deliberate):

- **Ignition** ``t0`` is the timestamp of the first frame with ``label == 1``. A sequence
  with no such frame has no fire.
- **Detection**: the first alert at a time ``t >= t0`` is the detection, whatever the
  label of the frame it fires on (smoke can thin out to a label-0 frame after ignition;
  an alert there still answers "did the operator get woken up for this fire"). Alerts
  after the detection are neither detections nor false alarms.
- **False alarms** are alerts at ``t < t0``, or every alert in a sequence with no fire.
  Every frame before ``t0`` has label 0 by construction of ``t0``, so this matches §4's
  "alert on a label-0 frame where no smoke is visible yet". Label-0 frames after ``t0``
  are not negative footage and alerts on them are never false alarms.
- **Negative exposure** of a sequence is the sum of the gaps between consecutive frames
  that are both before ``t0`` (all frames when there is no fire), each gap capped at
  ``max_gap_s``. The gap from the last pre-ignition frame to the ignition frame is not
  counted (its second frame is not negative). A sequence with a single negative frame
  therefore contributes 0 seconds; exposure is measured between frames, not per frame.
- **Penalised TTD** of a missed fire is ``(last timestamp - t0) / 60``: the time the
  detector stayed silent while smoke was on screen. A fire whose only labelled frame is
  the last frame and is missed gets a penalised TTD of 0; that is the data's resolution.
- **p90** uses numpy's default (linear interpolation) percentile.
- **F1** is computed as ``2 tp / (2 tp + fp + fn)``, so it is NaN only when that
  denominator is zero (no positive labels and no positive predictions). This equals
  ``2PR / (P + R)`` whenever both are defined, and is 0 (not NaN) when there are
  positives but no positive predictions, where precision is NaN.
- **Bootstrap**: resamples whose statistic is NaN (e.g. a resample containing no fires,
  for a TTD statistic) are dropped before taking percentiles. If every resample is NaN
  the interval is ``(nan, nan)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np
import numpy.typing as npt
import pandas as pd

from vigia.eval.alerting import AlertConfig, run_policy

__all__ = [
    "FrameMetrics",
    "ScoredSequence",
    "SequenceOutcome",
    "Summary",
    "bootstrap_over_sequences",
    "evaluate_sequence",
    "frame_metrics",
    "operating_curve",
    "paired_bootstrap_difference",
    "poisson_rate_interval",
    "summarise",
]

SECONDS_PER_DAY = 86_400.0
_NAN = float("nan")


def _ratio(num: float, den: float) -> float:
    return num / den if den != 0 else _NAN


# --------------------------------------------------------------------------- frame level


@dataclass(frozen=True)
class FrameMetrics:
    """Frame-level confusion counts and the metrics derived from them (§2)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def frame_metrics(labels: npt.ArrayLike, scores: npt.ArrayLike, threshold: float) -> FrameMetrics:
    """Accuracy, precision, recall and F1 of the predictions ``score >= threshold``.

    accuracy = (tp + tn) / N, precision = tp / (tp + fp), recall = tp / (tp + fn),
    F1 = 2 tp / (2 tp + fp + fn). A zero denominator gives NaN, never 0.
    """
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(f"labels and scores must be 1-D of equal length, got {y.shape}, {s.shape}")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    truth = y == 1
    pred = s >= threshold
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fn = int(np.sum(truth & ~pred))
    return FrameMetrics(
        accuracy=_ratio(tp + tn, tp + fp + tn + fn),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


# ------------------------------------------------------------------------ sequence level


@dataclass(frozen=True)
class SequenceOutcome:
    """Event-level result of one sequence.

    ``ttd_min``: minutes from ignition to the first alert at or after it; None if the
    fire was missed or there is no fire. ``penalised_ttd_min``: ``ttd_min`` if detected,
    minutes from ignition to the last frame if missed, None if there is no fire.
    ``false_alarms``: alerts in negative footage. ``negative_exposure_s``: seconds of
    negative footage (gaps capped at ``max_gap_s``).
    """

    sequence_id: str
    has_fire: bool
    detected: bool
    ttd_min: float | None
    penalised_ttd_min: float | None
    false_alarms: int
    negative_exposure_s: float


def evaluate_sequence(
    sequence_id: str,
    timestamps_s: npt.ArrayLike,
    labels: npt.ArrayLike,
    alerts: npt.ArrayLike,
    max_gap_s: float = 180.0,
) -> SequenceOutcome:
    """Detection, TTD, false alarms and negative exposure of one sequence (§3, §4).

    See the module docstring for how ignition, detection, false alarms and exposure are
    defined at the edges.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64)
    y = np.asarray(labels)
    al = np.asarray(alerts, dtype=np.bool_)
    if ts.ndim != 1 or ts.shape != y.shape or ts.shape != al.shape:
        raise ValueError(
            f"timestamps, labels and alerts must be 1-D of equal length, "
            f"got {ts.shape}, {y.shape}, {al.shape}"
        )
    if ts.size == 0:
        raise ValueError(f"sequence {sequence_id!r} has no frames")
    if not np.all(np.isfinite(ts)) or (ts.size > 1 and not np.all(np.diff(ts) > 0)):
        raise ValueError(f"timestamps of {sequence_id!r} must be finite, strictly increasing")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    if not max_gap_s > 0:
        raise ValueError(f"max_gap_s must be > 0, got {max_gap_s}")

    positive = np.flatnonzero(y == 1)
    has_fire = positive.size > 0
    n_negative = int(positive[0]) if has_fire else ts.size

    negative_ts = ts[:n_negative]
    exposure = float(np.sum(np.minimum(np.diff(negative_ts), max_gap_s)))
    false_alarms = int(np.sum(al[:n_negative]))

    if not has_fire:
        return SequenceOutcome(sequence_id, False, False, None, None, false_alarms, exposure)

    t0 = float(ts[n_negative])
    post = np.flatnonzero(al[n_negative:])
    if post.size > 0:
        ttd = (float(ts[n_negative + int(post[0])]) - t0) / 60.0
        return SequenceOutcome(sequence_id, True, True, ttd, ttd, false_alarms, exposure)
    penalised = (float(ts[-1]) - t0) / 60.0
    return SequenceOutcome(sequence_id, True, False, None, penalised, false_alarms, exposure)


# ------------------------------------------------------------------------ experiment level


@dataclass(frozen=True)
class Summary:
    """Experiment-level metrics over many sequences (§3, §4). Empty denominators -> NaN."""

    n_sequences: int
    n_fires: int
    detection_rate: float
    ttd_median_detected: float
    ttd_mean_detected: float
    ttd_mean_penalised: float
    ttd_p90_penalised: float
    false_alarms: int
    negative_exposure_days: float
    false_alarms_per_camera_day: float
    fa_rate_ci_low: float
    fa_rate_ci_high: float


SUMMARY_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Summary))


def _stat(values: list[float], fn: Callable[[npt.NDArray[np.float64]], float]) -> float:
    return fn(np.asarray(values, dtype=np.float64)) if values else _NAN


def summarise(outcomes: Sequence[SequenceOutcome], confidence: float = 0.95) -> Summary:
    """Aggregate sequence outcomes into the headline metrics.

    detection_rate = detected fires / fires; TTD statistics over detected fires (median,
    mean) or over all fires with misses penalised (mean, 90th percentile);
    false_alarms_per_camera_day = total false alarms / total negative exposure in days,
    with an exact Poisson interval at ``confidence``.
    """
    fires = [o for o in outcomes if o.has_fire]
    detected = [o.ttd_min for o in fires if o.detected and o.ttd_min is not None]
    penalised = [o.penalised_ttd_min for o in fires if o.penalised_ttd_min is not None]
    n_detected = sum(o.detected for o in fires)
    fa = sum(o.false_alarms for o in outcomes)
    days = sum(o.negative_exposure_s for o in outcomes) / SECONDS_PER_DAY
    if days > 0:
        ci_low, ci_high = poisson_rate_interval(fa, days, confidence)
    else:
        ci_low, ci_high = _NAN, _NAN
    return Summary(
        n_sequences=len(outcomes),
        n_fires=len(fires),
        detection_rate=_ratio(n_detected, len(fires)),
        ttd_median_detected=_stat(detected, lambda a: float(np.median(a))),
        ttd_mean_detected=_stat(detected, lambda a: float(np.mean(a))),
        ttd_mean_penalised=_stat(penalised, lambda a: float(np.mean(a))),
        ttd_p90_penalised=_stat(penalised, lambda a: float(np.percentile(a, 90))),
        false_alarms=fa,
        negative_exposure_days=days,
        false_alarms_per_camera_day=_ratio(fa, days),
        fa_rate_ci_low=ci_low,
        fa_rate_ci_high=ci_high,
    )


# ------------------------------------------------------------------- exact Poisson interval


def _gamma_p(s: float, x: float) -> float:
    """Regularised lower incomplete gamma P(s, x) for s > 0, x >= 0.

    Series expansion for x < s + 1, Lentz continued fraction for Q = 1 - P otherwise.
    """
    if x <= 0.0:
        return 0.0
    log_prefactor = s * math.log(x) - x - math.lgamma(s)
    if x < s + 1.0:
        term = 1.0 / s
        total = term
        a = s
        for _ in range(10_000):
            a += 1.0
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return min(1.0, total * math.exp(log_prefactor))
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return max(0.0, 1.0 - math.exp(log_prefactor) * h)


def _gamma_ppf(p: float, s: float) -> float:
    """Quantile of the standard Gamma(shape=s, scale=1) distribution, by bisection."""
    if p <= 0.0:
        return 0.0
    hi = max(1.0, s)
    while _gamma_p(s, hi) < p:
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _gamma_p(s, mid) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * hi:
            break
    return 0.5 * (lo + hi)


def poisson_rate_interval(
    count: int, exposure: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact (Garwood) confidence interval for a Poisson rate ``count / exposure``.

    lower = chi2.ppf(a/2, 2 count) / 2 / exposure (0 when count == 0),
    upper = chi2.ppf(1 - a/2, 2 count + 2) / 2 / exposure, with a = 1 - confidence.
    Uses chi2.ppf(p, 2s) / 2 == Gamma(s, 1).ppf(p), computed without scipy.
    """
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count}")
    if not (exposure > 0) or math.isinf(exposure):
        raise ValueError(f"exposure must be finite and > 0, got {exposure}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    k = int(count)
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else _gamma_ppf(alpha / 2.0, float(k))
    upper = _gamma_ppf(1.0 - alpha / 2.0, float(k + 1))
    return lower / exposure, upper / exposure


# ------------------------------------------------------------------------- operating curve


@dataclass(frozen=True, eq=False)
class ScoredSequence:
    """One sequence ready for evaluation: time-ordered timestamps (s), labels, scores."""

    sequence_id: str
    timestamps_s: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int_]
    scores: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        shapes = {self.timestamps_s.shape, self.labels.shape, self.scores.shape}
        if len(shapes) != 1 or self.timestamps_s.ndim != 1:
            raise ValueError(
                f"{self.sequence_id!r}: timestamps, labels and scores must be 1-D of equal "
                f"length, got {self.timestamps_s.shape}, {self.labels.shape}, "
                f"{self.scores.shape}"
            )


def evaluate_scored(
    sequences: Sequence[ScoredSequence], config: AlertConfig, max_gap_s: float = 180.0
) -> list[SequenceOutcome]:
    """Run a fresh alert policy on every sequence and evaluate it."""
    return [
        evaluate_sequence(
            s.sequence_id,
            s.timestamps_s,
            s.labels,
            run_policy(s.timestamps_s, s.scores, config),
            max_gap_s,
        )
        for s in sequences
    ]


def operating_curve(
    sequences: Sequence[ScoredSequence],
    thresholds: Sequence[float] | npt.NDArray[np.float64],
    k: int = 1,
    n: int = 1,
    cooldown_s: float = 0.0,
    max_gap_s: float = 180.0,
) -> pd.DataFrame:
    """One row per threshold: ``threshold`` followed by every :class:`Summary` field (§5)."""
    rows = []
    for threshold in thresholds:
        config = AlertConfig(threshold=float(threshold), k=k, n=n, cooldown_s=cooldown_s)
        summary = summarise(evaluate_scored(sequences, config, max_gap_s))
        rows.append({"threshold": float(threshold), **asdict(summary)})
    return pd.DataFrame(rows, columns=["threshold", *SUMMARY_FIELDS])


# ------------------------------------------------------------------------------ bootstrap

Statistic = Callable[[Sequence[SequenceOutcome]], float]


def _percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return _NAN, _NAN
    alpha = 1.0 - confidence
    low, high = np.percentile(arr, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def _check_bootstrap_args(n_items: int, n_resamples: int, confidence: float) -> None:
    if n_items == 0:
        raise ValueError("need at least one sequence outcome")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def bootstrap_over_sequences(
    outcomes: Sequence[SequenceOutcome],
    statistic: Statistic,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap over sequences: (statistic on all outcomes, low, high).

    Each resample draws ``len(outcomes)`` sequences with replacement. Resamples whose
    statistic is NaN are ignored when taking the percentiles.
    """
    _check_bootstrap_args(len(outcomes), n_resamples, confidence)
    rng = np.random.default_rng(seed)
    m = len(outcomes)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, m, size=m)
        values.append(float(statistic([outcomes[i] for i in idx])))
    low, high = _percentile_interval(values, confidence)
    return float(statistic(outcomes)), low, high


def paired_bootstrap_difference(
    outcomes_a: Sequence[SequenceOutcome],
    outcomes_b: Sequence[SequenceOutcome],
    statistic: Statistic,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap of ``statistic(a) - statistic(b)`` over the same sequences.

    Both lists must hold the same sequence ids in the same order; every resample applies
    the same indices to both. Returns (point difference, low, high); NaN resamples are
    ignored.
    """
    ids_a = [o.sequence_id for o in outcomes_a]
    ids_b = [o.sequence_id for o in outcomes_b]
    if ids_a != ids_b:
        raise ValueError("outcomes_a and outcomes_b must have identical sequence_id order")
    _check_bootstrap_args(len(outcomes_a), n_resamples, confidence)
    rng = np.random.default_rng(seed)
    m = len(outcomes_a)
    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, m, size=m)
        a = float(statistic([outcomes_a[i] for i in idx]))
        b = float(statistic([outcomes_b[i] for i in idx]))
        values.append(a - b)
    low, high = _percentile_interval(values, confidence)
    point = float(statistic(outcomes_a)) - float(statistic(outcomes_b))
    return point, low, high
