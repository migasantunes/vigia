import math
from itertools import pairwise

import numpy as np
import pytest

from vigia.eval.metrics import (
    ScoredSequence,
    SequenceOutcome,
    bootstrap_over_sequences,
    evaluate_sequence,
    frame_metrics,
    operating_curve,
    paired_bootstrap_difference,
    poisson_rate_interval,
    summarise,
)

# ------------------------------------------------------------------------- frame metrics


def test_frame_metrics_hand_computed() -> None:
    m = frame_metrics([1, 1, 0, 0, 1], [0.9, 0.4, 0.6, 0.1, 0.5], threshold=0.5)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_frame_metrics_no_positives_anywhere_is_nan() -> None:
    m = frame_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert m.accuracy == 1.0
    assert math.isnan(m.precision)
    assert math.isnan(m.recall)
    assert math.isnan(m.f1)


def test_frame_metrics_no_predicted_positives() -> None:
    m = frame_metrics([1, 1, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert math.isnan(m.precision)
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.accuracy == pytest.approx(1 / 3)


def test_frame_metrics_empty() -> None:
    m = frame_metrics([], [], threshold=0.5)
    assert math.isnan(m.accuracy)
    assert (m.tp, m.fp, m.tn, m.fn) == (0, 0, 0, 0)


def test_frame_metrics_rejects_bad_labels() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        frame_metrics([0, 2], [0.1, 0.2], threshold=0.5)


# --------------------------------------------------------------------- evaluate_sequence

TS = [0.0, 60.0, 120.0, 180.0, 240.0]
LABELS = [0, 0, 1, 1, 1]


def test_detected() -> None:
    o = evaluate_sequence("s", TS, LABELS, [False, False, False, True, False])
    assert o == SequenceOutcome("s", True, True, 1.0, 1.0, 0, 60.0)


def test_alert_on_ignition_frame_is_ttd_zero() -> None:
    o = evaluate_sequence("s", TS, LABELS, [False, False, True, True, True])
    assert o.ttd_min == 0.0
    assert o.false_alarms == 0


def test_missed_is_penalised_to_last_frame() -> None:
    o = evaluate_sequence("s", TS, LABELS, [False] * 5)
    assert o == SequenceOutcome("s", True, False, None, 2.0, 0, 60.0)


def test_alert_before_ignition_is_false_alarm_not_detection() -> None:
    o = evaluate_sequence("s", TS, LABELS, [False, True, False, False, False])
    assert o.detected is False
    assert o.ttd_min is None
    assert o.penalised_ttd_min == 2.0
    assert o.false_alarms == 1


def test_alert_after_ignition_on_label_zero_frame_is_detection() -> None:
    o = evaluate_sequence("s", [0.0, 60.0, 120.0], [0, 1, 0], [False, False, True])
    assert o.detected is True
    assert o.ttd_min == 1.0
    assert o.false_alarms == 0


def test_no_fire_sequence() -> None:
    o = evaluate_sequence("s", [0.0, 60.0, 120.0], [0, 0, 0], [True, False, True])
    assert o == SequenceOutcome("s", False, False, None, None, 2, 120.0)


def test_exposure_gaps_are_capped() -> None:
    o = evaluate_sequence("s", [0.0, 60.0, 1060.0, 1120.0], [0] * 4, [False] * 4, max_gap_s=180)
    assert o.negative_exposure_s == 60.0 + 180.0 + 60.0


def test_gap_into_ignition_frame_is_not_exposure() -> None:
    o = evaluate_sequence("s", [0.0, 100.0], [0, 1], [False, False])
    assert o.negative_exposure_s == 0.0


def test_single_frame_sequences() -> None:
    neg = evaluate_sequence("a", [5.0], [0], [True])
    assert neg == SequenceOutcome("a", False, False, None, None, 1, 0.0)
    pos = evaluate_sequence("b", [5.0], [1], [True])
    assert pos == SequenceOutcome("b", True, True, 0.0, 0.0, 0, 0.0)


def test_evaluate_sequence_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        evaluate_sequence("s", [0.0, 1.0], [0], [False])
    with pytest.raises(ValueError, match="no frames"):
        evaluate_sequence("s", [], [], [])
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_sequence("s", [1.0, 0.0], [0, 0], [False, False])


# ----------------------------------------------------------------------------- summarise


def _outcome(
    sid: str, ttd: float | None, pen: float | None, fa: int, exp: float
) -> SequenceOutcome:
    return SequenceOutcome(sid, pen is not None, ttd is not None, ttd, pen, fa, exp)


THREE_FIRES = [
    _outcome("a", 2.0, 2.0, 1, 21_600.0),
    _outcome("b", 4.0, 4.0, 0, 7_200.0),
    _outcome("c", None, 9.0, 0, 7_200.0),
    _outcome("d", None, None, 2, 7_200.0),
]


def test_summarise_three_fires_by_hand() -> None:
    s = summarise(THREE_FIRES)
    assert s.n_sequences == 4
    assert s.n_fires == 3
    assert s.detection_rate == pytest.approx(2 / 3)
    assert s.ttd_median_detected == 3.0
    assert s.ttd_mean_detected == 3.0
    assert s.ttd_mean_penalised == 5.0
    # linear percentile of [2, 4, 9] at 90%: position 1.8 -> 4 + 0.8 * 5
    assert s.ttd_p90_penalised == pytest.approx(8.0)
    assert s.false_alarms == 3
    assert s.negative_exposure_days == 0.5
    assert s.false_alarms_per_camera_day == 6.0
    low, high = poisson_rate_interval(3, 0.5)
    assert (s.fa_rate_ci_low, s.fa_rate_ci_high) == (low, high)
    assert low < 6.0 < high


def test_summarise_empty_is_nan() -> None:
    s = summarise([])
    assert (s.n_sequences, s.n_fires, s.false_alarms) == (0, 0, 0)
    for value in (
        s.detection_rate,
        s.ttd_median_detected,
        s.ttd_mean_penalised,
        s.false_alarms_per_camera_day,
        s.fa_rate_ci_low,
        s.fa_rate_ci_high,
    ):
        assert math.isnan(value)


def test_summarise_no_detections() -> None:
    s = summarise([_outcome("c", None, 9.0, 0, 0.0)])
    assert s.detection_rate == 0.0
    assert math.isnan(s.ttd_mean_detected)
    assert s.ttd_mean_penalised == 9.0
    assert math.isnan(s.false_alarms_per_camera_day)


# ------------------------------------------------------------------------------- poisson

REFERENCE = {
    0: (0.0, 3.688879),
    1: (0.025318, 5.571643),
    5: (1.623486, 11.668332),
    10: (4.795389, 18.390356),
}


@pytest.mark.parametrize(("count", "expected"), REFERENCE.items())
def test_poisson_reference_values(count: int, expected: tuple[float, float]) -> None:
    low, high = poisson_rate_interval(count, 1.0)
    assert low == pytest.approx(expected[0], abs=1e-4)
    assert high == pytest.approx(expected[1], abs=1e-4)


def test_poisson_reference_value_large_count() -> None:
    low, high = poisson_rate_interval(100, 1.0)
    assert low == pytest.approx(81.364, rel=1e-3)
    assert high == pytest.approx(121.627, rel=1e-3)


def test_poisson_zero_count_lower_is_exactly_zero() -> None:
    assert poisson_rate_interval(0, 3.0)[0] == 0.0


def test_poisson_monotonic_in_count() -> None:
    bounds = [poisson_rate_interval(c, 1.0) for c in range(60)]
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]
    assert all(a < b for a, b in pairwise(lows))
    assert all(a < b for a, b in pairwise(highs))
    assert all(lo <= c <= hi for c, (lo, hi) in enumerate(bounds))


def test_poisson_scales_with_exposure() -> None:
    low1, high1 = poisson_rate_interval(5, 1.0)
    low4, high4 = poisson_rate_interval(5, 4.0)
    assert low4 == pytest.approx(low1 / 4)
    assert high4 == pytest.approx(high1 / 4)


def test_poisson_narrower_at_lower_confidence() -> None:
    low95, high95 = poisson_rate_interval(5, 1.0, 0.95)
    low80, high80 = poisson_rate_interval(5, 1.0, 0.80)
    assert low95 < low80 < high80 < high95


@pytest.mark.parametrize(
    ("count", "exposure", "confidence", "match"),
    [
        (-1, 1.0, 0.95, "count"),
        (1.5, 1.0, 0.95, "count"),
        (1, 0.0, 0.95, "exposure"),
        (1, 1.0, 1.0, "confidence"),
        (1, 1.0, 0.0, "confidence"),
    ],
)
def test_poisson_validation(count: int, exposure: float, confidence: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        poisson_rate_interval(count, exposure, confidence)


def test_poisson_matches_scipy_on_a_grid() -> None:
    stats = pytest.importorskip("scipy.stats")
    for confidence in (0.8, 0.95, 0.99):
        alpha = 1 - confidence
        for count in [*range(30), 50, 100, 250, 1000, 5000]:
            low, high = poisson_rate_interval(count, 2.5, confidence)
            ref_low = 0.0 if count == 0 else stats.chi2.ppf(alpha / 2, 2 * count) / 2 / 2.5
            ref_high = stats.chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2 / 2.5
            assert low == pytest.approx(ref_low, rel=1e-9, abs=1e-12)
            assert high == pytest.approx(ref_high, rel=1e-9)


# ----------------------------------------------------------------------- operating curve


def _sequences(seed: int) -> list[ScoredSequence]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(12):
        length = int(rng.integers(5, 30))
        ts = np.cumsum(rng.uniform(30.0, 90.0, size=length))
        labels = np.zeros(length, dtype=np.int_)
        if i % 3 != 0:
            labels[int(rng.integers(1, length)) :] = 1
        scores = np.clip(0.4 * labels + rng.uniform(0.0, 0.7, size=length), 0.0, 1.0)
        out.append(ScoredSequence(f"seq{i}", ts, labels, scores))
    return out


def test_operating_curve_is_monotone_in_threshold() -> None:
    thresholds = np.linspace(0.0, 1.0, 21)
    curve = operating_curve(_sequences(7), thresholds)
    assert list(curve["threshold"]) == pytest.approx(list(thresholds))
    assert curve.columns[0] == "threshold"
    assert "fa_rate_ci_high" in curve.columns
    assert np.all(np.diff(curve["detection_rate"].to_numpy()) <= 0)
    assert np.all(np.diff(curve["false_alarms"].to_numpy()) <= 0)
    assert curve["detection_rate"].iloc[0] == 1.0
    assert curve["false_alarms"].iloc[-1] < curve["false_alarms"].iloc[0]


def test_operating_curve_single_sequence_exact() -> None:
    seq = ScoredSequence(
        "s",
        np.array([0.0, 60.0, 120.0, 180.0]),
        np.array([0, 0, 1, 1]),
        np.array([0.3, 0.6, 0.2, 0.8]),
    )
    curve = operating_curve([seq], [0.25, 0.5, 0.9])
    assert curve["false_alarms"].tolist() == [2, 1, 0]
    assert curve["detection_rate"].tolist() == [1.0, 1.0, 0.0]
    assert curve["ttd_mean_penalised"].tolist() == [1.0, 1.0, 1.0]
    assert curve["ttd_mean_detected"].iloc[:2].tolist() == [1.0, 1.0]


def test_scored_sequence_validates_lengths() -> None:
    with pytest.raises(ValueError, match="equal"):
        ScoredSequence("s", np.zeros(3), np.zeros(2, dtype=np.int_), np.zeros(3))


# ----------------------------------------------------------------------------- bootstrap


def detection_rate(outcomes: object) -> float:
    return summarise(outcomes).detection_rate  # type: ignore[arg-type]


def mean_detected_ttd(outcomes: object) -> float:
    return summarise(outcomes).ttd_mean_detected  # type: ignore[arg-type]


FIRES = [_outcome(f"f{i}", float(i), float(i), 0, 0.0) for i in range(6)] + [
    _outcome(f"m{i}", None, 10.0, 0, 0.0) for i in range(4)
]


def test_bootstrap_is_deterministic_and_contains_point() -> None:
    first = bootstrap_over_sequences(FIRES, detection_rate, n_resamples=500, seed=3)
    second = bootstrap_over_sequences(FIRES, detection_rate, n_resamples=500, seed=3)
    assert first == second
    point, low, high = first
    assert point == pytest.approx(0.6)
    assert low <= point <= high
    assert low < high


def mean_penalised_ttd(outcomes: object) -> float:
    return summarise(outcomes).ttd_mean_penalised  # type: ignore[arg-type]


def test_bootstrap_seed_changes_resamples() -> None:
    a = bootstrap_over_sequences(FIRES, mean_penalised_ttd, n_resamples=50, seed=3)
    b = bootstrap_over_sequences(FIRES, mean_penalised_ttd, n_resamples=50, seed=4)
    assert a[0] == b[0] == pytest.approx(5.5)
    assert a[1:] != b[1:]


def test_bootstrap_ignores_nan_resamples() -> None:
    outcomes = [_outcome("hit", 2.0, 2.0, 0, 0.0)] + [
        _outcome(f"m{i}", None, 5.0, 0, 0.0) for i in range(4)
    ]
    point, low, high = bootstrap_over_sequences(outcomes, mean_detected_ttd, 200, seed=0)
    assert (point, low, high) == (2.0, 2.0, 2.0)


def test_bootstrap_all_nan_gives_nan_interval() -> None:
    outcomes = [_outcome("m", None, 5.0, 0, 0.0)]
    point, low, high = bootstrap_over_sequences(outcomes, mean_detected_ttd, 20, seed=0)
    assert math.isnan(point)
    assert math.isnan(low)
    assert math.isnan(high)


def test_bootstrap_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_over_sequences([], detection_rate, 10, seed=0)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_over_sequences(FIRES, detection_rate, 0, seed=0)


def test_paired_bootstrap_identical_lists_is_zero() -> None:
    assert paired_bootstrap_difference(FIRES, FIRES, detection_rate, 200, seed=1) == (
        0.0,
        0.0,
        0.0,
    )


def test_paired_bootstrap_difference_direction() -> None:
    better = [_outcome(o.sequence_id, 0.0, 0.0, 0, 0.0) for o in FIRES]
    point, low, high = paired_bootstrap_difference(better, FIRES, detection_rate, 300, seed=1)
    assert point == pytest.approx(0.4)
    assert 0.0 < low <= point <= high
    again = paired_bootstrap_difference(better, FIRES, detection_rate, 300, seed=1)
    assert again == (point, low, high)


def test_paired_bootstrap_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="sequence_id"):
        paired_bootstrap_difference(FIRES, FIRES[::-1], detection_rate, 10, seed=0)
    with pytest.raises(ValueError, match="sequence_id"):
        paired_bootstrap_difference(FIRES, FIRES[:-1], detection_rate, 10, seed=0)
