import numpy as np
import pandas as pd
import pytest

from vigia.eval.alerting import AlertConfig
from vigia.eval.harness import evaluate, sanity_baselines

START = pd.Timestamp("2023-06-01 12:00:00", tz="UTC")


def _manifest() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    for i, label in enumerate([0, 0, 1, 1, 1]):
        rows.append(("figlib", "fire1", i, START + pd.Timedelta(seconds=60 * i), label))
    for i in range(3):
        rows.append(("pyrosdis", "clip1", i, START + pd.Timedelta(seconds=30 * i), 0))
    manifest = pd.DataFrame(
        rows, columns=["dataset", "sequence_id", "frame_idx", "timestamp", "label"]
    )
    manifest["label"] = manifest["label"].astype("int8")
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 0.1, 0.8, 0.2])
    # Shuffle rows (and scores with them) so grouping and frame ordering are exercised.
    order = np.array([6, 2, 0, 7, 4, 1, 5, 3])
    return manifest.iloc[order].reset_index(drop=True), scores[order]


def test_evaluate_hand_built_manifest() -> None:
    manifest, scores = _manifest()
    summary, outcomes = evaluate(manifest, scores, AlertConfig(threshold=0.5))
    assert [o.sequence_id for o in outcomes] == ["figlib/fire1", "pyrosdis/clip1"]
    fire, clip = outcomes
    assert (fire.detected, fire.ttd_min, fire.false_alarms, fire.negative_exposure_s) == (
        True,
        1.0,
        0,
        60.0,
    )
    assert (clip.has_fire, clip.false_alarms, clip.negative_exposure_s) == (False, 1, 60.0)
    assert summary.n_sequences == 2
    assert summary.n_fires == 1
    assert summary.detection_rate == 1.0
    assert summary.ttd_mean_penalised == 1.0
    assert summary.false_alarms == 1
    assert summary.negative_exposure_days == pytest.approx(120 / 86_400)
    assert summary.false_alarms_per_camera_day == pytest.approx(720.0)


def test_evaluate_accepts_series_positionally() -> None:
    manifest, scores = _manifest()
    series = pd.Series(scores, index=np.arange(len(scores))[::-1])
    summary, _ = evaluate(manifest, series, AlertConfig(threshold=0.5))
    assert summary.false_alarms == 1
    assert summary.detection_rate == 1.0


def test_fresh_policy_per_sequence() -> None:
    manifest, _ = _manifest()
    config = AlertConfig(threshold=0.5, cooldown_s=1e6)
    _, outcomes = evaluate(manifest, np.ones(len(manifest)), config)
    assert [o.false_alarms for o in outcomes] == [1, 1]


def test_evaluate_validates_inputs() -> None:
    manifest, scores = _manifest()
    with pytest.raises(ValueError, match="one value per manifest row"):
        evaluate(manifest, scores[:-1], AlertConfig(threshold=0.5))
    with pytest.raises(ValueError, match="missing columns"):
        evaluate(manifest.drop(columns="label"), scores, AlertConfig(threshold=0.5))


def test_sanity_baselines() -> None:
    manifest, _ = _manifest()
    table = sanity_baselines(manifest, AlertConfig(threshold=0.5)).set_index("baseline")
    assert list(table.index) == ["never_alert", "always_alert", "oracle"]

    never = table.loc["never_alert"]
    assert never["detection_rate"] == 0.0
    assert never["false_alarms"] == 0
    assert never["ttd_mean_penalised"] == 2.0

    always = table.loc["always_alert"]
    assert always["detection_rate"] == 1.0
    assert always["false_alarms"] == 5
    assert always["ttd_mean_detected"] == 0.0

    oracle = table.loc["oracle"]
    assert oracle["detection_rate"] == 1.0
    assert oracle["ttd_median_detected"] == 0.0
    assert oracle["ttd_mean_detected"] == 0.0
    assert oracle["ttd_mean_penalised"] == 0.0
    assert oracle["ttd_p90_penalised"] == 0.0
    assert oracle["false_alarms"] == 0
    assert oracle["false_alarms_per_camera_day"] == 0.0
    assert np.isfinite(oracle["fa_rate_ci_high"])


def test_always_alert_respects_cooldown() -> None:
    manifest, _ = _manifest()
    table = sanity_baselines(manifest, AlertConfig(threshold=0.5, cooldown_s=90.0))
    always = table.set_index("baseline").loc["always_alert"]
    # fire1: alerts at 0 s (false alarm) and 120 s (= ignition, TTD 0).
    # clip1: alert at 0 s only (30 s and 60 s are inside the cooldown).
    assert always["false_alarms"] == 2
    assert always["ttd_mean_detected"] == 0.0


def test_sanity_baselines_reject_zero_threshold() -> None:
    manifest, _ = _manifest()
    with pytest.raises(ValueError, match="threshold > 0"):
        sanity_baselines(manifest, AlertConfig(threshold=0.0))
