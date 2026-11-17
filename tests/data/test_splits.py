"""Tests for vigia.data.splits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from vigia.data.schema import build_manifest, make_frame_records
from vigia.data.splits import (
    LeakageError,
    assert_disjoint,
    assignment_from_lists,
    assignment_hash,
    leakage_report,
    leave_groups_out_folds,
    split_by_group,
)

MakeManifest = Callable[..., pd.DataFrame]
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def _per_sequence(manifest: pd.DataFrame, assignment: pd.Series) -> dict[tuple[str, str], str]:
    frame = manifest[["dataset", "sequence_id"]].assign(split=assignment.astype(str))
    grouped = frame.groupby(["dataset", "sequence_id"])["split"]
    assert (grouped.nunique() == 1).all()
    return {(str(d), str(s)): str(v) for (d, s), v in grouped.first().items()}


def _assert_groups_whole(manifest: pd.DataFrame, assignment: pd.Series, cols: list[str]) -> None:
    n_splits = assignment.groupby([manifest[c] for c in cols]).nunique()
    assert (n_splits == 1).all()


def _shuffled(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed)


def _hand(spec: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """Manifest from (sequence_id, camera_id, station_id, n_frames) in dataset 'd'."""
    t0 = pd.Timestamp("2022-01-01T00:00:00Z")
    recs: list[dict[str, object]] = []
    for seq, cam, st, n in spec:
        recs += make_frame_records(
            dataset="d",
            sequence_id=seq,
            camera_id=cam,
            station_id=st,
            timestamps=[t0 + pd.Timedelta(minutes=i) for i in range(n)],
            paths=[f"{seq}/{i}.jpg" for i in range(n)],
            labels=[0] * n,
            width=10,
            height=10,
        )
    return build_manifest(recs)


@pytest.mark.parametrize("key", ["station_id", "camera_id", ("dataset", "station_id")])
def test_split_keeps_groups_whole(make_manifest: MakeManifest, key: str | tuple[str, ...]) -> None:
    df = make_manifest(n_datasets=2, n_stations=5, cameras_per_station=2)
    assignment = split_by_group(df, key, FRACTIONS, seed=0)
    cols = [key] if isinstance(key, str) else list(key)
    _assert_groups_whole(df, assignment, cols)
    assert assignment.index.equals(df.index)
    assert set(assignment.unique()) == {"train", "val", "test"}
    assert str(assignment.dtype) == "string"


def test_split_deterministic_and_seed_sensitive(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=20, cameras_per_station=1)
    a = split_by_group(df, "station_id", FRACTIONS, seed=1)
    b = split_by_group(df, "station_id", FRACTIONS, seed=1)
    pd.testing.assert_series_equal(a, b)
    others = [split_by_group(df, "station_id", FRACTIONS, seed=s) for s in range(2, 6)]
    assert any(not o.equals(a) for o in others)


def test_split_independent_of_row_order(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_datasets=2, n_stations=6)
    a = split_by_group(df, "station_id", FRACTIONS, seed=7)
    shuffled = _shuffled(df, 3)
    b = split_by_group(shuffled, "station_id", FRACTIONS, seed=7)
    assert b.index.equals(shuffled.index)
    assert _per_sequence(df, a) == _per_sequence(shuffled, b)
    pd.testing.assert_series_equal(b.loc[df.index], a)


def test_split_sequence_fractions_on_100_groups(make_manifest: MakeManifest) -> None:
    df = make_manifest(
        n_stations=100, cameras_per_station=1, sequences_per_camera=1, frames_per_sequence=2
    )
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    assignment = split_by_group(df, "station_id", fractions, seed=0)
    per_seq = pd.Series(_per_sequence(df, assignment)).value_counts()
    assert per_seq.to_dict() == {"train": 70, "val": 15, "test": 15}


def test_split_fractions_count_sequences_not_rows() -> None:
    # Station A has one 100-frame sequence; B, C, D one 1-frame sequence each. Halving the
    # 103 rows would isolate A; halving the 4 sequences gives 2 + 2 for every seed.
    df = _hand([("a", "A", "A", 100), ("b", "B", "B", 1), ("c", "C", "C", 1), ("d", "D", "D", 1)])
    for seed in range(10):
        assignment = split_by_group(df, "station_id", {"x": 0.5, "y": 0.5}, seed=seed)
        counts = pd.Series(_per_sequence(df, assignment)).value_counts().to_dict()
        assert counts == {"x": 2, "y": 2}


def test_split_small_fraction_still_gets_a_group(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=3, cameras_per_station=1, sequences_per_camera=1)
    for seed in range(10):
        a = split_by_group(df, "station_id", {"train": 0.98, "val": 0.01, "test": 0.01}, seed)
        assert sorted(a.unique()) == ["test", "train", "val"]


def test_split_zero_fraction_split_is_empty(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=4)
    a = split_by_group(df, "station_id", {"train": 1.0, "unused": 0.0}, seed=0)
    assert set(a.unique()) == {"train"}


@pytest.mark.parametrize(
    "fractions",
    [{"a": 0.5, "b": 0.4}, {"a": 1.1, "b": -0.1}, {}, {"a": float("nan"), "b": 1.0}],
)
def test_split_rejects_bad_fractions(
    make_manifest: MakeManifest, fractions: dict[str, float]
) -> None:
    with pytest.raises(ValueError, match="fractions"):
        split_by_group(make_manifest(), "station_id", fractions, seed=0)


def test_split_tolerates_float_rounding(make_manifest: MakeManifest) -> None:
    split_by_group(make_manifest(n_stations=3), "station_id", {"a": 0.1, "b": 0.2, "c": 0.7}, 0)


def test_split_too_few_groups(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=2)
    with pytest.raises(ValueError, match="cannot fill 3"):
        split_by_group(df, "station_id", FRACTIONS, seed=0)


def test_split_rejects_key_not_constant_within_sequence(make_manifest: MakeManifest) -> None:
    with pytest.raises(ValueError, match="not constant within sequence"):
        split_by_group(make_manifest(), "frame_idx", {"a": 0.5, "b": 0.5}, seed=0)


def test_folds_each_group_tested_once(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=7, cameras_per_station=2)
    folds = leave_groups_out_folds(df, "station_id", n_folds=3, seed=0)
    assert len(folds) == 3
    times_tested = pd.Series(0, index=sorted(df["station_id"].unique()))
    for fold in folds:
        assert set(fold.unique()) == {"train", "test"}
        _assert_groups_whole(df, fold, ["station_id"])
        assert_disjoint(df, fold, ["station_id", "camera_id", "sequence_id"])
        tested = df.loc[fold == "test", "station_id"].unique()
        times_tested[list(tested)] += 1
    assert (times_tested == 1).all()


def test_folds_balanced_by_sequences(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=9, cameras_per_station=1)
    folds = leave_groups_out_folds(df, "station_id", n_folds=3, seed=4)
    n_test_seqs = [df.loc[f == "test", "sequence_id"].nunique() for f in folds]
    assert n_test_seqs == [6, 6, 6]


def test_folds_deterministic_and_order_independent(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=6)
    a = leave_groups_out_folds(df, "station_id", 3, seed=2)
    shuffled = _shuffled(df, 1)
    b = leave_groups_out_folds(shuffled, "station_id", 3, seed=2)
    for fa, fb in zip(a, b, strict=True):
        pd.testing.assert_series_equal(fb.loc[df.index], fa)


def test_folds_errors(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=2)
    with pytest.raises(ValueError, match="cannot fill 3 folds"):
        leave_groups_out_folds(df, "station_id", 3, seed=0)
    with pytest.raises(ValueError, match="at least 2"):
        leave_groups_out_folds(df, "station_id", 1, seed=0)


def test_assignment_from_lists(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=1, cameras_per_station=1, sequences_per_camera=3)
    lists = {"train": ["ds0-st0-cam0-seq0"], "test": ["ds0-st0-cam0-seq1"]}
    with pytest.warns(UserWarning, match=r"in no list.*ds0-st0-cam0-seq2"):
        a = assignment_from_lists(df, lists)
    per_seq = df.assign(split=a).groupby("sequence_id")["split"].first()
    assert per_seq["ds0-st0-cam0-seq0"] == "train"
    assert per_seq["ds0-st0-cam0-seq1"] == "test"
    assert a[df["sequence_id"] == "ds0-st0-cam0-seq2"].isna().all()


def test_assignment_from_lists_missing_ids(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=1, cameras_per_station=1, sequences_per_camera=1)
    lists = {"train": ["ds0-st0-cam0-seq0", "ghost"]}
    with pytest.raises(ValueError, match="ghost"):
        assignment_from_lists(df, lists)
    a = assignment_from_lists(df, lists, on_missing="ignore")
    assert (a == "train").all()


def test_assignment_from_lists_duplicate_id(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=1, cameras_per_station=1, sequences_per_camera=1)
    with pytest.raises(ValueError, match="appears in both"):
        assignment_from_lists(df, {"train": ["ds0-st0-cam0-seq0"], "test": ["ds0-st0-cam0-seq0"]})


def test_assignment_from_lists_dataset_scope(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_datasets=2, n_stations=1, cameras_per_station=1, sequences_per_camera=1)
    a = assignment_from_lists(df, {"train": ["ds0-st0-cam0-seq0"]}, dataset="ds0")
    assert (a[df["dataset"] == "ds0"] == "train").all()
    assert a[df["dataset"] == "ds1"].isna().all()
    with pytest.raises(ValueError, match="not found in dataset 'ds1'"):
        assignment_from_lists(df, {"train": ["ds0-st0-cam0-seq0"]}, dataset="ds1")


def test_assert_disjoint_passes_on_station_split(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=6, cameras_per_station=2)
    a = split_by_group(df, "station_id", FRACTIONS, seed=0)
    assert_disjoint(df, a, ["station_id", "camera_id", "sequence_id"])


def test_assert_disjoint_catches_station_leak_in_sequence_split(
    make_manifest: MakeManifest,
) -> None:
    # One station, one camera, four sequences: any sequence-level split into two non-empty
    # splits must put that station in both.
    df = make_manifest(n_stations=1, cameras_per_station=1, sequences_per_camera=4)
    a = split_by_group(df, "sequence_id", {"train": 0.5, "test": 0.5}, seed=0)
    assert_disjoint(df, a, ["sequence_id"])
    with pytest.raises(LeakageError, match=r"key 'station_id'.*'test' and 'train'.*ds0:ds0-st0"):
        assert_disjoint(df, a, ["sequence_id", "station_id"])


def test_assert_disjoint_ignores_na_rows(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=1, cameras_per_station=1, sequences_per_camera=2)
    a = pd.Series(pd.NA, index=df.index, dtype="string")
    a[df["sequence_id"] == "ds0-st0-cam0-seq0"] = "train"
    assert_disjoint(df, a, ["station_id"])


def test_keys_are_qualified_by_dataset() -> None:
    t0 = pd.Timestamp("2022-01-01T00:00:00Z")
    recs: list[dict[str, object]] = []
    for ds in ("a", "b"):
        recs += make_frame_records(
            dataset=ds,
            sequence_id="s",
            camera_id="cam",
            station_id="st",
            timestamps=[t0],
            paths=["p"],
            labels=[0],
            width=1,
            height=1,
        )
    df = build_manifest(recs)
    a = split_by_group(df, "dataset", {"train": 0.5, "test": 0.5}, seed=0)
    assert_disjoint(df, a, ["sequence_id", "camera_id", "station_id", "dataset"])


def test_leakage_report_hand_case() -> None:
    # station S1: cameras c1, c2; station S2: camera c3.
    df = _hand(
        [("q1", "c1", "S1", 2), ("q2", "c1", "S1", 1), ("q3", "c2", "S1", 1), ("q4", "c3", "S2", 3)]
    )
    split = {"q1": "train", "q2": "test", "q3": "train", "q4": "test"}
    a = pd.Series(df["sequence_id"].map(split), index=df.index, dtype="string")
    report = leakage_report(df, a, ["camera_id", "station_id"]).set_index(
        ["key", "split_a", "split_b"]
    )
    assert len(report) == 4
    # test: cameras {c1, c3}, rows: q2 (1 row, c1) + q4 (3 rows, c3) = 4.
    # train: cameras {c1, c2}, rows: q1 (2, c1) + q3 (1, c2) = 3.
    row = report.loc[("camera_id", "train", "test")]
    assert (row["n_shared"], row["n_a"], row["n_b"]) == (1, 2, 2)
    assert row["frac_of_b_seen_in_a"] == pytest.approx(0.5)
    assert row["frac_rows_b_with_key_in_a"] == pytest.approx(1 / 4)
    row = report.loc[("camera_id", "test", "train")]
    assert row["frac_rows_b_with_key_in_a"] == pytest.approx(2 / 3)
    row = report.loc[("station_id", "train", "test")]
    assert (row["n_shared"], row["n_a"], row["n_b"]) == (1, 1, 2)
    assert row["frac_of_b_seen_in_a"] == pytest.approx(0.5)
    assert row["frac_rows_b_with_key_in_a"] == pytest.approx(1 / 4)
    row = report.loc[("station_id", "test", "train")]
    assert row["frac_of_b_seen_in_a"] == pytest.approx(1.0)
    assert row["frac_rows_b_with_key_in_a"] == pytest.approx(1.0)


def test_assignment_hash(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_stations=4)
    a = split_by_group(df, "station_id", {"train": 0.5, "test": 0.5}, seed=0)
    h = assignment_hash(df, a)
    assert len(h) == 16
    assert int(h, 16) >= 0
    shuffled = _shuffled(df, 5)
    assert assignment_hash(shuffled, a.loc[shuffled.index]) == h
    reindexed = df.reset_index(drop=True)
    reindexed.index = np.arange(len(df)) * 10
    assert assignment_hash(reindexed, pd.Series(a.to_numpy(), index=reindexed.index)) == h
    moved = a.copy()
    seq = df["sequence_id"].iloc[0]
    current = moved[df["sequence_id"] == seq].iloc[0]
    moved[df["sequence_id"] == seq] = "test" if current == "train" else "train"
    assert assignment_hash(df, moved) != h
    to_na = a.copy()
    to_na[df["sequence_id"] == seq] = pd.NA
    assert assignment_hash(df, to_na) != h
