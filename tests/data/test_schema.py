"""Tests for vigia.data.schema."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vigia.data.schema import (
    MANIFEST_COLUMNS,
    MANIFEST_DTYPES,
    ManifestError,
    build_manifest,
    empty_manifest,
    make_frame_records,
    read_manifest,
    sequence_table,
    validate_manifest,
    write_manifest,
)

MakeManifest = Callable[..., pd.DataFrame]


def _hand_manifest() -> pd.DataFrame:
    """Two sequences: s1 (fire, 3 frames, ignition at frame 1) and s2 (no fire, 2 frames)."""
    t0 = pd.Timestamp("2021-07-01T12:00:00Z")
    recs = make_frame_records(
        dataset="d",
        sequence_id="s1",
        camera_id="c1",
        station_id="st1",
        timestamps=[t0, t0 + pd.Timedelta(minutes=1), t0 + pd.Timedelta(minutes=2)],
        paths=["s1/0.jpg", "s1/1.jpg", "s1/2.jpg"],
        labels=[0, 1, 1],
        width=100,
        height=50,
        offsets_s=[-60.0, 0.0, 60.0],
        boxes=[[], [[0.1, 0.2, 0.3, 0.4]], [[0.1, 0.2, 0.5, 0.6], [0.6, 0.6, 0.7, 0.7]]],
    )
    recs += make_frame_records(
        dataset="d",
        sequence_id="s2",
        camera_id="c2",
        station_id=None,
        timestamps=[t0, t0 + pd.Timedelta(minutes=1)],
        paths=["s2/0.jpg", "s2/1.jpg"],
        labels=[0, 0],
        width=100,
        height=50,
    )
    return build_manifest(recs)


def test_valid_manifest_passes(make_manifest: MakeManifest) -> None:
    df = make_manifest(n_datasets=2, n_stations=2, cameras_per_station=2)
    validate_manifest(df)
    assert tuple(df.columns) == MANIFEST_COLUMNS
    for col in MANIFEST_COLUMNS:
        assert df[col].dtype == MANIFEST_DTYPES[col], col


def test_build_sorts_and_resets_index() -> None:
    df = _hand_manifest()
    shuffled = df.sample(frac=1.0, random_state=0).to_dict("records")
    rebuilt = build_manifest(shuffled)
    pd.testing.assert_frame_equal(rebuilt, df)
    assert list(rebuilt.index) == list(range(len(df)))


def test_station_defaults_to_camera() -> None:
    df = _hand_manifest()
    assert (df.loc[df["sequence_id"] == "s2", "station_id"] == "c2").all()


def test_empty_manifest_has_contract_columns() -> None:
    df = empty_manifest()
    assert len(df) == 0
    assert tuple(df.columns) == MANIFEST_COLUMNS
    validate_manifest(df)
    assert build_manifest([]).columns.tolist() == list(MANIFEST_COLUMNS)


def _expect(df: pd.DataFrame, invariant: str, *also: str) -> None:
    with pytest.raises(ManifestError) as info:
        validate_manifest(df)
    msg = str(info.value)
    assert f"[{invariant}]" in msg, msg
    for fragment in also:
        assert fragment in msg, msg


def test_missing_column() -> None:
    _expect(_hand_manifest().drop(columns=["path"]), "required columns", "path")


def test_non_null_column_with_null() -> None:
    df = _hand_manifest()
    df["path"] = df["path"].astype(object)
    df.loc[1, "path"] = None
    _expect(df, "non-null", "'path'", "row 1", "'s1'")


def test_duplicate_frame_key() -> None:
    df = _hand_manifest()
    df.loc[1, "frame_idx"] = 0
    _expect(df, "unique (dataset, sequence_id, frame_idx)", "'s1'")


def test_timestamp_not_increasing() -> None:
    df = _hand_manifest()
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]
    _expect(df, "timestamp strictly increasing within sequence", "row 2", "'s1'")


def test_timestamp_order_follows_frame_idx_not_row_order() -> None:
    df = _hand_manifest().iloc[::-1]
    validate_manifest(df)


def test_camera_not_constant() -> None:
    df = _hand_manifest()
    df.loc[2, "camera_id"] = "other"
    _expect(df, "camera_id constant within sequence", "sequence_id='s1'")


def test_station_not_constant() -> None:
    df = _hand_manifest()
    df.loc[4, "station_id"] = "other"
    _expect(df, "station_id constant within sequence", "sequence_id='s2'")


def test_label_out_of_range() -> None:
    df = _hand_manifest()
    df.loc[0, "label"] = 2
    _expect(df, "label in {0,1}", "row 0")


def test_non_positive_size() -> None:
    df = _hand_manifest()
    df.loc[3, "height"] = 0
    _expect(df, "positive width/height", "row 3", "'s2'")


def test_negative_frame_idx() -> None:
    df = _hand_manifest()
    df.loc[0, "frame_idx"] = -1
    _expect(df, "frame_idx non-negative", "row 0")


def test_naive_timestamp() -> None:
    df = _hand_manifest()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    _expect(df, "timestamp UTC")


def test_non_utc_timestamp() -> None:
    df = _hand_manifest()
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Lisbon")
    _expect(df, "timestamp UTC")


def test_build_converts_other_timezones_to_utc() -> None:
    recs = _hand_manifest().to_dict("records")
    for r in recs:
        r["timestamp"] = r["timestamp"].tz_convert("America/Los_Angeles")
    df = build_manifest(recs)
    assert str(df["timestamp"].dt.tz) == "UTC"
    pd.testing.assert_frame_equal(df, _hand_manifest())


@pytest.mark.parametrize(
    ("cell", "reason"),
    [
        ([[0.3, 0.2, 0.1, 0.4]], "x0 < x1"),
        ([[0.1, 0.4, 0.3, 0.4]], "x0 < x1 and y0 < y1"),
        ([[0.1, 0.2, 1.2, 0.4]], "outside [0, 1]"),
        ([[-0.1, 0.2, 0.3, 0.4]], "outside [0, 1]"),
        ([[0.1, 0.2, 0.3]], "4 coordinates"),
        ([[0.1, 0.2, float("nan"), 0.4]], "non-finite"),
        ("not a list", "expected a list"),
    ],
)
def test_bad_boxes(cell: object, reason: str) -> None:
    df = _hand_manifest()
    df.at[1, "boxes"] = cell
    _expect(df, "boxes well-formed", "row 1", reason)


def test_offset_not_increasing() -> None:
    df = _hand_manifest()
    df.loc[2, "offset_s"] = 0.0
    _expect(df, "offset_s strictly increasing within sequence", "row 2")


def test_offset_partially_null() -> None:
    df = _hand_manifest()
    df.loc[0, "offset_s"] = np.nan
    _expect(df, "offset_s all-or-none within sequence", "sequence_id='s1'")


def test_error_lists_at_most_five_rows(make_manifest: MakeManifest) -> None:
    df = make_manifest()
    df["label"] = np.int8(3)
    _expect(df, "label in {0,1}", f"{len(df)} offending row(s)", f"and {len(df) - 5} more")


def test_parquet_round_trip_preserves_everything(tmp_path: Path) -> None:
    df = _hand_manifest()
    assert df.loc[0, "boxes"] == []
    assert df.loc[3, "boxes"] is None
    path = tmp_path / "m.parquet"
    write_manifest(df, path)
    back = read_manifest(path)
    pd.testing.assert_frame_equal(back, df)
    assert back.loc[0, "boxes"] == []
    assert back.loc[3, "boxes"] is None
    assert back.loc[2, "boxes"] == [[0.1, 0.2, 0.5, 0.6], [0.6, 0.6, 0.7, 0.7]]
    assert isinstance(back.loc[2, "boxes"], list)
    assert isinstance(back.loc[2, "boxes"][0], list)
    assert str(back["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert np.isnan(back.loc[3, "offset_s"])


@pytest.mark.parametrize("fill", [None, []])
def test_parquet_round_trip_all_boxes_same_kind(tmp_path: Path, fill: object) -> None:
    df = _hand_manifest()
    df["boxes"] = pd.Series([fill if fill is None else [] for _ in range(len(df))], dtype=object)
    path = tmp_path / "m.parquet"
    write_manifest(df, path)
    back = read_manifest(path)
    assert back["boxes"].tolist() == df["boxes"].tolist()


def test_parquet_round_trip_synthetic(tmp_path: Path, make_manifest: MakeManifest) -> None:
    df = make_manifest(n_datasets=2)
    path = tmp_path / "m.parquet"
    write_manifest(df, path)
    pd.testing.assert_frame_equal(read_manifest(path), df)


def test_write_refuses_invalid(tmp_path: Path) -> None:
    df = _hand_manifest()
    df.loc[0, "label"] = 5
    with pytest.raises(ManifestError):
        write_manifest(df, tmp_path / "m.parquet")


def test_sequence_table_hand_case() -> None:
    table = sequence_table(_hand_manifest())
    t0 = pd.Timestamp("2021-07-01T12:00:00Z")
    assert table["sequence_id"].tolist() == ["s1", "s2"]
    assert table["camera_id"].tolist() == ["c1", "c2"]
    assert table["station_id"].tolist() == ["st1", "c2"]
    assert table["n_frames"].tolist() == [3, 2]
    assert table["n_positive"].tolist() == [2, 0]
    assert table["has_fire"].tolist() == [True, False]
    assert table["start_timestamp"].tolist() == [t0, t0]
    assert table["end_timestamp"].tolist() == [
        t0 + pd.Timedelta(minutes=2),
        t0 + pd.Timedelta(minutes=1),
    ]
    assert table.loc[0, "ignition_timestamp"] == t0 + pd.Timedelta(minutes=1)
    assert pd.isna(table.loc[1, "ignition_timestamp"])
    assert str(table["ignition_timestamp"].dtype) == "datetime64[ns, UTC]"


def test_sequence_table_synthetic_counts(make_manifest: MakeManifest) -> None:
    df = make_manifest(
        n_stations=1, cameras_per_station=1, sequences_per_camera=2, frames_per_sequence=6
    )
    table = sequence_table(df)
    assert table["n_frames"].tolist() == [6, 6]
    assert table["n_positive"].tolist() == [3, 0]


def test_validation_scales_to_100k_rows(make_manifest: MakeManifest) -> None:
    df = make_manifest(
        n_stations=10, cameras_per_station=5, sequences_per_camera=20, frames_per_sequence=100
    )
    assert len(df) == 100_000
    validate_manifest(df)
