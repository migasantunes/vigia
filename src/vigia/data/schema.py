"""Manifest schema: column contract, validation, construction and Parquet IO.

The manifest is one row per frame (``docs/ARCHITECTURE.md`` §4). Every dataset adapter
produces one, and nothing downstream reads dataset-specific layouts.

In-memory representation chosen here (the contract fixes semantics, this module fixes dtypes):

- text columns use ``pd.StringDtype()``;
- ``frame_idx``, ``width``, ``height`` are ``int64``; ``label`` is ``int8``;
- ``timestamp`` is ``datetime64[ns, UTC]``;
- ``offset_s`` is ``float64`` with ``NaN`` for "no ignition time" (numpy-friendly, so metrics
  code can do arithmetic on it directly);
- ``boxes`` is an ``object`` column whose cells are either ``None`` (not annotated) or a
  Python ``list`` of ``[x0, y0, x1, y1]`` lists of ``float`` (possibly empty: annotated, no
  smoke). ``read_manifest`` normalises back to exactly this form.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "MANIFEST_COLUMNS",
    "MANIFEST_DTYPES",
    "NULLABLE_COLUMNS",
    "SORT_KEY",
    "ManifestError",
    "build_manifest",
    "empty_manifest",
    "make_frame_records",
    "read_manifest",
    "sequence_table",
    "validate_manifest",
    "write_manifest",
]

STRING_DTYPE = pd.StringDtype()
TIMESTAMP_DTYPE = pd.DatetimeTZDtype(unit="ns", tz="UTC")

MANIFEST_COLUMNS: tuple[str, ...] = (
    "dataset",
    "sequence_id",
    "camera_id",
    "station_id",
    "frame_idx",
    "timestamp",
    "offset_s",
    "label",
    "path",
    "width",
    "height",
    "boxes",
)
"""Manifest column names, in canonical order."""

MANIFEST_DTYPES: Mapping[str, Any] = {
    "dataset": STRING_DTYPE,
    "sequence_id": STRING_DTYPE,
    "camera_id": STRING_DTYPE,
    "station_id": STRING_DTYPE,
    "frame_idx": np.dtype("int64"),
    "timestamp": TIMESTAMP_DTYPE,
    "offset_s": np.dtype("float64"),
    "label": np.dtype("int8"),
    "path": STRING_DTYPE,
    "width": np.dtype("int64"),
    "height": np.dtype("int64"),
    "boxes": np.dtype("object"),
}
"""In-memory dtype of every manifest column (``boxes`` holds ``None`` or list of lists)."""

NULLABLE_COLUMNS: frozenset[str] = frozenset({"offset_s", "boxes"})
"""Columns allowed to hold nulls; every other column must be fully non-null."""

SORT_KEY: tuple[str, str, str] = ("dataset", "sequence_id", "frame_idx")
"""Canonical row order and the uniqueness key of a frame."""

SEQUENCE_KEY: tuple[str, str] = ("dataset", "sequence_id")

_MAX_REPORTED = 5


class ManifestError(ValueError):
    """A manifest violates the contract of ``docs/ARCHITECTURE.md`` §4."""


def _fail(invariant: str, detail: str) -> ManifestError:
    return ManifestError(f"manifest invariant violated [{invariant}]: {detail}")


def _rows_detail(df: pd.DataFrame, mask: pd.Series | np.ndarray[Any, Any]) -> str:
    """Describe the first offending rows by index label and (dataset, sequence_id, frame_idx)."""
    bad = df.loc[np.asarray(mask, dtype=bool)]
    parts = []
    cols = [c for c in SORT_KEY if c in df.columns]
    for idx, row in bad.head(_MAX_REPORTED).iterrows():
        ident = ", ".join(f"{c}={row[c]!r}" for c in cols)
        parts.append(f"row {idx!r} ({ident})")
    more = f" and {len(bad) - _MAX_REPORTED} more" if len(bad) > _MAX_REPORTED else ""
    return f"{len(bad)} offending row(s): " + "; ".join(parts) + more


def _sequences_detail(keys: Iterable[tuple[Any, Any]], n: int) -> str:
    shown = [f"(dataset={d!r}, sequence_id={s!r})" for d, s in list(keys)[:_MAX_REPORTED]]
    more = f" and {n - _MAX_REPORTED} more" if n > _MAX_REPORTED else ""
    return f"{n} offending sequence(s): " + ", ".join(shown) + more


def _box_problem(cell: object) -> str | None:
    """Return why a non-null ``boxes`` cell is malformed, or None if it is well formed."""
    if isinstance(cell, str | bytes) or not isinstance(cell, Sequence | np.ndarray):
        return f"expected a list of boxes, got {type(cell).__name__}"
    for box in cell:
        if isinstance(box, str | bytes) or not isinstance(box, Sequence | np.ndarray):
            return f"box {box!r} is not a list"
        if len(box) != 4:
            return f"box {list(box)!r} does not have 4 coordinates"
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            return f"box {list(box)!r} has non-numeric coordinates"
        coords = (x0, y0, x1, y1)
        if not all(math.isfinite(v) for v in coords):
            return f"box {list(coords)!r} has non-finite coordinates"
        if not all(0.0 <= v <= 1.0 for v in coords):
            return f"box {list(coords)!r} has coordinates outside [0, 1]"
        if not (x0 < x1 and y0 < y1):
            return f"box {list(coords)!r} does not satisfy x0 < x1 and y0 < y1"
    return None


def validate_manifest(df: pd.DataFrame) -> None:
    """Check every manifest invariant and raise ``ManifestError`` on the first violation.

    Checked, in order: required columns present; non-nullable columns have no nulls;
    ``timestamp`` is tz-aware UTC; ``label`` in {0, 1}; ``frame_idx`` is a non-negative
    integer; ``width`` and ``height`` are positive integers; ``(dataset, sequence_id,
    frame_idx)`` unique; ``camera_id`` and ``station_id`` constant within a sequence;
    ``timestamp`` strictly increasing with ``frame_idx`` within a sequence; ``offset_s`` is
    either null for the whole sequence or present for the whole sequence (the contract says
    it is null exactly when the sequence has no ignition time) and, when present, strictly
    increasing with ``frame_idx``; every non-null ``boxes`` cell is a list of
    ``[x0, y0, x1, y1]`` with finite coordinates in [0, 1], ``x0 < x1`` and ``y0 < y1``.

    ``frame_idx`` need not be contiguous (an adapter may drop unreadable frames). Extra
    columns are allowed. The error message names the invariant in brackets and lists the
    first offending rows or sequences. Everything except box validation is vectorised;
    box validation loops over non-null box cells only.
    """
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise _fail("required columns", f"missing column(s) {missing}")

    for col in MANIFEST_COLUMNS:
        if col in NULLABLE_COLUMNS:
            continue
        nulls = df[col].isna()
        if nulls.any():
            raise _fail("non-null", f"column {col!r} has nulls; " + _rows_detail(df, nulls))

    ts = df["timestamp"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype) or str(ts.dtype.tz) != "UTC":
        raise _fail("timestamp UTC", f"timestamp must be tz-aware UTC, got dtype {ts.dtype}")

    for col in ("label", "frame_idx", "width", "height"):
        if not pd.api.types.is_integer_dtype(df[col].dtype):
            raise _fail(f"{col} integer", f"column {col!r} has dtype {df[col].dtype}")

    bad_label = ~df["label"].isin([0, 1])
    if bad_label.any():
        raise _fail("label in {0,1}", _rows_detail(df, bad_label))

    bad_idx = df["frame_idx"] < 0
    if bad_idx.any():
        raise _fail("frame_idx non-negative", _rows_detail(df, bad_idx))

    bad_size = (df["width"] <= 0) | (df["height"] <= 0)
    if bad_size.any():
        raise _fail("positive width/height", _rows_detail(df, bad_size))

    dup = df.duplicated(subset=list(SORT_KEY), keep=False)
    if dup.any():
        raise _fail("unique (dataset, sequence_id, frame_idx)", _rows_detail(df, dup))

    seq_groups = df.groupby(list(SEQUENCE_KEY), sort=True, observed=True)
    for col in ("camera_id", "station_id"):
        nunique = seq_groups[col].nunique()
        bad = nunique[nunique > 1]
        if len(bad):
            raise _fail(f"{col} constant within sequence", _sequences_detail(bad.index, len(bad)))

    ordered = df.sort_values(list(SORT_KEY), kind="stable")
    codes = ordered.groupby(list(SEQUENCE_KEY), sort=False, observed=True).ngroup().to_numpy()
    same_seq = np.zeros(len(ordered), dtype=bool)
    same_seq[1:] = codes[1:] == codes[:-1]

    ts_values = ordered["timestamp"].to_numpy(dtype="datetime64[ns]")
    ts_step_ok = np.ones(len(ordered), dtype=bool)
    ts_step_ok[1:] = ts_values[1:] > ts_values[:-1]
    bad_ts = same_seq & ~ts_step_ok
    if bad_ts.any():
        raise _fail(
            "timestamp strictly increasing within sequence",
            _rows_detail(ordered, bad_ts) + " (each is not later than the previous frame)",
        )

    null_offset = ordered["offset_s"].isna()
    partial = null_offset.groupby(
        [ordered["dataset"], ordered["sequence_id"]], sort=True, observed=True
    ).agg(["any", "all"])
    partial = partial[partial["any"] & ~partial["all"]]
    if len(partial):
        raise _fail(
            "offset_s all-or-none within sequence",
            _sequences_detail(partial.index, len(partial)),
        )
    off = ordered["offset_s"].to_numpy(dtype=np.float64)
    off_step_ok = np.ones(len(ordered), dtype=bool)
    with np.errstate(invalid="ignore"):
        off_step_ok[1:] = off[1:] > off[:-1]
    bad_off = same_seq & ~null_offset.to_numpy() & ~off_step_ok
    if bad_off.any():
        raise _fail(
            "offset_s strictly increasing within sequence",
            _rows_detail(ordered, bad_off) + " (each is not larger than the previous frame)",
        )

    bad_boxes: list[tuple[Any, str]] = []
    boxes = df["boxes"]
    for idx, cell in zip(boxes.index, boxes.to_numpy(), strict=True):
        if cell is None or (isinstance(cell, float) and math.isnan(cell)):
            continue
        problem = _box_problem(cell)
        if problem is not None:
            bad_boxes.append((idx, problem))
    if bad_boxes:
        mask = df.index.isin([i for i, _ in bad_boxes])
        reasons = "; ".join(f"row {i!r}: {p}" for i, p in bad_boxes[:_MAX_REPORTED])
        raise _fail("boxes well-formed", _rows_detail(df, mask) + ". Reasons: " + reasons)


def empty_manifest() -> pd.DataFrame:
    """Return a zero-row manifest with the canonical columns and dtypes."""
    return pd.DataFrame(
        {col: pd.Series([], dtype=MANIFEST_DTYPES[col]) for col in MANIFEST_COLUMNS}
    )


def _normalise_boxes(cell: object) -> list[list[float]] | None:
    """Convert a boxes cell (list, tuple, numpy array, None/NaN) to None or list of lists."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    if isinstance(cell, str | bytes) or not isinstance(cell, Sequence | np.ndarray):
        raise _fail("boxes well-formed", f"expected a list of boxes, got {type(cell).__name__}")
    out: list[list[float]] = []
    for box in cell:
        if isinstance(box, str | bytes) or not isinstance(box, Sequence | np.ndarray):
            raise _fail("boxes well-formed", f"box {box!r} is not a list")
        out.append([float(v) for v in box])
    return out


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Cast manifest columns to canonical dtypes, keep extra columns after them."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise _fail("required columns", f"missing column(s) {missing}")
    out = pd.DataFrame(index=df.index)
    for col in MANIFEST_COLUMNS:
        series = df[col]
        if col == "timestamp":
            ts = pd.to_datetime(series)
            if not isinstance(ts.dtype, pd.DatetimeTZDtype):
                raise _fail("timestamp UTC", "timestamps must be tz-aware (got naive datetimes)")
            out[col] = ts.dt.tz_convert("UTC").astype(TIMESTAMP_DTYPE)
        elif col == "boxes":
            out[col] = pd.Series(
                [_normalise_boxes(c) for c in series.to_numpy()], index=df.index, dtype=object
            )
        elif col == "offset_s":
            out[col] = pd.to_numeric(series, errors="raise").astype(np.float64)
        else:
            nulls = series.isna()
            if nulls.any():
                raise _fail("non-null", f"column {col!r} has nulls; " + _rows_detail(df, nulls))
            out[col] = series.astype(MANIFEST_DTYPES[col])
    for col in df.columns:
        if col not in MANIFEST_COLUMNS:
            out[col] = df[col]
    return out


def make_frame_records(
    *,
    dataset: str,
    sequence_id: str,
    camera_id: str,
    station_id: str | None,
    timestamps: Sequence[Any],
    paths: Sequence[str],
    labels: Sequence[int],
    width: int,
    height: int,
    offsets_s: Sequence[float] | None = None,
    boxes: Sequence[Sequence[Sequence[float]] | None] | None = None,
) -> list[dict[str, object]]:
    """Return one record per frame of one sequence, ready for ``build_manifest``.

    ``frame_idx`` is the position in the given (time-ordered) lists. ``station_id=None``
    means unknown and is replaced by ``camera_id``, as the contract specifies.
    ``offsets_s=None`` means the sequence has no ignition time (all null); ``boxes=None``
    means no frame is annotated. All per-frame sequences must have the same length.
    """
    n = len(timestamps)
    lengths = {"paths": len(paths), "labels": len(labels)}
    if offsets_s is not None:
        lengths["offsets_s"] = len(offsets_s)
    if boxes is not None:
        lengths["boxes"] = len(boxes)
    wrong = {k: v for k, v in lengths.items() if v != n}
    if wrong:
        raise ValueError(f"{wrong} do not match len(timestamps)={n}")
    station = camera_id if station_id is None else station_id
    records: list[dict[str, object]] = []
    for i in range(n):
        frame_boxes = None if boxes is None else boxes[i]
        records.append(
            {
                "dataset": dataset,
                "sequence_id": sequence_id,
                "camera_id": camera_id,
                "station_id": station,
                "frame_idx": i,
                "timestamp": timestamps[i],
                "offset_s": math.nan if offsets_s is None else float(offsets_s[i]),
                "label": int(labels[i]),
                "path": paths[i],
                "width": width,
                "height": height,
                "boxes": None if frame_boxes is None else [list(b) for b in frame_boxes],
            }
        )
    return records


def build_manifest(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build a validated manifest from per-frame dict records.

    Each record must provide every column in ``MANIFEST_COLUMNS``; extra keys become extra
    columns. ``offset_s`` may be omitted or None (null). Values are coerced to the canonical
    dtypes (timestamps must be tz-aware; they are converted to UTC), rows are sorted by
    ``(dataset, sequence_id, frame_idx)`` and the index is reset. Raises ``ManifestError``
    if the result violates any invariant.
    """
    rows = []
    for rec in records:
        row = dict(rec)
        if row.get("offset_s") is None:
            row["offset_s"] = math.nan
        rows.append(row)
    if not rows:
        return empty_manifest()
    raw = pd.DataFrame.from_records(rows)
    if "boxes" in raw.columns:
        raw["boxes"] = pd.Series([r.get("boxes") for r in rows], dtype=object)
    df = _coerce(raw)
    df = df.sort_values(list(SORT_KEY), kind="stable").reset_index(drop=True)
    validate_manifest(df)
    return df


def _arrow_schema(df: pd.DataFrame) -> Any:
    import pyarrow as pa

    fields = [
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("camera_id", pa.string(), nullable=False),
        pa.field("station_id", pa.string(), nullable=False),
        pa.field("frame_idx", pa.int64(), nullable=False),
        pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("offset_s", pa.float64(), nullable=True),
        pa.field("label", pa.int8(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("width", pa.int64(), nullable=False),
        pa.field("height", pa.int64(), nullable=False),
        pa.field("boxes", pa.list_(pa.list_(pa.float64())), nullable=True),
    ]
    extra = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    if extra:
        inferred = pa.Schema.from_pandas(df[extra], preserve_index=False)
        fields.extend(inferred.field(c) for c in extra)
    return pa.schema(fields)


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Validate ``df`` and write it to Parquet with an explicit Arrow schema.

    ``boxes`` is stored as a nullable ``list<list<float64>>``: Arrow keeps a null list and
    an empty list distinct, so "not annotated" and "annotated, no smoke" survive the round
    trip. The explicit schema also fixes the column type when every cell is null or empty
    (otherwise pyarrow would infer ``null`` or ``list<null>``). ``offset_s`` NaN is written
    as a Parquet null. The index is not stored.
    """
    validate_manifest(df)
    canonical = _coerce(df)
    canonical.to_parquet(Path(path), engine="pyarrow", index=False, schema=_arrow_schema(canonical))


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest written by ``write_manifest`` and return it with canonical dtypes.

    pyarrow hands nested lists back as numpy object arrays; they are normalised to Python
    lists of float lists, and null cells to ``None`` (an empty stored list stays ``[]``).
    The result is validated; row order is preserved.
    """
    raw = pd.read_parquet(Path(path), engine="pyarrow")
    df = _coerce(raw.reset_index(drop=True))
    validate_manifest(df)
    return df


def sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise a manifest with one row per ``(dataset, sequence_id)``.

    Columns: ``dataset``, ``sequence_id``, ``camera_id``, ``station_id``, ``n_frames``,
    ``n_positive`` (frames with label 1), ``has_fire`` (``n_positive > 0``),
    ``start_timestamp``, ``end_timestamp`` (first and last capture time) and
    ``ignition_timestamp`` (timestamp of the earliest label-1 frame, NaT if none). Rows are
    sorted by ``(dataset, sequence_id)`` with a fresh index. Assumes a valid manifest.
    """
    keys = list(SEQUENCE_KEY)
    g = df.groupby(keys, sort=True, observed=True)
    table = g.agg(
        camera_id=("camera_id", "first"),
        station_id=("station_id", "first"),
        n_frames=("frame_idx", "size"),
        n_positive=("label", lambda s: int((s == 1).sum())),
        start_timestamp=("timestamp", "min"),
        end_timestamp=("timestamp", "max"),
    )
    positive = df.loc[df["label"] == 1]
    ignition = positive.groupby(keys, sort=True, observed=True)["timestamp"].min()
    table["ignition_timestamp"] = ignition.reindex(table.index).astype(TIMESTAMP_DTYPE)
    table["n_frames"] = table["n_frames"].astype(np.int64)
    table["n_positive"] = table["n_positive"].astype(np.int64)
    table["has_fire"] = table["n_positive"] > 0
    table = table.reset_index()
    order = [
        "dataset",
        "sequence_id",
        "camera_id",
        "station_id",
        "n_frames",
        "n_positive",
        "has_fire",
        "start_timestamp",
        "end_timestamp",
        "ignition_timestamp",
    ]
    return table[order]
