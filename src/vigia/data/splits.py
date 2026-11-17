"""Group-aware split assignment and leakage checks (``docs/ARCHITECTURE.md`` §5).

An *assignment* is a ``pd.Series`` of split names (``pd.StringDtype``) indexed like the
manifest, one entry per frame row; ``pd.NA`` means "in no split". Every function here
assigns whole groups, is deterministic for a given seed, and gives the same per-row result
whatever the manifest row order (groups are sorted by key before the seeded shuffle).

Identifier columns other than ``dataset`` are compared *qualified by dataset*: the contract
only makes ``sequence_id`` unique within a dataset, and two datasets that happen to reuse a
camera or station name do not share a physical camera. So for every key except ``dataset``
the value used is the pair ``(dataset, value)``, rendered as ``"dataset:value"`` in messages.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "LeakageError",
    "assert_disjoint",
    "assignment_from_lists",
    "assignment_hash",
    "leakage_report",
    "leave_groups_out_folds",
    "split_by_group",
]

_SPLIT_DTYPE = pd.StringDtype()
_SEQ = ["dataset", "sequence_id"]
_MAX_SHOWN = 5


class LeakageError(ValueError):
    """A key value appears in more than one split."""


def _key_columns(group_key: str | Sequence[str]) -> list[str]:
    cols = [group_key] if isinstance(group_key, str) else list(group_key)
    if not cols:
        raise ValueError("group_key must name at least one column")
    return cols


def _group_sizes(manifest: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return (groups sorted by key with their sequence count, per-row group number).

    Raises ValueError if a sequence spans several groups, because splitting one sequence
    across splits is leakage by construction.
    """
    missing = [c for c in [*cols, *_SEQ] if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing column(s) {missing}")
    if len(manifest) == 0:
        raise ValueError("manifest is empty")
    others = [c for c in cols if c not in _SEQ]
    per_seq = manifest.groupby(_SEQ, sort=True, observed=True)[others].nunique()
    spanning = per_seq[(per_seq > 1).any(axis=1)]
    if len(spanning):
        shown = [f"{d}:{s}" for d, s in spanning.index[:_MAX_SHOWN]]
        raise ValueError(
            f"group_key {cols} is not constant within sequence(s) {shown}; "
            "a sequence must belong to exactly one group"
        )
    keyed = manifest[list(dict.fromkeys(cols + _SEQ))].drop_duplicates()
    groups = (
        keyed.groupby(cols, sort=True, observed=True, dropna=False)
        .size()
        .rename("n_sequences")
        .reset_index()
    )
    row_group = manifest.groupby(cols, sort=True, observed=True, dropna=False).ngroup()
    return groups, row_group


def _check_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if not fractions:
        raise ValueError("fractions must name at least one split")
    out = {str(k): float(v) for k, v in fractions.items()}
    bad = {k: v for k, v in out.items() if not math.isfinite(v) or v < 0}
    if bad:
        raise ValueError(f"fractions must be finite and non-negative, got {bad}")
    total = math.fsum(out.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1 (tolerance 1e-9), got {total!r}")
    return out


def split_by_group(
    manifest: pd.DataFrame,
    group_key: str | Sequence[str],
    fractions: Mapping[str, float],
    seed: int,
) -> pd.Series:
    """Assign whole groups to named splits so each split's share of sequences tracks a target.

    ``group_key`` is a column name or a sequence of column names (e.g.
    ``("dataset", "station_id")``); a group is one distinct value of that key, and every row
    of a group gets the same split. ``fractions`` maps split name to its target share of
    *sequences* (distinct ``(dataset, sequence_id)``), must be non-negative and sum to 1.

    Algorithm: groups are sorted by key (row-order independence), permuted with
    ``numpy.random.default_rng(seed)``, then assigned one at a time to the split with the
    largest remaining deficit ``fraction * total_sequences - assigned_sequences`` (ties go to
    the split listed first in ``fractions``). To guarantee that every split with a positive
    fraction receives at least one group, once the number of unassigned groups equals the
    number of still-empty positive splits, each remaining group goes to the empty split with
    the largest deficit. Splits with fraction 0 receive nothing.

    Raises ValueError if fractions are invalid, if there are fewer groups than splits with a
    positive fraction, or if a sequence spans several groups. Returns a ``string`` Series
    named ``"split"`` indexed like ``manifest``.
    """
    fr = _check_fractions(fractions)
    cols = _key_columns(group_key)
    groups, row_group = _group_sizes(manifest, cols)
    positive = [name for name, f in fr.items() if f > 0]
    n_groups = len(groups)
    if n_groups < len(positive):
        raise ValueError(
            f"{n_groups} group(s) of {cols} cannot fill {len(positive)} split(s) "
            f"with positive fraction {positive}"
        )
    sizes = groups["n_sequences"].to_numpy(dtype=np.int64)
    total = int(sizes.sum())
    names = list(fr)
    targets = np.array([fr[n] * total for n in names], dtype=np.float64)
    assigned = np.zeros(len(names), dtype=np.float64)
    n_groups_in = np.zeros(len(names), dtype=np.int64)
    is_positive = np.array([fr[n] > 0 for n in names], dtype=bool)
    order = np.random.default_rng(seed).permutation(n_groups)
    group_split = np.empty(n_groups, dtype=np.int64)
    for step, g in enumerate(order):
        remaining = n_groups - step
        deficit = targets - assigned
        empty_positive = is_positive & (n_groups_in == 0)
        candidates = empty_positive if remaining <= int(empty_positive.sum()) else is_positive
        choice = int(np.argmax(np.where(candidates, deficit, -np.inf)))
        group_split[g] = choice
        assigned[choice] += sizes[g]
        n_groups_in[choice] += 1
    labels = np.array(names, dtype=object)[group_split[row_group.to_numpy()]]
    return pd.Series(labels, index=manifest.index, dtype=_SPLIT_DTYPE, name="split")


def leave_groups_out_folds(
    manifest: pd.DataFrame,
    group_key: str | Sequence[str],
    n_folds: int,
    seed: int,
) -> list[pd.Series]:
    """Return ``n_folds`` train/test assignments where each group is tested exactly once.

    Fold balancing: groups are sorted by key, permuted with ``default_rng(seed)``, then
    stably sorted by sequence count (largest first) and each is placed in the fold that
    currently holds the fewest sequences (ties to the lowest fold number). This is the
    longest-processing-time heuristic: fold sizes in sequences differ by at most the size of
    the largest group, and every fold gets at least one group. The seed decides how
    equal-sized groups are distributed. Fold ``i`` is ``"test"`` on its groups and
    ``"train"`` elsewhere. Raises ValueError if ``n_folds < 2`` or there are fewer groups
    than folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    cols = _key_columns(group_key)
    groups, row_group = _group_sizes(manifest, cols)
    n_groups = len(groups)
    if n_groups < n_folds:
        raise ValueError(f"{n_groups} group(s) of {cols} cannot fill {n_folds} folds")
    sizes = groups["n_sequences"].to_numpy(dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(n_groups)
    order = perm[np.argsort(-sizes[perm], kind="stable")]
    load = np.zeros(n_folds, dtype=np.int64)
    group_fold = np.empty(n_groups, dtype=np.int64)
    for g in order:
        f = int(np.argmin(load))
        group_fold[g] = f
        load[f] += sizes[g]
    row_fold = group_fold[row_group.to_numpy()]
    folds = []
    for f in range(n_folds):
        labels = np.where(row_fold == f, "test", "train").astype(object)
        folds.append(pd.Series(labels, index=manifest.index, dtype=_SPLIT_DTYPE, name="split"))
    return folds


def assignment_from_lists(
    manifest: pd.DataFrame,
    lists: Mapping[str, Iterable[str]],
    key: str = "sequence_id",
    dataset: str | None = None,
    on_missing: Literal["raise", "ignore"] = "raise",
) -> pd.Series:
    """Build an assignment from published id lists, e.g. the official FIgLib fire split.

    ``lists`` maps split name to the ids (values of column ``key``, compared as strings) in
    that split. If ``dataset`` is given only rows of that dataset are matched; other rows get
    NA silently. Matched-scope rows whose id is in no list get NA and trigger a
    ``UserWarning`` naming how many there are and the first few ids. Raises ValueError if an
    id appears in two lists, or (unless ``on_missing="ignore"``) if an id in the lists does
    not occur in the manifest scope.
    """
    if on_missing not in ("raise", "ignore"):
        raise ValueError(f"on_missing must be 'raise' or 'ignore', got {on_missing!r}")
    if key not in manifest.columns:
        raise ValueError(f"manifest has no column {key!r}")
    id_to_split: dict[str, str] = {}
    for split, ids in lists.items():
        for raw in ids:
            ident = str(raw)
            previous = id_to_split.get(ident)
            if previous is not None and previous != split:
                raise ValueError(f"id {ident!r} appears in both {previous!r} and {split!r}")
            id_to_split[ident] = str(split)

    in_scope = (
        np.ones(len(manifest), dtype=bool)
        if dataset is None
        else (manifest["dataset"] == dataset).to_numpy(dtype=bool, na_value=False)
    )
    values = manifest[key].astype(str).to_numpy(dtype=object)
    scope_values = set(values[in_scope])
    absent = sorted(set(id_to_split) - scope_values)
    if absent and on_missing == "raise":
        where = "manifest" if dataset is None else f"dataset {dataset!r}"
        raise ValueError(
            f"{len(absent)} listed id(s) not found in {where} column {key!r}: {absent[:_MAX_SHOWN]}"
        )
    mapped = [
        id_to_split.get(v) if scope else None for v, scope in zip(values, in_scope, strict=True)
    ]
    result = pd.Series(mapped, index=manifest.index, dtype=_SPLIT_DTYPE, name="split")
    unlisted_mask = in_scope & result.isna().to_numpy()
    if unlisted_mask.any():
        unlisted = sorted(set(values[unlisted_mask]))
        warnings.warn(
            f"{int(unlisted_mask.sum())} row(s) with {len(unlisted)} {key} value(s) are in no "
            f"list and get no split: {unlisted[:_MAX_SHOWN]}",
            UserWarning,
            stacklevel=2,
        )
    return result


def _qualified(manifest: pd.DataFrame, key: str) -> pd.Series:
    """Values of ``key`` as comparable strings, qualified by dataset unless key is dataset."""
    if key not in manifest.columns:
        raise ValueError(f"manifest has no column {key!r}")
    values = manifest[key].astype(str)
    if key == "dataset":
        return values
    return manifest["dataset"].astype(str) + ":" + values


def _check_assignment(manifest: pd.DataFrame, assignment: pd.Series) -> pd.Series:
    if not assignment.index.equals(manifest.index):
        raise ValueError("assignment must be indexed exactly like the manifest")
    return assignment.astype(_SPLIT_DTYPE)


def _split_value_sets(
    manifest: pd.DataFrame, assignment: pd.Series, key: str
) -> tuple[list[str], dict[str, set[str]], pd.Series, pd.Series]:
    split = _check_assignment(manifest, assignment)
    keep = split.notna().to_numpy(dtype=bool)
    values = _qualified(manifest, key)[keep]
    split = split[keep].astype(str)
    names = sorted(split.unique())
    sets = {n: set(values[split == n]) for n in names}
    return names, sets, values, split


def assert_disjoint(manifest: pd.DataFrame, assignment: pd.Series, keys: Sequence[str]) -> None:
    """Raise ``LeakageError`` if any value of any key appears in more than one split.

    Rows with NA assignment are ignored. Keys are checked in the given order and split pairs
    in sorted order; the error names the key, the two splits and up to 5 shared values
    (``"dataset:value"``).
    """
    for key in keys:
        names, sets, _, _ = _split_value_sets(manifest, assignment, key)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                shared = sorted(sets[a] & sets[b])
                if shared:
                    raise LeakageError(
                        f"key {key!r}: {len(shared)} value(s) shared between splits {a!r} and "
                        f"{b!r}, e.g. {shared[:_MAX_SHOWN]}"
                    )


def leakage_report(
    manifest: pd.DataFrame, assignment: pd.Series, keys: Sequence[str]
) -> pd.DataFrame:
    """Count shared key values between every ordered pair of distinct splits.

    One row per ``(key, split_a, split_b)`` with ``split_a != split_b`` (both orders, since
    the fractions are asymmetric), keys in the given order and split names sorted. Columns:
    ``n_shared`` (distinct values in both), ``n_a`` and ``n_b`` (distinct values in each),
    ``frac_of_b_seen_in_a = n_shared / n_b`` and ``frac_rows_b_with_key_in_a`` (share of
    split-b rows whose value also occurs in split a). Rows with NA assignment are ignored.
    """
    records: list[dict[str, Any]] = []
    for key in keys:
        names, sets, values, split = _split_value_sets(manifest, assignment, key)
        for a in names:
            for b in names:
                if a == b:
                    continue
                shared = sets[a] & sets[b]
                rows_b = values[split == b]
                n_b = len(sets[b])
                records.append(
                    {
                        "key": key,
                        "split_a": a,
                        "split_b": b,
                        "n_shared": len(shared),
                        "n_a": len(sets[a]),
                        "n_b": n_b,
                        "frac_of_b_seen_in_a": len(shared) / n_b if n_b else 0.0,
                        "frac_rows_b_with_key_in_a": float(rows_b.isin(sets[a]).mean())
                        if len(rows_b)
                        else 0.0,
                    }
                )
    columns = [
        "key",
        "split_a",
        "split_b",
        "n_shared",
        "n_a",
        "n_b",
        "frac_of_b_seen_in_a",
        "frac_rows_b_with_key_in_a",
    ]
    report = pd.DataFrame.from_records(records, columns=columns)
    return report.astype(
        {
            "n_shared": np.int64,
            "n_a": np.int64,
            "n_b": np.int64,
            "frac_of_b_seen_in_a": np.float64,
            "frac_rows_b_with_key_in_a": np.float64,
        }
    )


def assignment_hash(manifest: pd.DataFrame, assignment: pd.Series) -> str:
    """Return a 16-hex-character SHA-256 digest of the assignment, independent of row order.

    The digest covers the sorted distinct ``(dataset, sequence_id, split)`` triples, with NA
    written as ``<NA>``, one tab-separated triple per line in UTF-8. It changes whenever any
    sequence changes split (including to or from NA) and is unaffected by row order or index.
    """
    split = _check_assignment(manifest, assignment)
    triples = pd.DataFrame(
        {
            "dataset": manifest["dataset"].astype(str).to_numpy(dtype=object),
            "sequence_id": manifest["sequence_id"].astype(str).to_numpy(dtype=object),
            "split": split.fillna("<NA>").astype(str).to_numpy(dtype=object),
        }
    ).drop_duplicates()
    lines = sorted("\t".join(t) for t in triples.itertuples(index=False, name=None))
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    return digest[:16]
