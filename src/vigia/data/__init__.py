"""Data layer: manifest schema, group-aware splits and tiling.

The modules re-exported here are core-layer: they import only numpy, pandas and the
standard library (pyarrow is imported inside the Parquet IO functions). Dataset adapters
and transforms live in vision-layer modules of this package and are not imported here.
"""

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
from vigia.data.splits import (
    LeakageError,
    assert_disjoint,
    assignment_from_lists,
    assignment_hash,
    leakage_report,
    leave_groups_out_folds,
    split_by_group,
)
from vigia.data.tiling import boxes_to_tile_labels, tile_grid, tile_index_map

__all__ = [
    "MANIFEST_COLUMNS",
    "MANIFEST_DTYPES",
    "LeakageError",
    "ManifestError",
    "assert_disjoint",
    "assignment_from_lists",
    "assignment_hash",
    "boxes_to_tile_labels",
    "build_manifest",
    "empty_manifest",
    "leakage_report",
    "leave_groups_out_folds",
    "make_frame_records",
    "read_manifest",
    "sequence_table",
    "split_by_group",
    "tile_grid",
    "tile_index_map",
    "validate_manifest",
    "write_manifest",
]
