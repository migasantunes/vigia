"""Tests for vigia.data.tiling."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vigia.data.tiling import boxes_to_tile_labels, tile_grid, tile_index_map, tile_starts


def test_tiny_grid_even() -> None:
    # 5 x 7, tile 3, overlap 1 -> stride 2. Rows start at 0, 2 (2 + 3 = 5 = height);
    # columns at 0, 2, 4 (4 + 3 = 7 = width). Row-major.
    expected = np.array(
        [
            [0, 0, 3, 3],
            [0, 2, 3, 5],
            [0, 4, 3, 7],
            [2, 0, 5, 3],
            [2, 2, 5, 5],
            [2, 4, 5, 7],
        ]
    )
    grid = tile_grid(5, 7, tile=3, overlap=1)
    assert grid.dtype == np.int64
    np.testing.assert_array_equal(grid, expected)


def test_tiny_grid_last_row_and_column_shifted_inwards() -> None:
    # 6 x 8, tile 3, overlap 1 -> stride 2. Regular rows 0, 2 end at 5 < 6, so a third row
    # is shifted in to start at 6 - 3 = 3 (not 4, which would overrun). Columns: 0, 2, 4
    # end at 7 < 8, so a fourth starts at 8 - 3 = 5.
    np.testing.assert_array_equal(tile_starts(6, 3, 1), [0, 2, 3])
    np.testing.assert_array_equal(tile_starts(8, 3, 1), [0, 2, 4, 5])
    grid = tile_grid(6, 8, tile=3, overlap=1)
    assert grid.shape == (12, 4)
    np.testing.assert_array_equal(grid[3], [0, 5, 3, 8])
    np.testing.assert_array_equal(grid[-1], [3, 5, 6, 8])


def test_tile_equal_to_image() -> None:
    np.testing.assert_array_equal(tile_grid(4, 4, 4, 0), [[0, 0, 4, 4]])


def test_smokeynet_45_tiles() -> None:
    # SmokeyNet (Dewangan et al. 2022): 1040 x 1856 frames, 224 px tiles, 20 px overlap.
    # Stride 204. Rows: (1040 - 224) / 204 = 4 strides -> 5 rows, last ends at 4*204+224 =
    # 1040 exactly. Columns: (1856 - 224) / 204 = 8 strides -> 9 columns, last ends at
    # 8*204+224 = 1856 exactly. 5 x 9 = 45 and no shifted tile is needed.
    np.testing.assert_array_equal(tile_starts(1040, 224, 20), np.arange(5) * 204)
    np.testing.assert_array_equal(tile_starts(1856, 224, 20), np.arange(9) * 204)
    grid = tile_grid(1040, 1856, tile=224, overlap=20)
    assert grid.shape == (45, 4)
    assert grid[:, 2].max() == 1040
    assert grid[:, 3].max() == 1856


@pytest.mark.parametrize(
    ("height", "width", "tile", "overlap"),
    [
        (h, w, t, o)
        for h, w, t, o in itertools.product([7, 10, 31, 64], [7, 13, 50], [3, 5, 7], [0, 1, 2, 4])
        if t <= min(h, w) and o < t
    ],
)
def test_grid_properties_sweep(height: int, width: int, tile: int, overlap: int) -> None:
    grid = tile_grid(height, width, tile, overlap)
    assert grid.ndim == 2
    assert grid.shape[1] == 4
    assert (grid[:, 2] - grid[:, 0] == tile).all()
    assert (grid[:, 3] - grid[:, 1] == tile).all()
    assert (grid[:, :2] >= 0).all()
    assert (grid[:, 2] <= height).all()
    assert (grid[:, 3] <= width).all()
    assert (tile_index_map(grid, height, width) >= 1).all()
    order = np.lexsort((grid[:, 1], grid[:, 0]))
    np.testing.assert_array_equal(order, np.arange(len(grid)))
    stride = tile - overlap
    for length in (height, width):
        starts = tile_starts(length, tile, overlap)
        assert starts[0] == 0
        assert starts[-1] + tile == length
        assert (np.diff(starts) > 0).all()
        assert (np.diff(starts) <= stride).all()
        assert (np.diff(starts)[:-1] == stride).all()
        # Minimal: one fewer tile at the regular stride would not reach the edge.
        assert len(starts) == 1 or (len(starts) - 2) * stride + tile < length
    n_rows = len(tile_starts(height, tile, overlap))
    n_cols = len(tile_starts(width, tile, overlap))
    assert len(grid) == n_rows * n_cols


@pytest.mark.parametrize(
    ("height", "width", "tile", "overlap"),
    [(5, 7, 6, 0), (7, 5, 6, 0), (10, 10, 3, 3), (10, 10, 3, 4), (10, 10, 3, -1), (10, 10, 0, 0)],
)
def test_grid_errors(height: int, width: int, tile: int, overlap: int) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case hits a different message
        tile_grid(height, width, tile, overlap)


def test_tile_index_map_hand_case() -> None:
    grid = tile_grid(5, 7, tile=3, overlap=1)
    counts = tile_index_map(grid, 5, 7)
    assert counts.dtype == np.int32
    rows = np.array([1, 1, 2, 1, 1])
    cols = np.array([1, 1, 2, 1, 2, 1, 1])
    np.testing.assert_array_equal(counts, np.outer(rows, cols))


# Labels: 10 x 10 image, tile 5, overlap 0 -> tiles 0 [0:5, 0:5], 1 [0:5, 5:10],
# 2 [5:10, 0:5], 3 [5:10, 5:10]; each tile has area 25.
GRID_10 = tile_grid(10, 10, tile=5, overlap=0)


def test_labels_empty_boxes() -> None:
    for empty in (None, [], np.zeros((0, 4))):
        labels = boxes_to_tile_labels(empty, GRID_10, 10, 10)
        assert labels.dtype == np.int8
        np.testing.assert_array_equal(labels, [0, 0, 0, 0])


def test_labels_hand_case_and_x_first_order() -> None:
    # x0=0.6, y0=0.1, x1=0.8, y1=0.2 -> pixels x 6..8, y 1..2: top-right tile only.
    labels = boxes_to_tile_labels([[0.6, 0.1, 0.8, 0.2]], GRID_10, 10, 10)
    np.testing.assert_array_equal(labels, [0, 1, 0, 0])
    # Non-square image: 10 high, 20 wide, tile 10 -> tiles at x 0 and 10. Box x 0.6..0.7
    # is pixels 12..14, i.e. the right tile; with y/x swapped it would land elsewhere.
    grid = tile_grid(10, 20, tile=10, overlap=0)
    np.testing.assert_array_equal(
        boxes_to_tile_labels([[0.6, 0.1, 0.7, 0.2]], grid, 10, 20), [0, 1]
    )


def test_labels_box_touching_edge_is_not_overlap() -> None:
    # Pixels x 5..7, y 0..2: touches tile 0 along x = 5 with zero area.
    labels = boxes_to_tile_labels([[0.5, 0.0, 0.7, 0.2]], GRID_10, 10, 10)
    np.testing.assert_array_equal(labels, [0, 1, 0, 0])


def test_labels_centre_box_hits_all_four_tiles() -> None:
    # Pixels 4..6 in both axes: 1 px^2 in each tile, i.e. fraction 1/25 = 0.04.
    box = [[0.4, 0.4, 0.6, 0.6]]
    np.testing.assert_array_equal(boxes_to_tile_labels(box, GRID_10, 10, 10), [1, 1, 1, 1])
    np.testing.assert_array_equal(
        boxes_to_tile_labels(box, GRID_10, 10, 10, min_fraction=0.039), [1, 1, 1, 1]
    )
    # Strictly more than min_fraction: exactly 0.04 is not enough.
    np.testing.assert_array_equal(
        boxes_to_tile_labels(box, GRID_10, 10, 10, min_fraction=0.04), [0, 0, 0, 0]
    )


def test_labels_union_not_double_counted() -> None:
    # In tile 0: box A pixels [0,3)^2 (area 9), box B [2,4)^2 (area 4), overlap [2,3)^2
    # (area 1). Union = 12 -> 0.48 of the tile; a plain sum would give 13 -> 0.52.
    boxes = [[0.0, 0.0, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]
    assert boxes_to_tile_labels(boxes, GRID_10, 10, 10, min_fraction=0.47)[0] == 1
    assert boxes_to_tile_labels(boxes, GRID_10, 10, 10, min_fraction=0.48)[0] == 0
    assert boxes_to_tile_labels(boxes, GRID_10, 10, 10, min_fraction=0.5)[0] == 0
    # A duplicated box changes nothing.
    assert boxes_to_tile_labels(boxes * 3, GRID_10, 10, 10, min_fraction=0.48)[0] == 0


def test_labels_full_cover_and_fraction_one() -> None:
    whole = [[0.0, 0.0, 1.0, 1.0]]
    np.testing.assert_array_equal(
        boxes_to_tile_labels(whole, GRID_10, 10, 10, min_fraction=0.999), [1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        boxes_to_tile_labels(whole, GRID_10, 10, 10, min_fraction=1.0), [0, 0, 0, 0]
    )


def test_labels_overlapping_tiles() -> None:
    # 5 x 7, tile 3, overlap 1. Box pixels x 4..5, y 0..1 lies in tiles 1 (x 2..5) and 2
    # (x 4..7) of the top row only.
    grid = tile_grid(5, 7, tile=3, overlap=1)
    labels = boxes_to_tile_labels([[4 / 7, 0.0, 5 / 7, 1 / 5]], grid, 5, 7)
    np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0, 0])


def test_labels_errors() -> None:
    with pytest.raises(ValueError, match="min_fraction"):
        boxes_to_tile_labels([[0, 0, 1, 1]], GRID_10, 10, 10, min_fraction=1.5)
    with pytest.raises(ValueError, match="shape"):
        boxes_to_tile_labels([[0, 0, 1]], GRID_10, 10, 10)
