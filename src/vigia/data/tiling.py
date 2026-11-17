"""Tile grid, box-to-tile labels and tile coverage maps (``docs/ARCHITECTURE.md`` §6).

Pixel coordinates are half-open: a tile ``[y0, x0, y1, x1]`` covers rows ``y0 .. y1-1`` and
columns ``x0 .. x1-1``, i.e. the continuous region ``[y0, y1) x [x0, x1)``. Normalised box
coordinates map to pixels by multiplying x by ``width`` and y by ``height``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["boxes_to_tile_labels", "tile_grid", "tile_index_map", "tile_starts"]


def tile_starts(length: int, tile: int, overlap: int) -> NDArray[np.int64]:
    """Start offsets of tiles along one axis of ``length`` pixels.

    Starts are ``0, stride, 2*stride, ...`` with ``stride = tile - overlap``, using the
    fewest tiles that reach the end: ``n = ceil((length - tile) / stride) + 1``. If the last
    regular start would overrun, it is replaced by ``length - tile`` so the last tile ends
    exactly at the edge; it then overlaps its neighbour by more than ``overlap``. Starts are
    strictly increasing and every tile lies inside ``[0, length)``.
    """
    stride = tile - overlap
    n = -(-(length - tile) // stride) + 1
    starts = np.arange(n, dtype=np.int64) * stride
    starts[-1] = length - tile
    return starts


def tile_grid(height: int, width: int, tile: int, overlap: int) -> NDArray[np.int64]:
    """Return an ``(N, 4)`` int64 array of tiles ``[y0, x0, y1, x1]`` in row-major order.

    Every tile is exactly ``tile`` x ``tile`` pixels, the union of tiles is the whole image,
    consecutive tiles along an axis overlap by ``overlap`` pixels except that the last
    row/column is shifted inwards to end at the image edge (never padded, never out of
    bounds), see ``tile_starts``. For 1040 x 1856 with tile 224 and overlap 20 this gives
    5 x 9 = 45 tiles, matching SmokeyNet. Raises ValueError if ``tile`` exceeds either image
    side or ``overlap`` is not in ``[0, tile)``.
    """
    for name, value in (("height", height), ("width", width), ("tile", tile)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if tile > height or tile > width:
        raise ValueError(f"tile {tile} does not fit in a {height} x {width} image")
    if overlap < 0 or overlap >= tile:
        raise ValueError(f"overlap must be in [0, tile={tile}), got {overlap}")
    ys = tile_starts(height, tile, overlap)
    xs = tile_starts(width, tile, overlap)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    y0 = yy.ravel()
    x0 = xx.ravel()
    return np.stack([y0, x0, y0 + tile, x0 + tile], axis=1).astype(np.int64)


def _as_boxes(boxes: ArrayLike | Sequence[Sequence[float]] | None) -> NDArray[np.float64]:
    if boxes is None:
        return np.zeros((0, 4), dtype=np.float64)
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"boxes must have shape (B, 4), got {arr.shape}")
    return arr


def _union_area_in_tiles(px: NDArray[np.float64], grid: NDArray[np.int64]) -> NDArray[np.float64]:
    """Exact area of the union of pixel-space boxes ``px`` (x0, y0, x1, y1) inside each tile.

    Coordinate compression: the distinct box x-edges and y-edges cut the plane into
    rectangular cells, each either fully inside the union or fully outside. The covered area
    in a tile is the sum over covered cells of the cell/tile overlap, computed as
    ``overlap_y @ covered @ overlap_x.T``. No rasterisation, no double counting.
    """
    xe = np.unique(np.concatenate([px[:, 0], px[:, 2]]))
    ye = np.unique(np.concatenate([px[:, 1], px[:, 3]]))
    cx0, cx1 = xe[:-1], xe[1:]
    cy0, cy1 = ye[:-1], ye[1:]
    covered = np.zeros((len(cy0), len(cx0)), dtype=bool)
    for x0, y0, x1, y1 in px:
        rows = (cy0 >= y0) & (cy1 <= y1)
        cols = (cx0 >= x0) & (cx1 <= x1)
        covered |= rows[:, None] & cols[None, :]
    g = grid.astype(np.float64)
    ov_y = np.clip(
        np.minimum(g[:, 2:3], cy1[None, :]) - np.maximum(g[:, 0:1], cy0[None, :]), 0.0, None
    )
    ov_x = np.clip(
        np.minimum(g[:, 3:4], cx1[None, :]) - np.maximum(g[:, 1:2], cx0[None, :]), 0.0, None
    )
    area: NDArray[np.float64] = np.einsum("ni,ij,nj->n", ov_y, covered.astype(np.float64), ov_x)
    return area


def boxes_to_tile_labels(
    boxes: ArrayLike | Sequence[Sequence[float]] | None,
    grid: NDArray[np.int64],
    height: int,
    width: int,
    min_fraction: float = 0.0,
) -> NDArray[np.int8]:
    """Label each tile 1 if the union of smoke boxes covers more than ``min_fraction`` of it.

    ``boxes`` are normalised ``[x0, y0, x1, y1]`` (x first, as in the manifest), shape
    ``(B, 4)``; ``None`` or empty gives all zeros. ``grid`` is ``tile_grid`` output
    (``[y0, x0, y1, x1]`` in pixels). The comparison is strict: a tile is positive when
    ``covered_area > min_fraction * tile_area``, so ``min_fraction=0`` means any overlap of
    positive area (a box that only touches a tile edge does not count) and ``min_fraction=1``
    can never fire. Covered area is computed exactly in continuous pixel coordinates (see
    ``_union_area_in_tiles``), so overlapping boxes are not double counted and sub-pixel box
    edges are honoured. Raises ValueError if ``min_fraction`` is outside ``[0, 1]``.
    """
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in [0, 1], got {min_fraction}")
    grid = np.asarray(grid, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError(f"grid must have shape (N, 4), got {grid.shape}")
    arr = _as_boxes(boxes)
    labels = np.zeros(len(grid), dtype=np.int8)
    if len(arr) == 0 or len(grid) == 0:
        return labels
    px = arr * np.array([width, height, width, height], dtype=np.float64)
    px = px[(px[:, 2] > px[:, 0]) & (px[:, 3] > px[:, 1])]
    if len(px) == 0:
        return labels
    area = _union_area_in_tiles(px, grid)
    tile_area = (grid[:, 2] - grid[:, 0]) * (grid[:, 3] - grid[:, 1])
    labels[area > min_fraction * tile_area] = 1
    return labels


def tile_index_map(grid: NDArray[np.int64], height: int, width: int) -> NDArray[np.int32]:
    """Return an ``(height, width)`` int32 map counting how many tiles cover each pixel.

    Built with a 2-D difference array and cumulative sums, so cost is O(N + H*W). Tiles
    are clipped to the image if they extend beyond it.
    """
    grid = np.asarray(grid, dtype=np.int64)
    diff = np.zeros((height + 1, width + 1), dtype=np.int32)
    y0 = np.clip(grid[:, 0], 0, height)
    x0 = np.clip(grid[:, 1], 0, width)
    y1 = np.clip(grid[:, 2], 0, height)
    x1 = np.clip(grid[:, 3], 0, width)
    keep = (y1 > y0) & (x1 > x0)
    y0, x0, y1, x1 = y0[keep], x0[keep], y1[keep], x1[keep]
    np.add.at(diff, (y0, x0), 1)
    np.add.at(diff, (y0, x1), -1)
    np.add.at(diff, (y1, x0), -1)
    np.add.at(diff, (y1, x1), 1)
    counts = diff.cumsum(axis=0).cumsum(axis=1)[:height, :width]
    return np.asarray(counts, dtype=np.int32)
