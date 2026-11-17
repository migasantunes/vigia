"""Pyro-SDIS adapter (plan Phase 5): Hugging Face dataset -> manifest.

What is known about the dataset, from its dataset card (``pyronear/pyro-sdis`` on Hugging
Face, Apache-2.0):

- v2.0 has 12,378 images: 10,744 with smoke and 1,634 false-positive images without.
- Splits: train 11,249 / val 1,129.
- Per-image fields: ``image``, ``annotations`` (YOLO format, one line per box:
  ``class x_center y_center width height``, normalised; class 0 = smoke),
  ``image_name``, ``partner``, ``camera``, ``date`` (``YYYY-MM-DDTHH-MM-SS``), ``split``.
- Smoke sequences are capped at 10 images.

``yolo_to_xyxy`` (implemented here) converts the ``annotations`` field to the manifest
box format. The rest of the adapter is Phase 5.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

SMOKE_CLASS = 0


def yolo_to_xyxy(lines: str) -> list[list[float]]:
    """Convert YOLO annotation text to manifest boxes ``[x0, y0, x1, y1]`` in [0, 1].

    Each non-blank line must be ``class x_center y_center width height``: an integer class
    followed by four finite numbers, separated by whitespace, with ``width`` and ``height``
    non-negative. Coordinates are clipped to [0, 1] after conversion. Blank lines are
    skipped, so an empty string gives ``[]`` (annotated, no box). Only class 0 (smoke)
    exists in Pyro-SDIS; any other class is rejected rather than silently kept.

    Raises:
        ValueError: on a line with the wrong number of fields, a non-integer or non-smoke
            class, a non-numeric or non-finite coordinate, or a negative width/height.
    """
    boxes: list[list[float]] = []
    for lineno, raw in enumerate(lines.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"line {lineno}: expected 5 fields, got {len(fields)}: {raw!r}")
        try:
            cls = int(fields[0])
        except ValueError:
            raise ValueError(f"line {lineno}: class must be an integer: {raw!r}") from None
        if cls != SMOKE_CLASS:
            raise ValueError(f"line {lineno}: unknown class {cls} (only 0 = smoke): {raw!r}")
        try:
            xc, yc, w, h = (float(v) for v in fields[1:])
        except ValueError:
            raise ValueError(f"line {lineno}: coordinates must be numbers: {raw!r}") from None
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            raise ValueError(f"line {lineno}: coordinates must be finite: {raw!r}")
        if w < 0 or h < 0:
            raise ValueError(f"line {lineno}: width and height must be >= 0: {raw!r}")
        box = [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
        boxes.append([min(1.0, max(0.0, v)) for v in box])
    return boxes


def build_pyrosdis_manifest(root_or_hf_cache: Path) -> pd.DataFrame:
    """Build the Pyro-SDIS manifest from a local copy of ``pyronear/pyro-sdis``.

    Must return one row per image with exactly the columns of ``docs/ARCHITECTURE.md`` §4,
    passing ``vigia.data.schema.validate_manifest``:

    - ``dataset = "pyrosdis"``; ``camera_id`` from ``partner`` and ``camera`` together
      (camera names are only meaningful within a partner); ``station_id`` derived the same
      way unless the card or data reveal a finer site identity (then document it in
      docs/DATA.md).
    - ``timestamp`` parsed from ``date`` (``YYYY-MM-DDTHH-MM-SS``); the time zone is not
      stated on the card and must be established before results depend on it.
    - ``sequence_id``: images are grouped into sequences by camera and time; the grouping
      rule (and its check against the 10-image cap) is a Phase 5 decision recorded in
      docs/DATA.md. ``frame_idx`` follows ``timestamp`` within a sequence.
    - ``offset_s`` null (the card gives no ignition time).
    - ``label`` = 1 if ``yolo_to_xyxy(annotations)`` is non-empty, else 0.
    - ``boxes`` = ``yolo_to_xyxy(annotations)``; ``path`` relative to the root (images
      exported from the HF dataset if needed); ``width``/``height`` from the image.
    - The card's ``split`` is kept out of the manifest: protocol P3 uses all of
      Pyro-SDIS as a test set.
    """
    raise NotImplementedError("Phase 5: Pyro-SDIS manifest (build_pyrosdis_manifest)")
