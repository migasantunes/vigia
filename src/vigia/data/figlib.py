"""FIgLib adapter (plan Phase 0): raw download -> manifest (``docs/ARCHITECTURE.md`` §4).

What is known about the dataset, from the FIgLib & SmokeyNet paper (Dewangan et al.,
Remote Sensing, 2022):

- 315 fire sequences, 24,800 images, from 101 cameras at 30 stations (HPWREN network).
- About 81 images per sequence, roughly 40 minutes before to 40 minutes after ignition,
  about 60 s apart.
- Image sizes are 1536 x 2048 or 2048 x 3072.
- 270 fires are used after removing 45 (night, black-and-white, or questionable).
- Official split by fire: 144 train / 64 validation / 62 test fires.
- Bounding boxes / contours are available for 144 fires.
- Hosted at the HPWREN FIgLib site.

Directory and filename conventions are NOT verified yet. Phase 0 task: inspect a real
download, document the convention in docs/DATA.md, then implement the parser with tests
built from real example names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_figlib_manifest(root: Path) -> pd.DataFrame:
    """Build the FIgLib manifest from a download rooted at ``root``.

    Must return one row per image with exactly the columns of ``docs/ARCHITECTURE.md`` §4,
    passing ``vigia.data.schema.validate_manifest``:

    - ``dataset = "figlib"``; ``sequence_id`` = one fire sequence; ``camera_id`` and
      ``station_id`` taken from the sequence's camera (both must be recoverable, because
      protocol P2 splits by station).
    - ``frame_idx`` in time order; ``timestamp`` in UTC.
    - ``offset_s`` = seconds relative to the labelled ignition, if the source provides it.
    - ``label`` = 1 where smoke is visible, 0 otherwise, so that the ignition time used
      by the evaluation (first frame with ``label == 1``) matches the source's labelled
      ignition; how the source encodes this is to be verified in Phase 0.
    - ``path`` relative to ``root``; ``width``/``height`` read from the image header.
    - ``boxes`` normalised ``[x0, y0, x1, y1]`` for annotated fires, null otherwise.

    How each field is recovered from paths and files is to be established in Phase 0
    from a real download and documented in docs/DATA.md before this is implemented.
    """
    raise NotImplementedError(
        "Phase 0: FIgLib manifest (inspect a real download, document it in docs/DATA.md first)"
    )


def load_official_split(path: Path) -> dict[str, list[str]]:
    """Load the published FIgLib train/validation/test fire lists.

    Returns ``{"train": [...], "val": [...], "test": [...]}`` whose values are
    ``sequence_id`` values as produced by ``build_figlib_manifest``, each fire in exactly
    one split (expected sizes 144 / 64 / 62), ready for
    ``vigia.data.splits.assignment_from_lists``. Where the lists are published and in
    what file format is to be established in Phase 0.
    """
    raise NotImplementedError("Phase 0: FIgLib official split loader (load_official_split)")
