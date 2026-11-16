"""Shared test fixtures.

``make_manifest`` is a factory fixture that builds a small, valid, deterministic synthetic
manifest. Use it as::

    def test_something(make_manifest):
        df = make_manifest(n_stations=3, frames_per_sequence=8)

Layout (all ids globally unique and readable):

- datasets ``ds0, ds1, ...``; stations ``ds0-st0``; cameras ``ds0-st0-cam0``;
  sequences ``ds0-st0-cam0-seq0``.
- Within each camera, even-numbered sequences (``seq0``, ``seq2``, ...) are fire sequences
  and odd-numbered ones are no-fire sequences.
- Frames are 60 s apart; each sequence starts at its own hour so cameras never overlap.
- Fire sequence with ``F`` frames: ignition at frame ``F // 2``; ``offset_s`` runs
  ``(i - F // 2) * 60`` (negative before ignition, 0 at ignition); ``label`` is 0 before and
  1 from ignition on; positive frames carry one box ``[0.4, 0.5, 0.45 + 0.5 k / F, 0.6]`` that
  grows with ``k`` frames since ignition (always inside the image); negative frames carry
  ``[]`` (annotated, no smoke).
- No-fire sequence: ``offset_s`` null, ``label`` 0, ``boxes`` None (not annotated).
- Images are ``width`` x ``height`` (default 640 x 480).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from vigia.data.schema import build_manifest

MakeManifest = Callable[..., pd.DataFrame]


def build_synthetic_manifest(
    n_datasets: int = 1,
    n_stations: int = 2,
    cameras_per_station: int = 2,
    sequences_per_camera: int = 2,
    frames_per_sequence: int = 6,
    width: int = 640,
    height: int = 480,
) -> pd.DataFrame:
    """Build the synthetic manifest described in the module docstring."""
    if frames_per_sequence < 2:
        raise ValueError("frames_per_sequence must be at least 2 to have a pre-ignition frame")
    base = pd.Timestamp("2020-06-01T00:00:00Z")
    ignition = frames_per_sequence // 2
    records: list[dict[str, object]] = []
    seq_counter = 0
    for d in range(n_datasets):
        dataset = f"ds{d}"
        for s in range(n_stations):
            station = f"{dataset}-st{s}"
            for c in range(cameras_per_station):
                camera = f"{station}-cam{c}"
                for q in range(sequences_per_camera):
                    sequence = f"{camera}-seq{q}"
                    fire = q % 2 == 0
                    start = base + pd.Timedelta(hours=seq_counter)
                    seq_counter += 1
                    for i in range(frames_per_sequence):
                        if fire:
                            positive = i >= ignition
                            k = i - ignition
                            boxes: list[list[float]] | None = (
                                [[0.4, 0.5, 0.45 + 0.5 * k / frames_per_sequence, 0.6]]
                                if positive
                                else []
                            )
                            offset: float | None = float(k * 60)
                        else:
                            positive = False
                            boxes = None
                            offset = None
                        records.append(
                            {
                                "dataset": dataset,
                                "sequence_id": sequence,
                                "camera_id": camera,
                                "station_id": station,
                                "frame_idx": i,
                                "timestamp": start + pd.Timedelta(seconds=60 * i),
                                "offset_s": offset,
                                "label": int(positive),
                                "path": f"{camera}/{sequence}/{i:04d}.jpg",
                                "width": width,
                                "height": height,
                                "boxes": boxes,
                            }
                        )
    return build_manifest(records)


@pytest.fixture
def make_manifest() -> MakeManifest:
    """Factory fixture: ``make_manifest(**kwargs)`` calls ``build_synthetic_manifest``."""
    return build_synthetic_manifest
