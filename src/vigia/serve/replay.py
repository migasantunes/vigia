"""Replay a recorded ignition through the running service (plan Phase 6). The demo."""

from __future__ import annotations

from pathlib import Path


def replay_sequence(
    manifest_path: Path,
    sequence_id: str,
    url: str = "http://127.0.0.1:8000",
    speedup: float = 60.0,
) -> float | None:
    """Post one manifest sequence's frames to ``url`` and report when the alert fired.

    Frames are sent in ``frame_idx`` order to ``POST /cameras/{camera_id}/frames`` with
    their recorded timestamps, sleeping ``(t[i+1] - t[i]) / speedup`` between posts. Prints
    one line per frame (offset in minutes, score, alert) and, at the end, the minute
    relative to ignition at which the first alert fired.

    Returns:
        Minutes from labelled ignition to the first alert at or after it (as defined in
        ``docs/EVALUATION.md`` §3), or ``None`` if no such alert fired.
    """
    raise NotImplementedError("Phase 6: replay demo (replay_sequence)")
