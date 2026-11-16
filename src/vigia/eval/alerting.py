"""Streaming alert policy: turns per-frame scores into alert events for one camera.

This module is the single implementation used by both the evaluation harness and the
alert service (ARCHITECTURE.md §1, §8), so the reported numbers describe the deployed
behaviour.

Definitions implemented here:

- A frame is **hot** when ``score >= threshold`` (a score equal to the threshold is hot).
- An alert **fires** on the current frame when at least ``k`` of the last ``n`` frames,
  including the current one, are hot, and the camera is not inside the cooldown window.
- The camera is **inside the cooldown window** when an alert has fired before and
  ``timestamp_s - last_alert_time_s < cooldown_s``.
- When an alert fires, the hot-frame window is cleared, so a new alert needs ``k`` hot
  frames observed after the alert. Frames observed *during* the cooldown are new
  evidence: they enter the window as usual and can fire an alert on the first frame
  after the cooldown expires.
- Timestamps must be strictly increasing; a non-increasing timestamp raises ValueError.
  Scores must be finite (NaN or infinite scores raise ValueError rather than being
  silently treated as cold).

Irregular frame spacing: the ``k``-of-``n`` window counts **frames**, not seconds. With
irregularly spaced frames the window may therefore span a variable amount of time; only
the cooldown is expressed in seconds. Frame counts are used because scorers and
annotations are per frame, and a gap in the recording should neither create nor destroy
evidence.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["AlertConfig", "AlertPolicy", "run_policy"]


@dataclass(frozen=True)
class AlertConfig:
    """Alert policy parameters.

    ``threshold``: score at or above which a frame is hot, in [0, 1].
    ``k``, ``n``: fire when at least ``k`` of the last ``n`` frames are hot, 1 <= k <= n.
    ``cooldown_s``: after an alert, suppress further alerts for this many seconds (>= 0).
    """

    threshold: float
    k: int = 1
    n: int = 1
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if not (1 <= self.k <= self.n):
            raise ValueError(f"need 1 <= k <= n, got k={self.k}, n={self.n}")
        if not (self.cooldown_s >= 0.0) or math.isinf(self.cooldown_s):
            raise ValueError(f"cooldown_s must be finite and >= 0, got {self.cooldown_s}")


class AlertPolicy:
    """Streaming k-of-n alert state machine with cooldown for a single camera."""

    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self._window: deque[bool] = deque(maxlen=config.n)
        self._last_timestamp_s: float | None = None
        self._last_alert_s: float | None = None

    def reset(self) -> None:
        """Forget all history, as if no frame had been seen."""
        self._window.clear()
        self._last_timestamp_s = None
        self._last_alert_s = None

    def step(self, timestamp_s: float, score: float) -> bool:
        """Consume one frame; return True exactly when an alert fires on this frame."""
        timestamp_s = float(timestamp_s)
        score = float(score)
        if not math.isfinite(timestamp_s):
            raise ValueError(f"timestamp must be finite, got {timestamp_s}")
        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score}")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError(
                f"timestamps must be strictly increasing: {timestamp_s} after "
                f"{self._last_timestamp_s}"
            )
        self._last_timestamp_s = timestamp_s

        self._window.append(score >= self.config.threshold)
        if sum(self._window) < self.config.k:
            return False
        if (
            self._last_alert_s is not None
            and timestamp_s - self._last_alert_s < self.config.cooldown_s
        ):
            return False
        self._last_alert_s = timestamp_s
        self._window.clear()
        return True


def run_policy(
    timestamps_s: npt.ArrayLike, scores: npt.ArrayLike, config: AlertConfig
) -> npt.NDArray[np.bool_]:
    """Batch form of :class:`AlertPolicy`: True at each frame where an alert fires.

    Gives exactly the result of feeding the frames one at a time through a fresh
    ``AlertPolicy(config)``. Implemented independently (ring buffer over a precomputed
    hot-frame array) so that the equivalence test checks two implementations.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    if ts.ndim != 1 or sc.ndim != 1 or ts.shape != sc.shape:
        raise ValueError(f"need 1-D arrays of equal length, got {ts.shape} and {sc.shape}")
    if not (np.all(np.isfinite(ts)) and np.all(np.isfinite(sc))):
        raise ValueError("timestamps and scores must be finite")
    if ts.size > 1 and not np.all(np.diff(ts) > 0):
        raise ValueError("timestamps must be strictly increasing")

    hot = sc >= config.threshold
    alerts = np.zeros(ts.shape, dtype=np.bool_)
    ring = np.zeros(config.n, dtype=np.bool_)
    filled = 0
    head = 0
    hot_count = 0
    last_alert = -math.inf
    for i in range(ts.size):
        if filled == config.n:
            hot_count -= int(ring[head])
        else:
            filled += 1
        ring[head] = hot[i]
        hot_count += int(hot[i])
        head = (head + 1) % config.n
        if hot_count >= config.k and ts[i] - last_alert >= config.cooldown_s:
            alerts[i] = True
            last_alert = float(ts[i])
            ring[:] = False
            filled = 0
            head = 0
            hot_count = 0
    return alerts
