"""M0: OpenCV motion baseline (plan Phase 2). Needs the ``vision`` extra.

Early wildfire smoke is a slow, rising, low-contrast plume. This scorer looks for exactly
that kind of motion and nothing else; it has no learned parameters, which makes it the
floor every learned model must beat at a matched false-alarm rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from vigia.models.base import Frame, Scores

SkyMask = Literal["none", "above_horizon", "below_horizon"]
_SKY_MASKS: tuple[str, ...] = ("none", "above_horizon", "below_horizon")


@dataclass(frozen=True)
class MotionConfig:
    """Parameters of the motion baseline. All fractions are of the downscaled frame area.

    Defaults are untuned starting points; Phase 2 chooses values on validation fires only.

    Attributes:
        downscale: integer factor by which width and height are divided before processing.
        mog2_history: ``history`` of ``cv2.createBackgroundSubtractorMOG2``, in frames.
        mog2_var_threshold: ``varThreshold`` of MOG2 (squared Mahalanobis distance).
        min_blob_area_fraction: connected components smaller than this are discarded.
        max_blob_area_fraction: components larger than this (global lighting changes,
            camera shake) are discarded.
        upward_flow_weight: weight of the upward component of the mean blob flow in the
            blob's motion score, relative to its horizontal component (>= 0).
        max_flow_px: blobs whose mean flow magnitude (downscaled pixels per frame) exceeds
            this are discarded as too fast to be early smoke (vehicles, birds, insects).
        max_contrast: blobs whose mean absolute grayscale difference to the background
            model exceeds this (0-255 scale) are discarded as too opaque to be early smoke.
        score_area_fraction: surviving area fraction that maps to score 1.0.
        sky_mask: ``"none"`` uses the whole frame; ``"above_horizon"`` / ``"below_horizon"``
            keep only that region, with the horizon row estimated per sequence.
    """

    downscale: int = 4
    mog2_history: int = 20
    mog2_var_threshold: float = 16.0
    min_blob_area_fraction: float = 1e-5
    max_blob_area_fraction: float = 0.05
    upward_flow_weight: float = 2.0
    max_flow_px: float = 3.0
    max_contrast: float = 60.0
    score_area_fraction: float = 1e-3
    sky_mask: SkyMask = "none"

    def __post_init__(self) -> None:
        if self.downscale < 1:
            raise ValueError("downscale must be >= 1")
        if self.mog2_history < 1:
            raise ValueError("mog2_history must be >= 1")
        if self.mog2_var_threshold <= 0:
            raise ValueError("mog2_var_threshold must be positive")
        if not 0 <= self.min_blob_area_fraction < self.max_blob_area_fraction <= 1:
            raise ValueError("need 0 <= min_blob_area_fraction < max_blob_area_fraction <= 1")
        if self.upward_flow_weight < 0:
            raise ValueError("upward_flow_weight must be non-negative")
        if self.max_flow_px <= 0 or self.max_contrast <= 0 or self.score_area_fraction <= 0:
            raise ValueError("max_flow_px, max_contrast and score_area_fraction must be > 0")
        if self.sky_mask not in _SKY_MASKS:
            raise ValueError(f"sky_mask must be one of {_SKY_MASKS}, got {self.sky_mask!r}")


class MotionScorer:
    """Classical ``Scorer``: area of slow, upward, low-contrast motion, mapped to [0, 1].

    ``score_sequence`` re-initialises all state at the start of every call and processes
    frames strictly in order, so it is causal. For each frame t:

    1. Convert RGB to grayscale and downscale by ``downscale`` (``cv2.INTER_AREA``).
    2. Apply MOG2 (``mog2_history``, ``mog2_var_threshold``, shadows off) to get a binary
       foreground mask. Frame 0 only initialises the model and scores 0.
    3. Morphological open, then close (3x3 elliptical kernel) to remove speckle and fill
       small holes.
    4. Farneback dense optical flow between frames t-1 and t (grayscale, downscaled).
    5. Label connected components of the mask. Keep a blob if its area fraction lies in
       ``[min_blob_area_fraction, max_blob_area_fraction]``, its mean flow magnitude is
       at most ``max_flow_px``, its mean flow points upward (negative image y), its mean
       absolute difference to the MOG2 background image is at most ``max_contrast``, and
       it lies inside the ``sky_mask`` region.
    6. Weight each surviving blob's area by its motion score
       ``(upward_flow_weight * up + |horizontal|) / ((upward_flow_weight + 1) * magnitude)``
       (in [0, 1]), sum the weighted area fractions, and return
       ``min(1, total / score_area_fraction)``.

    ``context_frames`` is ``mog2_history``: the background model is a function of that many
    past frames (approximately; MOG2 decays older frames geometrically).
    """

    def __init__(self, config: MotionConfig | None = None, name: str = "motion") -> None:
        self.config = config if config is not None else MotionConfig()
        self.name = name
        self.context_frames = self.config.mog2_history

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        """frames: T images (H, W, 3, uint8, RGB). Returns (T,) float64 scores in [0, 1]."""
        raise NotImplementedError("Phase 2: OpenCV motion baseline (score_sequence)")
