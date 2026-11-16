"""The ``Scorer`` protocol, a causality check, and trivial reference scorers.

This module is part of no heavy layer: it depends on numpy only, so the protocol and the
causality check can be used by every test and by the evaluation harness.

Contract (``docs/ARCHITECTURE.md`` §7): a scorer receives T frames in time order, each an
``(H, W, 3)`` ``uint8`` RGB image, and returns a ``(T,)`` float array in [0, 1]. The score
for frame t may depend on frames ``0..t`` only. ``score_sequence`` must be a pure function of
its input: no state may carry over between calls (a stateful algorithm such as background
subtraction is re-initialised at the start of every call).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]
"""One camera frame: ``(H, W, 3)``, ``uint8``, RGB."""

Scores = NDArray[np.float64]
"""Per-frame scores: ``(T,)``, ``float64``, each in [0, 1]."""


@runtime_checkable
class Scorer(Protocol):
    """Anything that turns a time-ordered frame sequence into one causal score per frame."""

    name: str
    context_frames: int
    """How many frames (the current one included) the scorer looks at; 1 = single-frame."""

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        """frames: T images (H, W, 3, uint8, RGB), in time order.

        Returns shape (T,), float in [0, 1]: the score for frame t may use frames <= t only.
        """
        ...


class CausalityError(AssertionError):
    """A scorer's score at time t changed when only frames after t were changed."""


def validate_scores(scores: object, n_frames: int) -> Scores:
    """Check that ``scores`` honours the output contract and return it as ``float64``.

    Raises:
        ValueError: if ``scores`` is not a 1-D array of length ``n_frames``, contains a
            non-finite value, or has a value outside [0, 1].
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape != (n_frames,):
        raise ValueError(f"scores must have shape ({n_frames},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("scores must lie in [0, 1]")
    return arr


def check_causality(
    scorer: Scorer,
    frames: Sequence[Frame],
    rng: np.random.Generator,
    atol: float = 1e-6,
    n_cuts: int = 4,
) -> None:
    """Assert that ``scorer`` never uses future frames.

    Procedure:

    1. Score ``frames`` twice; the two results must be equal within ``atol`` (a scorer that
       is not deterministic cannot be checked this way, and that is reported as ValueError).
    2. Draw up to ``n_cuts`` distinct cut points t from ``0..T-2`` with ``rng``.
    3. For each t, replace every frame after t by a noisy copy (additive integer noise in
       [-128, 127], clipped to uint8), score the perturbed sequence and require
       ``scores[: t + 1]`` to equal the unperturbed scores within ``atol``.

    The input frames are never modified.

    Raises:
        ValueError: if fewer than 2 frames are given, the scores violate the output
            contract (see ``validate_scores``), or the scorer is not deterministic.
        CausalityError: if any score at or before a cut point changed.
    """
    n_frames = len(frames)
    if n_frames < 2:
        raise ValueError("check_causality needs at least 2 frames")
    if n_cuts < 1:
        raise ValueError("n_cuts must be >= 1")
    originals = [np.array(f, dtype=np.uint8, copy=True) for f in frames]

    reference = validate_scores(scorer.score_sequence(originals), n_frames)
    again = validate_scores(scorer.score_sequence(originals), n_frames)
    if not np.allclose(reference, again, rtol=0.0, atol=atol):
        raise ValueError(f"scorer {scorer.name!r} is not deterministic; cannot check causality")

    n_choices = n_frames - 1
    cuts = np.sort(rng.choice(n_choices, size=min(n_cuts, n_choices), replace=False))
    for cut in cuts.tolist():
        perturbed = list(originals[: cut + 1])
        for frame in originals[cut + 1 :]:
            noise = rng.integers(-128, 128, size=frame.shape, dtype=np.int16)
            noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            perturbed.append(noisy)
        scores = validate_scores(scorer.score_sequence(perturbed), n_frames)
        diff = np.abs(scores[: cut + 1] - reference[: cut + 1])
        if np.any(diff > atol):
            first_bad = int(np.argmax(diff > atol))
            raise CausalityError(
                f"scorer {scorer.name!r}: score at frame {first_bad} changed by "
                f"{diff[first_bad]:.3g} when only frames after {cut} were perturbed"
            )


def _check_frames(frames: Sequence[Frame]) -> None:
    for i, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame {i} must have shape (H, W, 3), got {frame.shape}")


class ConstantScorer:
    """Scores every frame with the same ``value``. Sanity baseline (never/always alert)."""

    context_frames: int = 1

    def __init__(self, value: float, name: str | None = None) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"value must lie in [0, 1], got {value}")
        self.value = float(value)
        self.name = name if name is not None else f"constant_{self.value:g}"

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        _check_frames(frames)
        return np.full(len(frames), self.value, dtype=np.float64)


class MeanBrightnessScorer:
    """Score = mean pixel value of the current frame / 255. Causal, single-frame, useless."""

    name: str = "mean_brightness"
    context_frames: int = 1

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        _check_frames(frames)
        return np.array([float(np.mean(f)) / 255.0 for f in frames], dtype=np.float64)


class PeekAheadScorer:
    """Deliberately broken: frame t gets the mean brightness of frame t+1 (the last frame
    uses itself). Exists only to prove that ``check_causality`` catches violations."""

    name: str = "peek_ahead"
    context_frames: int = 1

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        _check_frames(frames)
        n = len(frames)
        return np.array(
            [float(np.mean(frames[min(t + 1, n - 1)])) / 255.0 for t in range(n)],
            dtype=np.float64,
        )
