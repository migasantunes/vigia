"""Augmentation pipelines (plan Phase 3). Needs the ``vision`` extra (Albumentations).

Both builders read the ``transforms`` mapping of a resolved config (``vigia.config``) and
return an Albumentations pipeline; Albumentations is imported inside the functions.

Training augmentations, in order:

- Geometric: horizontal flip (p=0.5); small random scale/translate (scale 0.9-1.1,
  shift up to 5 % of each side) and rotation (at most 3 degrees, lookout cameras are
  level); random crop to the model input size when the frame is larger.
- Photometric: brightness/contrast, gamma, colour temperature and hue/saturation jitter;
  haze (a uniform fog overlay of random strength); mild Gaussian blur and JPEG
  compression (camera artefacts).
- No vertical flip: smoke rises, and upward motion is a feature the models may use.

Boxes are transformed with the image (``bbox_params`` in ``albumentations`` format, i.e.
normalised ``[x0, y0, x1, y1]`` as in the manifest), boxes left with less than 20 % of
their area are dropped, and tile labels are recomputed after the transform.

Temporal clips: every frame of a clip MUST receive the SAME geometric transform, or the
model learns apparent motion that the camera never produced. Use
``albumentations.ReplayCompose`` (sample on the last frame, replay on the others) or
``additional_targets`` (``{"image1": "image", ...}``) for the geometric part.
Photometric jitter may differ slightly between frames of a clip (a small per-frame
perturbation around one clip-level draw) to mimic auto-exposure flicker.

Evaluation applies no randomness: only the deterministic resize/crop and normalisation
the model expects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_train_transforms(config: Mapping[str, Any]) -> Any:
    """Return the training pipeline described in the module docstring.

    The returned object is called as ``pipeline(image=..., bboxes=..., labels=...)`` for a
    single frame, or with ``image1``, ``image2``, ... ``additional_targets`` for a clip, and
    returns the same keys. Probabilities and ranges come from ``config["transforms"]``.
    """
    raise NotImplementedError("Phase 3: training augmentations (build_train_transforms)")


def build_eval_transforms(config: Mapping[str, Any]) -> Any:
    """Return the deterministic evaluation pipeline (resize/crop + normalisation only)."""
    raise NotImplementedError("Phase 3: evaluation transforms (build_eval_transforms)")
