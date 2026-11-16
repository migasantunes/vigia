"""ONNX Runtime scorer (plan Phase 6). Needs the ``serve`` extra; never imports PyTorch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vigia.models.base import Frame, Scores


class OnnxScorer:
    """``Scorer`` backed by an ONNX file written by ``vigia.serve.export.export_onnx``.

    Reads tile grid, normalisation and ``context_frames`` from the ONNX metadata, and
    reproduces the PyTorch scorer's preprocessing with numpy (and OpenCV for resizing),
    so its scores match the PyTorch scorer within 1e-4. ``onnxruntime`` is imported in
    ``__init__``.
    """

    name: str
    context_frames: int

    def __init__(self, onnx_path: Path, name: str = "onnx") -> None:
        raise NotImplementedError("Phase 6: ONNX Runtime scorer (OnnxScorer)")

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        """frames: T images (H, W, 3, uint8, RGB). Returns (T,) float64 scores in [0, 1]."""
        raise NotImplementedError("Phase 6: ONNX Runtime scorer (score_sequence)")
