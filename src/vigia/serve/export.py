"""PyTorch -> ONNX export (plan Phase 6). Needs the ``train`` extra."""

from __future__ import annotations

from pathlib import Path


def export_onnx(checkpoint: Path, config_path: Path, output: Path, opset: int = 17) -> Path:
    """Export a trained scorer's model to ONNX and return ``output``.

    The graph input is the tile tensor the model's ``forward`` takes
    (``(B, N, 3, S, S)`` single-frame, ``(B, T, N, 3, S, S)`` temporal, S = tile size) with
    a dynamic batch axis; outputs are ``tile_logits (B, N)`` and ``image_logit (B,)``.
    The resolved config (tile grid, normalisation, ``context_frames``) is stored in the
    ONNX metadata so ``vigia.serve.runtime.OnnxScorer`` needs no other file.
    After export, the ONNX outputs are compared with PyTorch on a random input and the
    export fails if they differ by more than 1e-4.
    """
    raise NotImplementedError("Phase 6: ONNX export (export_onnx)")
