"""M1: single-frame tile classifier (plan Phase 3). Needs the ``train`` extra.

Pipeline for a batch of B images of size H x W (no time axis; T = 1 throughout):

1. Input frames ``(B, 3, H, W)`` float, normalised with the backbone's mean/std.
2. Tiling with ``vigia.data.tiling.tile_grid(H, W, tile_size, tile_overlap)`` gives N
   tiles: ``(B, N, 3, tile_size, tile_size)``, i.e. ``(B, N, 3, 224, 224)`` by default.
3. The timm backbone (``num_classes=0``, global average pooling) maps each tile to a
   feature vector: tile features ``(B, N, D)``, D = ``backbone.num_features``.
4. A linear tile head gives tile logits ``(B, N)``.
5. Pooling over tiles gives the image logit ``(B,)``:
   ``"max"`` = max over tile logits, ``"mean"`` = mean over tile logits,
   ``"attention"`` = softmax-weighted sum of tile logits, weights from a learned
   ``Linear(D, 1)`` on tile features.

Training targets are tile labels from ``vigia.data.tiling.boxes_to_tile_labels`` (frames
whose ``boxes`` is null get no tile loss) and the frame ``label`` for the image logit.
The frame score handed to the alert policy is ``sigmoid(image logit)``.

The same tensor layout is used by ``vigia.models.temporal`` with T = ``context_frames``;
this model is the T = 1 special case with no temporal or spatial module, so the two can be
compared as an ablation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vigia.models.base import Frame, Scores

if TYPE_CHECKING:
    import torch

Pooling = Literal["max", "mean", "attention"]
_POOLINGS: tuple[str, ...] = ("max", "mean", "attention")


@dataclass(frozen=True)
class SingleFrameConfig:
    """Hyper-parameters of the single-frame tile classifier.

    Attributes:
        backbone: timm model name, created with ``timm.create_model(backbone,
            pretrained=pretrained, num_classes=0)``.
        pretrained: load ImageNet weights for the backbone.
        tile_size: side of each square tile in pixels, after any resize of the frame.
        tile_overlap: pixels shared by neighbouring tiles (``0 <= overlap < tile_size``).
        pooling: how tile logits become the image logit (see module docstring).
        tile_loss_weight: weight of the tile BCE loss in the total loss.
        image_loss_weight: weight of the image BCE loss in the total loss.
    """

    backbone: str = "resnet34"
    pretrained: bool = True
    tile_size: int = 224
    tile_overlap: int = 20
    pooling: Pooling = "max"
    tile_loss_weight: float = 1.0
    image_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.tile_overlap < self.tile_size:
            raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < tile_size")
        if self.pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {_POOLINGS}, got {self.pooling!r}")
        if self.tile_loss_weight < 0 or self.image_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.tile_loss_weight == 0 and self.image_loss_weight == 0:
            raise ValueError("at least one loss weight must be positive")


def build_model(config: SingleFrameConfig) -> torch.nn.Module:
    """Build the single-frame tile classifier as a ``torch.nn.Module``.

    ``forward(tiles)`` takes ``(B, N, 3, tile_size, tile_size)`` and returns a pair
    ``(tile_logits (B, N), image_logit (B,))``. Tiling is done outside the module (by the
    data pipeline and by ``SingleFrameScorer``) so the exported ONNX graph has a fixed tile
    input. Imports torch and timm inside the function.
    """
    raise NotImplementedError("Phase 3: single-frame tile classifier (build_model)")


class SingleFrameScorer:
    """``Scorer`` wrapping a trained single-frame model (``context_frames = 1``).

    ``score_sequence`` scores each frame independently, so it is causal by construction:
    each frame is tiled with the config's grid, run through the model in batches on
    ``device``, and its score is ``sigmoid(image logit)``.
    """

    context_frames: int = 1

    def __init__(
        self,
        config: SingleFrameConfig,
        checkpoint: str | None = None,
        device: str = "cpu",
        name: str = "single_frame",
    ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.name = name

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        """frames: T images (H, W, 3, uint8, RGB). Returns (T,) float64 scores in [0, 1]."""
        raise NotImplementedError("Phase 3: single-frame scorer (score_sequence)")
