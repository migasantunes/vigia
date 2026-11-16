"""M2: spatiotemporal tile model (plan Phase 4). Needs the ``train`` extra.

Reference point: SmokeyNet (Dewangan et al., 2022) uses a ResNet34 tile encoder, an LSTM
across 2 frames, and a ViT across tiles, on 45 tiles of 224 px with 20 px overlap cut from a
1040 x 1856 crop, trained with a tile BCE loss plus an image BCE loss. This model shares
that layout so it can be ablated against it (temporal module, number of frames, spatial
aggregator, backbone), not so it can copy it.

Tensor shapes, for B clips of T = ``context_frames`` frames, each H x W, N tiles per frame:

1. Frames ``(B, T, 3, H, W)`` float, normalised with the backbone's mean/std, oldest first;
   the last frame is the one being scored.
2. Tiles ``(B, T, N, 3, 224, 224)`` from ``vigia.data.tiling.tile_grid(H, W, tile_size,
   tile_overlap)``; the same grid is used for every frame of a clip.
3. Tile features ``(B, T, N, D)``: the timm backbone (``num_classes=0``) applied to the
   ``B * T * N`` tiles, reshaped back.
4. Temporal module, run independently per tile over the T axis -> ``(B, N, D)``:
   ``"gru"`` = GRU with ``hidden_size`` units, last hidden state projected to D;
   ``"transformer"`` = transformer encoder over T tokens (learned time embedding), output
   token of the last frame; ``"difference"`` = ``MLP([f_T, f_T - f_{T-1}])``, no
   recurrence (needs T >= 2).
5. Spatial aggregator across the N tiles -> ``(B, N, D)``: ``"none"`` = identity;
   ``"transformer"`` = transformer encoder over N tokens with a learned tile-position
   embedding (one per grid cell, so the grid must be fixed for a given model).
6. Tile head -> tile logits ``(B, N)``; image head (on the mean-pooled tile tokens plus
   a learned [CLS]-style token when ``spatial == "transformer"``) -> image logit ``(B,)``.

Loss = ``tile_loss_weight * BCE(tile logits, tile labels of the last frame)
+ image_loss_weight * BCE(image logit, label of the last frame)``. The frame score handed
to the alert policy is ``sigmoid(image logit)``.

Causality: the clip for frame t is frames ``t - T + 1 .. t``. At the start of a sequence,
missing past frames are filled by repeating frame 0, never by looking ahead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vigia.models.base import Frame, Scores
from vigia.models.single_frame import Pooling

if TYPE_CHECKING:
    import torch

TemporalModule = Literal["gru", "transformer", "difference"]
SpatialAggregator = Literal["none", "transformer"]
_TEMPORAL: tuple[str, ...] = ("gru", "transformer", "difference")
_SPATIAL: tuple[str, ...] = ("none", "transformer")
_POOLINGS: tuple[str, ...] = ("max", "mean", "attention")


@dataclass(frozen=True)
class TemporalConfig:
    """Hyper-parameters of the spatiotemporal tile model.

    Attributes:
        backbone: timm model name for the per-tile encoder.
        pretrained: load ImageNet weights for the backbone.
        tile_size: side of each square tile in pixels.
        tile_overlap: pixels shared by neighbouring tiles (``0 <= overlap < tile_size``).
        pooling: how tile logits feed the image score when ``spatial == "none"``
            (same meaning as in ``SingleFrameConfig``); ignored otherwise.
        context_frames: T, frames per clip including the scored one (>= 1; >= 2 for
            ``temporal == "difference"``).
        temporal: temporal module applied per tile (see module docstring).
        hidden_size: width of the GRU state or transformer model dimension.
        spatial: aggregator across tiles (see module docstring).
        tile_loss_weight: weight of the tile BCE loss.
        image_loss_weight: weight of the image BCE loss.
    """

    backbone: str = "resnet34"
    pretrained: bool = True
    tile_size: int = 224
    tile_overlap: int = 20
    pooling: Pooling = "max"
    context_frames: int = 2
    temporal: TemporalModule = "gru"
    hidden_size: int = 256
    spatial: SpatialAggregator = "transformer"
    tile_loss_weight: float = 1.0
    image_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.tile_overlap < self.tile_size:
            raise ValueError("tile_overlap must satisfy 0 <= tile_overlap < tile_size")
        if self.pooling not in _POOLINGS:
            raise ValueError(f"pooling must be one of {_POOLINGS}, got {self.pooling!r}")
        if self.context_frames < 1:
            raise ValueError("context_frames must be >= 1")
        if self.temporal not in _TEMPORAL:
            raise ValueError(f"temporal must be one of {_TEMPORAL}, got {self.temporal!r}")
        if self.temporal == "difference" and self.context_frames < 2:
            raise ValueError("temporal='difference' needs context_frames >= 2")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.spatial not in _SPATIAL:
            raise ValueError(f"spatial must be one of {_SPATIAL}, got {self.spatial!r}")
        if self.tile_loss_weight < 0 or self.image_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.tile_loss_weight == 0 and self.image_loss_weight == 0:
            raise ValueError("at least one loss weight must be positive")


def build_model(config: TemporalConfig) -> torch.nn.Module:
    """Build the spatiotemporal model as a ``torch.nn.Module``.

    ``forward(tiles)`` takes ``(B, T, N, 3, tile_size, tile_size)`` and returns
    ``(tile_logits (B, N), image_logit (B,))`` for the last frame of each clip. Imports
    torch and timm inside the function.
    """
    raise NotImplementedError("Phase 4: spatiotemporal model (build_model)")


class TemporalScorer:
    """``Scorer`` wrapping a trained temporal model; ``context_frames = config.context_frames``.

    ``score_sequence`` builds, for every frame t, the causal clip described in the module
    docstring, runs the model in batches on ``device``, and returns ``sigmoid(image
    logit)`` per frame. Tile features of a frame may be cached and reused by the T clips
    that contain it; that is an optimisation and must not change the scores.
    """

    def __init__(
        self,
        config: TemporalConfig,
        checkpoint: str | None = None,
        device: str = "cpu",
        name: str = "temporal",
    ) -> None:
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.name = name
        self.context_frames = config.context_frames

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        """frames: T images (H, W, 3, uint8, RGB). Returns (T,) float64 scores in [0, 1]."""
        raise NotImplementedError("Phase 4: spatiotemporal scorer (score_sequence)")
