"""Scorers built on learned models, plus the ``Scorer`` protocol every detector implements.

Only ``vigia.models.base`` is dependency-free. ``single_frame`` and ``temporal`` need the
``train`` extra (PyTorch, timm) at call time, never at import time.
"""

from vigia.models.base import (
    CausalityError,
    ConstantScorer,
    MeanBrightnessScorer,
    PeekAheadScorer,
    Scorer,
    check_causality,
)

__all__ = [
    "CausalityError",
    "ConstantScorer",
    "MeanBrightnessScorer",
    "PeekAheadScorer",
    "Scorer",
    "check_causality",
]
