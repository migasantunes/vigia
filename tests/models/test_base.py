from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from vigia.classical.motion import MotionScorer
from vigia.models.base import (
    CausalityError,
    ConstantScorer,
    Frame,
    MeanBrightnessScorer,
    PeekAheadScorer,
    Scorer,
    Scores,
    check_causality,
    validate_scores,
)
from vigia.models.single_frame import SingleFrameConfig, SingleFrameScorer
from vigia.models.temporal import TemporalConfig, TemporalScorer


def _frames(n: int, seed: int = 0, shape: tuple[int, int] = (24, 32)) -> list[Frame]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8) for _ in range(n)]


CAUSAL: list[Scorer] = [ConstantScorer(0.0), ConstantScorer(0.7), MeanBrightnessScorer()]


@pytest.mark.parametrize(
    "scorer",
    [
        *CAUSAL,
        PeekAheadScorer(),
        MotionScorer(),
        SingleFrameScorer(SingleFrameConfig()),
        TemporalScorer(TemporalConfig()),
    ],
    ids=lambda s: s.name,
)
def test_scorers_satisfy_the_protocol(scorer: Scorer) -> None:
    assert isinstance(scorer, Scorer)
    assert isinstance(scorer.name, str)
    assert scorer.context_frames >= 1


def test_temporal_scorer_reports_its_context() -> None:
    assert TemporalScorer(TemporalConfig(context_frames=4)).context_frames == 4


@pytest.mark.parametrize("scorer", CAUSAL, ids=lambda s: s.name)
@pytest.mark.parametrize("n_frames", [2, 3, 10])
def test_check_causality_passes_for_causal_scorers(scorer: Scorer, n_frames: int) -> None:
    check_causality(scorer, _frames(n_frames), np.random.default_rng(1))


@pytest.mark.parametrize("n_frames", [2, 3, 10])
def test_check_causality_catches_peek_ahead(n_frames: int) -> None:
    with pytest.raises(CausalityError, match="peek_ahead"):
        check_causality(PeekAheadScorer(), _frames(n_frames), np.random.default_rng(1))


def test_check_causality_does_not_modify_frames() -> None:
    frames = _frames(6)
    copies = [f.copy() for f in frames]
    check_causality(MeanBrightnessScorer(), frames, np.random.default_rng(2))
    for a, b in zip(frames, copies, strict=True):
        np.testing.assert_array_equal(a, b)


def test_check_causality_needs_two_frames() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        check_causality(MeanBrightnessScorer(), _frames(1), np.random.default_rng(0))


class _Random:
    name = "random"
    context_frames = 1

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        return self._rng.random(len(frames))


class _WrongShape:
    name = "wrong_shape"
    context_frames = 1

    def score_sequence(self, frames: Sequence[Frame]) -> Scores:
        return np.zeros(len(frames) + 1)


def test_check_causality_rejects_nondeterministic_scorer() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        check_causality(_Random(), _frames(4), np.random.default_rng(0))


def test_check_causality_rejects_contract_violations() -> None:
    with pytest.raises(ValueError, match="shape"):
        check_causality(_WrongShape(), _frames(4), np.random.default_rng(0))


@pytest.mark.parametrize("scorer", [*CAUSAL, PeekAheadScorer()], ids=lambda s: s.name)
def test_score_shapes_and_ranges(scorer: Scorer) -> None:
    scores = scorer.score_sequence(_frames(7))
    assert scores.shape == (7,)
    assert scores.dtype == np.float64
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_empty_sequence_gives_empty_scores() -> None:
    assert MeanBrightnessScorer().score_sequence([]).shape == (0,)


def test_reference_scorer_values() -> None:
    frames = [np.full((4, 5, 3), v, dtype=np.uint8) for v in (0, 51, 255)]
    np.testing.assert_allclose(MeanBrightnessScorer().score_sequence(frames), [0.0, 0.2, 1.0])
    np.testing.assert_allclose(PeekAheadScorer().score_sequence(frames), [0.2, 1.0, 1.0])
    np.testing.assert_allclose(ConstantScorer(0.3).score_sequence(frames), [0.3] * 3)


def test_constant_scorer_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ConstantScorer(1.5)


def test_reference_scorers_reject_non_rgb_frames() -> None:
    with pytest.raises(ValueError, match="H, W, 3"):
        MeanBrightnessScorer().score_sequence([np.zeros((4, 5), dtype=np.uint8)])


@pytest.mark.parametrize(
    ("scores", "match"),
    [([0.1, 0.2], "shape"), ([0.1, np.nan, 0.2], "finite"), ([0.1, 1.2, 0.0], r"\[0, 1\]")],
)
def test_validate_scores(scores: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_scores(np.array(scores), 3)


@pytest.mark.parametrize(
    "scorer",
    [MotionScorer(), SingleFrameScorer(SingleFrameConfig()), TemporalScorer(TemporalConfig())],
    ids=lambda s: s.name,
)
def test_stub_scorers_name_their_phase(scorer: Scorer) -> None:
    with pytest.raises(NotImplementedError, match=r"^Phase \d"):
        scorer.score_sequence(_frames(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_overlap": 224},
        {"pooling": "median"},
        {"tile_loss_weight": 0.0, "image_loss_weight": 0.0},
    ],
)
def test_single_frame_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - the message varies per field
        SingleFrameConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_frames": 0},
        {"temporal": "lstm"},
        {"temporal": "difference", "context_frames": 1},
        {"spatial": "gnn"},
        {"hidden_size": 0},
    ],
)
def test_temporal_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - the message varies per field
        TemporalConfig(**kwargs)  # type: ignore[arg-type]
