import numpy as np
import pytest

from vigia.eval.alerting import AlertConfig, AlertPolicy, run_policy


def stream(config: AlertConfig, timestamps: list[float], scores: list[float]) -> list[bool]:
    policy = AlertPolicy(config)
    return [policy.step(t, s) for t, s in zip(timestamps, scores, strict=True)]


def test_score_equal_to_threshold_is_hot() -> None:
    config = AlertConfig(threshold=0.5)
    assert stream(config, [0.0, 1.0, 2.0], [0.5, 0.49999, 1.0]) == [True, False, True]


def test_default_config_fires_on_every_hot_frame() -> None:
    config = AlertConfig(threshold=0.3)
    scores = [0.9, 0.9, 0.1, 0.9, 0.3]
    assert stream(config, [0, 1, 2, 3, 4], scores) == [True, True, False, True, True]


def test_k_of_n_window() -> None:
    config = AlertConfig(threshold=0.5, k=2, n=3)
    hot = [1, 0, 0, 1, 0, 1, 1, 1]
    expected = [False, False, False, False, False, True, False, True]
    assert stream(config, list(range(8)), [float(h) for h in hot]) == expected


def test_cooldown_suppresses_until_expired() -> None:
    config = AlertConfig(threshold=0.5, cooldown_s=10.0)
    # 10 - 0 == cooldown, so t=10 is no longer suppressed.
    assert stream(config, [0, 5, 10, 15], [1, 1, 1, 1]) == [True, False, True, False]


def test_rearming_needs_fresh_hot_frames() -> None:
    config = AlertConfig(threshold=0.5, k=2, n=2, cooldown_s=10.0)
    # Fires at t=1; at t=12 only one fresh hot frame exists, so it must not fire yet.
    assert stream(config, [0, 1, 12, 13], [1, 1, 1, 1]) == [False, True, False, True]


def test_hot_frames_during_cooldown_count_after_it() -> None:
    config = AlertConfig(threshold=0.5, k=2, n=2, cooldown_s=10.0)
    expected = [False, True, False, False, True]
    assert stream(config, [0, 1, 2, 3, 12], [1, 1, 1, 1, 1]) == expected


@pytest.mark.parametrize("timestamps", [[0.0, 1.0, 1.0], [0.0, 2.0, 1.0]])
def test_non_increasing_timestamps_raise(timestamps: list[float]) -> None:
    policy = AlertPolicy(AlertConfig(threshold=0.5))
    policy.step(timestamps[0], 0.0)
    policy.step(timestamps[1], 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        policy.step(timestamps[2], 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        run_policy(timestamps, [0.0, 0.0, 0.0], AlertConfig(threshold=0.5))


def test_nan_score_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        AlertPolicy(AlertConfig(threshold=0.5)).step(0.0, float("nan"))


def test_reset_forgets_history() -> None:
    policy = AlertPolicy(AlertConfig(threshold=0.5, k=2, n=2, cooldown_s=100.0))
    policy.step(10.0, 1.0)
    policy.reset()
    assert policy.step(0.0, 1.0) is False
    assert policy.step(1.0, 1.0) is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": -0.01}, "threshold"),
        ({"threshold": 1.01}, "threshold"),
        ({"threshold": 0.5, "k": 0}, "k <= n"),
        ({"threshold": 0.5, "k": 3, "n": 2}, "k <= n"),
        ({"threshold": 0.5, "cooldown_s": -1.0}, "cooldown_s"),
        ({"threshold": float("nan")}, "threshold"),
    ],
)
def test_config_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AlertConfig(**kwargs)  # type: ignore[arg-type]


def test_config_bounds_are_valid() -> None:
    AlertConfig(threshold=0.0, k=3, n=3, cooldown_s=0.0)
    AlertConfig(threshold=1.0)


def test_run_policy_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        run_policy([0.0, 1.0], [0.5], AlertConfig(threshold=0.5))


def test_run_policy_empty() -> None:
    out = run_policy([], [], AlertConfig(threshold=0.5))
    assert out.shape == (0,)
    assert out.dtype == np.bool_


def test_run_policy_matches_streaming_on_random_inputs() -> None:
    rng = np.random.default_rng(20261116)
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for _ in range(200):
        length = int(rng.integers(0, 60))
        timestamps = np.cumsum(rng.uniform(0.5, 120.0, size=length))
        if rng.random() < 0.5:
            scores = rng.choice(levels, size=length)
            threshold = float(rng.choice(levels))
        else:
            scores = rng.random(length)
            threshold = float(rng.random())
        n = int(rng.integers(1, 6))
        k = int(rng.integers(1, n + 1))
        cooldown = 0.0 if rng.random() < 0.3 else float(rng.uniform(0.0, 600.0))
        config = AlertConfig(threshold=threshold, k=k, n=n, cooldown_s=cooldown)
        expected = stream(config, timestamps.tolist(), scores.tolist())
        assert run_policy(timestamps, scores, config).tolist() == expected
