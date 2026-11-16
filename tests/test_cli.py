from __future__ import annotations

import pytest

from vigia import __version__
from vigia.cli import EXIT_NOT_IMPLEMENTED, PHASES, main

STUB_INVOCATIONS: dict[str, list[str]] = {
    "manifest": ["manifest", "figlib", "--root", "data/raw/figlib", "--output", "m.parquet"],
    "audit-split": ["audit-split", "--manifest", "m.parquet", "--split", "lists.txt"],
    "score": [
        "score",
        "configs/experiment/p1_single_frame.yaml",
        "--manifest",
        "m.parquet",
        "--output",
        "s.parquet",
        "--set",
        "eval.policy.k=2",
    ],
    "evaluate": [
        "evaluate",
        "configs/eval/default.yaml",
        "--scores",
        "s.parquet",
        "--manifest",
        "m.parquet",
        "--output",
        "report",
    ],
    "train": [
        "train",
        "configs/experiment/p1_temporal.yaml",
        "--set",
        "train.lr=0.001",
        "--set",
        "train.seed=1",
    ],
    "export": ["export", "c.yaml", "--checkpoint", "best.ckpt", "--output", "m.onnx"],
    "serve": ["serve", "c.yaml", "--port", "8080"],
    "replay": ["replay", "--manifest", "m.parquet", "--sequence", "fire-1", "--speedup", "120"],
}


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_every_stub_has_an_invocation() -> None:
    assert set(STUB_INVOCATIONS) == set(PHASES)


@pytest.mark.parametrize("command", sorted(STUB_INVOCATIONS))
def test_stub_subcommands_exit_2_with_phase(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(STUB_INVOCATIONS[command]) == EXIT_NOT_IMPLEMENTED == 2
    err = capsys.readouterr().err
    assert f"not implemented yet (Phase {PHASES[command]}, see docs/PLAN.md)" in err


def test_phase_numbers_follow_the_plan() -> None:
    assert PHASES == {
        "manifest": 0,
        "audit-split": 0,
        "evaluate": 1,
        "score": 2,
        "train": 3,
        "export": 6,
        "serve": 6,
        "replay": 6,
    }


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in [*PHASES, "version"]:
        assert command in out


@pytest.mark.parametrize("command", sorted(STUB_INVOCATIONS))
def test_subcommand_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert "usage: vigia" in capsys.readouterr().out


def test_missing_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_missing_required_argument_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["manifest", "figlib"])
    assert exc.value.code == 2


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main(["manifest", "nope", "--root", "r", "--output", "o"])
