"""The ``vigia`` command. Subcommands map to the phases of ``docs/PLAN.md``.

Only ``version`` does anything yet. Every other subcommand parses its documented
arguments and exits with code 2 and a "not implemented yet" message naming its phase, until
it is wired to the library code of that phase.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vigia import __version__

EXIT_NOT_IMPLEMENTED = 2

PHASES: dict[str, int] = {
    "manifest": 0,
    "audit-split": 0,
    "evaluate": 1,
    "score": 2,
    "train": 3,
    "export": 6,
    "serve": 6,
    "replay": 6,
}
"""Phase of ``docs/PLAN.md`` in which each subcommand gets wired up."""


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    # A repeatable option, not a trailing positional: argparse before Python 3.13
    # rejects positionals that come after options ("cfg --out x key=value").
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable, e.g. --set train.lr=0.001 --set eval.policy.k=2",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand and its arguments."""
    parser = argparse.ArgumentParser(
        prog="vigia",
        description="Wildfire smoke detection for fixed lookout cameras.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    sub.add_parser("version", help="print the package version")

    p = sub.add_parser("manifest", help="build a dataset manifest (Phase 0; Pyro-SDIS: Phase 5)")
    p.add_argument("dataset", choices=["figlib", "pyrosdis"], help="dataset adapter to run")
    p.add_argument("--root", type=Path, required=True, help="dataset download root")
    p.add_argument("--output", type=Path, required=True, help="output Parquet manifest")

    p = sub.add_parser(
        "audit-split", help="leakage report of a split by camera and station (Phase 0)"
    )
    p.add_argument("--manifest", type=Path, required=True, help="Parquet manifest")
    p.add_argument("--split", type=Path, required=True, help="official fire lists file")
    p.add_argument(
        "--keys",
        nargs="+",
        default=["camera_id", "station_id"],
        help="manifest columns to audit (default: camera_id station_id)",
    )

    p = sub.add_parser("score", help="run a scorer over a manifest (Phase 2)")
    p.add_argument("config", type=Path, help="experiment config YAML")
    p.add_argument("--manifest", type=Path, required=True, help="Parquet manifest")
    p.add_argument("--checkpoint", type=Path, default=None, help="model checkpoint, if any")
    p.add_argument("--output", type=Path, required=True, help="output Parquet scores file")
    _add_overrides(p)

    p = sub.add_parser("evaluate", help="scores + manifest -> evaluation report (Phase 1)")
    p.add_argument("config", type=Path, help="experiment or eval config YAML")
    p.add_argument("--scores", type=Path, required=True, help="Parquet scores file")
    p.add_argument("--manifest", type=Path, required=True, help="Parquet manifest")
    p.add_argument("--output", type=Path, required=True, help="report directory")
    _add_overrides(p)

    p = sub.add_parser("train", help="train a model from a config (Phases 3-4)")
    p.add_argument("config", type=Path, help="experiment config YAML")
    _add_overrides(p)

    p = sub.add_parser("export", help="export a checkpoint to ONNX (Phase 6)")
    p.add_argument("config", type=Path, help="experiment config YAML")
    p.add_argument("--checkpoint", type=Path, required=True, help="model checkpoint")
    p.add_argument("--output", type=Path, required=True, help="output .onnx file")

    p = sub.add_parser("serve", help="run the alert service (Phase 6)")
    p.add_argument("config", type=Path, help="serve config YAML")
    p.add_argument("--host", default="127.0.0.1", help="bind address")
    p.add_argument("--port", type=int, default=8000, help="bind port")

    p = sub.add_parser("replay", help="replay a recorded ignition through the service (Phase 6)")
    p.add_argument("--manifest", type=Path, required=True, help="Parquet manifest")
    p.add_argument("--sequence", required=True, help="sequence_id to replay")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="service base URL")
    p.add_argument("--speedup", type=float, default=60.0, help="replay speed factor")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``vigia`` console script. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    command: str = args.command
    if command == "version":
        print(__version__)
        return 0
    phase = PHASES[command]
    print(
        f"vigia {command}: not implemented yet (Phase {phase}, see docs/PLAN.md)",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    raise SystemExit(main())
