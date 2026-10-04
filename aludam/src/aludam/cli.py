"""Command line front end.

    aludam list
    aludam predict text_det "some text to classify"
    aludam predict img_det ./photo.png
    aludam predict vdo_det ./clip.mp4 --json

Every subcommand prints the same four fields, for every modality.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__, weights
from .exceptions import AludamError
from .registry import DETECTOR_NAMES, load_detector


def _read_text_operand(operand: str) -> str:
    """``@path`` reads the file; anything else is used verbatim."""
    if operand.startswith("@"):
        with open(operand[1:], "r", encoding="utf-8") as fh:
            return fh.read()
    return operand


def _cmd_list(_args: argparse.Namespace) -> int:
    for name in DETECTOR_NAMES:
        print(name)
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    detector = load_detector(args.detector)
    value = (
        _read_text_operand(args.input)
        if detector.media_type == "TEXT"
        else args.input
    )
    result = detector.predict(value)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result)
        print(
            f"  detector     = {result.detector} media={result.media_type} "
            f"backend={result.backend} elapsed_ms={result.elapsed_ms}"
        )
        print(
            f"  calibrated   = {result.calibrated}"
            f"{'  <- thresholds NOT fitted' if not result.calibrated else ''}"
        )
        if result.signals:
            print(f"  signals      = {', '.join(result.signals)}")
        if result.kind != "unknown":
            print(f"  kind         = {result.kind}")
        if result.upstream_label:
            print(
                f"  upstream     = {result.upstream_label}"
                f"  (kept for comparison; does not decide label)"
            )
    return 0


def _detector_name(value: str) -> str:
    """Validate one detector operand.

    Used as an argparse ``type`` rather than ``choices=`` because with
    ``nargs="*"`` argparse validates ``choices`` against the *whole* list, so
    `aludam fetch --dry-run` would reject the empty list as "invalid choice".
    """
    if value not in DETECTOR_NAMES:
        raise argparse.ArgumentTypeError(
            f"unknown detector {value!r}; choose from "
            f"{', '.join(DETECTOR_NAMES)}"
        )
    return value


def _cmd_fetch(args: argparse.Namespace) -> int:
    """``aludam fetch [detectors...] [--dry-run]`` (ALUDAM_PLAN.md 6.3)."""
    names = list(args.detectors) or None
    if args.dry_run:
        print(weights.summary(names))
        print()
        print("Run `aludam fetch` to download the missing weights.")
        return 0
    report = weights.fetch(names, dry_run=False, quiet=False)
    print()
    print(weights.summary(list(report)))
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    print(weights.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aludam",
        description="Uniform deepfake / AI-content detection.",
    )
    parser.add_argument(
        "--version", action="version", version=f"aludam {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="list available detectors")
    p_list.set_defaults(func=_cmd_list)

    p_pred = sub.add_parser("predict", help="run one detector")
    p_pred.add_argument("detector", choices=list(DETECTOR_NAMES))
    p_pred.add_argument(
        "input",
        help='text, @file (text), or a path to an image / video / audio file',
    )
    p_pred.add_argument(
        "--json", action="store_true", help="emit the full JSON result"
    )
    p_pred.set_defaults(func=_cmd_predict)

    p_fetch = sub.add_parser(
        "fetch",
        help="download detector weights (wheels never ship them)",
        description=(
            "Download the weights each detector needs. Already-present "
            "weights are skipped, so this is safe to re-run."
        ),
    )
    p_fetch.add_argument(
        "detectors", nargs="*", metavar="detector", type=_detector_name,
        help="which detectors to fetch (default: all)",
    )
    p_fetch.add_argument(
        "--dry-run", action="store_true",
        help="report what would be downloaded without downloading",
    )
    p_fetch.set_defaults(func=_cmd_fetch)

    p_status = sub.add_parser(
        "status", help="show which weights are present and which are missing",
    )
    p_status.set_defaults(func=_cmd_status)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except AludamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
