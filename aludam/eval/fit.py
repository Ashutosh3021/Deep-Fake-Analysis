"""Fit the abstain band on a held-out calibration split.

Decides the two numbers ``decide()`` uses: ``low`` (score <= low -> real) and
``high`` (score >= high -> ai), in between -> uncertain.

Two independent constraints, not one:

  * ``target_fpr`` bounds how often an *authentic* input is called ai. This is
    the number that decides whether a false accusation ships. ``high`` is set
    to the smallest score at which at most ``target_fpr`` of negatives reach it.
  * ``target_fnr`` bounds how often a *synthetic* input is called real. Left
    unconstrained, pushing ``high`` up to hit 1% FPR can drag ``low`` with it
    and quietly manufacture false negatives - the exact failure finding 12
    recorded. ``low`` is therefore set from the positive distribution.

The two constraints can be mutually unsatisfiable: if the score distributions
overlap too heavily, the band that meets the FPR budget also fails the FNR
budget. That is a *result*, not an error - it means this detector's score
cannot support the operating point, and the artifact says so rather than
shipping a band that was tuned to look tidy.

Run:
  python eval/fit.py --calibration eval/runs/mage_valid.jsonl \\
                     --detector text_det --target-fpr 0.01 --target-fnr 0.05

The artifact is written into the package so an installed ``aludam`` picks it up
without the eval tree present; see ``aludam.thresholds.load_artifact``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE.parent / "src" / "aludam" / "data" / "thresholds.json"

MIN_CAL_ROWS = 200
MIN_PER_CLASS = 50


def _quantile_hi(neg_scores: Sequence[float], target: float) -> float:
    """Smallest score such that at most ``target`` of negatives reach it."""
    import numpy as np

    arr = np.sort(np.asarray(neg_scores, dtype=np.float64))
    if len(arr) == 0:
        raise ValueError("no negative scores")
    # index of the value below which (1-target) of the data lies
    k = int(np.floor((1.0 - target) * (len(arr) - 1)))
    k = max(0, min(len(arr) - 1, k))
    return float(arr[k])


def _quantile_lo(pos_scores: Sequence[float], target: float) -> float:
    """Largest score such that at most ``target`` of positives fall below it."""
    import numpy as np

    arr = np.sort(np.asarray(pos_scores, dtype=np.float64))
    if len(arr) == 0:
        raise ValueError("no positive scores")
    k = int(np.floor(target * (len(arr) - 1)))
    k = max(0, min(len(arr) - 1, k))
    return float(arr[k])


def rates_at(
    scores: Sequence[float], labels: Sequence[int], low: float, high: float
) -> Dict[str, float]:
    """Empirical FPR / FNR / abstain rate of a band on the data it came from."""
    n = len(scores)
    neg = [s for s, y in zip(scores, labels) if y == 0]
    pos = [s for s, y in zip(scores, labels) if y == 1]
    if not neg or not pos:
        return {}
    fpr = sum(1 for s in neg if s >= high) / len(neg)
    fnr = sum(1 for s in pos if s <= low) / len(pos)
    abstain = sum(1 for s in scores if low < s < high) / n
    return {
        "fpr": round(fpr, 5),
        "fnr": round(fnr, 5),
        "abstain_rate": round(abstain, 5),
    }


def fit_band(
    scores: Sequence[float],
    labels: Sequence[int],
    target_fpr: float = 0.01,
    target_fnr: float = 0.05,
    source: str = "",
) -> Dict[str, Any]:
    """Return an artifact dict describing the fitted band.

    Raises ValueError when there is too little data to fit at all. When the
    two budgets conflict the artifact is still returned, with
    ``satisfies=False`` and the measured violations spelled out - refusing to
    return anything would hide the finding.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    n = len(scores)
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = n - n_pos

    out: Dict[str, Any] = {
        "detector": None,
        "calibration_split": None,
        "source": source,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_calibration": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "target_fpr": target_fpr,
        "target_fnr": target_fnr,
    }

    if n < MIN_CAL_ROWS or n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        raise ValueError(
            f"too little calibration data: n={n} pos={n_pos} neg={n_neg} "
            f"(need n>={MIN_CAL_ROWS}, >= {MIN_PER_CLASS} per class)"
        )

    neg_scores = [s for s, y in zip(scores, labels) if y == 0]
    pos_scores = [s for s, y in zip(scores, labels) if y == 1]

    # Each class gives one edge at its own budget:
    #   A = largest `low` still meeting the FNR budget (P(pos <= low) <= fnr)
    #   B = smallest `high` still meeting the FPR budget (P(neg >= high) <= fpr)
    #
    # Taking them straight gives low=A, high=B, which is correct only while
    # A < B (heavily overlapping scores). On separable data A sits *above* B -
    # e.g. A=0.70, B=0.34 - and the pair is an inverted band that Thresholds
    # would reject. That crossing is not a failure, it means the two budgets
    # have slack: the abstain band belongs in the gap between the
    # distributions, so the edges swap to low=B, high=A, which still meets both
    # budgets (each edge is now the *looser* one) and puts the uncertain region
    # where no sample actually lives.
    a = _quantile_lo(pos_scores, target_fnr)
    b = _quantile_hi(neg_scores, target_fpr)
    low, high = (a, b) if a <= b else (b, a)
    if low >= high:  # a == b exactly: zero-width band
        high = min(1.0, high + 1e-6)

    out["low"] = round(float(low), 4)
    out["high"] = round(float(high), 4)
    out["calibration_rates"] = rates_at(scores, labels, low, high)

    # Overlap check: does the band hold on calibration at all?
    violated = []
    r = out["calibration_rates"]
    if r["fpr"] > target_fpr + 1e-9:
        violated.append("fpr")
    if r["fnr"] > target_fnr + 1e-9:
        violated.append("fnr")
    out["satisfies"] = not violated
    out["violated"] = violated
    out["band_separable"] = bool(low < high)
    if low >= high:
        out["note"] = (
            "low >= high: the score distributions overlap so heavily that the "
            "FPR and FNR budgets define an empty or inverted band. No band "
            "from this score meets both constraints."
        )

    # Score quality on the calibration split - if this is near 0.5, no
    # threshold choice can save it, and the report should say so first.
    try:
        from .metrics import auroc as _auroc

        out["calibration_auroc"] = round(_auroc(scores, labels), 4)
    except Exception:
        pass

    # Monotone calibrator for a future calibrated `confidence`.
    try:
        from .metrics import fit_calibrator, ece, reliability_bins

        cal = fit_calibrator(scores, labels, method="isotonic")
        probs = [float(cal(float(s))) for s in scores]
        out["ece_raw"] = round(ece([float(s) for s in scores], labels), 4)
        out["ece_isotonic"] = round(ece(probs, labels), 4)
        out["reliability"] = [
            [round(m, 4), round(o, 4), int(c)]
            for m, o, c in reliability_bins(probs, labels)
        ]
        out["calibrator"] = "isotonic"
    except Exception as exc:
        out["calibrator_error"] = f"{type(exc).__name__}: {exc}"

    return out


def write_artifact(artifact: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[artifact["detector"]] = artifact
    path.write_text(json.dumps(existing, indent=2, sort_keys=True),
                    encoding="utf-8")


def load_scores(path: Path, split: Optional[str] = None) -> Dict[str, Any]:
    """Score/label pairs from a run's JSONL, deduplicated by key.

    A dict rather than a list-with-pop: popping the last element removes the
    wrong record when keys interleave (a, b, a) and would silently drop row b
    while keeping both copies of a.
    """
    by_key: Dict[str, Any] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("score") is None or rec.get("label") is None:
                continue
            if split and rec.get("split") != split:
                continue
            by_key[str(rec.get("key"))] = rec

    scores = [float(r["score"]) for r in by_key.values()]
    labels = [int(r["label"]) for r in by_key.values()]
    return {"scores": scores, "labels": labels}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval.fit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True,
                    help="JSONL of calibration-split scores (mage:valid)")
    ap.add_argument("--detector", default="text_det")
    ap.add_argument("--split", default=None,
                    help="only use records with this split value")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--target-fnr", type=float, default=0.05)
    ap.add_argument("--out", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the artifact, do not write it")
    args = ap.parse_args(argv)

    data = load_scores(Path(args.calibration), args.split)
    if not data["scores"]:
        print(f"no scored rows in {args.calibration}")
        return 1

    try:
        art = fit_band(
            data["scores"], data["labels"],
            target_fpr=args.target_fpr, target_fnr=args.target_fnr,
            source=str(args.calibration),
        )
    except ValueError as exc:
        print(f"cannot fit: {exc}")
        return 1

    art["detector"] = args.detector
    art["calibration_split"] = args.split

    print(json.dumps({k: v for k, v in art.items()
                      if k != "reliability"}, indent=2))
    if not art["satisfies"]:
        print(f"\nWARNING: band does not meet target on calibration "
              f"(violated: {art['violated']})")
    if not art.get("band_separable", True):
        print("WARNING: band is empty or inverted - see artifact 'note'")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    write_artifact(art, Path(args.out))
    print(f"\nartifact -> {args.out}")
    print("restart or call aludam.thresholds.load_artifact() to activate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
