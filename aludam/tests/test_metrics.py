"""Tests for the Phase 0 metrics/calibration helpers.

Self-contained: no network, no model weights, no torch. Verifies the numbers
that every Phase 0 gate is decided on, because a mis-wired metric would let a
bad model swap through the gate.

Run:  python tests/test_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import metrics as M  # noqa: E402

PASS = 0
FAIL = 0


def check(condition: bool, label: str) -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}")


def raises(fn, exc_type=ValueError):
    try:
        fn()
    except exc_type:
        return True
    except Exception:
        return False
    return False


# --- 1. AUROC ------------------------------------------------------------

print("\n[1] AUROC")

labels = [0] * 50 + [1] * 50
perfect = [i / 100 for i in range(100)]
check(M.auroc(perfect, labels) == 1.0, "perfect separation -> 1.0")
check(M.auroc([1 - x for x in perfect], labels) == 0.0, "inverted scores -> 0.0")
check(abs(M.auroc([0.5] * 100, labels) - 0.5) < 1e-9,
      "constant scores -> 0.5 (ties)")

# label convention: higher score must mean class 1, or every gate is inverted
check(M.auroc([float(y) for y in labels], labels) == 1.0,
      "class 1 is the positive class (score=1 => label=1 => 1.0)")
check(M.auroc([float(1 - y) for y in labels], labels) == 0.0,
      "score=1 => label=0 gives 0.0 (catches an inverted convention)")

check(raises(lambda: M.auroc(perfect, [0] * 100)),
      "single class raises (AUROC undefined)")
check(raises(lambda: M.auroc([0.1, 0.9], [0, 1])),
      "too few rows raises")
check(raises(lambda: M.auroc([float("nan")] * 100, labels)),
      "NaN score raises")
check(raises(lambda: M.auroc(perfect[:50], labels)),
      "length mismatch raises")
check(raises(lambda: M.auroc(perfect, [0, 1, 2] * 33 + [0])),
      "non-binary labels raise")
check(raises(lambda: M.auroc([0.1] * 20 + [0.9] * 8, [0] * 20 + [1] * 8)),
      "below MIN_POS raises")


# --- 2. TPR @ FPR --------------------------------------------------------

print("\n[2] tpr_at_fpr")

check(M.tpr_at_fpr(perfect, labels, 0.01) == 1.0,
      "perfect model reaches TPR 1.0 at 1% FPR")
check(M.tpr_at_fpr([0.5] * 100, labels, 0.01) == 0.0,
      "constant score -> TPR 0 at 1% FPR (a tie must accept everyone "
      "or no one, so no TPR is purchasable)")
check(raises(lambda: M.tpr_at_fpr(perfect, labels, 1.5)),
      "target_fpr > 1 raises")
check(raises(lambda: M.tpr_at_fpr(perfect, labels, -0.1)),
      "target_fpr < 0 raises")

fpr, tpr, thr = M.roc_curve_points(perfect, labels)
check(len(fpr) == len(tpr) == len(thr),
      "roc_curve_points returns three equal-length arrays")
check(fpr[-1] == 1.0 and tpr[-1] == 1.0,
      "curve ends at the accept-everyone point (1,1)")
check(np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0),
      "fpr and tpr are both non-decreasing")


# --- 3. grouped bootstrap -------------------------------------------------

print("\n[3] grouped bootstrap")

groups = [f"doc{i // 5}" for i in range(100)]  # 20 groups of 5 rows
point, lo, hi = M.bootstrap_auroc(perfect, labels, n_boot=200, seed=1)
check(lo <= point <= hi, "interval brackets the point estimate")
check(0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0, "interval inside [0,1]")

p2, lo2, hi2 = M.bootstrap_auroc(perfect, labels, n_boot=200, seed=1)
check((p2, lo2, hi2) == (point, lo, hi), "seed makes the interval reproducible")

pg, log_, hig = M.bootstrap_auroc(perfect, labels, groups=groups,
                                  n_boot=200, seed=1)
check(log_ <= pg <= hig, "grouped interval brackets the point estimate")

# grouping must not silently widen: it changes the resampling unit
check(raises(lambda: M.bootstrap_auroc(perfect, labels, n_boot=100, seed=1,
                                       groups=groups[:50])),
      "groups length mismatch raises")
check(raises(lambda: M.bootstrap_auroc(perfect, labels, n_boot=100, seed=1,
                                       groups=["a"] * 3 + ["b"] * 97)),
      "<5 distinct groups raises (interval would be meaningless)")


# --- 4. paired bootstrap -------------------------------------------------

print("\n[4] paired bootstrap delta")

rng = np.random.default_rng(0)
noise = rng.normal(0, 0.03, 100)
a = [min(1.0, max(0.0, perfect[i] + noise[i])) for i in range(100)]
b = rng.normal(0.5, 0.3, 100).tolist()

d, lo, hi = M.paired_bootstrap_delta(perfect, perfect, labels,
                                     n_boot=200, seed=2)
check(abs(d) < 1e-12, "identical models -> delta exactly 0")
check(lo <= 0 <= hi, "identical models -> interval straddles 0 (no fake gain)")

d2, lo2, hi2 = M.paired_bootstrap_delta(perfect, b, labels,
                                        n_boot=200, seed=2)
check(d2 > 0, "A better than B -> positive delta")
check(lo2 > 0, "clear win -> lower bound > 0 (the gate we actually use)")

check(raises(lambda: M.paired_bootstrap_delta(perfect, b[:50], labels)),
      "different-length models raise")
check(raises(lambda: M.paired_bootstrap_delta(perfect, b, [1] * 100)),
      "mismatched labels raise")


# --- 5. calibration ------------------------------------------------------

print("\n[5] calibrator + reliability")

iso = M.fit_calibrator(perfect, labels, method="isotonic")
probs = np.asarray(iso(perfect), dtype=float)
check(np.all((probs >= 0) & (probs <= 1)), "isotonic output in [0,1]")
check(np.all(np.diff(probs[::10]) >= -1e-9), "isotonic is monotone in score")
check(float(np.asarray(iso([0.0]))[0]) < float(np.asarray(iso([1.0]))[0]),
      "isotonic orders 0.0 below 1.0")

platt = M.fit_calibrator(perfect, labels, method="platt")
p = np.asarray(platt(perfect), dtype=float)
check(np.all((p >= 0) & (p <= 1)), "platt output in [0,1]")
check(len(p) == 100, "platt returns one value per row")
check(raises(lambda: M.fit_calibrator(perfect, labels, method="bogus")),
      "unknown method raises")

check(abs(M.ece([0.5] * 100, labels)) < 1e-9,
      "all-prob-0.5 on balanced labels -> ECE 0")
check(abs(M.ece([float(y) for y in labels], labels)) < 1e-9,
      "perfectly confident and correct -> ECE 0")
check(M.ece(perfect, labels) > 0.1,
      "well-ranked but uncalibrated scores still show ECE "
      "(a good AUROC is not a calibrated probability)")
check(M.ece([0.9] * 50 + [0.1] * 50, [0] * 50 + [1] * 50) > 0.5,
      "overconfident and wrong -> large ECE")

bins = M.reliability_bins([0.1] * 50 + [0.9] * 50, labels, n_bins=10)
check(sum(n for _, _, n in bins) == 100, "reliability bins account for every row")


# --- summary -------------------------------------------------------------

print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
