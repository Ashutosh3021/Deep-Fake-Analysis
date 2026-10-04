"""Tests for Phase 0 threshold fitting and artifact loading.

The dangerous failure here is the inverse of the one thresholds.py guards:
registering a band that looks calibrated but does not meet the operating
point it was fitted for. That would flip `publish_blocked()` to "allow" and
ship a release nobody can defend, so `load_artifact` refuses such entries and
that refusal is asserted below.

Run:  python tests/test_fit.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random  # noqa: E402

from eval.fit import (  # noqa: E402
    fit_band,
    rates_at,
    write_artifact,
    load_scores,
    MIN_CAL_ROWS,
)

from aludam import thresholds as T  # noqa: E402

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


def separable(n=1000, seed=7):
    """Well-separated scores: fitting should be easy and succeed."""
    rng = random.Random(seed)
    scores, labels = [], []
    for _ in range(n // 2):
        scores.append(min(0.999, max(0.0, rng.gauss(0.80, 0.06))))
        labels.append(1)
        scores.append(min(0.999, max(0.0, rng.gauss(0.20, 0.06))))
        labels.append(0)
    return scores, labels


def overlapping(n=800, seed=11):
    """Same distribution for both classes: no band can satisfy both targets."""
    rng = random.Random(seed)
    scores, labels = [], []
    for _ in range(n // 2):
        scores.append(min(0.999, max(0.0, rng.gauss(0.50, 0.30))))
        labels.append(1)
        scores.append(min(0.999, max(0.0, rng.gauss(0.50, 0.30))))
        labels.append(0)
    return scores, labels


print("\n[1] fitting a separable band")

scores, labels = separable()
art = fit_band(scores, labels, target_fpr=0.01, target_fnr=0.05,
               source="synthetic")
check(0 <= art["low"] < art["high"] <= 1, "low < high, both in [0,1]")
check(art["satisfies"] is True, "satisfies both targets")
check(art["violated"] == [], "nothing violated")
check(art["calibration_rates"]["fpr"] <= 0.01, "FPR <= target_fpr")
check(art["calibration_rates"]["fnr"] <= 0.05, "FNRate <= target_fnr")
check(art["calibration_auroc"] > 0.99, "calibration AUROC near 1")
check(art["n_calibration"] == len(scores), "row count recorded")

print("\n[2] FPR budget actually binds")

# Raising the FPR target must move `high` down (more positives accepted),
# never up. If it moves the wrong way the quantile is inverted.
low_tight = fit_band(scores, labels, target_fpr=0.001, target_fnr=0.05)
low_loose = fit_band(scores, labels, target_fpr=0.10, target_fnr=0.05)
check(low_tight["high"] >= low_loose["high"],
      "tighter FPR target => higher or equal `high`")

# And FNR target binds `low` the same way.
fnr_tight = fit_band(scores, labels, target_fpr=0.01, target_fnr=0.001)
fnr_loose = fit_band(scores, labels, target_fpr=0.01, target_fnr=0.20)
check(fnr_tight["low"] <= fnr_loose["low"],
      "tighter FNR target => lower or equal `low`")

print("\n[3] unworkable operating point is reported, not hidden")

ov, ovl = overlapping()
art_ov = fit_band(ov, ovl, target_fpr=0.01, target_fnr=0.01)
revealed = (
    art_ov["satisfies"] is False
    or art_ov["band_separable"] is False
    or art_ov["calibration_rates"].get("abstain_rate", 0) >= 0.5
)
check(revealed,
      "failure shows up as violated target, inverted band, or abstain>=50%")
check(art_ov["calibration_auroc"] < 0.75, "AUROC shows the overlap")
check("note" in art_ov or art_ov["violated"]
      or art_ov["calibration_rates"]["abstain_rate"] >= 0.5,
      "reason recorded in artifact")

print("\n[4] refuses insufficient data")

tiny_s = [0.1] * 10 + [0.9] * 10
tiny_l = [0] * 10 + [1] * 10
check(raises(lambda: fit_band(tiny_s, tiny_l)),
      f"raises below MIN_CAL_ROWS={MIN_CAL_ROWS}")
check(raises(lambda: fit_band(scores[:-1], labels)),
      "raises on length mismatch")
one_class = [0.9] * 500
check(raises(lambda: fit_band(one_class, [0] * 500)),
      "raises when one class is missing")

print("\n[5] rates_at")

r = rates_at(scores, labels, art["low"], art["high"])
check(set(r) == {"fpr", "fnr", "abstain_rate"}, "reports all three rates")
check(0 <= r["abstain_rate"] <= 1, "abstain rate in [0,1]")
check(r["fpr"] == sum(1 for s, y in zip(scores, labels)
                      if y == 0 and s >= art["high"]) / labels.count(0),
      "FPR computed against negatives only")

print("\n[6] artifact round-trip registers calibrated thresholds")

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "thresholds.json"
    art["detector"] = "text_det"
    art["calibration_split"] = "mage:valid"
    write_artifact(art, path)

    # Start from a clean slate: the module may have loaded a real artifact.
    T._FITTED.clear()
    T.ARTIFACT_REJECTED.clear()
    T._ARTIFACT_ATTEMPTED = False

    loaded = T.load_artifact(path)
    check("text_det" in loaded, "detector present after load")
    check(T.thresholds_for("text_det").calibrated is True,
          "thresholds_for reports calibrated=True")
    check(T.thresholds_for("text_det").low == art["low"], "low round-trips")
    check(T.thresholds_for("text_det").high == art["high"], "high round-trips")
    check(T.thresholds_for("img_det") is T.PLACEHOLDER,
          "unknown detector still gets PLACEHOLDER")
    blocked, offenders = T.publish_blocked(["text_det", "img_det"])
    check(blocked and offenders == ("img_det",),
          "publish still blocked by the unfitted modality")

print("\n[7] refuses an artifact entry that fails its own target")

with tempfile.TemporaryDirectory() as td:
    bad = Path(td) / "t.json"
    bad.write_text(json.dumps({
        "vdo_det": {"low": 0.2, "high": 0.8, "satisfies": False,
                    "violated": ["fpr", "fnr"], "source": "x"},
        "aud_det": {"low": 0.7, "high": 0.3, "satisfies": True},
        "text_det": {"n_calibration": 1},
        "notanobject": 5,
    }), encoding="utf-8")

    T._FITTED.clear()
    T.ARTIFACT_REJECTED.clear()
    T._ARTIFACT_ATTEMPTED = False

    loaded = T.load_artifact(bad)
    check("vdo_det" not in loaded, "satisfies=false entry refused")
    check("aud_det" not in loaded, "inverted band refused")
    check("text_det" not in loaded, "entry without low/high refused")
    check("notanobject" not in loaded, "non-object entry refused")
    check(set(T.ARTIFACT_REJECTED) == {"vdo_det", "aud_det", "text_det",
                                       "notanobject"},
          "every refusal carries a reason")
    check(T.thresholds_for("vdo_det") is T.PLACEHOLDER,
          "refused entry stays on PLACEHOLDER")
    blocked, offenders = T.publish_blocked(["vdo_det"])
    check(blocked, "a refused band keeps publishing blocked")

print("\n[8] missing and corrupt artifacts are not errors")

with tempfile.TemporaryDirectory() as td:
    T._ARTIFACT_ATTEMPTED = False
    T._FITTED.clear()
    check(T.load_artifact(Path(td) / "nope.json") == {},
          "missing file -> {} (Phase 0 simply has not run)")

    corrupt = Path(td) / "c.json"
    corrupt.write_text("{not json", encoding="utf-8")
    T._ARTIFACT_ATTEMPTED = False
    check(T.load_artifact(corrupt) == {}, "corrupt file -> {}, no raise")

    wrongtype = Path(td) / "w.json"
    wrongtype.write_text("[1,2,3]", encoding="utf-8")
    T._ARTIFACT_ATTEMPTED = False
    check(T.load_artifact(wrongtype) == {}, "non-object root -> {}")

# Restore a clean state for anything that imports thresholds afterwards.
T._FITTED.clear()
T.ARTIFACT_REJECTED.clear()
T._ARTIFACT_ATTEMPTED = False

print("\n[9] load_scores deduplicates by key (last write wins)")

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "r.jsonl"
    p.write_text(
        '{"key":"a","score":0.1,"label":0}\n'
        '{"key":"a","score":0.9,"label":1}\n'
        '{"key":"b","score":0.4,"label":1}\n'
        '{"key":"c","score":null,"label":0}\n'
        "not json\n"
        "\n",
        encoding="utf-8",
    )
    d = load_scores(p)
    check(len(d["scores"]) == 2, "deduped + skipped score=null + bad line")
    check(d["scores"][0] == 0.9, "last write wins for a duplicated key")
    check(d["labels"] == [1, 1], "labels follow their own records")

    # Interleaved keys: a naive pop-the-last dedup drops b and keeps both a's.
    p2 = Path(td) / "i.jsonl"
    p2.write_text(
        '{"key":"a","score":0.1,"label":0}\n'
        '{"key":"b","score":0.4,"label":1}\n'
        '{"key":"a","score":0.9,"label":1}\n',
        encoding="utf-8",
    )
    d2 = load_scores(p2)
    check(len(d2["scores"]) == 2, "interleaved keys deduplicated")
    check(0.4 in d2["scores"], "interleaved row b not dropped")
    check(d2["scores"].count(0.9) == 1, "only the winning copy of a kept")

print("\n" + "=" * 60)
print(f"{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
