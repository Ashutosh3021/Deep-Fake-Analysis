"""Contract tests for the aludam API.

Self-contained and fast: no torch, no transformers, no model weights. It
exercises the guarantees simultaneously for all four modalities:

1. ``load_detector`` accepts exactly the four specified names.
2. All four return the SAME ``DetectionResult`` class.
3. ``score`` and ``confidence`` are both 0.0..1.0.
4. While thresholds are **uncalibrated** (the shipped state) ``label`` is the
   upstream detector's verdict, so an unfitted band can never flip a fake to
   ``real``; the score-derived verdict is reported as
   ``details["placeholder_label"]``. Once Phase 0 fits the band,
   ``verdict = f(score)`` becomes the rule again.
5. ``confidence`` is flagged uncalibrated (``runtime["calibrated"] is False``).
6. The printed field format is identical across modalities.

Run:  python tests/test_contract.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aludam import (  # noqa: E402
    DETECTOR_NAMES,
    KIND_ALTERED,
    KIND_BOTH,
    KIND_GENERATED,
    KIND_UNKNOWN,
    VALID_KINDS,
    VALID_LABELS,
    DetectionResult,
    InvalidInputError,
    UnknownDetectorError,
    al,
    available_detectors,
    load_detector,
)
from aludam.normalize import normalize  # noqa: E402
from aludam.thresholds import (  # noqa: E402
    PLACEHOLDER,
    decide,
    decide_uncalibrated,
    margin,
    publish_blocked,
)

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


def raises(fn, exc_type=ValueError) -> bool:
    try:
        fn()
    except exc_type:
        return True
    except Exception:
        return False
    return False


# --- 1. detector names ---------------------------------------------------

print("\n[1] detector names")
check(DETECTOR_NAMES == ("text_det", "img_det", "vdo_det", "aud_det"),
      "exactly the four specified names, in order")
check(available_detectors() == DETECTOR_NAMES, "available_detectors() mirrors")

for bad in ("text", "IMAGE_DET", "", "image"):
    try:
        load_detector(bad)
        check(False, f"load_detector({bad!r}) should raise")
    except UnknownDetectorError as exc:
        check("text_det" in str(exc),
              f"load_detector({bad!r}) raises with valid names listed")


# --- 2. DetectionResult construction -------------------------------------

print("\n[2] DetectionResult construction")

good = DetectionResult(
    score=0.8012, label="ai", confidence=0.3125,
    explanation="Synthetic / machine-generated content (score 0.80).",
    kind=KIND_GENERATED,
)
check(good.score == 0.8012, "score preserved at 4 dp")
check(good.confidence == 0.3125, "confidence preserved at 4 dp")
check(good.label == "ai", "label is 'ai'")
check(good.kind == "generated", "kind is 'generated'")
check(good.runtime["calibrated"] is False, "runtime.calibrated ships False")
check(good.calibrated is False, "calibrated property is False")

for bad_kwargs, what in [
    (dict(score=80.12, label="ai", confidence=0.5, explanation="x"),
     "score > 1 rejected"),
    (dict(score=0.5, label="ai", confidence=80.12, explanation="x"),
     "0-100 confidence rejected (not silently clamped)"),
    (dict(score=-0.1, label="ai", confidence=0.5, explanation="x"),
     "negative score rejected"),
    (dict(score=0.5, label="FAKE", confidence=0.5, explanation="x"),
     "native label rejected"),
    (dict(score=0.5, label="ai", confidence=0.5, explanation="x", kind="spoofed"),
     "unknown kind rejected"),
    (dict(score=0.5, label="ai", confidence=0.5, explanation="x",
          runtime={"calibrated": "yes"}),
     "non-bool calibrated rejected"),
    (dict(score=0.5, label="ai", confidence=0.5, explanation="x",
          signals=("A", "B"), kind=KIND_BOTH),
     "valid: signals+kind accepted"),
]:
    try:
        DetectionResult(**bad_kwargs)
        check(what.startswith("valid:"), what)
    except (ValueError, TypeError) as exc:
        check(not what.startswith("valid:"), f"{what} -> {type(exc).__name__}")


# --- 3. uniform repr across modalities -----------------------------------

print("\n[3] uniform printed format (all four modalities)")

RAW = {
    "text_det": dict(
        media_type="TEXT",
        raw=dict(
            label="AI_GENERATED", ai_probability=0.8012, confidence=88.4,
            perplexity_signal=0.91, stylometry_signal=0.74, pattern_signal=0.88,
            watermark_signal=0.12,
            metrics={"word_count": 240,
                     "perplexity_details": {"perplexity": 12.3, "burstiness": 0.41}},
            notes="Consistent lexical uniformity.",
        ),
    ),
    "img_det": dict(
        media_type="IMAGE",
        raw=dict(
            label="FAKE", p_synthetic=0.9431, confidence=76.0,
            fake_type=["face_swap", "ai_edited_region"],
            family_scores={"provenance": 0.0, "fully_ai_generated": 0.11,
                           "ai_edited_region": 0.87, "face_swap": 0.96,
                           "model_signal_raw": 0.9, "global_heuristic": 0.55,
                           "fused_score": 0.94, "relative_evidence": 1.18},
            faces_detected=1,
            file_hash="abc123",
            notes="Two regions disagree with the global generator fingerprint.",
            tier=2, tier_name="Manipulation",
        ),
    ),
    "vdo_det": dict(
        media_type="VIDEO",
        raw=dict(
            label="FAKE", confidence=88.0, frames_analyzed=48,
            fake_type=["temporal_inconsistency"],
            family_scores={"fused_score": 0.912, "fully_ai_generated": 0.4,
                           "ai_edited_region": 0.2, "face_swap": 0.2,
                           "temporal_inconsistency": 0.93,
                           "audio_visual_desync": 0.12,
                           "temporal_details": {"cuts": 3},
                           "v2_lip_sync": 0.81, "v2_rppg": 0.33},
            suspicious_intervals=[{"start": 1.2, "end": 2.4}],
            file_hash="def456",
            notes="Inconsistent temporal flow across 3 cut points.",
        ),
    ),
    "aud_det": dict(
        media_type="AUDIO",
        raw=dict(
            label="SYNTHETIC", fake_probability=0.8734, confidence=82.0,
            signal_source="neural",
            # real keys from final_audio_detector.py:533-546 - note the
            # *_count fields, which must NOT surface as signals.
            feature_summary={"raw_score": 0.87, "quality_weighted_score": 0.83,
                             "segment_count": 8, "suspicious_segment_count": 3,
                             "v2_amff": 0.91, "v2_tdnn": 0.77,
                             "v2_aasist2": None},
            suspicious_segments=[{"start": 0.5, "end": 1.5}],
            notes="Narrow-band vocoder residue at 6.2 kHz.",
        ),
    ),
}

results = {}
for name, spec in RAW.items():
    results[name] = normalize(
        spec["raw"], detector=name, media_type=spec["media_type"],
        backend="neural", elapsed_ms=12,
    )

classes = {type(r) for r in results.values()}
check(len(classes) == 1 and DetectionResult in classes,
      "all four return the identical DetectionResult class")

for name, r in results.items():
    check(0.0 <= r.score <= 1.0, f"{name}: score in [0,1] (got {r.score})")
    check(0.0 <= r.confidence <= 1.0,
          f"{name}: confidence in [0,1] (got {r.confidence})")
    check(r.label in VALID_LABELS, f"{name}: label in vocabulary (got {r.label!r})")
    check(r.kind in VALID_KINDS, f"{name}: kind in vocabulary (got {r.kind!r})")
    check(isinstance(r.explanation, str) and len(r.explanation) > 20,
          f"{name}: explanation is non-trivial ({len(r.explanation)} chars)")
    check(len(r.explanation) <= 280,
          f"{name}: explanation <= 280 chars")
    check(r.runtime["calibrated"] is False,
          f"{name}: runtime.calibrated is False")
    check(r.calibrated is False, f"{name}: calibrated property False")

reps = {name: repr(r) for name, r in results.items()}
check(all(rep.startswith("DetectionResult(score=") for rep in reps.values()),
      "every repr starts with DetectionResult(score=")
for name, rep in reps.items():
    for f in ("label=", "confidence=", "explanation="):
        check(f in rep, f"{name}: repr contains {f}")

print("\n  --- sample output, all four, same format ---")
for name in ("text_det", "img_det", "vdo_det", "aud_det"):
    print(f"  {results[name]!r}")
print("  ---")


# --- 4. the decision invariant -------------------------------------------

print("\n[4] decision rule (UNCALIBRATED: upstream verdict decides)")

for name, r in results.items():
    check(r.runtime["calibrated"] is False, f"{name}: ships uncalibrated")
    up = r.details["upstream_label_uniform"]
    expected, reason = decide_uncalibrated(up, r.score, PLACEHOLDER)
    check(r.label == expected,
          f"{name}: label {r.label!r} == decide_uncalibrated(upstream={up!r}, "
          f"score={r.score}) -> {expected!r}")
    check(r.details["label_reason"] == reason,
          f"{name}: reason recorded ({r.details['label_reason']})")
    check(r.details["label_basis"] == "upstream",
          f"{name}: label_basis is 'upstream' while uncalibrated")
    ph, _ = decide(r.score, PLACEHOLDER)
    check(r.details["placeholder_label"] == ph,
          f"{name}: placeholder verdict still reported ({ph!r})")
    check("Uncalibrated" in r.explanation,
          f"{name}: explanation discloses it is uncalibrated")

# THE guarantee: an unfitted band must never turn a fake into a real.
print("  -- placeholder safety matrix --")
for upstream in ("ai", "real", "uncertain"):
    for score in (0.0, 0.4499, 0.45, 0.5, 0.5499, 0.55, 1.0):
        got, _ = decide_uncalibrated(upstream, score, PLACEHOLDER)
        if upstream == "ai":
            check(got != "real",
                  f"upstream=ai score={score}: never 'real' (got {got!r})")
        if upstream == "real":
            check(got != "ai",
                  f"upstream=real score={score}: placeholder never escalates "
                  f"to 'ai' (got {got!r})")
        if got == "uncertain":
            in_band = PLACEHOLDER.low < score < PLACEHOLDER.high
            check(upstream == "uncertain" or in_band,
                  f"upstream={upstream} score={score}: 'uncertain' only from "
                  f"abstention or the band")

check(raises(lambda: decide_uncalibrated("FAKE", 0.5, PLACEHOLDER)),
      "non-uniform upstream label raises")

# Boundary behaviour of the score band itself (unchanged, and still what
# Phase 0 will fit against)
for score, want in [(0.0, "real"), (0.4499, "real"), (0.45, "real"),
                    (0.4501, "uncertain"), (0.5, "uncertain"),
                    (0.5499, "uncertain"), (0.55, "ai"), (1.0, "ai")]:
    got, _ = decide(score, PLACEHOLDER)
    check(got == want, f"decide({score}) == {want!r} (got {got!r})")

# confidence must equal the margin, not the (decorative) upstream value
for name, r in results.items():
    check(r.confidence == round(margin(r.score, PLACEHOLDER), 4),
          f"{name}: confidence == margin(score)")

# thresholds echoed back for auditability
t = results["img_det"].thresholds
check(t.get("calibrated") is False and "low" in t and "high" in t,
      "details.thresholds echoes low/high/calibrated")


# --- 5. upstream verdicts drive the label while uncalibrated ---------------

print("\n[5] upstream label decides while uncalibrated, but stays auditable")

check(results["img_det"].upstream_label == "FAKE",
      f"upstream_label kept (got {results['img_det'].upstream_label!r})")
check(results["text_det"].upstream_label == "AI_GENERATED", "text upstream kept")
check(results["aud_det"].upstream_label == "SYNTHETIC", "audio upstream kept")
check(results["vdo_det"].upstream_label == "FAKE", "video upstream kept")
check(results["img_det"].details.get("upstream_label_uniform") == "ai",
      "uniform upstream mapping recorded for Phase 0 comparison")
check(results["text_det"].details["upstream_confidence_0_100"] == 88.4,
      "native 0-100 confidence preserved raw")
check(results["img_det"].details["label_basis"] == "upstream",
      "label_basis recorded as 'upstream' (not stale, not 'score')")

# The case that motivated the whole rule: upstream said FAKE, score is low,
# and the placeholder band would have flipped it to 'real'.
low_score_image = normalize(
    {"label": "FAKE", "p_synthetic": 0.42, "confidence": 60.0,
     "fake_type": ["face_swap"], "family_scores": {"face_swap": 0.62,
                                                   "fused_score": 0.42},
     "faces_detected": 1, "notes": "Marginal signal."},
    detector="img_det", media_type="IMAGE", backend="neural", elapsed_ms=9,
)
check(low_score_image.upstream_label == "FAKE", "upstream still says FAKE")
check(low_score_image.score <= PLACEHOLDER.low,
      "score 0.42 sits below the placeholder band")
check(low_score_image.details["placeholder_label"] == "real",
      "placeholder WOULD have said 'real' - reported, not used")
check(low_score_image.label == "ai",
      f"but label stays 'ai' (got {low_score_image.label!r}) - "
      "placeholder never flips a fake to real")
check(low_score_image.runtime["calibrated"] is False,
      "result is flagged uncalibrated")

# The converse: an uncalibrated placeholder must not escalate either.
real_image = normalize(
    {"label": "AUTHENTIC", "p_synthetic": 0.92, "confidence": 70.0,
     "fake_type": [], "family_scores": {"provenance": 0.0,
                                        "fused_score": 0.92},
     "faces_detected": 0, "notes": ""},
    detector="img_det", media_type="IMAGE", backend="neural", elapsed_ms=5,
)
check(real_image.details["placeholder_label"] == "ai",
      "placeholder would have said 'ai'")
check(real_image.label == "real",
      f"upstream AUTHENTIC stays 'real' (got {real_image.label!r})")

# Publishing gate: no modality may ship uncalibrated.
blocked, offenders = publish_blocked(DETECTOR_NAMES)
check(blocked is True, "publish_blocked() blocks while any modality is raw")
check(set(offenders) == set(DETECTOR_NAMES),
      f"all four modalities are offenders (got {offenders})")


# --- 6. kind vocabulary ----------------------------------------------------

print("\n[6] kind: generated vs altered vs both, kept out of the verdict")

check(results["img_det"].kind == KIND_ALTERED, "face_swap -> kind='altered'")
check(results["vdo_det"].kind == KIND_ALTERED, "temporal -> kind='altered'")
check(results["text_det"].kind == KIND_GENERATED, "text signals -> kind='generated'")
check(results["aud_det"].kind == KIND_GENERATED, "audio -> kind='generated'")

both = normalize(
    {"label": "FAKE", "p_synthetic": 0.9, "confidence": 80.0,
     "fake_type": ["face_swap", "fully_ai_generated"],
     "family_scores": {"face_swap": 0.9, "fully_ai_generated": 0.85,
                       "fused_score": 0.9},
     "faces_detected": 1, "notes": ""},
    detector="img_det", media_type="IMAGE", backend="neural", elapsed_ms=5,
)
check(both.kind == KIND_BOTH, "both families -> kind='both'")

none_fired = normalize(
    {"label": "AUTHENTIC", "p_synthetic": 0.1, "confidence": 71.0,
     "fake_type": [], "family_scores": {"face_swap": 0.04, "provenance": 0.0,
                                        "fused_score": 0.1},
     "faces_detected": 0, "notes": ""},
    detector="img_det", media_type="IMAGE", backend="neural", elapsed_ms=5,
)
check(none_fired.kind == KIND_UNKNOWN, "no signals -> kind='unknown'")
check(none_fired.label == "real", "AUTHENTIC + score 0.1 -> label 'real'")
check(none_fired.explanation.startswith("No strong evidence"),
      "authentic opener does not over-claim")
check("no AI provenance tag found" in none_fired.explanation,
      f"provenance absence reported (got: {none_fired.explanation})")

# UNCERTAIN abstention
unc = normalize(
    {"label": "UNCERTAIN", "ai_probability": 0.53, "confidence": 41.0,
     "metrics": {"word_count": 40}, "notes": ""},
    detector="text_det", media_type="TEXT", backend="neural", elapsed_ms=5,
)
check(unc.label == "uncertain", "score 0.53 inside band -> 'uncertain'")
check(unc.confidence == 0.0, "margin is 0 inside the band")
check("inconclusive band" in unc.explanation,
      f"abstention explains itself (got: {unc.explanation})")


# --- 7. signal hygiene ------------------------------------------------------

print("\n[7] signal hygiene")
check("FACE_SWAP" in results["img_det"].signals
      and results["img_det"].signals[0] == "FACE_SWAP",
      "signals ranked by score, FACE_SWAP first")
check("FUSED_SCORE" not in results["img_det"].signals
      and "RELATIVE_EVIDENCE" not in results["img_det"].signals,
      "fused/meta scores excluded from signals")
check(not any("COUNT" in s for s in results["aud_det"].signals),
      f"audio counters excluded (got {results['aud_det'].signals})")
check("V2_AMFF" in results["aud_det"].signals,
      f"audio v2 model score reported (got {results['aud_det'].signals})")
check("AV_SYNC_MISMATCH" in results["vdo_det"].signals
      or "TEMPORAL_INCONSISTENCY" in results["vdo_det"].signals,
      f"video signals mapped (got {results['vdo_det'].signals})")


# --- 8. errors raise, never degrade ----------------------------------------

print("\n[8] errors raise, never degrade")
try:
    normalize({"error": "empty_text"}, detector="text_det",
              media_type="TEXT", backend="neural", elapsed_ms=1)
    check(False, "upstream error dict should raise")
except InvalidInputError as exc:
    check("empty_text" in str(exc), "upstream error -> InvalidInputError")

try:
    normalize({"ai_probability": 0.4}, detector="text_det",
              media_type="TEXT", backend="neural", elapsed_ms=1)
    check(False, "missing label should raise")
except InvalidInputError:
    check(True, "missing label -> InvalidInputError")

try:
    normalize({"label": "MAYBE_SYNTHETIC", "ai_probability": 0.9,
               "confidence": 50}, detector="text_det",
              media_type="TEXT", backend="neural", elapsed_ms=1)
    check(False, "unknown upstream label should raise (schema drift)")
except InvalidInputError:
    check(True, "unknown upstream label -> InvalidInputError (loud)")

check(callable(al.detect), "al.detect exported")
check(al.detect.__module__.endswith("al"), "al.detect lives in aludam.al")

# --- summary -----------------------------------------------------------------

print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
