"""Adapters: raw detector dicts -> :class:`aludam.result.DetectionResult`.

The four upstream detectors return four different dict shapes. This module is
the single place where they are unified, so nothing downstream ever has to
branch on modality.

Upstream shapes (all verified against the dataclasses' ``to_dict()``):

===========  ==========================================  ==================
modality     native labels                               score key
===========  ==========================================  ==================
text         AI_GENERATED / HUMAN_WRITTEN / UNCERTAIN    ai_probability
image        FAKE / AUTHENTIC / UNCERTAIN                p_synthetic
audio        SYNTHETIC / AUTHENTIC / UNCERTAIN           fake_probability
video        FAKE / AUTHENTIC / UNCERTAIN                family_scores.fused_score
===========  ==========================================  ==================

All four emit ``confidence`` on a **0-100** scale; it is divided by 100 here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .exceptions import InvalidInputError
from .result import (
    LABEL_AI,
    LABEL_REAL,
    LABEL_UNCERTAIN,
    KIND_ALTERED,
    KIND_BOTH,
    KIND_GENERATED,
    KIND_UNKNOWN,
    DetectionResult,
)
from .thresholds import Thresholds, decide, decide_uncalibrated, margin, thresholds_for

__all__ = ["normalize", "EXPLANATION_CHAR_LIMIT"]

EXPLANATION_CHAR_LIMIT = 280

# Score at or above which a signal is reported as having "fired".
SIGNAL_THRESHOLD = 0.5

# Keys that are NOT signals even though they are numeric: fused/meta scores
# (redundant with `score` itself), counters, and aggregate readouts.
# Without this filter, audio's `segment_count: 3` would rank first as a
# "signal scored 3.00", and image's `fused_score` would shadow its own score.
#
# NB: `model_signal_raw` and `global_heuristic` are deliberately NOT here --
# they are individual detector inputs to the fusion, not the fused output.
NON_SIGNAL_KEYS = frozenset(
    {
        "fused_score",
        "relative_evidence",
        "raw_score",
        "quality_weighted_score",
        "segment_count",
        "suspicious_segment_count",
        "amff_score",
    }
)


# Native label -> uniform label.
_LABEL_MAP = {
    # synthetic / generated side
    "AI_GENERATED": LABEL_AI,
    "FAKE": LABEL_AI,
    "SYNTHETIC": LABEL_AI,
    "LIKELY_SYNTHETIC": LABEL_AI,
    # authentic side
    "HUMAN_WRITTEN": LABEL_REAL,
    "AUTHENTIC": LABEL_REAL,
    "LIKELY_AUTHENTIC": LABEL_REAL,
    # abstentions
    "UNCERTAIN": LABEL_UNCERTAIN,
    "INDETERMINATE": LABEL_UNCERTAIN,
}

# Upstream family/signal key -> public signal name.
# (ALUDAM_PLAN.md sec 7.1 signal-name mapping table.)
SIGNAL_NAMES = {
    "face_swap": "FACE_SWAP",
    "fully_ai_generated": "FULLY_AI_GENERATED",
    "ai_edited_region": "SPLICE_EDIT",
    "splice": "SPLICE_EDIT",
    "temporal_inconsistency": "TEMPORAL_INCONSISTENCY",
    # NB: the video detector's family key is `audio_visual_desync`
    # (final_video_detector.py:814), NOT `audio_visual_sync`.
    "audio_visual_desync": "AV_SYNC_MISMATCH",
    "audio_visual_sync": "AV_SYNC_MISMATCH",
    "provenance_ai_detected": "AI_PROVENANCE_TAG",
    "provenance_ai_detected": "AI_PROVENANCE_TAG",
    "c2pa_provenance": "C2PA_PROVENANCE",
    "exif_metadata": "GENERATOR_SIGNATURE",
    "provenance": "AI_PROVENANCE_TAG",
    # text-side signals
    "perplexity": "PERPLEXITY_ANOMALY",
    "detectgpt": "DETECTGPT_CURVATURE",
    "stylometry": "STYLOMETRY_ANOMALY",
    "pattern": "AI_PHRASE_PATTERN",
    "watermark": "WATERMARK_DETECTED",
    # flat text verdict keys
    "stylometry_signal": "STYLOMETRY_ANOMALY",
    "pattern_signal": "AI_PHRASE_PATTERN",
    "watermark_signal": "WATERMARK_DETECTED",
    "detectgpt_signal": "DETECTGPT_CURVATURE",
    "perplexity_signal": "PERPLEXITY_ANOMALY",
    # music / video extras (Phase 3)
    "AI_MUSIC": "AI_MUSIC",
    "T2V_GENERATED": "T2V_GENERATED",
}

# Signals whose presence means "this content was machine-generated".
# Includes each modality's dedicated generator-detector families:
#   image  - HF pipeline (model_signal_raw), global fingerprint, MSCA-FFT, FreqNet
#   audio  - AMFF / TDNN / AASIST2 anti-spoofing heads
#   text   - perplexity, DetectGPT, stylometry, phrasing, watermark
_GENERATION_SIGNALS = frozenset(
    {
        # cross-modality
        "FULLY_AI_GENERATED",
        "AI_GENERATED",
        "T2V_GENERATED",
        "AI_MUSIC",
        "AI_PHRASE_PATTERN",
        "STYLOMETRY_ANOMALY",
        "WATERMARK_DETECTED",
        "PERPLEXITY_ANOMALY",
        "DETECTGPT_CURVATURE",
        "GENERATOR_SIGNATURE",
        "AI_PROVENANCE_TAG",
        "C2PA_PROVENANCE",
        # image generator detectors
        "MODEL_SIGNAL_RAW",
        "GLOBAL_HEURISTIC",
        "V2_MSCA_FFT",
        "V2_FREQNET",
        # audio generator detectors (anti-spoofing heads)
        "V2_AMFF",
        "V2_TDNN",
        "V2_AASIST2",
    }
)

# Signals whose presence means "real content was altered".
#   image/video - face swap, splice/edit, AV desync, temporal inconsistency
#   video v2    - lip-sync and rPPG integrity both detect swapped faces
_MANIPULATION_SIGNALS = frozenset(
    {
        "FACE_SWAP",
        "SPLICE_EDIT",
        "AV_SYNC_MISMATCH",
        "TEMPORAL_INCONSISTENCY",
        "V2_LIP_SYNC",
        "V2_RPPG",
    }
)

# Evidence strings that are honest about which layer produced the verdict.
_AI_OPENERS = {
    KIND_ALTERED: "Altered (manipulated) content",
    KIND_GENERATED: "Synthetic / machine-generated content",
    KIND_BOTH: "Synthetic content - evidence of both generation and alteration",
    KIND_UNKNOWN: "Synthetic / machine-generated content",
}
_REAL_OPENER = "No strong evidence of synthesis or alteration"
_UNCERTAIN_OPENER = "Inconclusive - evidence is weak or conflicting"


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN
        return 0.0
    return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)


def _map_label(native: str) -> str:
    """Validate the upstream label vocabulary; map to the uniform one.

    Only used for ``details["upstream_label_uniform"]`` so Phase 0 can compare
    the upstream rule against the score-derived rule. Unknown labels RAISE:
    that is schema drift, not an input property to guess around.
    """
    if native in _LABEL_MAP:
        return _LABEL_MAP[native]
    raise InvalidInputError(
        f"Unrecognised upstream label {native!r}. Known: {sorted(_LABEL_MAP)}"
    )


def _kind_for(signals: Sequence[str], fired_families: Sequence[str]) -> str:
    """``generated`` / ``altered`` / ``both`` / ``unknown``.

    Per ALUDAM_PLAN.md sec 7.1 this is kept OUT of the verdict: ``label``
    answers accept / reject / abstain, ``kind`` answers *what kind* of
    synthetic content this is. Both families firing yields ``both``.

    Priority: the detector's own ``fake_type`` list (which families it says
    fired) over the numeric signal ranking. Using every numeric family score
    instead made image read as ``both`` almost always, because the global
    generator detectors fire alongside any manipulation.
    """
    if fired_families:
        names = {SIGNAL_NAMES.get(str(f), str(f).upper()) for f in fired_families}
    else:
        names = set(signals)
    generation = bool(names & _GENERATION_SIGNALS)
    alteration = bool(names & _MANIPULATION_SIGNALS)
    if generation and alteration:
        return KIND_BOTH
    if alteration:
        return KIND_ALTERED
    if generation:
        return KIND_GENERATED
    return KIND_UNKNOWN


def _opener_for(label: str, kind: str) -> str:
    if label == LABEL_AI:
        return _AI_OPENERS.get(kind, _AI_OPENERS[KIND_UNKNOWN])
    if label == LABEL_REAL:
        return _REAL_OPENER
    return _UNCERTAIN_OPENER


def _ranked(
    items: Iterable[Tuple[str, float]], threshold: float = SIGNAL_THRESHOLD
) -> List[Tuple[str, float]]:
    """Map raw keys -> public names, drop non-signals, sort by score desc.

    Values must be probabilities in [0, 1]. Counters (``segment_count: 3``) and
    other out-of-range numbers are dropped rather than reported as signals.
    """
    out: List[Tuple[str, float]] = []
    seen = set()
    for key, value in items:
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score != score:  # NaN
            continue
        if not 0.0 <= score <= 1.0:
            continue
        if key in NON_SIGNAL_KEYS:
            continue
        name = SIGNAL_NAMES.get(key, str(key).upper())
        if score < threshold or name in seen:
            continue
        seen.add(name)
        out.append((name, round(score, 4)))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


def _signal_names(ranked: Sequence[Tuple[str, float]]) -> Tuple[str, ...]:
    return tuple(name for name, _ in ranked)


def _signal_clause(ranked: Sequence[Tuple[str, float]], limit: int = 3) -> str:
    if not ranked:
        return "no signal crossed its firing threshold"
    picked = ranked[:limit]
    body = ", ".join(f"{n} {s:.2f}" for n, s in picked)
    if len(ranked) > limit:
        body += f" (+{len(ranked) - limit} more)"
    return body


def _clip(text: str, limit: int = EXPLANATION_CHAR_LIMIT) -> str:
    """Collapse whitespace and hard-truncate to ``limit`` ASCII-safe chars.

    Output must stay ASCII: ``explanation`` flows into JSON, logs and Windows
    consoles, all of which choke on characters outside the local codec.
    """
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    head = text[: limit - 3].rstrip()
    if not head:
        return text[:limit]
    return head + "..."


# ---------------------------------------------------------------------------
# per-modality extraction
# ---------------------------------------------------------------------------

def _extract_text(raw: Mapping[str, Any]) -> Dict[str, Any]:
    signals = _ranked(
        [
            ("perplexity", raw.get("perplexity_signal")),
            ("detectgpt", raw.get("detectgpt_signal")),
            ("stylometry", raw.get("stylometry_signal")),
            ("pattern", raw.get("pattern_signal")),
            ("watermark", raw.get("watermark_signal")),
        ]
    )
    metrics = raw.get("metrics") or {}
    words = metrics.get("word_count")
    evidence_bits = []
    if words is not None:
        evidence_bits.append(f"{words} words")
    pdetails = metrics.get("perplexity_details") or {}
    if "perplexity" in pdetails:
        evidence_bits.append(f"perplexity {pdetails['perplexity']}")
    if "burstiness" in pdetails:
        evidence_bits.append(f"burstiness {pdetails['burstiness']}")
    return {
        # text has no fake_type; kind falls back to the signal names.
        "fired_families": (),
        "signals": signals,
        "evidence": "; ".join(evidence_bits),
        "details": {
            "metrics": metrics,
            "suspicious_spans": raw.get("suspicious_spans") or [],
            "notes": raw.get("notes", ""),
        },
    }


def _extract_media(raw: Mapping[str, Any], media_type: str) -> Dict[str, Any]:
    """Shared path for image / audio / video (family-score based)."""
    family = dict(raw.get("family_scores") or {})
    items: List[Tuple[str, float]] = []

    # image/video expose a `fake_type` list of fired families
    for fired in (raw.get("fake_type") or []):
        key = str(fired)
        val = family.get(key)
        if val is None:
            val = 1.0  # family fired but didn't publish a numeric score
        items.append((key, val))

    # plus every numeric family score
    for key, val in family.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            items.append((str(key), float(val)))

    # audio has flat feature_summary instead of family_scores
    for key, val in (raw.get("feature_summary") or {}).items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            items.append((str(key), float(val)))

    # C2PA/EXIF provenance findings surface for image
    for finding in (raw.get("provenance_findings") or []):
        if isinstance(finding, Mapping):
            name = finding.get("name") or finding.get("signal")
            score = finding.get("score")
            if name is not None and score is not None:
                items.append((str(name), float(score)))

    signals = _ranked(items)

    score = _clamp01(raw.get("p_synthetic", raw.get("fake_probability", 0.5)))
    if media_type == "VIDEO":
        score = _clamp01(family.get("fused_score", score))

    evidence_bits = []
    if raw.get("faces_detected") is not None:
        evidence_bits.append(f"{raw['faces_detected']} face(s) detected")
    if raw.get("frames_analyzed") is not None:
        evidence_bits.append(f"{raw['frames_analyzed']} frames analysed")
    intervals = raw.get("suspicious_intervals") or []
    if intervals:
        evidence_bits.append(f"{len(intervals)} suspicious interval(s)")
    segments = raw.get("suspicious_segments") or []
    if segments:
        evidence_bits.append(f"{len(segments)} suspicious segment(s)")
    if media_type == "IMAGE":
        # ImageVerdict has no separate metadata block; provenance state lives
        # in family_scores["provenance"] (1.0 = AI-tagged, 0.0 = no/absent).
        prov = family.get("provenance")
        if isinstance(prov, (int, float)) and prov >= SIGNAL_THRESHOLD:
            evidence_bits.append("provenance marks AI-generated (C2PA/EXIF tag)")
        elif isinstance(prov, (int, float)):
            evidence_bits.append("no AI provenance tag found")
    if raw.get("signal_source"):
        evidence_bits.append(f"source: {raw['signal_source']}")

    details: Dict[str, Any] = {
        "family_scores": family,
        "notes": raw.get("notes", ""),
        "file_hash": raw.get("file_hash", ""),
    }
    if raw.get("fake_type") is not None:
        details["fired_families"] = list(raw["fake_type"])
    if raw.get("reasons"):
        details["reasons"] = raw["reasons"]
    if media_type == "IMAGE":
        details["faces_detected"] = raw.get("faces_detected")
        details["tier"] = raw.get("tier")
        details["tier_name"] = raw.get("tier_name")
    if media_type == "VIDEO":
        details["frames_analyzed"] = raw.get("frames_analyzed")
        details["suspicious_intervals"] = intervals
    if media_type == "AUDIO":
        details["signal_source"] = raw.get("signal_source")
        details["suspicious_segments"] = segments
        details["feature_summary"] = raw.get("feature_summary") or {}

    return {
        "score": score,
        "signals": signals,
        # The detector's authoritative list of which families it says fired.
        "fired_families": tuple(raw.get("fake_type") or ()),
        "evidence": "; ".join(evidence_bits),
        "details": details,
    }


# ---------------------------------------------------------------------------

def normalize(
    raw: Mapping[str, Any],
    *,
    detector: str,
    media_type: str,
    backend: str,
    elapsed_ms: int,
    degraded: bool = False,
    thresholds: "Thresholds" = None,  # type: ignore[assignment]
) -> DetectionResult:
    """Convert one upstream detector dict into a :class:`DetectionResult`.

    While ``thresholds`` is uncalibrated the reported ``label`` is the
    upstream detector's own verdict (legacy-faithful); the score-derived
    verdict is still computed and published under
    ``details["placeholder_label"]``. Once Phase 0 fits the band, ``label``
    becomes a pure function of ``score`` again. See
    :mod:`aludam.thresholds`.

    Raises:
        InvalidInputError: if ``raw`` carries an upstream ``error`` key, or an
            unrecognised upstream label (schema drift).
    """
    if thresholds is None:
        thresholds = thresholds_for(detector)

    if not isinstance(raw, Mapping):
        raise InvalidInputError(f"Detector returned {type(raw).__name__}, expected dict")

    if "error" in raw:
        raise InvalidInputError(str(raw["error"]))

    native_label = str(raw.get("label", "")).strip()
    if not native_label:
        raise InvalidInputError("Detector result has no 'label'")

    # Validates the upstream vocabulary (loud on schema drift) but does NOT
    # decide the reported label - see thresholds.decide().
    upstream_uniform = _map_label(native_label)

    if media_type == "TEXT":
        extracted = _extract_text(raw)
        score = _clamp01(raw.get("ai_probability", 0.5))
    else:
        extracted = _extract_media(raw, media_type)
        score = extracted["score"]

    signals: Sequence[Tuple[str, float]] = extracted["signals"]
    signal_names = _signal_names(signals)

    # The placeholder verdict is always computed - Phase 0 evaluates raw
    # scores with AUROC, and details must show what the band *would* say.
    placeholder_label, placeholder_reason = decide(score, thresholds)

    if thresholds.calibrated:
        # Fitted band: the one place a label is computed.
        label, label_reason = placeholder_label, placeholder_reason
        label_basis = "score"
    else:
        # Unfitted band must not decide. Report the upstream verdict instead,
        # which is what legacy actually does. May push toward 'uncertain',
        # never from fake to 'real'.
        label, label_reason = decide_uncalibrated(
            upstream_uniform, score, thresholds
        )
        label_basis = "upstream"

    # Interim margin, NOT a calibrated probability. See thresholds.margin().
    confidence = margin(score, thresholds)

    kind = _kind_for(signal_names, extracted.get("fired_families", ()))

    opener = _opener_for(label, kind)
    notes = str(extracted["details"].get("notes", "") or "").strip()
    parts = [f"{opener} (score {score:.2f})."]
    if label == LABEL_UNCERTAIN:
        # Both cases are honest: name the band when the score is in it, and
        # say the detector abstained when that is what happened.
        if thresholds.low < score < thresholds.high:
            parts.append(
                f"Score {score:.2f} falls inside the inconclusive band "
                f"({thresholds.low:.2f}-{thresholds.high:.2f})."
            )
        if label_reason == "upstream_abstained":
            parts.append("The upstream detector returned UNCERTAIN.")
    if not thresholds.calibrated:
        # Keep this terse: EXPLANATION_CHAR_LIMIT is 280, and the evidence
        # below must not be the thing that gets clipped.
        parts.append("Uncalibrated: verdict is the upstream detector's own.")
    parts.append(f"Signals: {_signal_clause(signals)}.")
    if extracted["evidence"]:
        parts.append(f"Evidence: {extracted['evidence']}.")
    if not signals:
        parts.append("No signal crossed its firing threshold.")
    if notes:
        parts.append(notes)

    details = dict(extracted["details"])
    details["upstream_label"] = native_label
    details["upstream_label_uniform"] = upstream_uniform
    details["upstream_confidence_0_100"] = raw.get("confidence")
    details["signal_scores"] = {n: s for n, s in signals}
    details["media_type"] = media_type
    details["thresholds"] = thresholds.to_dict()
    details["label_basis"] = label_basis
    details["label_reason"] = label_reason
    details["placeholder_label"] = placeholder_label
    details["placeholder_reason"] = placeholder_reason
    if raw.get("file_hash"):
        details["file_hash"] = raw["file_hash"]

    return DetectionResult(
        score=score,
        label=label,
        confidence=confidence,
        explanation=_clip(" ".join(parts)),
        kind=kind,
        signals=signal_names,
        detector=detector,
        media_type=media_type,
        runtime={
            "calibrated": thresholds.calibrated,
            "backend": backend,
            "degraded": degraded,
        },
        elapsed_ms=elapsed_ms,
        details=details,
    )
