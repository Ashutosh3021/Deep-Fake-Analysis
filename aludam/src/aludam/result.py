"""``DetectionResult`` - the single, uniform result type.

Every detector returns this same class, for every modality, so callers can
write one line of formatting and have it work for text, images, video and
audio alike::

    from aludam import load_detector

    det = load_detector("text_det")
    result = det.predict("my name is ...")
    print(result)
    # DetectionResult(score=0.8012, label='ai', confidence=0.3125,
    #                 explanation="High pattern-signal ...")

Field scales
------------
``score`` and ``confidence`` are BOTH 0.0 .. 1.0.

The single decision statistic
-----------------------------
``score`` is the decision statistic, and ``label`` is a **pure function of
score**::

    label = "real"      if score <= thresholds.low
          = "uncertain" if low < score < thresholds.high
          = "ai"        if score >= thresholds.high

So ``r.score > 0.7`` can never contradict ``r.label``. The thresholds live in
:mod:`aludam.thresholds`; the ones shipped are explicit **placeholders**
(``runtime["calibrated"] is False``) until Phase 0 fits them on a held-out
calibration split.

The upstream detectors' own verdicts used a *different* rule (any detection
family crossing its own threshold wins), so they can disagree with ``score``.
They are preserved verbatim in ``details["upstream_label"]`` and are never
used to decide ``label``. Which scoring rule should feed ``score`` is decided
in Phase 0 by AUROC; this class does not care.

``confidence`` - interim definition
-----------------------------------
``confidence`` is currently a **margin**: how far ``score`` sits outside the
abstain band. It is NOT a calibrated probability that the verdict is correct.
Check ``runtime["calibrated"]`` - it ships ``False``.

It is not the upstream value. The four detectors emit ``confidence`` on a
0-100 scale, but for three of them that number is a deterministic function of
their own score (video: ``0.5 + 0.5 * score``; text and audio:
``0.5 + |score - 0.5|`` times a length penalty), i.e. decorative. The
unmodified upstream value is kept in ``details["native_confidence"]``.

Label vs kind
-------------
``label`` answers *what to do* (accept / reject / abstain).
``kind`` answers *what kind of synthetic content* it is - keeping
``generated`` and ``altered`` distinct without overloading the verdict, per
ALUDAM_PLAN.md sec 7.1:

- ``generated`` - machine-authored from scratch
- ``altered``   - real content that was modified (face swap, splice, desync)
- ``both``      - evidence for each side
- ``unknown``   - no usable kind evidence

Vocabulary
----------
``label``: ``ai`` / ``real`` / ``uncertain``. ``kind``: ``generated`` /
``altered`` / ``both`` / ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "DetectionResult",
    "LABEL_AI",
    "LABEL_REAL",
    "LABEL_UNCERTAIN",
    "KIND_GENERATED",
    "KIND_ALTERED",
    "KIND_BOTH",
    "KIND_UNKNOWN",
    "VALID_LABELS",
    "VALID_KINDS",
]

LABEL_AI = "ai"
LABEL_REAL = "real"
LABEL_UNCERTAIN = "uncertain"

KIND_GENERATED = "generated"
KIND_ALTERED = "altered"
KIND_BOTH = "both"
KIND_UNKNOWN = "unknown"

VALID_LABELS = frozenset({LABEL_AI, LABEL_REAL, LABEL_UNCERTAIN})
VALID_KINDS = frozenset({KIND_GENERATED, KIND_ALTERED, KIND_BOTH, KIND_UNKNOWN})

_REPR_EXPLANATION_LIMIT = 96

_DEFAULT_RUNTIME: Dict[str, Any] = {
    "calibrated": False,
    "backend": "light",
    "degraded": False,
}


@dataclass(frozen=True)
class DetectionResult:
    """A scored, explained verdict. Identical shape for every modality.

    Attributes:
        score: 0.0..1.0 - the decision statistic. 0.0 reads authentic,
            1.0 reads synthetic. NOT a confidence.
        label: ``"ai"``, ``"real"`` or ``"uncertain"`` - derived from ``score``
            via :mod:`aludam.thresholds`. Always consistent with ``score``.
        confidence: 0.0..1.0 - interim **margin** (distance from the abstain
            band), NOT calibrated. See ``runtime["calibrated"]``.
        explanation: human-readable justification built from the signals that
            actually fired, plus concrete evidence (frame counts, face counts,
            metadata state). Not a template.
        kind: ``"generated"`` / ``"altered"`` / ``"both"`` / ``"unknown"`` -
            the *type* of synthetic content, independent of the verdict.
        signals: names of the signals that fired, highest-scoring first.
        detector: which detector produced this (``text_det`` ...).
        media_type: ``TEXT`` / ``IMAGE`` / ``VIDEO`` / ``AUDIO``.
        runtime: how this result was produced.

            - ``calibrated``: bool - are thresholds/confidence fitted? Ships
              ``False``.
            - ``backend``: ``"neural"`` or ``"light"``.
            - ``degraded``: True only when a model LOADED and then failed
              mid-inference. Missing deps/weights RAISE instead - they never
              set this flag.

        elapsed_ms: wall-clock milliseconds for this prediction.
        details: modality-specific raw evidence (family scores, metadata,
            ``upstream_label``, native confidence, thresholds used, ...).
    """

    score: float
    label: str
    confidence: float
    explanation: str
    kind: str = KIND_UNKNOWN
    signals: Tuple[str, ...] = ()
    detector: str = ""
    media_type: str = ""
    runtime: Mapping[str, Any] = field(default_factory=lambda: dict(_DEFAULT_RUNTIME))
    elapsed_ms: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate BEFORE rounding. Clamping an out-of-range value would hide a
        # caller bug (most likely an un-normalised 0-100 confidence) behind a
        # confident-looking number - the failure mode this package exists to
        # remove.
        for name in ("score", "confidence"):
            raw = getattr(self, name)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a number, got {raw!r}") from exc
            if value != value:
                raise ValueError(f"{name} is NaN")
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {value!r}."
                )
            object.__setattr__(self, name, round(value, 4))

        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "details", dict(self.details))

        runtime = dict(_DEFAULT_RUNTIME)
        runtime.update(self.runtime)
        object.__setattr__(self, "runtime", runtime)

        if self.label not in VALID_LABELS:
            raise ValueError(
                f"label must be one of {sorted(VALID_LABELS)}, got {self.label!r}"
            )
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(VALID_KINDS)}, got {self.kind!r}"
            )
        if not isinstance(self.explanation, str):
            raise TypeError("explanation must be a str")
        if not isinstance(self.runtime.get("calibrated"), bool):
            raise TypeError("runtime['calibrated'] must be a bool")

    # -- convenience --------------------------------------------------

    @property
    def is_ai(self) -> bool:
        """True only for a positive ``ai`` verdict."""
        return self.label == LABEL_AI

    @property
    def is_uncertain(self) -> bool:
        return self.label == LABEL_UNCERTAIN

    @property
    def calibrated(self) -> bool:
        """Are the thresholds/confidence fitted? Ships ``False``."""
        return bool(self.runtime.get("calibrated", False))

    @property
    def backend(self) -> str:
        return str(self.runtime.get("backend", "light"))

    @property
    def degraded(self) -> bool:
        return bool(self.runtime.get("degraded", False))

    @property
    def upstream_label(self) -> str:
        """The upstream detector's own verdict (``FAKE``, ``SYNTHETIC``, ...).

        Kept for comparison only; it does not decide ``label``.
        """
        return str(
            self.details.get("upstream_label") or self.details.get("native_label", "")
        )

    # Backwards-compatible aliases for the earlier names.
    native_label = upstream_label

    @property
    def thresholds(self) -> Mapping[str, Any]:
        return dict(self.details.get("thresholds") or {})

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view. ``details`` is inlined, not nested twice."""
        out: Dict[str, Any] = {
            "score": self.score,
            "label": self.label,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "kind": self.kind,
            "signals": list(self.signals),
            "detector": self.detector,
            "media_type": self.media_type,
            "runtime": dict(self.runtime),
            "elapsed_ms": self.elapsed_ms,
        }
        out.update(self.details)
        return out

    def __repr__(self) -> str:
        expl = self.explanation
        if len(expl) > _REPR_EXPLANATION_LIMIT:
            expl = expl[: _REPR_EXPLANATION_LIMIT - 3].rstrip() + "..."
        return (
            f"{type(self).__name__}(score={self.score!r}, label={self.label!r}, "
            f"confidence={self.confidence!r}, explanation={expl!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
