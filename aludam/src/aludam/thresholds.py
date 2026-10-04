"""Decision thresholds - placeholder until Phase 0 fits real ones.

Two invariants live here, and which one applies depends on calibration:

**Calibrated (Phase 0 has fitted the band):**

    ``verdict = f(score)``

so a caller doing ``if r.score > 0.7`` can never contradict ``r.label``.

**Uncalibrated (the shipped state):**

the band is an unfitted placeholder, and an unfitted number must not decide
anything. The verdict is therefore the *upstream detector's own label* - what
legacy actually reports - while the score-derived verdict is still computed
and reported under ``details["placeholder_label"]`` for comparison.

That temporarily reintroduces a ``score`` / ``label`` mismatch. It is
deliberate and bounded:

* it is flagged by ``runtime["calibrated"] = False``;
* ``decide_uncalibrated`` may only push toward ``uncertain`` (INCONCLUSIVE),
  never flip a fake to ``real``;
* it ends the moment Phase 0 fits the band, at which point ``label`` becomes a
  pure function of ``score`` again.

A placeholder that labelled upstream-FAKE input ``real`` is exactly the harm
this avoids - see ALUDAM_PLAN.md sec 0.1 finding 12.

Calibration status
------------------
``calibrated = False`` on the shipped defaults. Nothing here has been fitted to
data, so these are explicit placeholders - see ALUDAM_PLAN.md sec 8 Phase 0:
Phase 2 builds the INCONCLUSIVE mechanism with placeholder thresholds flagged
``calibrated: false``; Phase 0 fits them on the calibration split.

Splits (dev / calibration / test) are defined by the eval harness, not here.
The test split must not be touched until the final gate.

Publishing gate
---------------
Phase 4 must not run while any modality is still uncalibrated;
:func:`publish_blocked` is that check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

__all__ = [
    "Thresholds",
    "PLACEHOLDER",
    "decide",
    "decide_uncalibrated",
    "margin",
    "calibrated_thresholds",
    "register_thresholds",
    "thresholds_for",
    "uncalibrated_detectors",
    "publish_blocked",
    "artifact_path",
    "load_artifact",
]

# Label vocabulary (re-exported for convenience).
LABEL_AI = "ai"
LABEL_REAL = "real"
LABEL_UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Thresholds:
    """Score decision bands.

    ``score <= low`` -> ``real``
    ``score >= high`` -> ``ai``
    ``low < score < high`` -> ``uncertain`` (the INCONCLUSIVE band)

    Attributes:
        low: lower edge of the abstain band.
        high: upper edge of the abstain band.
        calibrated: True only after fitting on real labels. Shipped default
            is False; see module docstring.
        source: where these numbers came from (provenance for reports).
    """

    low: float = 0.45
    high: float = 0.55
    calibrated: bool = False
    source: str = "placeholder defaults (NOT fitted; Phase 0 fits these)"

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= 1.0:
            raise ValueError(f"low must be in [0,1], got {self.low}")
        if not 0.0 <= self.high <= 1.0:
            raise ValueError(f"high must be in [0,1], got {self.high}")
        if self.low >= self.high:
            raise ValueError(
                f"low must be strictly below high, got low={self.low} high={self.high}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "calibrated": self.calibrated,
            "source": self.source,
        }


# The shipped default. Phase 0 replaces this via `calibrated_thresholds()`.
PLACEHOLDER = Thresholds()


def decide(score: float, thresholds: Thresholds = PLACEHOLDER) -> Tuple[str, str]:
    """Map a score to ``(label, reason)``.

    The returned label is a pure function of ``score`` and ``thresholds`` -
    this is the only place a score-derived label is computed.

    Only the *result* may be reported while ``thresholds.calibrated`` is
    False; the verdict itself then comes from :func:`decide_uncalibrated`.
    """
    if thresholds.low < score < thresholds.high:
        return LABEL_UNCERTAIN, "in_inconclusive_band"
    if score >= thresholds.high:
        return LABEL_AI, "at_or_above_high_threshold"
    return LABEL_REAL, "at_or_below_low_threshold"


def decide_uncalibrated(
    upstream_label: str,
    score: float,
    thresholds: Thresholds = PLACEHOLDER,
) -> Tuple[str, str]:
    """Verdict while the score band is an unfitted placeholder.

    ``upstream_label`` must already be in the uniform vocabulary
    (``ai`` / ``real`` / ``uncertain``).

    Rules, in order:

    1. ``uncertain`` anywhere - either the detector abstained or the score
       landed in the placeholder's abstain band - yields ``uncertain``. A
       placeholder is allowed to push *toward* INCONCLUSIVE.
    2. Otherwise the upstream verdict stands, so an uncalibrated placeholder
       can never flip a fake to ``real``.
    """
    if upstream_label not in (LABEL_AI, LABEL_REAL, LABEL_UNCERTAIN):
        raise ValueError(
            f"upstream_label must be uniform "
            f"({LABEL_AI}|{LABEL_REAL}|{LABEL_UNCERTAIN}), got {upstream_label!r}"
        )

    placeholder_label, _ = decide(score, thresholds)

    if upstream_label == LABEL_UNCERTAIN or placeholder_label == LABEL_UNCERTAIN:
        if upstream_label == LABEL_UNCERTAIN:
            return LABEL_UNCERTAIN, "upstream_abstained"
        return LABEL_UNCERTAIN, "placeholder_band_abstention"
    if upstream_label == LABEL_AI:
        # Reached only when placeholder_label is 'real' or 'ai'; both safe.
        return LABEL_AI, "upstream_verdict (thresholds uncalibrated)"
    return LABEL_REAL, "upstream_verdict (thresholds uncalibrated)"


def margin(score: float, thresholds: Thresholds = PLACEHOLDER) -> float:
    """How far ``score`` sits outside the abstain band, in [0, 1].

    0.0 means "inside the band" (no decision made), 1.0 means as far from a
    decision boundary as the score range allows.

    This is an *interim* stand-in for confidence. It is a margin, not a
    calibrated probability that the verdict is correct - that is fitted on the
    Phase 0 calibration split and gated by a reliability check. Flagging it as
    such is why ``runtime["calibrated"]`` ships ``False``.
    """
    if thresholds.low < score < thresholds.high:
        return 0.0
    if score >= thresholds.high:
        span = 1.0 - thresholds.high
        if span <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (score - thresholds.high) / span))
    span = thresholds.low
    if span <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (thresholds.low - score) / span))


def calibrated_thresholds(low: float, high: float, source: str) -> Thresholds:
    """Build a Thresholds instance that Phase 0 marks as fitted."""
    return replace(
        PLACEHOLDER, low=low, high=high, calibrated=True, source=source
    )


# Per-detector fitted bands. Empty until Phase 0 fits them, so every detector
# resolves to PLACEHOLDER and reports calibrated=False.
_FITTED: Dict[str, Thresholds] = {}

# Whether the shipped artifact has been looked for. Distinguished from "found"
# so a missing file is only stat'd once instead of on every thresholds_for().
_ARTIFACT_ATTEMPTED = False


def register_thresholds(detector: str, thresholds: Thresholds) -> None:
    """Attach a fitted band to a detector.

    Raises if the band is not marked calibrated - registering a placeholder
    under a fitted name is precisely the mistake this whole module guards.
    """
    if not thresholds.calibrated:
        raise ValueError(
            f"refusing to register uncalibrated thresholds for {detector!r}: "
            f"{thresholds.source}"
        )
    if not detector or not isinstance(detector, str):
        raise ValueError(f"detector must be a non-empty name, got {detector!r}")
    _FITTED[detector] = thresholds


def thresholds_for(detector: str) -> Thresholds:
    """Fitted band for ``detector``, or PLACEHOLDER when Phase 0 has not run.

    The shipped artifact (if present) is loaded on first use, so an installed
    ``aludam`` gets calibrated bands without the eval tree. A missing or
    malformed artifact is not an error - it just means nothing is fitted yet.
    """
    if detector not in _FITTED and not _ARTIFACT_ATTEMPTED:
        load_artifact()
    return _FITTED.get(detector, PLACEHOLDER)


def uncalibrated_detectors(detectors: Iterable[str]) -> Tuple[str, ...]:
    """Detectors still running placeholder thresholds, in input order."""
    return tuple(d for d in detectors if not thresholds_for(d).calibrated)


def publish_blocked(detectors: Iterable[str]) -> Tuple[bool, Tuple[str, ...]]:
    """Phase 4 gate: ``(blocked, offenders)``.

    Publishing while any modality reports ``calibrated: false`` would ship
    verdicts from numbers nobody fitted. Blocked until every modality is
    calibrated.
    """
    offenders = uncalibrated_detectors(detectors)
    return (bool(offenders), offenders)


# --------------------------------------------------------------------------
# Shipped artifact
# --------------------------------------------------------------------------
# Per-detector reasons an artifact entry was refused, so a report can say
# "these are uncalibrated because the fitted band failed its own target"
# rather than just "not fitted".
ARTIFACT_REJECTED: Dict[str, str] = {}


def artifact_path() -> Path:
    """Where eval/fit.py writes fitted bands.

    Lives inside the package so an installed wheel carries fitted thresholds
    with it; the eval tree is not installed and must not be a runtime
    dependency.
    """
    return Path(__file__).resolve().parent / "data" / "thresholds.json"


def load_artifact(path: Optional[Path] = None) -> Dict[str, Thresholds]:
    """Load fitted bands from the artifact, registering the usable ones.

    An entry is refused (left on PLACEHOLDER, so the modality stays blocked
    for publishing) when:

    * the band is structurally invalid - inverted or out of range, which
      :class:`Thresholds` rejects anyway; or
    * the artifact records ``satisfies: false`` - the band does not meet the
      FPR/FNR target it was fitted for, on the very data it was fitted on.
      Registering that as calibrated would flip ``publish_blocked`` to allow
      a release nobody can defend.

    Malformed or absent files are not errors: they mean Phase 0 has not run.
    """
    global _ARTIFACT_ATTEMPTED
    _ARTIFACT_ATTEMPTED = True

    p = Path(path) if path else artifact_path()
    if not p.exists():
        return {}

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt artifact must not take the detectors down with it.
        return {}
    if not isinstance(raw, dict):
        return {}

    loaded: Dict[str, Thresholds] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            ARTIFACT_REJECTED[name] = "artifact entry is not an object"
            continue

        if entry.get("satisfies") is False:
            ARTIFACT_REJECTED[name] = (
                "fitted band does not meet its target on calibration "
                "(violated: %s)" % entry.get("violated")
            )
            continue

        try:
            band = calibrated_thresholds(
                low=float(entry["low"]),
                high=float(entry["high"]),
                source=str(entry.get("source") or p),
            )
        except KeyError:
            ARTIFACT_REJECTED[name] = "artifact entry has no low/high"
            continue
        except Exception as exc:
            ARTIFACT_REJECTED[name] = f"{type(exc).__name__}: {exc}"
            continue

        register_thresholds(name, band)
        loaded[name] = band
    return loaded
