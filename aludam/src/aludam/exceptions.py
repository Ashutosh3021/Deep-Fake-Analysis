"""aludam exception hierarchy.

Two failure classes are deliberately distinct, per ALUDAM_PLAN.md sec 6.3:

``AludamDepsMissing``
    A Python package the requested detector needs is not installed.
    Fix: ``pip install aludam-fullplate``.

``AludamWeightsMissing``
    A detector's trained weights are not present in the local cache.
    Fix: ``aludam fetch``.

Both RAISE. Neither degrades into a heuristic score. A missing dependency or
missing weight is a setup error, not a signal about the input - returning a
confident-looking number in that situation is the exact failure mode that made
accuracy collapse invisibly in the first place (root cause #2/#3).
"""

from __future__ import annotations

__all__ = [
    "AludamError",
    "AludamDepsMissing",
    "AludamWeightsMissing",
    "UnknownDetectorError",
    "InvalidInputError",
]


class AludamError(Exception):
    """Base class for every aludam error."""


class AludamDepsMissing(AludamError):
    """A required Python package is not installed.

    Raised instead of silently falling back to a heuristic-only path.
    """

    def __init__(self, missing: list[str], detector: str) -> None:
        self.missing = sorted(missing)
        self.detector = detector
        pkgs = ", ".join(self.missing)
        super().__init__(
            f"Detector {detector!r} needs packages that are not installed: {pkgs}.\n"
            f"  Fix: pip install 'aludam[neural]'   (or: pip install aludam-fullplate)"
        )


class AludamWeightsMissing(AludamError):
    """A detector's trained weights are absent from the local cache.

    Raised instead of ``runtime.degraded = True``. See ALUDAM_PLAN.md sec 6.3.
    """

    def __init__(self, detector: str, weight_id: str, size_hint: str = "") -> None:
        self.detector = detector
        self.weight_id = weight_id
        size = f" (~{size_hint})" if size_hint else ""
        super().__init__(
            f"Weights for detector {detector!r} are not downloaded{size}.\n"
            f"  Weight id: {weight_id}\n"
            f"  Fix: aludam fetch {detector}\n"
            f"       (or python -c \"import aludam; aludam.fetch({detector!r})\")"
        )


class UnknownDetectorError(AludamError, ValueError):
    """``load_detector()`` was called with an unrecognised detector name."""


class InvalidInputError(AludamError, ValueError):
    """The input passed to ``predict()`` is empty or otherwise unusable."""
