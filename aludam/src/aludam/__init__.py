"""aludam - uniform deepfake / AI-content detection.

One entry point, four modalities, one result type::

    from aludam import load_detector

    det1 = load_detector("text_det");  r1 = det1.predict(text)
    det2 = load_detector("img_det");   r2 = det2.predict(image_path)
    det3 = load_detector("vdo_det");   r3 = det3.predict(video_path)
    det4 = load_detector("aud_det");   r4 = det4.predict(audio_path)

    print(r1)   # DetectionResult(score=0.8012, label='ai', confidence=0.3125,
    print(r2)   #                 explanation="...")   # same shape, every time

Or, if you prefer one call::

    from aludam import al
    al.detect(type="img_det", src="./photo.png")

Guarantees
----------
* Every detector returns the **same** :class:`~aludam.result.DetectionResult`
  class, so one line of formatting works for all modalities.
* ``score`` and ``confidence`` are both 0.0..1.0.
* ``label`` is a **pure function of ``score``** (``verdict = f(score)``), so
  ``r.score > 0.7`` can never contradict ``r.label``.
* ``label`` is always ``"ai"``, ``"real"`` or ``"uncertain"``; ``kind`` carries
  the generated-vs-altered distinction separately.
* Missing dependencies and missing weights **raise**
  (:class:`~aludam.exceptions.AludamDepsMissing`,
  :class:`~aludam.exceptions.AludamWeightsMissing`). They never degrade into a
  heuristic-looking number.
* Thresholds ship **uncalibrated**. Check ``result.runtime["calibrated"]``
  (``False`` until Phase 0 fits them on held-out labels).
"""

from __future__ import annotations

from . import al
from .exceptions import (
    AludamDepsMissing,
    AludamError,
    AludamWeightsMissing,
    InvalidInputError,
    UnknownDetectorError,
)
from .registry import DETECTOR_NAMES, Detector, available_detectors, load_detector
from .result import (
    KIND_ALTERED,
    KIND_BOTH,
    KIND_GENERATED,
    KIND_UNKNOWN,
    LABEL_AI,
    LABEL_REAL,
    LABEL_UNCERTAIN,
    VALID_KINDS,
    VALID_LABELS,
    DetectionResult,
)
from .thresholds import PLACEHOLDER, Thresholds
from .weights import fetch

__version__ = "1.0.0"

__all__ = [
    # primary API
    "load_detector",
    "al",
    "available_detectors",
    "DETECTOR_NAMES",
    "Detector",
    "DetectionResult",
    # weights (Must-Fix #4: explicit fetch, never shipped in the wheel)
    "fetch",
    # vocabularies
    "LABEL_AI",
    "LABEL_REAL",
    "LABEL_UNCERTAIN",
    "VALID_LABELS",
    "KIND_GENERATED",
    "KIND_ALTERED",
    "KIND_BOTH",
    "KIND_UNKNOWN",
    "VALID_KINDS",
    # thresholds
    "Thresholds",
    "PLACEHOLDER",
    # errors
    "AludamError",
    "AludamDepsMissing",
    "AludamWeightsMissing",
    "InvalidInputError",
    "UnknownDetectorError",
    "__version__",
]
