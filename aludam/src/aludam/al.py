"""``al.detect()`` - the one-call front door.

    from aludam import al
    al.detect(type="img_det", src="./photo.png")

This is deliberately a thin wrapper over :func:`aludam.load_detector`, which
owns caching, model lifecycle and input validation. One source of truth: there
is no second code path that could drift from the lower-level API.

``type`` is one of the four detector names; ``src`` is text for ``text_det``
and a filesystem path for the other three.
"""

from __future__ import annotations

from typing import Any

from .registry import load_detector
from .result import DetectionResult

__all__ = ["detect"]


def detect(*, type: str, src: Any) -> DetectionResult:  # noqa: A002 - API name
    """Run one detector and return a :class:`~aludam.result.DetectionResult`.

    Args:
        type: ``"text_det"``, ``"img_det"``, ``"vdo_det"`` or ``"aud_det"``.
        src: text (for ``text_det``) or a filesystem path (for the others).

    Returns:
        The uniform :class:`~aludam.result.DetectionResult`.

    Raises:
        UnknownDetectorError: ``type`` is not one of the four.
        InvalidInputError: empty text, missing or empty file.
        AludamDepsMissing / AludamWeightsMissing: setup problems.
    """
    return load_detector(type).predict(src)
