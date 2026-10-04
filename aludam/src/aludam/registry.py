"""Detector registry - ``load_detector(name)``.

The public entry point::

    from aludam import load_detector

    det = load_detector("text_det")   # "img_det" | "vdo_det" | "aud_det"
    result = det.predict(text_or_path)

Detectors are imported lazily and wrapped once, so the expensive model
construction happens on first ``load_detector`` and is reused afterwards.

Models live in the repo's ``models/`` directory and are imported by flat module
name (``import forensics_core``, ``from final_image_detector import ...``), so
that directory must be on ``sys.path``. It is located by:

1. ``$ALUDAM_MODELS_DIR`` if set, else
2. walking up from this file looking for ``models/final_text_detector.py``.

No environment variables are mutated here. In particular ``HF_HUB_OFFLINE`` is
left alone - forcing it silently switched transformers to cache-only loading,
which is exactly the class of invisible degradation aludam must not repeat.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .exceptions import (
    AludamDepsMissing,
    AludamError,
    InvalidInputError,
    UnknownDetectorError,
)
from .normalize import normalize
from .result import DetectionResult
from . import weights

__all__ = [
    "load_detector",
    "available_detectors",
    "Detector",
    "DETECTOR_NAMES",
]

# Public detector name -> (module, singleton attribute, media type).
# Names are exactly the four specified in the API contract; aliases are
# deliberately not accepted so a typo fails loudly instead of guessing.
_SPECS: Dict[str, Tuple[str, str, str]] = {
    "text_det": ("final_text_detector", "final_text_detector", "TEXT"),
    "img_det": ("final_image_detector", "final_image_detector", "IMAGE"),
    "vdo_det": ("final_video_detector", "final_video_detector", "VIDEO"),
    "aud_det": ("final_audio_detector", "final_audio_detector", "AUDIO"),
}

DETECTOR_NAMES: Tuple[str, ...] = tuple(_SPECS)

# Images at or below this on either axis skip the classifier entirely
# (ALUDAM_PLAN.md sec 7.1). Below it the model sits at chance, so its score
# would be noise presented as a verdict.
MIN_IMAGE_SIDE = 128

# Importable module name -> the pip package that provides it.
_IMPORTABLE_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "soundfile": "soundfile",
    "fitz": "pymupdf",
    "docx": "python-docx",
    "pypdf": "pypdf",
}

# heaviest first, so the first reported missing package is the one to install
_NEURAL_PIP_ORDER = (
    "torch",
    "transformers",
    "timm",
    "peft",
    "librosa",
    "onnxruntime",
    "scipy",
)

_models_dir_guard = threading.Lock()
_wrappers_guard = threading.Lock()
_wrappers: Dict[str, "Detector"] = {}
_models_dir: Optional[str] = None


def available_detectors() -> Tuple[str, ...]:
    """The four detector names accepted by :func:`load_detector`."""
    return DETECTOR_NAMES


# ---------------------------------------------------------------------------

def _candidate_repo_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` until a directory containing ``models/`` shows up."""
    for parent in (start, *start.parents):
        probe = parent / "models" / "final_text_detector.py"
        if probe.is_file():
            return parent
    return None


def resolve_models_dir() -> str:
    """Return the directory holding the four ``final_*_detector`` modules.

    Cached after the first successful resolution.

    Raises:
        AludamError: if the directory cannot be found.
    """
    global _models_dir
    if _models_dir:
        return _models_dir

    with _models_dir_guard:
        if _models_dir:
            return _models_dir

        found: Optional[str] = None

        env_dir = os.environ.get("ALUDAM_MODELS_DIR", "").strip()
        if env_dir:
            candidate = Path(env_dir).expanduser()
            if (candidate / "final_text_detector.py").is_file():
                found = str(candidate.resolve())
            else:
                raise AludamError(
                    f"ALUDAM_MODELS_DIR={env_dir!r} does not contain "
                    f"final_text_detector.py. Set it to the directory that holds "
                    f"final_text_detector.py, final_image_detector.py, "
                    f"final_audio_detector.py, final_video_detector.py."
                )

        if found is None:
            root = _candidate_repo_root(Path(__file__).resolve())
            if root is not None:
                found = str(root / "models")

        if found is None:
            raise AludamError(
                "Could not locate the detector models directory.\n"
                "  Fix: set ALUDAM_MODELS_DIR to the directory containing "
                "final_text_detector.py."
            )

        _models_dir = found
        return found


def _ensure_models_on_path() -> str:
    """Prepend the models directory to ``sys.path`` and return it."""
    models_dir = resolve_models_dir()
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    return models_dir


def _missing_packages(exc: ModuleNotFoundError) -> list[str]:
    """Turn a ``ModuleNotFoundError`` into the pip packages that would fix it."""
    missing = exc.name or ""
    # A missing submodule of a present package still means the package is there;
    # report the top-level component.
    top = missing.split(".")[0] if missing else ""
    if not top:
        return ["<unknown>"]
    pip_name = _IMPORTABLE_TO_PIP.get(top, top)
    ordered = [p for p in _NEURAL_PIP_ORDER if p == pip_name]
    ordered += [pip_name] if pip_name not in ordered else []
    return ordered


# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detector:
    """A single modality's detector behind the uniform ``predict`` contract.

    Attributes:
        name: one of the four public names (``text_det`` ...).
        media_type: ``TEXT`` / ``IMAGE`` / ``VIDEO`` / ``AUDIO``.
        backend: ``"neural"`` - aludam has no light-tier detector today; the
            light tier was scoped out of Phase 0 (ALUDAM_PLAN.md sec 5.2).
    """

    name: str
    media_type: str
    backend: str = "neural"
    _predict_raw: Any = None

    def predict(self, inp: Any) -> DetectionResult:
        """Run detection on ``inp`` and return a :class:`DetectionResult`.

        Args:
            inp: ``str`` for ``text_det``; a filesystem path (``str`` or
                ``os.PathLike``) for the other three.

        Returns:
            The uniform :class:`DetectionResult`.

        Raises:
            InvalidInputError: empty text, missing path, or empty file.
            AludamDepsMissing / AludamWeightsMissing: setup problems.
        """
        cleaned = self._validate(inp)

        # Quality gate first: below the model's stated minimum the detector is
        # at chance, so running it would produce a confident-looking number
        # with no information behind it. See ALUDAM_PLAN.md sec 7.1.
        guarded = self._guard(cleaned)
        if guarded is not None:
            return guarded

        started = time.perf_counter()
        raw = self._predict_raw(cleaned)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return normalize(
            raw,
            detector=self.name,
            media_type=self.media_type,
            backend=self.backend,
            elapsed_ms=elapsed_ms,
        )

    # -- input handling ------------------------------------------------

    def _validate(self, inp: Any) -> Any:
        if self.media_type == "TEXT":
            if not isinstance(inp, str):
                raise InvalidInputError(
                    f"{self.name} expects a str, got {type(inp).__name__}"
                )
            if not inp.strip():
                raise InvalidInputError(f"{self.name} received empty text")
            return inp

        if isinstance(inp, bytes):
            raise InvalidInputError(
                f"{self.name} expects a filesystem path, not raw bytes"
            )
        if not isinstance(inp, (str, os.PathLike)):
            raise InvalidInputError(
                f"{self.name} expects a path (str or os.PathLike), "
                f"got {type(inp).__name__}"
            )
        path = os.fspath(inp)
        if not path.strip():
            raise InvalidInputError(f"{self.name} received an empty path")
        if not os.path.isfile(path):
            raise InvalidInputError(f"Input file not found: {path}")
        if os.path.getsize(path) == 0:
            raise InvalidInputError(f"Input file is empty (0 bytes): {path}")
        return path

    def _guard(self, path: str) -> Optional[DetectionResult]:
        """Return a quality-gate result, or ``None`` to run the detector.

        Images smaller than ``MIN_IMAGE_SIDE`` on either axis are returned as
        ``uncertain`` with ``details["reason"] = "input_below_model_minimum"``
        and the detector is never invoked. The model card is explicit that
        below 128x128 the classifier sits at ~chance, so its output there is
        noise dressed as a verdict - the 9% false-positive figure that
        motivated this guard comes from exactly that regime.

        The score is 0.5, the maximally uninformative value: this is not a
        measurement, and pretending to one would be worse than saying so.
        """
        if self.media_type != "IMAGE":
            return None

        w = h = 0
        try:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001 - unreadable/not an image: not our gate
            return None

        if w >= MIN_IMAGE_SIDE and h >= MIN_IMAGE_SIDE:
            return None

        started = time.perf_counter()
        raw = {
            "label": "UNCERTAIN",
            "ai_probability": 0.5,
            "confidence": 0.0,
            "notes": (
                f"Image is {w}x{h} px, below the {MIN_IMAGE_SIDE}x"
                f"{MIN_IMAGE_SIDE} px model minimum, so the detector was "
                f"skipped rather than asked for a score it cannot support."
            ),
        }
        result = normalize(
            raw,
            detector=self.name,
            media_type=self.media_type,
            backend=self.backend,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        # DetectionResult is frozen, but `details` is a dict - mutating it is
        # how the gate reason reaches the caller without a second code path
        # through normalize().
        result.details["reason"] = "input_below_model_minimum"
        result.details["image_size"] = [w, h]
        result.details["min_image_side"] = MIN_IMAGE_SIDE
        return result

    def __repr__(self) -> str:
        return f"Detector(name={self.name!r}, media_type={self.media_type!r})"


# ---------------------------------------------------------------------------

def _build(name: str) -> Detector:
    module_name, attr_name, media_type = _SPECS[name]

    # Verify weights BEFORE importing the detector module. The upstream
    # modules record their own load failures and keep going
    # (final_image_detector logs "will rely on forensic heuristics only",
    # final_audio_detector falls through to _heuristic_predict), so once the
    # module is imported the silent degradation has already happened and
    # there is nothing left to raise about. See ALUDAM_PLAN.md sec 6.3.
    weights.ensure(name)

    _ensure_models_on_path()

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise AludamDepsMissing(_missing_packages(exc), name) from exc
    except Exception as exc:  # noqa: BLE001 - surface construction failures verbatim
        raise AludamError(
            f"Failed to import detector {name!r} from module {module_name!r}: {exc}"
        ) from exc

    instance = getattr(module, attr_name, None)
    if instance is None:
        raise AludamError(
            f"Module {module_name!r} has no attribute {attr_name!r}; "
            f"expected a pre-built detector singleton."
        )

    return Detector(
        name=name,
        media_type=media_type,
        backend="neural",
        _predict_raw=instance.predict,
    )


def load_detector(name: str) -> Detector:
    """Load a detector by name.

    Args:
        name: ``"text_det"``, ``"img_det"``, ``"vdo_det"`` or ``"aud_det"``.

    Returns:
        The cached :class:`Detector` for that modality.

    Raises:
        UnknownDetectorError: ``name`` is not one of the four.
        AludamDepsMissing: a Python package the detector needs is absent.
        AludamWeightsMissing: the detector's weights are not downloaded yet.
        AludamError: the models directory cannot be located, or import failed.
    """
    if not isinstance(name, str):
        raise UnknownDetectorError(
            f"Detector name must be a str, got {type(name).__name__}. "
            f"Valid names: {', '.join(DETECTOR_NAMES)}"
        )

    key = name.strip()
    if key not in _SPECS:
        raise UnknownDetectorError(
            f"Unknown detector {name!r}. Valid names: {', '.join(DETECTOR_NAMES)}"
        )

    cached = _wrappers.get(key)
    if cached is not None:
        return cached

    with _wrappers_guard:
        cached = _wrappers.get(key)
        if cached is not None:
            return cached
        built = _build(key)
        _wrappers[key] = built
        return built


def reset_cache() -> None:
    """Drop cached detector wrappers (used by tests and ``aludam`` CLI)."""
    with _wrappers_guard:
        _wrappers.clear()


def raw_result(detector: Detector) -> Mapping[str, Any]:  # pragma: no cover
    """Escape hatch for Phase 0 measurement: return the un-normalised dict."""
    raise NotImplementedError(
        "Call the underlying detector instance directly for raw output."
    )
