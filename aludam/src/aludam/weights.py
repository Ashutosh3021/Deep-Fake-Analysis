"""Weight inventory, verification and the explicit fetch step.

ALUDAM_PLAN.md sec 6.3 (Must-Fix #4). Three rules are load-bearing here:

1. **Wheels ship code, never weights.** PyPI cannot carry multi-GB blobs and
   Hugging Face already hosts them. So weights arrive by an explicit
   ``aludam fetch``, not by being unpacked from the wheel.
2. **Missing weights RAISE.** :class:`AludamWeightsMissing` is raised by
   :func:`ensure` before a detector module is even imported, because the
   upstream modules swallow their own load failures - ``final_audio_detector``
   records ``_classifier_load_error`` and carries on, and
   ``final_image_detector`` logs "Could not load pretrained model ... will rely
   on forensic heuristics only". Both then return a confident-looking number
   from an unfit heuristic. Checking the cache *first* is what makes raising
   possible at all: by the time the module has been imported the silent
   degradation has already happened.
3. **``runtime.degraded`` stays reserved for a model that loaded and then
   failed mid-inference.** Weights that were never there are a setup error.

Everything here works offline. Presence of a Hugging Face weight is decided by
inspecting the hub cache directory directly rather than calling into
``huggingface_hub`` - that keeps ``aludam fetch --dry-run`` and ``ensure()``
usable in the light tier, where the hub library is deliberately not installed.
Only an actual download needs it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .exceptions import AludamDepsMissing, AludamWeightsMissing

__all__ = [
    "Weight",
    "WeightStatus",
    "REQUIRED_WEIGHTS",
    "OPTIONAL_WEIGHTS",
    "cache_dir",
    "status",
    "missing",
    "ensure",
    "fetch",
    "summary",
]


@dataclass(frozen=True)
class Weight:
    """One downloadable/verifiable artefact a detector needs."""

    detector: str
    kind: str          # "hf" (hugging face repo) or "file" (repo-relative path)
    ref: str           # repo id, or path relative to the models directory
    size_hint: str = ""
    required: bool = True

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.ref}"

    @property
    def label(self) -> str:
        role = "required" if self.required else "optional"
        size = f" ~{self.size_hint}" if self.size_hint else ""
        return f"{self.id}  [{role}{size}]"


@dataclass(frozen=True)
class WeightStatus:
    weight: Weight
    present: bool
    where: str = ""
    bytes_on_disk: int = 0
    note: str = ""

    @property
    def size_text(self) -> str:
        if self.bytes_on_disk <= 0:
            return self.weight.size_hint or "?"
        mb = self.bytes_on_disk / 1e6
        if mb >= 1000:
            return f"{mb / 1000:.2f} GB"
        return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# Inventory. Sizes are approximate, for reporting only - never for validation.
#
# Derived from the shipped detector modules:
#   final_text_detector.py:39   PROBE_MODEL_ID          -> gpt2
#   final_image_detector.py:53  PRIMARY_MODEL_ID        -> umm-maybe/...
#   final_audio_detector.py:65  PRETRAINED_MODEL_ID     -> MelodyMachine/...
#   final_audio_detector.py:68  AUDIO_CLASSIFIER_PATH   -> models/...pkl
#   final_video_detector.py     imports no model at all: pure forensics.
# ---------------------------------------------------------------------------

REQUIRED_WEIGHTS: Dict[str, Tuple[Weight, ...]] = {
    "text_det": (
        Weight("text_det", "hf", "gpt2", "551 MB"),
    ),
    "img_det": (
        Weight("img_det", "hf", "umm-maybe/AI-image-detector", "348 MB"),
    ),
    "aud_det": (
        Weight("aud_det", "hf",
               "MelodyMachine/Deepfake-audio-detection-V2", "378 MB"),
    ),
    "vdo_det": (),
}

# Trained on the user's own machine by final_audio_detector's ``train_classifier``
# entry point, so it cannot be fetched - only reported. Listed so that
# ``aludam fetch`` is honest about what it will and will not produce.
OPTIONAL_WEIGHTS: Dict[str, Tuple[Weight, ...]] = {
    "aud_det": (
        Weight("aud_det", "file", "models/audio_rf_classifier.pkl",
               "small", required=False),
    ),
}


def _hf_cache_dir() -> Path:
    """Where ``huggingface_hub`` keeps its snapshots, without importing it.

    Honours the same variables the library honours, in its precedence order,
    so a user who has pointed their cache somewhere else is not lied to.
    """
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(var, "").strip()
        if raw:
            return Path(raw).expanduser()
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _models_root() -> Optional[Path]:
    """The directory holding the ``final_*_detector`` modules, or None."""
    try:
        from .registry import resolve_models_dir
        return Path(resolve_models_dir())
    except Exception:  # noqa: BLE001 - inventory must not itself fail
        return None


def _hf_repo_dirname(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def _hf_present(repo: str) -> Tuple[bool, str, int]:
    root = _hf_cache_dir() / _hf_repo_dirname(repo)
    if not root.is_dir():
        return False, "", 0
    snaps = root / "snapshots"
    files = [p for p in snaps.rglob("*") if p.is_file()] if snaps.is_dir() else []
    if not files:
        # A partially downloaded repo leaves refs/ but no snapshot files.
        return False, str(root), 0
    return True, str(snaps), sum(p.stat().st_size for p in files)


def _file_present(rel: str) -> Tuple[bool, str, int]:
    """Resolve a repo-relative path like ``models/foo.pkl``."""
    root = _models_root()
    candidates: List[Path] = []
    p = Path(rel)
    if root is not None:
        # ``models/audio_rf_classifier.pkl`` is written relative to the repo
        # root, while ``root`` is the ``models`` directory inside it.
        candidates.append(root.parent / p)
        candidates.append(root / p.name)
    candidates.append(p)
    for c in candidates:
        try:
            if c.is_file():
                return True, str(c), c.stat().st_size
        except OSError:
            continue
    return False, "", 0


def _check(w: Weight) -> WeightStatus:
    if w.kind == "hf":
        ok, where, size = _hf_present(w.ref)
        note = "" if ok else f"not in {_hf_cache_dir()}"
    elif w.kind == "file":
        ok, where, size = _file_present(w.ref)
        note = "" if ok else "file not found"
    else:  # pragma: no cover - inventory is closed, this cannot happen
        ok, where, size, note = False, "", 0, f"unknown weight kind {w.kind!r}"
    return WeightStatus(w, ok, where, size, note)


def _weights_for(detector: str, *, include_optional: bool = True) -> Tuple[Weight, ...]:
    req = REQUIRED_WEIGHTS.get(detector, ())
    if not include_optional:
        return req
    return req + OPTIONAL_WEIGHTS.get(detector, ())


def status(
    detector: str,
    *,
    include_optional: bool = True,
) -> List[WeightStatus]:
    """Presence report for one detector, required weights first."""
    return [_check(w) for w in _weights_for(detector, include_optional=include_optional)]


def missing(detector: str) -> List[Weight]:
    """Required weights for ``detector`` that are not present."""
    return [s.weight for s in status(detector, include_optional=False)
            if not s.present]


def ensure(detector: str) -> None:
    """Raise :class:`AludamWeightsMissing` if ``detector``'s weights are absent.

    Called by :func:`aludam.registry.load_detector` *before* the detector
    module is imported, so the upstream code's own silent fallback never gets
    a chance to run.

    Raises:
        AludamWeightsMissing: a required weight is absent.
        AludamDepsMissing: ``huggingface_hub`` is needed but not installed
            (only reachable through :func:`fetch`, not here - verification
            never needs the library).
    """
    absent = missing(detector)
    if not absent:
        return
    first = absent[0]
    raise AludamWeightsMissing(
        detector=detector,
        weight_id=first.id,
        size_hint=first.size_hint,
    )


def cache_dir() -> Path:
    """The Hugging Face hub cache directory this process would write to."""
    return _hf_cache_dir()


def _human_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} TB"  # pragma: no cover


def fetch(
    detectors: Optional[Sequence[str]] = None,
    *,
    dry_run: bool = False,
    quiet: bool = False,
) -> Dict[str, List[WeightStatus]]:
    """Download the missing weights for one or more detectors.

    Already-present weights are skipped rather than re-fetched, so this is
    safe to run repeatedly and cheap to put in a setup script.

    Args:
        detectors: detector names; ``None`` fetches every detector.
        dry_run: report what would be downloaded and touch nothing.
        quiet: suppress the human-readable progress lines.

    Returns:
        ``{detector: [WeightStatus, ...]}`` after the attempt.

    Raises:
        AludamDepsMissing: ``huggingface_hub`` is not installed.
        AludamWeightsMissing: a download failed, so weights are still absent.
    """
    names: List[str] = list(detectors) if detectors else list(REQUIRED_WEIGHTS)
    unknown = [n for n in names if n not in REQUIRED_WEIGHTS]
    if unknown:
        from .exceptions import UnknownDetectorError
        raise UnknownDetectorError(
            f"Unknown detector(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(REQUIRED_WEIGHTS)}"
        )

    report: Dict[str, List[WeightStatus]] = {}

    # Anything to download? Resolve the plan before importing the library so
    # --dry-run works with no network and no hub installation.
    plan: List[Tuple[str, Weight]] = []
    for name in names:
        for w in _weights_for(name):
            st = _check(w)
            if not st.present and w.kind == "hf":
                plan.append((name, w))

    if dry_run:
        if not plan:
            _say(quiet, "Nothing to download - all required weights present.")
        else:
            _say(quiet, f"Would download {len(plan)} weight(s) into {cache_dir()}:")
            for name, w in plan:
                _say(quiet, f"  {name:10} {w.id}  ~{w.size_hint or '?'}")
        return {n: status(n) for n in names}

    if plan:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise AludamDepsMissing(["huggingface_hub"], names[0]) from exc

        seen: Dict[str, str] = {}
        for name, w in plan:
            if w.ref in seen:
                continue
            seen[w.ref] = name
            _say(quiet, f"fetching {w.ref} (~{w.size_hint or '?'}) ...")
            try:
                snapshot_download(w.ref)
            except Exception as exc:  # noqa: BLE001 - surface verbatim
                _say(quiet, f"  FAILED: {type(exc).__name__}: {exc}")
                # Fall through: the status check below decides what to raise.

    report = {n: status(n) for n in names}

    # Re-verify and raise if anything required is still absent - a download
    # that half-succeeded must not look like success.
    still_missing = [(n, w) for n in names for w in missing(n)]
    if still_missing:
        name, w = still_missing[0]
        raise AludamWeightsMissing(
            detector=name, weight_id=w.id, size_hint=w.size_hint)

    if not quiet:
        total = sum(s.bytes_on_disk for sts in report.values() for s in sts)
        _say(quiet, f"cache: {cache_dir()}")
        _say(quiet, f"total on disk: {_human_bytes(total)}")
    return report


def summary(detectors: Optional[Sequence[str]] = None) -> str:
    """A short human-readable table of every weight's presence.

    Used by ``aludam fetch --dry-run`` and by ``aludam status`` so that one
    place answers "what is this install missing?"
    """
    names = list(detectors) if detectors else list(REQUIRED_WEIGHTS)
    lines = [f"cache: {cache_dir()}", ""]
    lines.append(f"{'detector':10} {'weight':52} {'state':9} {'size':10}")
    lines.append("-" * 84)
    grand = 0
    for name in names:
        sts = status(name)
        if not sts:
            # No weights at all (vdo_det is pure forensics). Printed rather
            # than skipped: a detector absent from `aludam status` reads as
            # broken or unsupported, not as "nothing to fetch".
            lines.append(
                f"{name:10} {'(no downloadable weights - heuristic only)':52} "
                f"{'n/a':9} {'-':10}"
            )
            continue
        for st in sts:
            state = "present" if st.present else "MISSING"
            grand += st.bytes_on_disk
            lines.append(
                f"{st.weight.detector:10} {st.weight.id:52} {state:9} "
                f"{st.size_text:10}"
            )
    lines.append("-" * 84)
    lines.append(f"{'total':10} {'':52} {'':9} {_human_bytes(grand):10}")
    return "\n".join(lines)


def _say(quiet: bool, msg: str) -> None:
    if not quiet:
        print(msg)
