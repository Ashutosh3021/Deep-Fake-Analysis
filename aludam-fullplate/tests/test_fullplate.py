"""Fullplate contract tests: same API as aludam, one dependency list.

Self-contained and fast: no torch, no transformers, no model weights, no
network. It checks the two things that make `aludam-fullplate` what it is:

1. It is a thin wrapper - one dependency (`aludam[neural]`), versioned in
   lockstep with `aludam`, whose `neural` extra is the single source of truth
   for the heavy stack (ALUDAM_PLAN.md sec 6.2).
2. The import surface is *identical*: `aludam_fullplate` re-exports `aludam`,
   so `DetectionResult`, `load_detector` and the errors are the same objects,
   not look-alikes.

Plus two release guards: THIRD_PARTY_NOTICES has no unresolved row in either
package, and every shipped file is ASCII (non-ASCII raises UnicodeEncodeError
on cp1252 consoles and would break JSON/log sinks).

Run:  python tests/test_fullplate.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../aludam-fullplate
REPO = ROOT.parent                           # repository root
ALUDAM_DIR = REPO / "aludam"

sys.path.insert(0, str(ALUDAM_DIR / "src"))
sys.path.insert(0, str(ROOT / "src"))

import aludam  # noqa: E402
import aludam_fullplate  # noqa: E402
from aludam_fullplate import (  # noqa: E402
    DETECTOR_NAMES,
    DetectionResult,
    InvalidInputError,
    UnknownDetectorError,
    al,
    load_detector,
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


def load_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def dep_name(spec: str) -> str:
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec)
    return m.group(0) if m else ""


def spec_after_extras(spec: str) -> str:
    return spec.split("]", 1)[1] if "]" in spec else spec


def spec_satisfies(spec: str, version: str) -> bool:
    """Does ``version`` satisfy a ``~=X.Y.Z`` pin? The only form we write."""
    if not spec.startswith("~="):
        return False
    base = spec[2:].split(".")
    cur = version.split(".")
    if len(base) != 3 or len(cur) != 3:
        return False
    if not all(p.isdigit() for p in base + cur):
        return False
    b = tuple(int(p) for p in base)
    c = tuple(int(p) for p in cur)
    return c[:2] == b[:2] and c >= b


ALUDAM_TOML = load_toml(ALUDAM_DIR / "pyproject.toml")
FULLPLATE_TOML = load_toml(ROOT / "pyproject.toml")

al_proj = ALUDAM_TOML["project"]
fp_proj = FULLPLATE_TOML["project"]

al_deps = list(al_proj.get("dependencies", []))
fp_deps = list(fp_proj.get("dependencies", []))
neural_deps = list(
    (al_proj.get("optional-dependencies") or {}).get("neural", [])
)

BASE_REQUIRED = {"pillow", "numpy", "pypdf", "python-docx"}
NEURAL_REQUIRED = {
    "torch",
    "transformers",
    "huggingface_hub",
    "timm",
    "peft",
    "opencv-python-headless",
    "librosa",
    "soundfile",
    "scipy",
    "onnxruntime",
}

# Every weight row that must exist in aludam's THIRD_PARTY_NOTICES: the three
# the shipped inventory fetches, the four Phase 3 swaps, and the YuNet face
# model (ALUDAM_PLAN.md sec 4).
WEIGHT_ROWS = [
    "gpt2",
    "umm-maybe/AI-image-detector",
    "MelodyMachine/Deepfake-audio-detection-V2",
    "AICodexLab/answerdotai-ModernBERT-base-ai-detector",
    "Bombek1/ai-image-detector-siglip-dinov2",
    "gaetanbrison/deepfake-detector-resnet50-t2v-sora-veo2",
    "lofcz/ai-music-detector",
    "face_detection_yunet",
]

# --- 1. packaging metadata ------------------------------------------------

print("\n[1] packaging metadata")

check(al_proj["name"] == "aludam", "aludam pyproject name is 'aludam'")
check(fp_proj["name"] == "aludam-fullplate",
      "fullplate pyproject name is 'aludam-fullplate'")

al_version = str(al_proj["version"])
fp_version = str(fp_proj["version"])
check(fp_version == al_version,
      f"versions in lockstep ({fp_version} == {al_version})")
check(aludam.__version__ == al_version,
      f"aludam.__version__ ({aludam.__version__}) == pyproject ({al_version})")
check(aludam_fullplate.__version__ == fp_version,
      "aludam_fullplate.__version__ == fullplate pyproject version")

check(len(fp_deps) == 1,
      f"fullplate declares exactly one dependency, got {len(fp_deps)}")
if fp_deps:
    d = fp_deps[0]
    check(dep_name(d) == "aludam",
          f"fullplate dependency is aludam itself ({d!r})")
    check("[" in d and d.split("[", 1)[1].startswith("neural]"),
          "fullplate dependency requests the [neural] extra")
    pin = spec_after_extras(d)
    check(spec_satisfies(pin, al_version),
          f"pin {pin!r} is satisfied by aludam {al_version}")

check({dep_name(d) for d in al_deps} == BASE_REQUIRED,
      "aludam base deps are exactly pillow/numpy/pypdf/python-docx")
check({dep_name(d) for d in neural_deps} == NEURAL_REQUIRED,
      "aludam [neural] extra is exactly the ten heavy packages")

# --- 2. identical import surface ------------------------------------------

print("\n[2] identical import surface")

check(aludam_fullplate.TIER == "neural", "TIER == 'neural'")

same = [n for n in aludam.__all__
        if getattr(aludam_fullplate, n, None) is getattr(aludam, n, None)]
check(len(same) == len(aludam.__all__),
      f"all {len(aludam.__all__)} exported names are the same objects "
      f"({len(same)} matched)")

check(aludam_fullplate.__all__ == [*aludam.__all__, "TIER"],
      "__all__ is aludam's plus TIER")
check(aludam_fullplate.al is aludam.al, "al.detect entry is the same module")
check(aludam_fullplate.load_detector is aludam.load_detector,
      "load_detector is the same function")
check(aludam_fullplate.DetectionResult is aludam.DetectionResult,
      "DetectionResult is the same class")

# --- 3. DetectionResult through the fullplate import ----------------------

print("\n[3] DetectionResult through the fullplate import")

good = DetectionResult(
    score=0.8012, label="ai", confidence=0.3125,
    explanation="Synthetic / machine-generated content (score 0.80).",
    kind="generated",
)
check(good.score == 0.8012, "score preserved at 4 dp")
check(good.confidence == 0.3125, "confidence preserved at 4 dp")
check(good.label == "ai", "label is 'ai'")
check(good.kind == "generated", "kind is 'generated'")
check(good.is_ai is True, "is_ai True for an 'ai' verdict")
check(good.runtime["calibrated"] is False, "runtime.calibrated ships False")
check(good.calibrated is False, "calibrated property is False")
check(good.backend == "light", "backend defaults to 'light' until normalize()")

for bad_kwargs, what, exc in [
    (dict(score=80.12, label="ai", confidence=0.5, explanation="x"),
     "score > 1 rejected", ValueError),
    (dict(score=0.5, label="ai", confidence=80.12, explanation="x"),
     "0-100 confidence rejected", ValueError),
    (dict(score=-0.1, label="ai", confidence=0.5, explanation="x"),
     "negative score rejected", ValueError),
    (dict(score=0.5, label="FAKE", confidence=0.5, explanation="x"),
     "native label rejected", ValueError),
    (dict(score=0.5, label="ai", confidence=0.5, explanation="x",
          kind="spoofed"),
     "unknown kind rejected", ValueError),
    (dict(score=0.5, label="ai", confidence=0.5, explanation=123),
     "non-str explanation rejected", TypeError),
    (dict(score=0.5, label="ai", confidence=0.5, explanation="x",
          runtime={"calibrated": "yes"}),
     "non-bool calibrated rejected", TypeError),
]:
    check(raises(lambda k=bad_kwargs: DetectionResult(**k), exc), what)

text = repr(good)
check(text.startswith("DetectionResult(score=0.8012"), "repr starts with fields")
try:
    text.encode("ascii")
    check(True, "repr is ASCII (cp1252-safe)")
except UnicodeEncodeError:
    check(False, "repr is ASCII (cp1252-safe)")

try:
    json.dumps(good.to_dict())
    check(True, "to_dict() is JSON-serialisable")
except (TypeError, ValueError):
    check(False, "to_dict() is JSON-serialisable")

uncertain = DetectionResult(
    score=0.5, label="uncertain", confidence=0.0, explanation="abstain")
check(uncertain.is_uncertain is True, "is_uncertain True for 'uncertain'")

# --- 4. detector names -----------------------------------------------------

print("\n[4] detector names")

check(DETECTOR_NAMES == ("text_det", "img_det", "vdo_det", "aud_det"),
      "exactly the four specified names, in order")
check(aludam_fullplate.available_detectors() == DETECTOR_NAMES,
      "available_detectors() mirrors")

for bad in ("text", "", "image"):
    try:
        load_detector(bad)
        check(False, f"load_detector({bad!r}) raises UnknownDetectorError")
    except UnknownDetectorError as exc:
        check("text_det" in str(exc),
              f"load_detector({bad!r}) raises with valid names listed")
    except Exception:
        check(False, f"load_detector({bad!r}) raised the wrong type")

try:
    load_detector(123)
    check(False, "non-str detector name raises")
except UnknownDetectorError:
    check(True, "non-str detector name raises")

check(InvalidInputError is aludam.InvalidInputError,
      "InvalidInputError is the same class")
check(callable(al.detect), "al.detect exported via fullplate")

# --- 5. THIRD_PARTY_NOTICES: no unresolved rows ----------------------------

print("\n[5] THIRD_PARTY_NOTICES")

for path in (ALUDAM_DIR / "THIRD_PARTY_NOTICES",
             ROOT / "THIRD_PARTY_NOTICES"):
    rel = path.relative_to(REPO).as_posix()
    try:
        raw = path.read_bytes()
    except OSError:
        check(False, f"{rel} exists")
        continue
    check(True, f"{rel} exists")
    try:
        raw.decode("ascii")
        check(True, f"{rel} is ASCII")
    except UnicodeDecodeError:
        check(False, f"{rel} is ASCII")
    check(b"PENDING" not in raw.upper(),
          f"{rel} has zero unresolved (PENDING) rows")

notices = (ALUDAM_DIR / "THIRD_PARTY_NOTICES").read_bytes().decode(
    "ascii", errors="replace")
low = notices.lower()

missing_deps = sorted(
    d for d in {dep_name(x) for x in al_deps + neural_deps}
    if d.lower() not in low
)
check(not missing_deps,
      "every declared dependency appears in aludam's notices"
      + (f" (missing: {', '.join(missing_deps)})" if missing_deps else ""))

missing_weights = sorted(
    w for w in WEIGHT_ROWS if w.lower() not in low
)
check(not missing_weights,
      "every weight row appears in aludam's notices"
      + (f" (missing: {', '.join(missing_weights)})" if missing_weights else ""))

# --- 6. ASCII guard: every shipped file in both packages --------------------

print("\n[6] ASCII guard (both package trees)")

SKIP_DIRS = {"__pycache__", ".pytest_cache", "build", "dist", "eval"}
SUFFIXES = {".py", ".toml", ".md", ".in", ".txt", ".typed"}
NAMES = {"LICENSE", "THIRD_PARTY_NOTICES", ".gitignore"}

ascii_files = 0
ascii_bad = []
for base in (ALUDAM_DIR, ROOT):
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info")
               for part in path.parts):
            continue
        if path.suffix not in SUFFIXES and path.name not in NAMES:
            continue
        ascii_files += 1
        try:
            path.read_bytes().decode("ascii")
        except UnicodeDecodeError:
            ascii_bad.append(path.relative_to(REPO).as_posix())

check(ascii_files > 0, f"scanned {ascii_files} shipped files")
check(not ascii_bad,
      "all shipped files are ASCII"
      + (f" (offenders: {', '.join(ascii_bad)})" if ascii_bad else ""))

# --- summary ---------------------------------------------------------------

print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
