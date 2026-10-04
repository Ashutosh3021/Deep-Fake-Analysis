"""Tests for the weight inventory and fetch step (Must-Fix #4).

The guarantee under test is narrow and important: **missing weights raise,
they never degrade.** The upstream modules swallow their own load failures,
so aludam has to detect absence from the cache *before* importing a detector
module. Everything here runs against a temporary cache directory so the
suite gives the same answer on a machine that has the weights and one that
does not.

Run:  python tests/test_weights.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aludam import (  # noqa: E402
    DETECTOR_NAMES,
    AludamWeightsMissing,
    UnknownDetectorError,
    load_detector,
)
from aludam import cli, weights  # noqa: E402

PASS = 0
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def empty_cache() -> Path:
    d = Path(tempfile.mkdtemp(prefix="aludam-hf-empty-"))
    return d


def populated_cache() -> Path:
    """A fake hub cache holding a snapshot for every required HF weight."""
    root = Path(tempfile.mkdtemp(prefix="aludam-hf-full-"))
    for det, ws in weights.REQUIRED_WEIGHTS.items():
        for w in ws:
            if w.kind != "hf":
                continue
            snap = root / f"models--{w.ref.replace('/', '--')}" / "snapshots" / "deadbeef"
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}", encoding="utf-8")
    return root


print("[1] inventory covers exactly the four detectors")
check(set(weights.REQUIRED_WEIGHTS) == set(DETECTOR_NAMES),
      "REQUIRED_WEIGHTS keys == DETECTOR_NAMES")
for name in DETECTOR_NAMES:
    check(isinstance(weights.REQUIRED_WEIGHTS[name], tuple),
          f"{name} has a weight tuple")

print("\n[2] vdo_det ships no model weights (pure forensics)")
check(weights.REQUIRED_WEIGHTS["vdo_det"] == (),
      "vdo_det requires no downloadable weights")

print("\n[3] every required weight is an hf or file kind with a size hint")
ok = True
for det, ws in weights.REQUIRED_WEIGHTS.items():
    for w in ws:
        if w.kind not in ("hf", "file") or not w.ref:
            ok = False
            print(f"     bad weight: {w}")
        if not w.required:
            ok = False
            print(f"     not required: {w}")
check(ok, "all required weights are well-formed")
check(any(not w.required for det in weights.OPTIONAL_WEIGHTS
          for w in weights.OPTIONAL_WEIGHTS[det]),
      "optional weights are declared as optional")

print("\n[4] hub cache directory is resolved without importing huggingface_hub")
with mock.patch.object(weights, "_hf_cache_dir", return_value=Path("/tmp/x")):
    check(weights.cache_dir() == Path("/tmp/x"), "cache_dir() honours the patch")
name = weights._hf_repo_dirname("org/model")
check(name == "models--org--model", f"repo -> cache dirname ({name})")

print("\n[5] empty cache -> every required HF weight reported missing")
tmp = empty_cache()
try:
    with mock.patch.object(weights, "_hf_cache_dir", return_value=tmp):
        for det in ("text_det", "img_det", "aud_det"):
            st = weights.status(det, include_optional=False)
            check(len(st) > 0, f"{det} reports status")
            check(all(not s.present for s in st), f"{det} weights absent")
            check([w.id for w in weights.missing(det)],
                  f"{det} missing() non-empty")
        check(weights.missing("vdo_det") == [],
              "vdo_det never reports missing weights")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[6] ensure() raises AludamWeightsMissing, naming the fix")
tmp = empty_cache()
try:
    with mock.patch.object(weights, "_hf_cache_dir", return_value=tmp):
        try:
            weights.ensure("text_det")
            check(False, "ensure() should have raised")
        except AludamWeightsMissing as exc:
            check(True, "ensure() raises AludamWeightsMissing")
            check(exc.detector == "text_det", "exception carries the detector")
            check("gpt2" in exc.weight_id, f"names the weight ({exc.weight_id})")
            check("aludam fetch" in str(exc),
                  "message tells the user how to fix it")
        check([w.id for w in weights.missing("aud_det")]
              == ["hf:MelodyMachine/Deepfake-audio-detection-V2"],
              "aud_det's optional .pkl is NOT treated as required")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[7] populated cache -> ensure() passes for every detector")
tmp = populated_cache()
try:
    with mock.patch.object(weights, "_hf_cache_dir", return_value=tmp):
        for det in DETECTOR_NAMES:
            try:
                weights.ensure(det)
                check(True, f"{det} ensure() ok")
            except AludamWeightsMissing as exc:  # pragma: no cover
                check(False, f"{det} unexpectedly missing: {exc}")
        check(all(s.present for det in ("text_det", "img_det", "aud_det")
                  for s in weights.status(det, include_optional=False)),
              "all required weights present")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[8] load_detector raises BEFORE the detector module is imported")
tmp = empty_cache()
try:
    from aludam import registry
    registry.reset_cache()
    with mock.patch.object(weights, "_hf_cache_dir", return_value=tmp):
        try:
            load_detector("text_det")
            check(False, "load_detector should have raised")
        except AludamWeightsMissing:
            check(True, "load_detector raises AludamWeightsMissing")
        check("final_text_detector" not in sys.modules,
              "detector module was never imported (no silent degradation)")
    registry.reset_cache()
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n[9] fetch() is honest about what it would and would not do")
tmp = empty_cache()
try:
    with mock.patch.object(weights, "_hf_cache_dir", return_value=tmp):
        rep = weights.fetch(["text_det"], dry_run=True, quiet=True)
        check("text_det" in rep, "dry-run returns a per-detector report")
        check(not (tmp / "models--gpt2").exists(),
              "dry-run downloads nothing")
        check(weights.missing("text_det"), "dry-run leaves weights missing")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

try:
    weights.fetch(["not_a_detector"], dry_run=True, quiet=True)
    check(False, "unknown detector should raise")
except UnknownDetectorError:
    check(True, "fetch() rejects unknown detectors")
except AludamWeightsMissing:  # pragma: no cover
    check(False, "raised the wrong error for an unknown detector")

print("\n[10] summary() renders every detector, present or not")
text = weights.summary()
check("cache:" in text, "summary names the cache directory")
for det in DETECTOR_NAMES:
    check(det in text, f"summary mentions {det}")
check("hf:gpt2" in text, "summary lists the text probe weight")
check("MISSING" in text or "present" in text, "summary reports state")

print("\n[11] CLI: status and fetch --dry-run both exit 0")
for argv, label in ((["status"], "aludam status"),
                    (["fetch", "--dry-run"], "aludam fetch --dry-run")):
    try:
        code = cli.main(argv)
        check(code == 0, f"{label} exits 0 (got {code})")
    except SystemExit as exc:  # pragma: no cover
        check(False, f"{label} raised SystemExit({exc.code})")

try:
    cli.main(["fetch", "bogus"])
    check(False, "bogus detector should be rejected")
except SystemExit:
    check(True, "aludam fetch bogus -> argparse error")
except Exception as exc:  # pragma: no cover
    check(False, f"bogus detector gave {type(exc).__name__}: {exc}")

print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(1 if FAIL else 0)
