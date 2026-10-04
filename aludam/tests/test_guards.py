"""Quality-gate tests: the <128 px image guard (ALUDAM_PLAN.md sec 7.1).

The point of the guard is that the detector is *never called* below the
model's stated minimum. So the detector stub here is written to raise if it
is invoked at all - a test that only checked the returned label would still
pass if the model had run and its output been overwritten afterwards.

Run:  python tests/test_guards.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aludam import InvalidInputError, load_detector  # noqa: E402
from aludam.registry import MIN_IMAGE_SIDE, Detector  # noqa: E402

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


def must_not_run(_path):
    raise AssertionError("detector ran - the quality gate failed to skip it")


def make_png(directory: Path, name: str, w: int, h: int) -> str:
    from PIL import Image
    p = directory / name
    Image.new("RGB", (w, h), (10, 120, 200)).save(p)
    return str(p)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="aludam-guard-"))

    print(f"[1] MIN_IMAGE_SIDE is {MIN_IMAGE_SIDE}")
    check(MIN_IMAGE_SIDE == 128, "guard threshold is 128 px")

    img = Detector(name="img_det", media_type="IMAGE", _predict_raw=must_not_run)

    print("\n[2] undersized image -> gate result, detector never called")
    small = make_png(tmp, "small.png", 64, 48)
    try:
        r = img.predict(small)
        check(True, "no exception")
    except AssertionError as exc:  # pragma: no cover
        check(False, f"detector ran: {exc}")
        return finish()
    check(r.label == "uncertain", f"label is uncertain (got {r.label!r})")
    check(r.score == 0.5, f"score is the neutral 0.5 (got {r.score})")
    check(r.details.get("reason") == "input_below_model_minimum",
          f"reason recorded (got {r.details.get('reason')!r})")
    check(r.details.get("image_size") == [64, 48],
          f"actual size recorded (got {r.details.get('image_size')})")
    check(r.detector == "img_det", "detector name preserved")
    check(r.media_type == "IMAGE", "media_type preserved")
    check(0.0 <= r.confidence <= 1.0, "confidence stays in 0..1")
    check(r.explanation.strip(), "explanation is non-empty")

    print("\n[3] exactly at the threshold -> detector runs")
    at = make_png(tmp, "at.png", MIN_IMAGE_SIDE, MIN_IMAGE_SIDE)
    ran = []

    def spy(path):
        ran.append(path)
        return {"label": "AUTHENTIC", "p_synthetic": 0.1, "confidence": 80}

    img2 = Detector(name="img_det", media_type="IMAGE", _predict_raw=spy)
    try:
        r2 = img2.predict(at)
        check(len(ran) == 1, "detector invoked at the threshold")
        check(r2.details.get("reason") is None,
              "no quality-gate reason on a passing image")
        check(r2.label == "real", f"label passed through (got {r2.label!r})")
    except Exception as exc:  # pragma: no cover
        check(False, f"{type(exc).__name__}: {exc}")

    print("\n[4] just below on one axis only -> still gated")
    onethin = make_png(tmp, "thin.png", 1000, 64)
    try:
        r3 = img.predict(onethin)
        check(r3.details.get("reason") == "input_below_model_minimum",
              "1000x64 is gated (either axis counts)")
    except AssertionError:  # pragma: no cover
        check(False, "detector ran on an undersized image")

    print("\n[5] text and other media types are unaffected")
    txt = Detector(name="text_det", media_type="TEXT",
                   _predict_raw=lambda t: must_not_run(t))
    try:
        txt.predict("some text")
        check(False, "text should have reached the detector (and raised)")
    except AssertionError:
        check(True, "text path does not go through the image guard")

    print("\n[6] unreadable / non-image file is not this gate's business")
    junk = tmp / "junk.bin"
    junk.write_bytes(b"\x00\x01\x02\x03" * 16)
    ran2 = []

    def spy2(path):
        ran2.append(path)
        return {"label": "AUTHENTIC", "p_synthetic": 0.2, "confidence": 70}

    img3 = Detector(name="img_det", media_type="IMAGE", _predict_raw=spy2)
    r4 = img3.predict(str(junk))
    check(len(ran2) == 1,
          "unopenable file falls through to the detector (its own error path)")
    check(r4 is not None, "detector result returned")

    print("\n[7] the real load_detector still reports calibrated state")
    try:
        det = load_detector("img_det")
        check(det.name == "img_det", "img_det loads")
        rr = det.predict(small)
        check(rr.label == "uncertain",
              f"real detector gated too (got {rr.label!r})")
        check(rr.details.get("reason") == "input_below_model_minimum",
              "real detector records the gate reason")
        check(rr.calibrated is False, "still uncalibrated (D4)")
    except Exception as exc:  # pragma: no cover
        check(False, f"load/predict failed: {type(exc).__name__}: {exc}")

    print("\n[8] missing file still raises InvalidInputError, not a gate")
    try:
        img.predict(str(tmp / "nope.png"))
        check(False, "missing file should raise")
    except InvalidInputError:
        check(True, "missing file -> InvalidInputError")

    return finish()


def finish() -> int:
    print(f"\n{'=' * 60}\n  {PASS} passed, {FAIL} failed\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
