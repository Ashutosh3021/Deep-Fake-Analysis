"""
DeepGuard AI - Pipeline Smoke Test
==================================
Verifies that every detector in the real-time deepfake pipeline can be
imported, instantiated, and run a prediction against a synthetic in-memory
input. Any failure is reported with its exact root cause (missing package,
broken import, hard-coded path, missing binary, etc.).

Usage (from project root):

    python -m unittest tests.test_detectors -v     # verbose
    python tests/test_detectors.py                 # plain

Environment overrides:

    DEEPFAKE_ALLOW_DOWNLOADS=1    allow transformers / retina-face to
                                  download weights from the network.
                                  Default is OFFLINE mode: model downloads
                                  fail fast and every detector falls back
                                  to its heuristic path (by design).
"""

import os
import sys
import shutil
import tempfile
import importlib
import unittest
import warnings
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
BACKEND_DIR = os.path.join(ROOT, "backend")

for _p in (ROOT, MODELS_DIR, BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ALLOW_DOWNLOADS = os.environ.get("DEEPFAKE_ALLOW_DOWNLOADS", "") == "1"
if not ALLOW_DOWNLOADS:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings("ignore")


def report(check, status, detail=""):
    flag = {"OK": "[ OK ]", "FAIL": "[FAIL]", "WARN": "[WARN]"}[status]
    print(f"{flag} {check}" + (f"  ->  {detail}" if detail else ""))


def safe_import(mod):
    """Return (module, "") on success or (None, error message) on failure."""
    try:
        return importlib.import_module(mod), ""
    except ImportError as e:
        return None, f"{mod}: {e} (missing package: {getattr(e, 'name', '?')})"
    except Exception as e:
        return None, f"{mod}: {type(e).__name__}: {e}"


def make_synthetic_image(size=256):
    """Random-noise BGR image saved as JPEG."""
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    path = os.path.join(tempfile_dir(), "synthetic_test_image.jpg")
    import cv2
    cv2.imwrite(path, img)
    return path


def make_synthetic_audio(duration_sec=1.2, sr=16000):
    """Silence WAV file (valid, analyzable audio)."""
    import soundfile as sf
    data = (np.random.randn(int(sr * duration_sec)) * 1e-4).astype(np.float32)
    path = os.path.join(tempfile_dir(), "synthetic_test_audio.wav")
    sf.write(path, data, sr)
    return path


def make_synthetic_video(frames=16, size=64):
    """Tiny AVI video (MJPG codec ships with OpenCV)."""
    import cv2
    path = os.path.join(tempfile_dir(), "synthetic_test_video.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (size, size))
    for _ in range(frames):
        frame = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path if os.path.exists(path) and os.path.getsize(path) > 0 else None


def tempfile_dir():
    d = os.path.join(tempfile.gettempdir(), "deepguard_tests")
    os.makedirs(d, exist_ok=True)
    return d


class PipelineSmokeTests(unittest.TestCase):

    longMessage = True

    def setUp(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        report(name, "OK" if ok else "FAIL", detail)
        if not ok:
            self.failures.append(f"{name}: {detail}")

    def warn(self, name, detail=""):
        report(name, "WARN", detail)

    # ======================================================================
    # 1. Imports - the base health check. Catches missing packages and any
    #    broken top-level code before we go further.
    # ======================================================================
    def test_01_active_module_imports(self):
        modules = [
            "forensics_core",
            "final_image_detector",
            "final_video_detector",
            "final_audio_detector",
            "final_text_detector",
            "nn_modules",
            "query_assistant",
            "api",  # Flask backend
        ]
        loaded = {}
        for mod in modules:
            m, err = safe_import(mod)
            loaded[mod] = m
            self.check(f"import {mod}", m is not None, err or "ok")
        self.assertFalse(self.failures, "\n".join(self.failures))

    # ======================================================================
    # 2. forensics_core - low-level forensic primitives.
    # ======================================================================
    def test_02_forensics_core(self):
        fc = sys.modules.get("forensics_core")
        self.assertTrue(fc is not None, "forensics_core must be imported first")

        import cv2
        gray = cv2.cvtColor(cv2.imread(make_synthetic_image()), cv2.COLOR_BGR2GRAY)
        for fn, arg in [
            ("fft_periodicity_score", gray),
            ("texture_uniformity_score", gray),
            ("histogram_smoothness_score", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
            ("noise_residual_consistency", gray),
        ]:
            try:
                finding = getattr(fc, fn)(arg)
                ok = isinstance(finding, dict) and "score" in finding and "signal" in finding
                self.check(f"forensics_core.{fn}", ok,
                           "" if ok else f"bad return: {finding}")
            except Exception as e:
                self.check(f"forensics_core.{fn}", False, f"{type(e).__name__}: {e}")

        # Full orchestration run on a synthetic image.
        result = fc.full_image_forensics(make_synthetic_image())
        required_keys = {"file_hash", "provenance_findings", "global_findings",
                         "splice_findings", "face_analyses", "faces_detected"}
        self.check("forensics_core.full_image_forensics",
                   isinstance(result, dict) and required_keys.issubset(result.keys()),
                   "error" if isinstance(result, dict) and "error" in result else str(result)[:200])
        self.assertFalse(self.failures, "\n".join(self.failures))

    # ======================================================================
    # 3. End-to-end predictions on synthetic inputs.
    # ======================================================================
    def test_03_image_detector_predict(self):
        mod = sys.modules.get("final_image_detector")
        self.assertTrue(mod is not None, "final_image_detector must be imported first")
        try:
            detector = mod.FinalImageDetector()
        except Exception as e:
            self.check("FinalImageDetector()", False, f"{type(e).__name__}: {e}")
            return

        note = getattr(detector, "_model_load_error", None)
        if note:
            self.warn("FinalImageDetector pretrained model",
                      f"offline/unavailable -> heuristic mode: {note[:160]}")

        try:
            verdict = detector.predict(make_synthetic_image())
            required_keys = {"label", "confidence", "family_scores", "reasons"}
            ok = isinstance(verdict, dict) and required_keys.issubset(verdict.keys())
            self.check("FinalImageDetector.predict(image)", ok, str(verdict)[:200])
        except Exception as e:
            self.check("FinalImageDetector.predict(image)", False,
                       f"{type(e).__name__}: {e}")
        self.assertFalse(self.failures, "\n".join(self.failures))

    def test_04_text_detector_predict(self):
        mod = sys.modules.get("final_text_detector")
        self.assertTrue(mod is not None, "final_text_detector must be imported first")
        try:
            detector = mod.FinalTextDetector()
        except Exception as e:
            self.check("FinalTextDetector()", False, f"{type(e).__name__}: {e}")
            return

        note = getattr(detector, "_load_error", None)
        if note:
            self.warn("FinalTextDetector probe LM (GPT-2)",
                      f"offline/unavailable -> stylometric mode: {note[:160]}")

        sample = (
            "The quick brown fox jumps over the lazy dog while the sun rises "
            "over the distant hills. People often gather in the town square "
            "to discuss the news of the day and share stories about their "
            "families and friends. The old library on the corner has been "
            "there for more than a century, and its shelves hold thousands "
            "of books that tell the history of the entire region. Every "
            "autumn, the trees in the park change color from green to shades "
            "of orange and gold, and children collect the fallen leaves on "
            "their way home from school."
        )
        try:
            verdict = detector.predict(sample)
            required_keys = {"label", "confidence", "ai_probability", "metrics"}
            ok = isinstance(verdict, dict) and required_keys.issubset(verdict.keys())
            self.check("FinalTextDetector.predict(text)", ok, str(verdict)[:200])
        except Exception as e:
            self.check("FinalTextDetector.predict(text)", False,
                       f"{type(e).__name__}: {e}")
        self.assertFalse(self.failures, "\n".join(self.failures))

    def test_05_audio_detector_predict(self):
        mod = sys.modules.get("final_audio_detector")
        self.assertTrue(mod is not None, "final_audio_detector must be imported first")
        try:
            detector = mod.FinalAudioDetector()
        except Exception as e:
            self.check("FinalAudioDetector()", False, f"{type(e).__name__}: {e}")
            return

        path = make_synthetic_audio()
        try:
            verdict = detector.predict(path)
            required_keys = {"label", "confidence", "fake_probability", "signal_source"}
            ok = isinstance(verdict, dict) and required_keys.issubset(verdict.keys())
            self.check("FinalAudioDetector.predict(audio)", ok, str(verdict)[:200])
        except Exception as e:
            self.check("FinalAudioDetector.predict(audio)", False,
                       f"{type(e).__name__}: {e}")
        self.assertFalse(self.failures, "\n".join(self.failures))

    def test_06_video_detector_predict(self):
        mod = sys.modules.get("final_video_detector")
        self.assertTrue(mod is not None, "final_video_detector must be imported first")
        try:
            detector = mod.FinalVideoDetector()
        except Exception as e:
            self.check("FinalVideoDetector()", False, f"{type(e).__name__}: {e}")
            return

        path = make_synthetic_video()
        if path is None:
            self.check("FinalVideoDetector.predict(video)", False,
                       "could not create a synthetic AVI with cv2.VideoWriter")
            self.assertFalse(self.failures, "\n".join(self.failures))
            return
        try:
            verdict = detector.predict(path)
            required_keys = {"label", "confidence", "family_scores", "notes"}
            ok = isinstance(verdict, dict) and required_keys.issubset(verdict.keys())
            detail = str(verdict)[:200] if not ok else \
                str(verdict.get("notes", ""))[:160]
            self.check("FinalVideoDetector.predict(video)", ok, detail)
        except Exception as e:
            self.check("FinalVideoDetector.predict(video)", False,
                       f"{type(e).__name__}: {e}")
        self.assertFalse(self.failures, "\n".join(self.failures))

    # ======================================================================
    # 4. Environment / known-issue probes - warn instead of fail, since these
    #    are environment-dependent, but report them loudly.
    # ======================================================================
    def test_07_core_package_versions(self):
        import platform
        self.warn("python", platform.python_version())
        for pkg, attr in [("numpy", "__version__"), ("cv2", "__version__"),
                          ("torch", "__version__"), ("transformers", "__version__"),
                          ("librosa", "__version__"), ("mediapipe", "__version__")]:
            try:
                m = importlib.import_module(pkg)
                self.warn(pkg, getattr(m, attr, "?"))
            except Exception as e:
                self.check(f"import {pkg}", False, f"{type(e).__name__}: {e}")
        try:
            import torch
            self.warn("torch device", torch.cuda.get_device_name(0) if torch.cuda.is_available()
                      else "cpu (CUDA not available)")
        except Exception:
            pass
        self.assertFalse(self.failures, "\n".join(self.failures))

    def test_08_video_av_sync_ffmpeg(self):
        if shutil.which("ffmpeg"):
            self.warn("ffmpeg", "found on PATH - AV-sync available")
        else:
            self.warn("ffmpeg", "NOT on PATH - FinalVideoDetector._audio_visual_sync "
                                "will be skipped")

    def test_09_video_tmpdir_windows_bug(self):
        # final_video_detector._per_frame_forensics hard-codes the POSIX path
        # "/tmp/deepguard_video_frames". This only exists on POSIX systems.
        tmp = "/tmp/deepguard_video_frames"
        try:
            os.makedirs(tmp, exist_ok=True)
            self.check(f"video per-frame temp dir ('{tmp}')", True,
                       "POSIX path is writable")
            try:
                shutil.rmtree(tmp)
            except Exception:
                pass
        except Exception as e:
            self.check(
                f"video per-frame temp dir ('{tmp}')", False,
                f"{type(e).__name__}: {e} -- final_video_detector.py hard-codes "
                f"'{tmp}'; per-frame forensics will raise at runtime. Fix: use "
                f"tempfile.gettempdir() instead of '/tmp'."
            )
        self.assertFalse(self.failures, "\n".join(self.failures))

    def test_10_query_assistant_gemini(self):
        qa = sys.modules.get("query_assistant")
        self.assertTrue(qa is not None, "query_assistant must be imported first")
        key = getattr(qa, "GEMINI_API_KEY", "")
        if key:
            self.warn("query_assistant.GEMINI_API_KEY", "set (len={})".format(len(key)))
        else:
            self.warn("query_assistant.GEMINI_API_KEY",
                      "NOT set - analyze_image()/answer_text_query() will raise "
                      "ValueError at runtime (add it to .env)")

    # ======================================================================
    # 5. Legacy files - informational. Not part of the active API pipeline,
    #    so reported as WARN, not FAIL.
    # ======================================================================
    def test_11_legacy_modules_status(self):
        legacy = [
            ("models/video_detector.py",
             "legacy - imports 'app.ml.*' which does not exist in this repo"),
            ("models/bin/*.py",
             "legacy - requires tensorflow/torchvision/pandas/matplotlib/seaborn "
             "(removed from requirements.txt)"),
        ]
        for name, why in legacy:
            self.warn(f"legacy {name}", why)


def _main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineSmokeTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n" + "=" * 70)
    print("SUMMARY")
    if result.failures or result.errors:
        print("SOME CHECKS FAILED - see [FAIL] lines and tracebacks above.")
        print("Each [FAIL] line names the exact module/function and root cause.")
        sys.exit(1)
    print("All smoke tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    _main()