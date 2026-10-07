"""Tests for notebooks/img_score_supervisor.py (crash-proof scoring).

A stub worker speaks the same protocol as img_score_worker.py but misbehaves on
command, driven by the text inside each fake "image" file:

    ok / hang / crash / sigterm / bloat / flaky

Run:  python -m unittest tests.test_score_isolation -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import img_score_supervisor as sup  # noqa: E402

STUB = textwrap.dedent('''
    import json, os, signal, sys, time
    P = "@@RESULT@@ "
    out = sys.stdout
    def send(o): out.write(P + json.dumps(o) + "\\n"); out.flush()
    send({"ready": True, "device": "cpu", "notes": ["stub"]})
    hold = []
    for line in sys.stdin:
        req = json.loads(line)
        mode = open(req["path"]).read().strip()
        if mode == "hang":
            time.sleep(120)
        elif mode == "crash":
            os._exit(7)
        elif mode == "sigterm":
            os.kill(os.getpid(), signal.SIGTERM); time.sleep(5)
        elif mode == "bloat":
            while True:
                hold.append(b"x" * (20 * 1024 * 1024)); time.sleep(0.02)
        elif mode.startswith("flaky:"):
            marker = mode.split(":", 1)[1]
            if not os.path.exists(marker):
                open(marker, "w").write("1"); os._exit(9)
        elif mode == "error":
            send({"i": req["i"], "error": "ValueError: nope"}); continue
        send({"i": req["i"], "rec": {"score": 0.5, "up": "x", "conf": 0.9,
              "cal": True, "ms": 1, "guard": 0, "fam": {}}})
''')


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.stub = self.dir / "stub_worker.py"
        self.stub.write_text(STUB)
        self.scores = self.dir / "scores.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_modes(self, modes, **kw):
        rows = []
        for i, m in enumerate(modes):
            (self.dir / f"f{i}.txt").write_text(m)
            rows.append({"i": i, "label": i % 2, "role": "t"})
        logs: list = []
        stats, info = sup.score_all_isolated(
            rows, self.dir, self.scores, repo=self.dir,
            payload_path_fn=lambda d, i: d / f"f{i}.txt",
            worker_cmd=[sys.executable, str(self.stub)],
            log=logs.append, every=1,
            per_image_timeout_s=kw.pop("timeout", 3), **kw)
        recs = [json.loads(x) for x in self.scores.read_text().splitlines()]
        return stats, info, recs, logs

    def test_all_ok(self):
        stats, info, recs, _ = self.run_modes(["ok"] * 5)
        self.assertEqual(stats, {"scored": 5, "errors": 0, "guarded": 0})
        self.assertEqual(info["device_used"], "cpu")
        self.assertEqual(len(recs), 5)

    def test_error_result_is_recorded_not_fatal(self):
        stats, _, recs, _ = self.run_modes(["ok", "error", "ok"])
        self.assertEqual((stats["scored"], stats["errors"]), (2, 1))
        self.assertEqual(recs[1]["error"], "ValueError: nope")

    def test_sigterm_like_exit_143_is_contained(self):
        stats, info, recs, logs = self.run_modes(["ok", "sigterm", "ok", "ok"])
        self.assertEqual((stats["scored"], stats["errors"]), (3, 1))
        self.assertIn("signal 15", recs[1]["error"])
        self.assertTrue(any("WORKER LOST on i=1" in m for m in logs))
        self.assertEqual(info["worker_losses"], 2)  # tried twice, then skipped

    def test_hard_crash_is_contained(self):
        stats, _, recs, _ = self.run_modes(["crash", "ok", "ok"])
        self.assertEqual((stats["scored"], stats["errors"]), (2, 1))
        self.assertIn("exit code 7", recs[0]["error"])

    def test_hang_is_killed_by_timeout(self):
        stats, _, recs, _ = self.run_modes(["hang", "ok"], timeout=1)
        self.assertEqual((stats["scored"], stats["errors"]), (1, 1))
        self.assertIn("worker_timeout", recs[0]["error"])

    @unittest.skipUnless(Path("/proc/self/status").exists(), "needs /proc")
    def test_memory_hog_is_killed_before_it_grows_unbounded(self):
        stats, info, recs, _ = self.run_modes(
            ["ok", "bloat", "ok"], timeout=20, rss_limit_mb=250)
        self.assertEqual((stats["scored"], stats["errors"]), (2, 1))
        self.assertIn("worker_rss", recs[1]["error"])
        self.assertLess(info["peak_rss_mb"], 1500)  # never ran away

    def test_one_off_crash_is_retried_in_a_fresh_worker(self):
        marker = self.dir / "marker"
        stats, info, recs, _ = self.run_modes([f"flaky:{marker}", "ok"])
        self.assertEqual((stats["scored"], stats["errors"]), (2, 0))
        self.assertEqual(info["worker_losses"], 1)

    def test_recycle_replaces_worker(self):
        _, info, _, _ = self.run_modes(["ok"] * 7, recycle_every=3)
        self.assertEqual(info["respawns"], 3)  # 3 + 3 + 1

    def test_resume_skips_done_rows(self):
        self.run_modes(["ok"] * 3)
        stats, _, recs, _ = self.run_modes(["ok"] * 5)
        self.assertEqual(stats["scored"], 2)
        self.assertEqual(len(recs), 5)

    def test_broken_worker_fails_fast_instead_of_writing_garbage(self):
        bad = self.dir / "bad.py"
        bad.write_text("import sys; sys.exit(3)")
        (self.dir / "f0.txt").write_text("ok")
        with self.assertRaises(sup.WorkerStartError):
            sup.score_all_isolated(
                [{"i": 0, "label": 0}], self.dir, self.scores, repo=self.dir,
                payload_path_fn=lambda d, i: d / f"f{i}.txt",
                worker_cmd=[sys.executable, str(bad)], log=lambda m: None)


FAKE_ALUDAM = textwrap.dedent('''
    import os, signal
    from types import SimpleNamespace
    def fetch(names, quiet=True): pass
    class _Det:
        def predict(self, path):
            raw = open(path, "rb").read()
            if b"POISON" in raw:                      # simulate runner-killer
                os.kill(os.getpid(), signal.SIGTERM)
            from PIL import Image
            Image.open(path).convert("RGB")           # real decode
            return SimpleNamespace(score=0.7, label="fake", confidence=0.9,
                details={"family_scores": {"a": 0.1}}, runtime={"calibrated": True},
                elapsed_ms=3)
    def load_detector(name): return _Det()
''')


class RealWorkerEndToEnd(unittest.TestCase):
    """Real img_score_worker.py + real _header_guard, fake model package."""

    def test_poison_image_does_not_stop_the_run(self):
        try:
            import PIL  # noqa: F401, PLC0415
            from PIL import Image  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"PIL unavailable: {exc}")
        import os  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pkg = d / "fakepkg" / "aludam"
            pkg.mkdir(parents=True)
            (pkg / "__init__.py").write_text(FAKE_ALUDAM)
            Image.new("RGB", (64, 64), "red").save(d / "a.jpg")
            Image.new("RGB", (64, 64), "blue").save(d / "b.png")
            (d / "garbage.jpg").write_bytes(b"not an image at all")
            (d / "poison.jpg").write_bytes((d / "a.jpg").read_bytes() + b"POISON")  # valid image, passes the guard
            (d / "c.jpg").write_bytes((d / "a.jpg").read_bytes())
            names = ["a.jpg", "garbage.jpg", "poison.jpg", "b.png", "c.jpg"]
            rows = [{"i": i, "label": i % 2} for i in range(len(names))]
            old = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = str(d / "fakepkg") + (
                os.pathsep + old if old else "")
            try:
                stats, info = sup.score_all_isolated(
                    rows, d, d / "scores.jsonl", repo=d,
                    payload_path_fn=lambda dd, i: dd / names[i],
                    device="cpu", log=lambda m: None, every=1,
                    per_image_timeout_s=60)
            finally:
                if old is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old
            recs = [json.loads(x) for x in (d / "scores.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), 5)                      # nothing lost
        self.assertEqual(stats["scored"], 3)                # a, b, c
        self.assertIn("payload_unreadable", recs[1]["error"])
        self.assertIn("signal 15", recs[2]["error"])        # poison contained
        self.assertAlmostEqual(recs[4]["score"], 0.7)       # run continued
        self.assertEqual(info["device_used"], "cpu")


if __name__ == "__main__":
    unittest.main()
