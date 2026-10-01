"""Regression tests for gunicorn --preload fork/import corruption.

Production symptom (Render logs):

    Error loading image detector: cannot import name 'final_image_detector'
    from partially initialized module 'final_image_detector' (most likely due
    to a circular import)
    Could not load probe LM (partially initialized module 'numpy' has no
    attribute 'ndarray'). Falling back to stylometric-only mode.

Cause: under `gunicorn --preload` the app module is imported in the master and
its warm-up thread starts importing model modules *before* the worker forks.
The thread does not survive the fork, so the child inherits sys.modules
entries whose spec is still `_initializing` -- modules no thread will ever
finish. They must be dropped before anything imports them.

Every case runs in a fresh subprocess: the assertions are about process-wide
import state, and a child keeps the run single-threaded and deterministic.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")

PROLOGUE = (
    "import importlib.machinery, os, sys, threading, types\n"
    f"sys.path.insert(0, {BACKEND!r})\n"
    # Fake "gunicorn is running" so importing api does not start the heavy
    # warm-up thread: these tests assert import *wiring*, not model loading.
    "sys.modules['gunicorn'] = types.ModuleType('gunicorn')\n"
)


def run_child(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", PROLOGUE + body],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )


PARTIAL_FACTORY = """
def partial(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__spec__._initializing = True   # what an interrupted import looks like
    sys.modules[name] = mod
    return mod
"""


class WarmupWiringTests(unittest.TestCase):
    def test_no_warmup_thread_at_import_under_gunicorn(self):
        """The master must never warm up: its thread dies at the fork."""
        proc = run_child(
            "base = threading.active_count()\n"
            "import api\n"
            "assert threading.active_count() == base, 'warm-up started at import'\n"
            "assert api._warmup_state['pid'] is None, 'warm-up state set at import'\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_standalone_warmup_still_starts_at_import(self):
        """`python api.py` has no fork, so the eager warm-up is still wanted."""
        proc = run_child(
            "sys.modules.pop('gunicorn', None)\n"
            "import api\n"
            "assert api._warmup_state['pid'] == os.getpid(), 'no warm-up at import'\n"
            "assert api._fork_recovered is False, 'recovery must not run standalone'\n"
            "os._exit(0)\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_ensure_warmup_recovers_modules_orphaned_by_fork(self):
        """post_fork -> ensure_warmup drops the fork's half-imported modules."""
        proc = run_child(
            PARTIAL_FACTORY
            + "import api\n"
            "partial('stale_alpha')\n"
            "partial('stale_beta')\n"
            "api._warmup_models = lambda: None   # keep the child light\n"
            "api.ensure_warmup()\n"
            "assert 'stale_alpha' not in sys.modules, 'stale_alpha survived'\n"
            "assert 'stale_beta' not in sys.modules, 'stale_beta survived'\n"
            "assert api._fork_recovered is True\n"
            "assert api._warmup_state['pid'] == os.getpid()\n"
            "api.ensure_warmup()   # second call must be a no-op\n"
            "assert api._fork_recovered is True\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class ImportRetryTests(unittest.TestCase):
    def test_retry_recovers_the_module_being_imported(self):
        """A half-imported target is dropped and re-imported on retry."""
        probe = os.path.join(ROOT, "tests", "_stale_probe.py")
        with open(probe, "w") as fh:
            fh.write("VALUE = 'recovered'\n")
        try:
            proc = run_child(
                PARTIAL_FACTORY
                + "import api\n"
                f"sys.path.insert(0, {os.path.join(ROOT, 'tests')!r})\n"
                "partial('_stale_probe')\n"
                "value = api._import_model('_stale_probe', 'VALUE')\n"
                "assert value == 'recovered', value\n"
                "assert sys.modules['_stale_probe'] is not None\n"
                "assert getattr(sys.modules['_stale_probe'].__spec__, "
                "'_initializing', False) is False, 'still half-imported'\n"
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        finally:
            if os.path.exists(probe):
                os.remove(probe)

    def test_retry_never_touches_modules_it_does_not_own(self):
        """Purging a module some other thread is importing breaks the import
        system (KeyError in _find_and_load), so only owned modules are eligible."""
        proc = run_child(
            PARTIAL_FACTORY
            + "import api\n"
            "partial('unrelated_probe')\n"
            "try:\n"
            "    api._import_model('definitely_missing_module_xyz', 'x')\n"
            "except ModuleNotFoundError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('missing module did not raise')\n"
            "assert 'unrelated_probe' in sys.modules, 'unrelated module purged'\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_clean_process_reports_no_partial_modules(self):
        proc = run_child(
            "import api\n"
            "assert api._partial_module_names() == [], "
            "str(api._partial_module_names())\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class LoaderTests(unittest.TestCase):
    def test_image_detector_loads_through_the_loader(self):
        """End to end: the real (locked, retrying) loader path returns a model."""
        proc = run_child(
            "import api\n"
            "detector = api.get_image_detector()\n"
            "assert detector is not None, 'get_image_detector returned None'\n"
            "assert type(detector).__name__ == 'FinalImageDetector'\n"
            "assert api.get_image_detector() is detector, 'not cached'\n"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
