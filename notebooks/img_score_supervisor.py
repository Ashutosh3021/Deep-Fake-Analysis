"""Crash-proof scoring: supervise an isolated worker instead of scoring in-process.

Why: on CI the score stage died mid-run (SIGTERM / exit 143) and took the whole
runner with it, so no progress was saved and every retry restarted from image 0.
Here the parent process stays tiny; the heavy decode + model run lives in a
worker that is

  * watched  - killed if its RSS passes a limit or one image takes too long,
  * retried  - a lost worker is replaced and the image retried once, to tell a
               poison file (dies twice) from a leak (dies once),
  * recycled - replaced every ``recycle_every`` images to bound memory growth,
  * logged   - every kill says which image, how big, and how much memory.

The JSONL record format is identical to ``img_eval_core.score_all``.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

RESULT_PREFIX = "@@RESULT@@ "
HERE = Path(__file__).resolve().parent


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def mem_total_mb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return None


def rss_mb(pid: int) -> Optional[float]:
    """Resident memory of ``pid`` in MB (Linux /proc, else psutil, else None)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil  # type: ignore

        return psutil.Process(pid).memory_info().rss / 1048576.0
    except Exception:  # noqa: BLE001
        return None


def default_rss_limit_mb() -> Optional[float]:
    """``SCORE_MAX_RSS_MB`` if set, else 55% of physical RAM, else no limit."""
    env = os.environ.get("SCORE_MAX_RSS_MB")
    if env:
        return float(env)
    total = mem_total_mb()
    return total * 0.55 if total else None


def _exit_text(rc: Optional[int]) -> str:
    if rc is None:
        return "exit code unknown"
    if rc < 0:
        return f"killed by signal {-rc}"
    return f"exit code {rc}"


class WorkerStartError(RuntimeError):
    pass


class Worker:
    """One child process speaking the JSON-lines protocol."""

    def __init__(self, cmd: Sequence[str], env: Dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        self.q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.peak_mb = 0.0
        self.last_mb = 0.0
        self.served = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                if line.startswith(RESULT_PREFIX):
                    try:
                        self.q.put(("res", json.loads(line[len(RESULT_PREFIX):])))
                        continue
                    except ValueError:
                        pass
                sys.stderr.write(line)  # stray library output stays visible
        finally:
            try:
                self.proc.stdout.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self.q.put(("eof", None))

    def _close_stdin(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass

    def request(self, payload: Optional[dict], *, timeout_s: float,
                rss_limit_mb: Optional[float]) -> Tuple[str, Any]:
        """Returns ('ok', obj) | ('timeout'|'rss'|'died', detail)."""
        if payload is not None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                return "died", _exit_text(self.proc.poll())
        t0 = time.time()
        while True:
            try:
                kind, val = self.q.get(timeout=0.1)
            except queue.Empty:
                kind, val = "", None
            if kind == "res":
                cur = rss_mb(self.proc.pid)  # always sample once per image
                if cur is not None:
                    self.last_mb = cur
                    self.peak_mb = max(self.peak_mb, cur)
                return "ok", val
            if kind == "eof":
                try:
                    rc = self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    rc = None
                return "died", _exit_text(rc)
            cur = rss_mb(self.proc.pid)
            if cur is not None:
                self.last_mb = cur
                self.peak_mb = max(self.peak_mb, cur)
                if rss_limit_mb and cur > rss_limit_mb:
                    self.kill()
                    return "rss", f"rss {cur:.0f} MB > limit {rss_limit_mb:.0f} MB"
            if time.time() - t0 > timeout_s:
                self.kill()
                return "timeout", f"no result after {timeout_s:.0f}s"

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self._close_stdin()

    def close(self) -> None:
        try:
            self._close_stdin()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.kill()


def _load_done(scores_path: Path) -> set:
    done: set = set()
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["i"])
            except Exception:  # noqa: BLE001 - tolerate trailing partial line
                continue
    return done


def score_all_isolated(
    selected: Sequence[Dict[str, Any]],
    payload_dir: Path,
    scores_path: Path,
    *,
    repo: Path,
    payload_path_fn: Callable[[Path, int], Optional[Path]],
    device: str = "auto",
    log: Callable[[str], None] = _log,
    every: int = 25,
    recycle_every: int = 250,
    per_image_timeout_s: float = 240.0,
    startup_timeout_s: float = 900.0,
    rss_limit_mb: Optional[float] = None,
    max_consecutive_failures: int = 15,
    worker_cmd: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Score ``selected`` through a supervised worker. Resumable JSONL output.

    Returns ``(stats, info)``; ``info`` carries device_used / notes / counters.
    """
    cmd = list(worker_cmd) if worker_cmd else [
        sys.executable, "-u", str(HERE / "img_score_worker.py"),
        str(repo), device]
    env = dict(os.environ)
    env.setdefault("ALUDAM_REPO", str(repo))
    env.setdefault("ALUDAM_MODELS_DIR", str(Path(repo) / "models"))
    env["PYTHONUNBUFFERED"] = "1"

    done = _load_done(scores_path)
    todo = [r for r in selected if int(r["i"]) not in done]
    log(f"[score] {len(todo)} to score, {len(done)} already done "
        f"(isolated worker; rss limit "
        f"{'none' if not rss_limit_mb else f'{rss_limit_mb:.0f} MB'}, "
        f"recycle every {recycle_every}, per-image timeout "
        f"{per_image_timeout_s:.0f}s)")
    stats = {"scored": 0, "errors": 0, "guarded": 0}
    info: Dict[str, Any] = {"device_used": None, "notes": [],
                            "worker_losses": 0, "respawns": 0,
                            "peak_rss_mb": 0.0}
    if not todo:
        return stats, info
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Optional[Worker]] = {"w": None}

    def spawn() -> Worker:
        last: Any = None
        for attempt in (1, 2):
            w = Worker(cmd, env)
            kind, val = w.request(None, timeout_s=startup_timeout_s,
                                  rss_limit_mb=None)
            if kind == "ok" and isinstance(val, dict) and val.get("ready"):
                info["device_used"] = val.get("device")
                info["notes"] = list(val.get("notes") or [])
                info["respawns"] += 1
                state["w"] = w
                return w
            last = (kind, val)
            w.kill()
            log(f"[score] worker failed to start (attempt {attempt}/2): {last}")
        raise WorkerStartError(f"scoring worker could not start: {last}")

    consec_fail = 0
    t0 = time.time()
    try:
        with scores_path.open("a", encoding="utf-8") as fh:
            for n, r in enumerate(todo, 1):
                i = int(r["i"])
                path = payload_path_fn(payload_dir, i)
                rec: Dict[str, Any] = {
                    "i": i, "y": int(r["label"]), "role": r.get("role"),
                    "model_name": r.get("model_name"),
                    "real_source": r.get("real_source"),
                }
                if path is None:
                    rec["error"] = "payload_missing"
                    stats["errors"] += 1
                else:
                    result: Optional[dict] = None
                    why = ("", "")
                    for attempt in (1, 2):
                        w = state["w"]
                        if w is not None and w.served >= recycle_every:
                            w.close()
                            state["w"] = w = None
                        if w is None:
                            w = spawn()
                        kind, val = w.request(
                            {"i": i, "path": str(path)},
                            timeout_s=per_image_timeout_s,
                            rss_limit_mb=rss_limit_mb)
                        w.served += 1
                        info["peak_rss_mb"] = max(info["peak_rss_mb"], w.peak_mb)
                        if kind == "ok" and isinstance(val, dict):
                            result = val
                            if rss_limit_mb and w.last_mb > 0.85 * rss_limit_mb:
                                # creeping toward the limit: replace the worker
                                # between images instead of waiting for a kill
                                log(f"[score] worker rss {w.last_mb:.0f} MB "
                                    f"near limit {rss_limit_mb:.0f} MB; "
                                    "recycling before next image")
                                w.served = recycle_every
                            break
                        info["worker_losses"] += 1
                        why = (kind, str(val))
                        try:
                            size = path.stat().st_size
                        except OSError:
                            size = -1
                        log(f"[score] WORKER LOST on i={i} "
                            f"(attempt {attempt}/2): {kind}: {val} | "
                            f"file {path.name} {size} bytes | worker peak "
                            f"rss {w.peak_mb:.0f} MB | served {w.served} "
                            f"images this worker")
                        w.kill()
                        state["w"] = None
                    if result is None:
                        rec["error"] = f"worker_{why[0]}: {why[1]}"
                        stats["errors"] += 1
                        consec_fail += 1
                        if consec_fail >= max_consecutive_failures:
                            raise RuntimeError(
                                f"{consec_fail} consecutive images lost their "
                                "worker; stopping instead of writing garbage "
                                "(the environment is broken, not the images)")
                    else:
                        consec_fail = 0
                        if "error" in result:
                            rec["error"] = result["error"]
                            stats["errors"] += 1
                        else:
                            rec.update(result["rec"])
                            stats["scored"] += 1
                            stats["guarded"] += 1 if rec.get("guard") else 0
                fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
                fh.flush()
                if n % every == 0 or n == len(todo):
                    rate = n / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - n) / max(rate, 1e-6)
                    w = state["w"]
                    cur = f"{w.last_mb:.0f}" if w else "-"
                    log(f"[score] {n}/{len(todo)}  {rate:.2f} img/s  "
                        f"eta {eta:.0f}s  rss={cur}MB "
                        f"peak={info['peak_rss_mb']:.0f}MB "
                        f"losses={info['worker_losses']} "
                        f"respawns={info['respawns']}")
    finally:
        w = state["w"]
        if w is not None:
            w.close()
    log(f"[score] done: {stats}")
    return stats, info
