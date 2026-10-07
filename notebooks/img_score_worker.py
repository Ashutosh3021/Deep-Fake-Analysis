"""Scoring worker: runs in its own process so a bad image can only kill *it*.

Spawned by ``img_score_supervisor``. Protocol (one JSON object per line):

    worker -> parent   ``@@RESULT@@ {"ready": true, "device": ..., "notes": [...]}``
    parent -> worker   ``{"i": 12, "path": "/.../off000012.jpg"}``
    worker -> parent   ``@@RESULT@@ {"i": 12, "rec": {...}}``  or  ``{"i": 12, "error": "..."}``

Anything a library prints to stdout is redirected to stderr, so it can never
corrupt the protocol channel. EOF on stdin = clean shutdown.

Usage (internal): python img_score_worker.py <repo_root> <device>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESULT_PREFIX = "@@RESULT@@ "


def main() -> int:
    repo, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "auto")

    proto = sys.stdout          # the real stdout is the protocol channel
    sys.stdout = sys.stderr     # stray prints from libraries go to stderr

    def send(obj: dict) -> None:
        proto.write(RESULT_PREFIX + json.dumps(obj, ensure_ascii=True) + "\n")
        proto.flush()

    os.environ.setdefault("ALUDAM_REPO", repo)
    os.environ.setdefault("ALUDAM_MODELS_DIR", str(Path(repo) / "models"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import img_eval_core as core  # noqa: E402

    detector, chosen, notes = core.get_detector(device)
    send({"ready": True, "device": chosen, "notes": list(notes)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        i, path = req["i"], Path(req["path"])
        try:
            guard_err = core._header_guard(path)  # noqa: SLF001
            if guard_err:
                send({"i": i, "error": guard_err})
                continue
            res = detector.predict(str(path))
            fam_raw = dict(res.details.get("family_scores") or {})
            fam = {k: v for k, v in fam_raw.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
            guarded = res.details.get("reason") == "input_below_model_minimum"
            send({"i": i, "rec": {
                "score": float(res.score),
                "up": res.label,
                "conf": float(res.confidence),
                "cal": bool(res.runtime.get("calibrated")),
                "ms": int(res.elapsed_ms),
                "guard": 1 if guarded else 0,
                "fam": fam,
            }})
        except Exception as exc:  # noqa: BLE001 - reported, worker stays alive
            send({"i": i, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
