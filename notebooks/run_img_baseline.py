"""Run the img_det baseline and write notebooks/result.txt.

Local stand-in for img_det_baseline.ipynb (identical steps and config).

Usage:
    .venv\\Scripts\\python.exe -u notebooks\\run_img_baseline.py            # all steps
    .venv\\Scripts\\python.exe -u notebooks\\run_img_baseline.py payload    # one step
    .venv\\Scripts\\python.exe -u notebooks\\run_img_baseline.py score
    .venv\\Scripts\\python.exe -u notebooks\\run_img_baseline.py analyze

Steps: index (implicit, cache-backed) -> payload -> score -> analyze.
Every step is resumable from disk; step stats persist in a state file so
the stages can run as separate processes (GitHub Actions checkpoints).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

CONFIG = dict(
    per_class=750,     # rows per class (plan default 750)
    seed=20261004,     # selection + split seed
    device="auto",     # "auto" = CUDA when available, else CPU
    workers=8,         # parallel metadata readers (HF-friendly)
    score_every=25,    # progress log interval, in images
)

VALID_STEPS = ("index", "payload", "score", "analyze")


def _find_repo() -> Path:
    for c in [Path.cwd(), *Path.cwd().parents]:
        if ((c / "aludam").is_dir() and (c / "models").is_dir()
                and (c / "notebooks").is_dir()):
            return c
    raise SystemExit(
        "Repo root not found. Run from inside the Deep-Fake-Analysis folder.")


REPO = _find_repo()

import os  # noqa: E402

os.environ["ALUDAM_REPO"] = str(REPO)
os.environ["ALUDAM_MODELS_DIR"] = str(REPO / "models")
sys.path.insert(0, str(REPO / "notebooks"))

import img_eval_core as core  # noqa: E402

NEED = {"numpy": "numpy", "pandas": "pandas", "PIL": "Pillow",
        "cv2": "opencv-python", "pyarrow": "pyarrow", "fsspec": "fsspec",
        "sklearn": "scikit-learn", "torch": "torch",
        "transformers": "transformers", "aludam": "aludam"}
missing = [p for m, p in NEED.items() if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        "Missing packages: %s\nInstall first, e.g.\n  pip install %s"
        % (missing, " ".join(missing)))

DATA_DIR = REPO / "aludam" / "eval" / "data" / "index"
PAYLOAD_DIR = DATA_DIR / "img_payload"
SCORES_PATH = DATA_DIR / "img_scores.jsonl"
STATE_PATH = DATA_DIR / "img_run_state.json"
OUT_PATH = REPO / "notebooks" / "result.txt"


def state_save(**updates: object) -> None:
    state: Dict[str, Any] = {}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - rebuild
            state = {}
    state.update(updates)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def state_load() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def fs_stats(selected) -> Dict[str, Any]:
    """Fallback stats derived from disk when the state file is missing."""
    n_files = sum(1 for r in selected
                  if any(PAYLOAD_DIR.glob(f"off{int(r['i']):06d}.*")))
    payload = {"cached": n_files,
               "failed": max(len(selected) - n_files, 0)}
    scored = errors = guarded = 0
    if SCORES_PATH.exists():
        for line in SCORES_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "score" in rec:
                scored += 1
                guarded += 1 if rec.get("guard") else 0
            else:
                errors += 1
    return {"payload_stats": payload,
            "score_stats": {"scored": scored, "errors": errors,
                            "guarded": guarded}}


def prepare():
    """Index (cache-backed) + selection. Deterministic; cheap after step 1."""
    index_rows = core.build_index(REPO, workers=CONFIG["workers"])
    print("indexed rows:", len(index_rows))
    selected, splits, evidence, warns = core.select_and_split(
        index_rows, per_class=CONFIG["per_class"], seed=CONFIG["seed"])
    print("selected:", len(selected), "| warnings:", len(warns))
    return selected, splits, evidence, warns


def do_payload(selected) -> Dict[str, int]:
    stats = core.fetch_payload(selected, PAYLOAD_DIR)
    print("payload:", stats)
    state_save(payload_stats=stats)
    return stats


def do_score(selected) -> Dict[str, Any]:
    detector, device_used, device_notes = core.get_detector(CONFIG["device"])
    print("device:", device_used)
    for n in device_notes:
        print(" -", n)
    stats = core.score_all(detector, selected, PAYLOAD_DIR, SCORES_PATH,
                           every=CONFIG["score_every"])
    print("scores:", stats)
    state_save(score_stats=stats, device_used=device_used,
               device_notes=list(device_notes))
    return {"score_stats": stats, "device_used": device_used,
            "device_notes": device_notes}


def do_analyze(selected, splits, evidence, warns) -> str:
    st = state_load()
    fb = fs_stats(selected)
    payload_stats = st.get("payload_stats") or fb["payload_stats"]
    score_stats = st.get("score_stats") or fb["score_stats"]
    device_used = st.get("device_used") or "unknown"
    device_notes = st.get("device_notes") or []
    text = core.analyze_and_write(
        repo=REPO, selected=selected, splits=splits, evidence=evidence,
        warnings_=warns, payload_stats=payload_stats, score_stats=score_stats,
        scores_path=SCORES_PATH, out_path=OUT_PATH,
        per_class=CONFIG["per_class"], seed=CONFIG["seed"],
        device_used=device_used, device_notes=device_notes)
    print(text)
    print("RESULT FILE:", OUT_PATH)
    return text


def main() -> None:
    args = [a.lower() for a in sys.argv[1:]] or ["all"]
    if args == ["all"]:
        args = ["payload", "score", "analyze"]
    bad = [a for a in args if a not in VALID_STEPS]
    if bad:
        raise SystemExit(f"unknown step(s) {bad}; valid: {VALID_STEPS} or 'all'")
    if "index" in args and len(args) == 1:
        args = []  # index alone = just build+select, nothing else to do

    print("repo:", REPO)
    print("config:", CONFIG, "| steps:", args or ["index"])
    selected, splits, evidence, warns = prepare()
    if "payload" in args:
        do_payload(selected)
    if "score" in args:
        do_score(selected)
    if "analyze" in args:
        do_analyze(selected, splits, evidence, warns)


if __name__ == "__main__":
    main()
