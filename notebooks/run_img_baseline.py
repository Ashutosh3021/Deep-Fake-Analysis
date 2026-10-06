"""Run the img_det baseline end to end and write notebooks/result.txt.

Local stand-in for img_det_baseline.ipynb (identical steps and config) for
environments without Jupyter. Colab users should run the notebook instead.

Usage:
    .venv\\Scripts\\python.exe -u notebooks\\run_img_baseline.py

Resumable at every stage - just re-run if interrupted.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CONFIG = dict(
    per_class=750,     # rows per class (plan default 750)
    seed=20261004,     # selection + split seed
    device="auto",     # "auto" = CUDA when available, else CPU
    workers=8,         # parallel metadata readers (HF-friendly)
    score_every=25,    # progress log interval, in images
)


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

print("repo:", REPO)
print("config:", CONFIG)

# Step 1: metadata-only index (payload column never read).
index_rows = core.build_index(REPO, workers=CONFIG["workers"])
print("indexed rows:", len(index_rows))

# Step 2: label evidence + source-grouped dev/cal/test split.
selected, splits, evidence, warns = core.select_and_split(
    index_rows, per_class=CONFIG["per_class"], seed=CONFIG["seed"])
print("selected:", len(selected), "| warnings:", len(warns))

# Step 3: download only the selected images (resumable).
PAYLOAD_DIR = REPO / "aludam" / "eval" / "data" / "index" / "img_payload"
payload_stats = core.fetch_payload(selected, PAYLOAD_DIR)
print("payload:", payload_stats)

# Step 4: load aludam img_det (weights fetched once if missing).
detector, device_used, device_notes = core.get_detector(CONFIG["device"])
print("device:", device_used)
for n in device_notes:
    print(" -", n)

# Step 5: score every selected image (resumable JSONL).
SCORES_PATH = REPO / "aludam" / "eval" / "data" / "index" / "img_scores.jsonl"
score_stats = core.score_all(detector, selected, PAYLOAD_DIR, SCORES_PATH,
                             every=CONFIG["score_every"])
print("scores:", score_stats)

# Step 6: metrics -> diagnosis -> dual-budget fit -> test verify.
OUT_PATH = REPO / "notebooks" / "result.txt"
text = core.analyze_and_write(
    repo=REPO, selected=selected, splits=splits, evidence=evidence,
    warnings_=warns, payload_stats=payload_stats, score_stats=score_stats,
    scores_path=SCORES_PATH, out_path=OUT_PATH,
    per_class=CONFIG["per_class"], seed=CONFIG["seed"],
    device_used=device_used, device_notes=device_notes)
print(text)
print("RESULT FILE:", OUT_PATH)
