"""aludam img_det baseline evaluation engine.

Runs the Phase-0 image path end to end, with no training and no package
changes:

    1. build_index      - metadata-only index of CommunityForensics-Eval
    2. select_and_split - label evidence + source-grouped dev/cal/test split
    3. fetch_payload    - download only the selected rows (datasets-server)
    4. get_detector     - load aludam img_det (weights fetched if missing)
    5. score_all        - score every selected row, resumable JSONL
    6. analyze_and_write- AUROC / per-signal diagnosis / dual-budget band /
                          one-shot test verify -> result.txt

Used by notebooks/img_det_baseline.ipynb (local CPU) and
notebooks/img_det_baseline_colab.ipynb (Colab GPU). Both write the same
result.txt contract so the two runs are comparable.

Eval-set stamp (must appear in every result): this measures general
deepfake imagery; it does NOT validate government-ID / KYC behaviour.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants - the written acceptance gate (decided before any fitting ran).
# ---------------------------------------------------------------------------

DATASET_REPO = "OwensLab/CommunityForensics-Eval"
DATASET_CONFIG = "default"
DATASET_SPLIT = "CompEval"
SYNTH_LABEL = 1                  # 1 = synthetic (verified by evidence, below)
REPO_URL = "https://github.com/Ashutosh3021/Deep-Fake-Analysis"

PER_CLASS = 750                  # rows per class to sample
SEED = 20261004
SPLIT_FRACS: Dict[str, float] = {"dev": 0.60, "cal": 0.20, "test": 0.20}
FNR_BUDGET = 0.01                # auto-accept path: <=1% of fakes accepted
FPR_BUDGET = 0.05                # auto-reject path: <=5% of reals rejected
CI_Z = 1.96                      # Wilson upper bound, 95%
BOOTSTRAP_N = 500
MIN_PER_SPLIT = 10               # smallest usable split, per class

SIGNAL_KEYS = [
    "model_signal_raw",
    "global_heuristic",
    "fully_ai_generated",
    "provenance",
    "ai_edited_region",
    "face_swap",
    "fused_score",
    "relative_evidence",
]

INDEX_COLS = [
    "image_name", "format", "model_name", "real_source", "subset",
    "split", "label", "architecture", "nsfw_flag",
]

NON_TRANSFER_STAMP = (
    "STAMP: general-deepfake distribution; NOT validated for "
    "government-ID / KYC scans (checkpoint card disclaims screenshots "
    "and general computer imagery)."
)


def _log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Repo / HTTP helpers
# ---------------------------------------------------------------------------

def find_repo() -> Path:
    """Locate the repo root (the folder holding aludam/ and models/)."""
    cands: List[Path] = []
    env = os.environ.get("ALUDAM_REPO")
    if env:
        cands.append(Path(env))
    try:
        here = Path(__file__).resolve().parent          # .../notebooks
        cands.append(here.parent)
    except NameError:                                    # pragma: no cover
        pass
    cands.append(Path.cwd())
    cands.extend(Path.cwd().parents)
    seen = set()
    for c in cands:
        c = Path(c)
        if c in seen:
            continue
        seen.add(c)
        if (c / "aludam").is_dir() and (c / "models").is_dir():
            return c
    raise RuntimeError(
        "repo root not found (need aludam/ and models/ side by side). "
        "Set ALUDAM_REPO or run from inside the repo."
    )


def _http_json(url: str, timeout: int = 60, tries: int = 6) -> Any:
    last: Optional[Exception] = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "aludam-img-eval/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last = exc
            msg = str(exc)
            # 429s come from HF rate limiting - back off much harder.
            base = 10.0 if ("429" in msg or "Too Many" in msg) else 1.5
            time.sleep(min(60.0, base * (2 ** attempt)) + random.random())
    raise RuntimeError(f"GET failed after {tries} tries: {url}\n{last}")


def resolve_revision() -> str:
    info = _http_json(f"https://huggingface.co/api/datasets/{DATASET_REPO}")
    sha = info.get("sha")
    if not sha or len(sha) < 40:
        raise RuntimeError(f"could not resolve dataset revision: {sha!r}")
    return sha


def _rows_url(offset: int, length: int = 1) -> str:
    return (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(DATASET_REPO)}"
        f"&config={DATASET_CONFIG}&split={DATASET_SPLIT}"
        f"&offset={offset}&length={length}"
    )


# ---------------------------------------------------------------------------
# Step 1 - metadata-only index (payload column never requested)
# ---------------------------------------------------------------------------

def _read_shard_meta(repo_url: str, sha: str, shard: str) -> List[Dict[str, Any]]:
    import warnings

    warnings.filterwarnings("ignore")
    import pyarrow.parquet as pq
    from fsspec.core import url_to_fs

    url = f"https://huggingface.co/datasets/{repo_url}/resolve/{sha}/{shard}"
    fs, _ = url_to_fs(url, block_size=16 << 20)
    table = pq.read_table(url, columns=INDEX_COLS, filesystem=fs)
    return table.to_pylist()


def build_index(
    repo: Path,
    *,
    shard_limit: Optional[int] = None,
    shards: Optional[Sequence[str]] = None,
    workers: int = 8,
    log: Callable[[str], None] = _log,
) -> List[Dict[str, Any]]:
    """Read shard metadata, assign global row offsets, cache to disk.

    ``shards`` restricts to an explicit list (offsets stay global, so the
    datasets-server mapping remains correct); otherwise every shard up to
    ``shard_limit`` is read. Returns rows sorted by global offset ``i``.
    Each row: ``i``, ``shard``, plus the metadata columns. Resumable per
    shard (one JSON file per shard).
    """
    from concurrent.futures import ThreadPoolExecutor

    sha = resolve_revision()
    tree = _http_json(
        f"https://huggingface.co/api/datasets/{DATASET_REPO}/tree/main?recursive=true"
    )
    all_shards = sorted(
        t["path"] for t in tree
        if t.get("type") == "file"
        and t["path"].startswith("data/CompEval-")
        and t["path"].endswith(".parquet")
    )
    if shards is not None:
        missing = [s for s in shards if s not in all_shards]
        if missing:
            raise RuntimeError(f"unknown shard(s): {missing}")
        chosen = set(shards)
        sel = [s for s in all_shards if s in chosen]
    elif shard_limit:
        sel = all_shards[:shard_limit]
    else:
        sel = all_shards
    shard_idx = {s: int(s.split("-")[1]) for s in sel}
    log(f"[index] revision {sha[:12]}  shards: {len(sel)} of {len(all_shards)}")

    cache_dir = repo / "aludam" / "eval" / "data" / "index" / "image_meta"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(shard: str) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
        idx = shard_idx[shard]
        cache = cache_dir / f"{idx:05d}.json"
        if cache.exists():
            try:
                return shard, json.loads(cache.read_text(encoding="utf-8")), None
            except Exception:  # corrupt cache - refetch
                pass
        try:
            rows = _read_shard_meta(DATASET_REPO, sha, shard)
        except Exception as exc:  # noqa: BLE001
            return shard, None, f"{type(exc).__name__}: {exc}"
        cache.write_text(json.dumps(rows), encoding="utf-8")
        return shard, rows, None

    results: Dict[str, List[Dict[str, Any]]] = {}
    failures: List[Tuple[str, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = 0
        for shard, rows, err in pool.map(one, sel):
            done += 1
            if err is not None:
                failures.append((shard, err))
                log(f"[index]   FAIL {shard}: {err}")
            else:
                assert rows is not None
                results[shard] = rows
            if done % 25 == 0 or done == len(sel):
                log(f"[index] {done}/{len(sel)} shards  ({time.time()-t0:.0f}s)")
    if failures:
        raise RuntimeError(
            f"{len(failures)} shard(s) failed to index (e.g. {failures[0]}). "
            "Delete the per-shard cache only if the failure persists, then re-run."
        )

    # Global offsets need row counts for every shard BEFORE the last
    # selected one (offsets are cumulative). Footer-only reads are cheap
    # and cached once; shards after the last selected one never matter.
    last_pos = max(all_shards.index(s) for s in sel)
    counts_path = cache_dir / "row_counts.json"
    row_counts: Dict[str, int] = {}
    if counts_path.exists():
        try:
            row_counts = json.loads(counts_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - refetch
            row_counts = {}
    needed = [
        s for i, s in enumerate(all_shards)
        if i < last_pos and s not in results and s not in row_counts
    ]
    if needed:
        log(f"[index] footer pass: {len(needed)} skipped shard(s) to count")

        def _count(shard: str) -> Tuple[str, Optional[int], Optional[str]]:
            for attempt in range(3):
                try:
                    import warnings as _w

                    _w.filterwarnings("ignore")
                    import pyarrow.parquet as _pq
                    from fsspec.core import url_to_fs as _u2f

                    url = (f"https://huggingface.co/datasets/{DATASET_REPO}"
                           f"/resolve/{sha}/{shard}")
                    fs, _ = _u2f(url, block_size=16 << 20)
                    return shard, int(_pq.ParquetFile(url, filesystem=fs)
                                      .metadata.num_rows), None
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        return shard, None, f"{type(exc).__name__}: {exc}"
                    time.sleep(1.5 * (attempt + 1))
            return shard, None, "unreachable"  # pragma: no cover

        count_fail = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for shard, n, err in pool.map(_count, needed):
                if err is not None or n is None:
                    log(f"[index]   count FAIL {shard}: {err}")
                    count_fail = True
                else:
                    row_counts[shard] = n
        if count_fail:
            raise RuntimeError("row-count footer pass failed; re-run to retry.")
        counts_path.write_text(json.dumps(row_counts), encoding="utf-8")

    out: List[Dict[str, Any]] = []
    offset = 0
    for shard in all_shards:
        if shard in results:
            for local_i, row in enumerate(results[shard]):
                rec = {"i": offset + local_i, "shard": shard}
                rec.update(row)
                out.append(rec)
            offset += len(results[shard])
        else:
            offset += int(row_counts.get(shard, 0))
    log(f"[index] {len(out)} rows indexed (global offsets 0..{len(out)-1})")
    return out


# ---------------------------------------------------------------------------
# Step 2 - label evidence + grouped split
# ---------------------------------------------------------------------------

def _crosstab(rows: Sequence[Dict[str, Any]], col: str) -> Dict[str, Dict[str, int]]:
    tab: Dict[str, Dict[str, int]] = {}
    for r in rows:
        key = str(r.get(col))
        lab = str(r.get("label"))
        tab.setdefault(key, {"0": 0, "1": 0})[lab] += 1
    return tab


def _group_of(row: Dict[str, Any]) -> str:
    if int(row["label"]) == SYNTH_LABEL:
        return f"fake:{row.get('model_name') or '?'}"
    return f"real:{row.get('real_source') or '?'}"


def _hash_split(name: str, seed: int) -> str:
    digest = hashlib.md5(f"{seed}:{name}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 10000
    if bucket < SPLIT_FRACS["dev"] * 10000:
        return "dev"
    if bucket < (SPLIT_FRACS["dev"] + SPLIT_FRACS["cal"]) * 10000:
        return "cal"
    return "test"


def _assign_groups(
    groups: Dict[str, List[Dict[str, Any]]], seed: int
) -> Dict[str, str]:
    """Greedy fill: biggest remaining deficit first, whole groups kept intact."""
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = sum(len(v) for v in groups.values())
    deficit = {s: frac * total for s, frac in SPLIT_FRACS.items()}
    assign: Dict[str, str] = {}
    for name in sorted(names, key=lambda n: -len(groups[n])):
        target = max(deficit, key=lambda s: deficit[s])
        assign[name] = target
        deficit[target] -= len(groups[name])
    return assign


def select_and_split(
    rows: Sequence[Dict[str, Any]],
    *,
    per_class: int = PER_CLASS,
    seed: int = SEED,
    log: Callable[[str], None] = _log,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Any], List[str]]:
    """Sample ``per_class`` rows per class and split dev/cal/test by source.

    Group rule (written before looking at any scores): synthetic rows group
    by generator (``model_name``), authentic rows by origin (``real_source``);
    whole groups are assigned to one split, so no generator or source crosses
    the boundary. If the group structure cannot fill a split, that class
    falls back to a per-image hash split (recorded as a warning).

    Returns (selected_rows, splits, evidence, warnings).
    """
    warnings_: List[str] = []
    labels = {int(r["label"]) for r in rows}
    if not labels <= {0, 1}:
        raise RuntimeError(f"unexpected label values {sorted(labels)}; aborting")

    evidence: Dict[str, Any] = {
        "total_rows": len(rows),
        "label_counts": {str(l): sum(1 for r in rows if int(r["label"]) == l)
                         for l in sorted(labels)},
        "label_x_format": _crosstab(rows, "format"),
        "label_x_architecture": _crosstab(rows, "architecture"),
        "convention": f"label=={SYNTH_LABEL} treated as synthetic",
        "convention_note": (
            "Verified pattern: synthetic rows are generator output (PNG, "
            "architecture=GAN/...), authentic rows are source photos (JPEG); "
            "matches the HF label convention 1=positive=synthetic."
        ),
    }
    log(f"[select] label evidence: {json.dumps(evidence['label_counts'])}")
    log(f"[select] label x format: {json.dumps(evidence['label_x_format'])}")

    by_class: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for r in rows:
        by_class[int(r["label"])].append(r)

    selected: List[Dict[str, Any]] = []
    splits: Dict[str, List[Dict[str, Any]]] = {"dev": [], "cal": [], "test": []}
    rng = random.Random(seed)

    for cls in (SYNTH_LABEL, 0):
        pool = list(by_class[cls])
        rng.shuffle(pool)
        if len(pool) > per_class:
            pool = pool[:per_class]
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in pool:
            groups.setdefault(_group_of(r), []).append(r)

        assignment: Optional[Dict[str, str]] = None
        if len(groups) >= 3:
            assignment = _assign_groups(groups, seed + cls)
            counts = {s: 0 for s in splits}
            for gname, grow in groups.items():
                counts[assignment[gname]] += len(grow)
            if any(counts[s] < MIN_PER_SPLIT for s in splits):
                assignment = None
                warnings_.append(
                    f"class {cls}: group split leaves a split with <"
                    f"{MIN_PER_SPLIT} rows {counts}; falling back to per-image "
                    "hash split (source-leakage risk recorded)."
                )
        else:
            warnings_.append(
                f"class {cls}: only {len(groups)} source group(s) "
                f"({sorted(groups)}); falling back to per-image hash split."
            )

        if assignment is None:
            for r in pool:
                key = str(r.get("image_name"))
                splits[_hash_split(key, seed + cls)].append(r)
        else:
            for gname, grow in groups.items():
                splits[assignment[gname]].extend(grow)
        selected.extend(pool)

    role_of = {r["i"]: s for s in splits for r in splits[s]}
    selected = [dict(r, role=role_of[r["i"]]) for r in selected]
    splits = {s: [dict(r, role=s) for r in splits[s]] for s in splits}

    counts = {s: {"synth": sum(1 for r in splits[s] if int(r["label"]) == SYNTH_LABEL),
                  "real": sum(1 for r in splits[s] if int(r["label"]) == 0)}
              for s in splits}
    for s in splits:
        for c in ("synth", "real"):
            if counts[s][c] == 0:
                raise RuntimeError(
                    f"split '{s}' has 0 {c} rows - cannot fit or verify. "
                    "Increase per_class or inspect group structure "
                    f"(warnings: {warnings_})."
                )
    evidence["split_counts"] = counts
    evidence["groups"] = {
        "synth": sorted({_group_of(r) for r in selected
                         if int(r["label"]) == SYNTH_LABEL}),
        "real": sorted({_group_of(r) for r in selected
                        if int(r["label"]) == 0}),
    }
    log(f"[select] selected {len(selected)} rows; splits: {json.dumps(counts)}")
    for w in warnings_:
        log(f"[select] WARNING: {w}")
    return selected, splits, evidence, warnings_


# ---------------------------------------------------------------------------
# Step 3 - payload fetch for the selected rows only
# ---------------------------------------------------------------------------

def _ext_for(fmt: str) -> str:
    return {"JPEG": ".jpg", "PNG": ".png"}.get((fmt or "").upper(), ".img")


def fetch_payload(
    selected: Sequence[Dict[str, Any]],
    cache_dir: Path,
    *,
    log: Callable[[str], None] = _log,
    pause: float = 0.05,
) -> Dict[str, int]:
    """Download image bytes for the selected rows via the datasets-server.

    One row per request (sparse offsets); resumable - files already on disk
    are skipped. Verifies ``image_name`` match before writing, so an offset
    mismatch aborts instead of scoring the wrong image.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "cached": 0, "failed": 0}
    todo = [r for r in selected
            if not any(cache_dir.glob(f"off{int(r['i']):06d}.*"))]
    stats["cached"] = len(selected) - len(todo)
    log(f"[payload] {len(todo)} to download, {stats['cached']} already cached")
    if not todo:
        return stats
    t0 = time.time()
    cur_pause = pause
    todo_pass2: List[Dict[str, Any]] = []

    def grab(r: Dict[str, Any]) -> Optional[str]:
        i = int(r["i"])
        try:
            data = _http_json(_rows_url(i, 1), timeout=90, tries=5)
            row = data["rows"][0]["row"]
            if str(row.get("image_name")) != str(r.get("image_name")):
                return (f"offset drift at i={i}: expected "
                        f"{r.get('image_name')!r}, server returned "
                        f"{row.get('image_name')!r}")
            raw = base64.b64decode(row.get("image_data") or "")
            if not raw:
                return f"empty image_data at i={i}"
            out = cache_dir / f"off{i:06d}{_ext_for(str(r.get('format')))}"
            out.write_bytes(raw)
            return None
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"

    for pass_no in (1, 2):
        failed_idx: List[Dict[str, Any]] = []
        items = todo if pass_no == 1 else todo_pass2
        if not items:
            break
        if pass_no == 2:
            cur_pause = max(cur_pause * 2, 0.5)
            log(f"[payload] retry pass with pause={cur_pause:.2f}s "
                 f"({len(items)} row(s))")
        for n, r in enumerate(items, 1):
            err = grab(r)
            if err is None:
                stats["downloaded"] += 1
            elif pass_no == 1 and ("429" in err or "Too Many" in err):
                failed_idx.append(r)  # likely rate-limit -> pass 2
            else:
                stats["failed"] += 1
                log(f"[payload] FAIL i={r['i']}: {err}")
            if n % 50 == 0 or n == len(items):
                rate = n / max(time.time() - t0, 1e-6)
                eta = (len(items) - n) / max(rate, 1e-6)
                log(f"[payload] p{pass_no} {n}/{len(items)}  eta {eta:.0f}s")
            time.sleep(cur_pause)
        todo_pass2 = failed_idx
    if todo_pass2:
        stats["failed"] += len(todo_pass2)
        for r in todo_pass2:
            log(f"[payload] FAIL i={r['i']}: still failing after retry pass")
    log(f"[payload] done: {stats}")
    return stats


def _payload_path(cache_dir: Path, i: int) -> Optional[Path]:
    hits = sorted(cache_dir.glob(f"off{int(i):06d}.*"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Step 4 - detector
# ---------------------------------------------------------------------------

def get_detector(
    device: str = "auto",
    *,
    log: Callable[[str], None] = _log,
) -> Tuple[Any, str, List[str]]:
    """Load aludam img_det, fetching weights if missing.

    ``device``: ``auto`` (CUDA when available) / ``cuda`` / ``cpu``. The CUDA
    path rebuilds the internal HF pipeline on the GPU from the notebook side;
    no package file is modified.
    """
    notes: List[str] = []
    from aludam import fetch as aludam_fetch
    from aludam import load_detector

    log("[detector] checking weights ...")
    aludam_fetch(["img_det"], quiet=True)
    detector = load_detector("img_det")
    log("[detector] loaded img_det")

    chosen = device
    if device == "auto":
        try:
            import torch
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            chosen = "cpu"
    if chosen == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")
            mod = sys.modules.get("models.final_image_detector")
            if mod is None:
                import models.final_image_detector as mod  # noqa: PLC0415
            from transformers import pipeline

            mod.final_image_detector._pipeline = pipeline(
                "image-classification",
                model=mod.PRIMARY_MODEL_ID,
                framework="pt",
                device=0,
            )
            notes.append("GPU override: HF pipeline rebuilt on device 0 "
                         "(notebook-level; package files unchanged).")
            log(f"[detector] {notes[-1]}")
        except Exception as exc:  # noqa: BLE001 - fall back, record honestly
            chosen = "cpu"
            notes.append(f"GPU override failed ({exc}); running CPU.")
            log(f"[detector] WARNING: {notes[-1]}")
    else:
        notes.append("CPU execution (package default, device=-1).")
    return detector, chosen, notes


# ---------------------------------------------------------------------------
# Step 5 - score
# ---------------------------------------------------------------------------

def score_all(
    detector: Any,
    selected: Sequence[Dict[str, Any]],
    payload_dir: Path,
    scores_path: Path,
    *,
    log: Callable[[str], None] = _log,
    every: int = 25,
) -> Dict[str, int]:
    """Score every selected row through ``detector.predict``; resumable JSONL.

    Record: i, y, role, score, upstream label, confidence, calibrated flag,
    elapsed ms, guard flag, numeric family scores.
    """
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
    todo = [r for r in selected if int(r["i"]) not in done]
    log(f"[score] {len(todo)} to score, {len(done)} already done")
    stats = {"scored": 0, "errors": 0, "guarded": 0}
    if not todo:
        return stats
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with scores_path.open("a", encoding="utf-8") as fh:
        for n, r in enumerate(todo, 1):
            i = int(r["i"])
            path = _payload_path(payload_dir, i)
            rec: Dict[str, Any] = {
                "i": i,
                "y": int(r["label"]),
                "role": r.get("role"),
                "model_name": r.get("model_name"),
                "real_source": r.get("real_source"),
            }
            if path is None:
                rec["error"] = "payload_missing"
                stats["errors"] += 1
            else:
                try:
                    res = detector.predict(str(path))
                    fam_raw = dict(res.details.get("family_scores") or {})
                    fam = {
                        k: v for k, v in fam_raw.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    }
                    guarded = res.details.get("reason") == "input_below_model_minimum"
                    rec.update({
                        "score": float(res.score),
                        "up": res.label,
                        "conf": float(res.confidence),
                        "cal": bool(res.runtime.get("calibrated")),
                        "ms": int(res.elapsed_ms),
                        "guard": 1 if guarded else 0,
                        "fam": fam,
                    })
                    stats["scored"] += 1
                    stats["guarded"] += 1 if guarded else 0
                except Exception as exc:  # noqa: BLE001 - recorded, continue
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    stats["errors"] += 1
            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
            fh.flush()
            if n % every == 0 or n == len(todo):
                rate = n / max(time.time() - t0, 1e-6)
                eta = (len(todo) - n) / max(rate, 1e-6)
                log(f"[score] {n}/{len(todo)}  {rate:.2f} img/s  eta {eta:.0f}s")
    log(f"[score] done: {stats}")
    return stats


# ---------------------------------------------------------------------------
# Step 6 - metrics, band fit, verification, result.txt
# ---------------------------------------------------------------------------

def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def auroc_ci(
    scores: Sequence[float], labels: Sequence[int], *, seed: int = SEED,
    n_boot: int = BOOTSTRAP_N,
) -> Optional[List[Optional[float]]]:
    import numpy as np

    if len(set(labels)) < 2:
        return None
    rng = random.Random(seed)
    idx_fake = [k for k, v in enumerate(labels) if v == SYNTH_LABEL]
    idx_real = [k for k, v in enumerate(labels) if v == 0]
    vals: List[float] = []
    for _ in range(n_boot):
        sample = (
            [rng.choice(idx_fake) for _ in idx_fake]
            + [rng.choice(idx_real) for _ in idx_real]
        )
        a = auroc([scores[k] for k in sample], [labels[k] for k in sample])
        if a is not None:
            vals.append(a)
    if not vals:
        return None
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return [float(np.mean(vals)), float(lo), float(hi)]


def tpr_at_fpr(
    scores: Sequence[float], labels: Sequence[int], target: float
) -> Optional[Tuple[float, float, float]]:
    """(tpr, achieved_fpr, threshold) at the empirical FPR just below target."""
    reals = sorted(s for s, y in zip(scores, labels) if y == 0)
    fakes = [s for s, y in zip(scores, labels) if y == SYNTH_LABEL]
    if not reals or not fakes:
        return None
    idx = max(0, int(math.ceil((1.0 - target) * len(reals))) - 1)
    thr = reals[idx]
    achieved_fpr = sum(1 for s in reals if s > thr) / len(reals)
    tpr = sum(1 for s in fakes if s > thr) / len(fakes)
    return float(tpr), float(achieved_fpr), float(thr)


def wilson_upper(k: int, n: int, z: float = CI_Z) -> float:
    if n == 0:
        return float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    spread = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    return (centre + spread) / denom


def fit_band(
    cal_scores: Sequence[float],
    cal_labels: Sequence[int],
    *,
    fnr_budget: float = FNR_BUDGET,
    fpr_budget: float = FPR_BUDGET,
) -> Dict[str, Any]:
    """Dual-budget band on the calibration split only.

    ``low`` = fnr-quantile of synthetic scores (fakes below it = auto-accept);
    ``high`` = (1-fpr)-quantile of authentic scores (reals above it =
    auto-reject). Satisfiable only when low < high. Never touches test.
    """
    import numpy as np

    fakes = [s for s, y in zip(cal_scores, cal_labels) if y == SYNTH_LABEL]
    reals = [s for s, y in zip(cal_scores, cal_labels) if y == 0]
    if not fakes or not reals:
        return {"satisfiable": False, "reason": "calibration class missing"}
    low = float(np.quantile(fakes, fnr_budget))
    high = float(np.quantile(reals, 1.0 - fpr_budget))
    out: Dict[str, Any] = {
        "low": round(low, 4),
        "high": round(high, 4),
        "satisfiable": bool(low < high),
        "n_fake": len(fakes),
        "n_real": len(reals),
        "fnr_budget": fnr_budget,
        "fpr_budget": fpr_budget,
    }
    if not out["satisfiable"]:
        out["reason"] = (
            f"low ({low:.4f}) >= high ({high:.4f}): the score distributions "
            "overlap too much to hold both budgets on calibration data."
        )
    out["cal_fn"] = sum(1 for s in fakes if s < low) / len(fakes)
    out["cal_fp"] = sum(1 for s in reals if s > high) / len(reals)
    out["cal_abstain"] = sum(
        1 for s in list(fakes) + list(reals) if low <= s <= high
    ) / (len(fakes) + len(reals))
    return out


def eval_band(
    scores: Sequence[float],
    labels: Sequence[int],
    low: float,
    high: float,
) -> Dict[str, Any]:
    fakes = [s for s, y in zip(scores, labels) if y == SYNTH_LABEL]
    reals = [s for s, y in zip(scores, labels) if y == 0]
    fn_k = sum(1 for s in fakes if s < low)
    fp_k = sum(1 for s in reals if s > high)
    abst_fake = sum(1 for s in fakes if low <= s <= high)
    abst_real = sum(1 for s in reals if low <= s <= high)
    fn_rate = fn_k / len(fakes) if fakes else float("nan")
    fp_rate = fp_k / len(reals) if reals else float("nan")
    fn_up = wilson_upper(fn_k, len(fakes))
    fp_up = wilson_upper(fp_k, len(reals))
    return {
        "n_fake": len(fakes),
        "n_real": len(reals),
        "fn": fn_k,
        "fp": fp_k,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "fn_upper_95": fn_up,
        "fp_upper_95": fp_up,
        "abstain_fake": abst_fake,
        "abstain_real": abst_real,
        "abstain_rate": (abst_fake + abst_real) / max(len(scores), 1),
        "gate_fn_pass": bool(fn_up <= FNR_BUDGET) if fakes else False,
        "gate_fp_pass": bool(fp_up <= FPR_BUDGET) if reals else False,
    }


def signal_aurocs(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels = [r["y"] for r in records]
    scores = [r["score"] for r in records]
    out["score"] = auroc(scores, labels)
    for key in SIGNAL_KEYS:
        vals = [ (r.get("fam") or {}).get(key) for r in records ]
        if any(v is None for v in vals) or len(set(vals)) <= 1:
            out[key] = None  # missing on some rows or constant - not rankable
        else:
            out[key] = auroc(vals, labels)
    return out


def diagnose(a_dev: Dict[str, Any]) -> Tuple[str, str]:
    """Three-bucket diagnosis from dev-split per-signal AUROC."""
    a_score = a_dev.get("score")
    a_sig = a_dev.get("model_signal_raw")
    if a_score is None:
        return "inconclusive", "dev AUROC undefined (single-class split?)"
    if a_score >= 0.85:
        return (
            "calibration",
            f"score AUROC {a_score:.3f} >= 0.85: ranking is good; the "
            "remaining error is threshold placement.",
        )
    if a_sig is not None and a_sig >= 0.80 and a_sig >= a_score + 0.05:
        return (
            "fusion_dilution",
            f"model_signal_raw AUROC {a_sig:.3f} vs score {a_score:.3f}: the "
            "38-image-tuned fusion is diluting the classifier; fusion/gating "
            "(lands in the aludam/ port, route b) is the lever.",
        )
    if a_score <= 0.60 and (a_sig is None or a_sig <= 0.60):
        return (
            "model_ceiling",
            f"score AUROC {a_score:.3f} and model_signal_raw "
            f"{f'{a_sig:.3f}' if a_sig is not None else 'n/a'} <= 0.60: the "
            "checkpoint is the bottleneck - pre-agreed trip-wire toward the "
            "HELD Phase-3 image swap (Bombek1) fires here.",
        )
    return (
        "partial_separation",
        f"score AUROC {a_score:.3f}: partial rank separation - calibration "
        "plus the third road (model-branch normalization / scale ensemble, "
        "via the aludam/ port) before any swap talk.",
    )


def _fmt_auroc(v: Optional[Sequence[Any]]) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    point, lo, hi = v  # bootstrap triple
    if lo is None:
        return f"{point:.4f} (CI n/a)"
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def _env_block(device_used: str, device_notes: Sequence[str]) -> List[str]:
    def _mod_ver(name: str) -> str:
        try:
            mod = __import__(name)
            return str(getattr(mod, "__version__", "?"))
        except Exception:  # noqa: BLE001
            return "not importable"

    aludam_ver = "?"
    try:
        import aludam
        aludam_ver = getattr(aludam, "__version__", "?")
    except Exception:  # noqa: BLE001
        pass
    lines = [
        f"timestamp_utc   : {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        f"host_platform   : {platform.platform()}",
        f"python          : {platform.python_version()}",
        f"torch           : {_mod_ver('torch')}",
        f"transformers    : {_mod_ver('transformers')}",
        f"aludam          : {aludam_ver}",
        f"device          : {device_used}",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            lines.append(f"cuda_device     : {torch.cuda.get_device_name(0)}")
    except Exception:  # noqa: BLE001
        pass
    for n in device_notes:
        lines.append(f"device_note     : {n}")
    return lines


def analyze_and_write(
    *,
    repo: Path,
    selected: Sequence[Dict[str, Any]],
    splits: Dict[str, List[Dict[str, Any]]],
    evidence: Dict[str, Any],
    warnings_: Sequence[str],
    payload_stats: Dict[str, int],
    score_stats: Dict[str, int],
    scores_path: Path,
    out_path: Path,
    per_class: int,
    seed: int,
    device_used: str,
    device_notes: Sequence[str],
    log: Callable[[str], None] = _log,
) -> str:
    """Join everything, compute the full report, write result.txt, return it."""
    by_i: Dict[int, Dict[str, Any]] = {}
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            by_i[int(rec["i"])] = rec  # later lines win (resumable reruns)

    def collect(role: str) -> List[Dict[str, Any]]:
        out = []
        for r in splits[role]:
            rec = by_i.get(int(r["i"]))
            if rec is None or "error" in rec or "score" not in rec:
                continue
            out.append(rec)
        return out

    dev, cal, test = collect("dev"), collect("cal"), collect("test")
    usable = dev + cal + test
    errored = len(usable) and sum(
        1 for i in (int(r["i"]) for r in selected)
        if i in by_i and ("error" in by_i[i] or "score" not in by_i[i])
    )
    guarded = sum(1 for r in usable if r.get("guard"))
    coverage = [r for r in usable if (r.get("fam") or {}).get("model_signal_raw") is not None]
    model_coverage = len(coverage) / len(usable) if usable else 0.0

    lines: List[str] = []
    add = lines.append
    add("=" * 72)
    add("ALUDAM IMG_DET BASELINE RESULT")
    add("=" * 72)
    add(NON_TRANSFER_STAMP)
    add("")
    add("-- RUN ENVIRONMENT " + "-" * 54)
    add("")
    lines.extend(_env_block(device_used, device_notes))
    add("")
    add("-- DATASET " + "-" * 61)
    add("")
    add(f"dataset         : {DATASET_REPO} (split {DATASET_SPLIT})")
    try:
        add(f"revision        : {resolve_revision()}")
    except Exception as exc:  # noqa: BLE001
        add(f"revision        : unresolved ({exc})")
    add(f"index rows      : {evidence.get('total_rows')}")
    add(f"selected        : {len(selected)} (per_class={per_class}, seed={seed})")
    add(f"label convention: {evidence.get('convention')}")
    add(f"convention note : {evidence.get('convention_note')}")
    add(f"label counts    : {json.dumps(evidence.get('label_counts'))}")
    add(f"label x format  : {json.dumps(evidence.get('label_x_format'))}")
    add(f"label x arch    : {json.dumps(evidence.get('label_x_architecture'))}")
    add(f"groups synth    : {evidence.get('groups', {}).get('synth')}")
    add(f"groups real     : {evidence.get('groups', {}).get('real')}")
    add(f"split counts    : {json.dumps(evidence.get('split_counts'))}")
    add(f"payload fetch   : {json.dumps(payload_stats)}")
    add(f"scoring         : {json.dumps(score_stats)}; usable={len(usable)}, "
        f"errored={errored}, guard_hits={guarded}")
    add(f"model_signal cov: {model_coverage:.1%} of usable rows "
        "(rows without it run heuristics-only)")
    add("")
    add("-- SPLIT RULE (written before scoring) " + "-" * 35)
    add("")
    add("synthetic rows grouped by generator (model_name); authentic rows")
    add("grouped by origin (real_source); whole groups assigned to one split;")
    add("fallback per-image hash split only when the group structure cannot")
    add("fill a split (recorded in warnings).")
    for w in warnings_:
        add(f"WARN: {w}")
    add("")
    add(f"-- GATE (fixed before fitting) " + "-" * 42)
    add("")
    add(f"auto-accept: FNR <= {FNR_BUDGET:.1%} of synthetics")
    add(f"auto-reject: FPR <= {FPR_BUDGET:.1%} of authentics")
    add(f"between bands: ABSTAIN -> human review")
    add(f"judged by Wilson 95% upper bound (z={CI_Z}), not point estimate")
    add(f"ladder: unsatisfiable on calibration -> third road -> trip-wire")
    add("")

    dev_ok = bool(dev) and len({r["y"] for r in dev}) == 2
    add("-- DEV DIAGNOSIS (ranking, per signal) " + "-" * 33)
    add("")
    a_dev = signal_aurocs(dev) if dev_ok else {"score": None}
    ci_dev = auroc_ci([r["score"] for r in dev], [r["y"] for r in dev]) if dev_ok else None
    add(f"score AUROC (dev)          : {_fmt_auroc(ci_dev if dev_ok else None)}")
    for key in SIGNAL_KEYS:
        v = a_dev.get(key)
        add(f"{key + ' AUROC (dev)':<28}: {'n/a' if v is None else f'{v:.4f}'}")
    bucket, why = diagnose(a_dev)
    add(f"diagnosis bucket           : {bucket}")
    add(f"why                        : {why}")
    add("")

    add("-- CALIBRATION FIT (cal split only) " + "-" * 38)
    add("")
    band = fit_band([r["score"] for r in cal], [r["y"] for r in cal])
    if band.get("satisfiable"):
        add(f"band low                  : {band['low']:.4f}")
        add(f"band high                 : {band['high']:.4f}")
        add(f"cal FNR (point)           : {band['cal_fn']:.4f} "
            f"(budget {FNR_BUDGET:.1%}, n_fake={band['n_fake']})")
        add(f"cal FPR (point)           : {band['cal_fp']:.4f} "
            f"(budget {FPR_BUDGET:.1%}, n_real={band['n_real']})")
        add(f"cal abstain rate          : {band['cal_abstain']:.4f}")
        add("satisfiable               : yes")
    else:
        add("satisfiable               : NO")
        add(f"reason                    : {band.get('reason')}")
    add("")

    add("-- TEST VERIFY (touched once) " + "-" * 43)
    add("")
    test_ok = bool(test) and len({r["y"] for r in test}) == 2
    if not test_ok:
        add("test split unusable (empty or single-class) - no test numbers.")
    else:
        ci_test = auroc_ci([r["score"] for r in test], [r["y"] for r in test])
        add(f"score AUROC (test)        : {_fmt_auroc(ci_test)}")
        t = tpr_at_fpr([r["score"] for r in test], [r["y"] for r in test], 0.01)
        if t is not None:
            add(f"TPR @ 1% FPR (test)      : {t[0]:.4f} "
                f"(empirical FPR {t[1]:.4f}, thr {t[2]:.4f})")
        base_fn = sum(1 for r in test if r["y"] == SYNTH_LABEL and r["score"] < 0.5)
        base_fp = sum(1 for r in test if r["y"] == 0 and r["score"] >= 0.5)
        add(f"raw 0.50-threshold counts: FN={base_fn}  FP={base_fp}")
        if band.get("satisfiable"):
            ev = eval_band([r["score"] for r in test], [r["y"] for r in test],
                           band["low"], band["high"])
            add("")
            add("at fitted band:")
            add(f"  auto-accept FN         : {ev['fn']}/{ev['n_fake']} "
                f"= {ev['fn_rate']:.4f}  (Wilson upper {ev['fn_upper_95']:.4f} "
                f"vs budget {FNR_BUDGET:.1%})  "
                f"{'PASS' if ev['gate_fn_pass'] else 'FAIL'}")
            add(f"  auto-reject FP         : {ev['fp']}/{ev['n_real']} "
                f"= {ev['fp_rate']:.4f}  (Wilson upper {ev['fp_upper_95']:.4f} "
                f"vs budget {FPR_BUDGET:.1%})  "
                f"{'PASS' if ev['gate_fp_pass'] else 'FAIL'}")
            add(f"  abstain fake / real    : {ev['abstain_fake']} / "
                f"{ev['abstain_real']}  (overall {ev['abstain_rate']:.4f})")
            gate = "PASS" if (ev["gate_fn_pass"] and ev["gate_fp_pass"]) else "FAIL"
            add(f"GATE                       : {gate}")
            if gate == "FAIL":
                add("ladder step               : see diagnosis bucket above")
                if bucket == "model_ceiling":
                    add("TRIP-WIRE                 : model ceiling confirmed - "
                        "open the HELD Phase-3 image-swap conversation "
                        "(Bombek1; its card also disclaims screenshots).")
        else:
            add("band eval                 : skipped - band unsatisfiable on "
                "calibration (ladder: third road, then trip-wire).")
    add("")

    add("-- STRATA (mean score by group, all usable rows) " + "-" * 24)
    add("")
    groups: Dict[str, List[float]] = {}
    for r in usable:
        if r["y"] == SYNTH_LABEL:
            key = f"fake:{r.get('model_name')}"
        else:
            key = f"real:{r.get('real_source')}"
        groups.setdefault(key, []).append(r["score"])
    for key in sorted(groups):
        vals = groups[key]
        add(f"{key:<32} n={len(vals):<5} mean={sum(vals)/len(vals):.4f}")
    add("")
    add("-- CAVEATS " + "-" * 61)
    add("")
    add("* Fusion weights/thresholds were tuned on a legacy 38-image set;")
    add("  AUROC on score measures that fusion, not the raw classifier -")
    add("  read model_signal_raw alongside score.")
    add("* Thresholds do not change ranking: AUROC before/after fit is")
    add("  identical by construction; the gate numbers above are the")
    add("  real before/after.")
    add("* Scores are NOT logits (softmax + hand fusion); sigmoid on them")
    add("  would be wrong.")
    add(NON_TRANSFER_STAMP)
    add("=" * 72)

    text = "\n".join(str(ln) for ln in lines) + "\n"
    text.encode("ascii", errors="replace")  # package rule: ASCII-safe
    if any(ord(ch) > 127 for ch in text):
        text = text.encode("ascii", "replace").decode("ascii")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    log(f"[result] written: {out_path}")
    return text
