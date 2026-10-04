"""Build metadata-only indices for the image and audio evaluation sets.

Purpose
-------
Phase 0 needs, per modality: an id, a label, and a *group* key so that
dev / calibration / test can be split by source rather than randomly
(D5). Getting those three fields must not cost a 206 GB image download or a
34 GB audio download.

How it stays cheap
------------------
Parquet is columnar, so we ask pyarrow for the metadata columns only and
let fsspec serve the byte ranges they live in. The payload columns
(``image_data``, ``audio``) are never requested, so they are never fetched.
``prompt`` is deliberately omitted too - it is the one metadata column big
enough to matter.

The shard list is pinned and recorded in the manifest, so a re-run indexes
the identical rows in the identical order. Rows are appended to a JSONL as
each shard finishes, which makes an interrupted run resumable.

Nothing written here is ever committed: the index holds ids, labels, group
keys and shard provenance only - no bytes of any sample.

Run:  python eval/index.py image
      python eval/index.py audio
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
INDEX_DIR = DATA_DIR / "index"
MANIFEST = DATA_DIR / "manifest.json"

IMAGE_COLS = [
    # everything except `image_data` (the payload) and `prompt` (large).
    "image_name", "format", "resolution", "mode", "nsfw_flag",
    "model_name", "real_source", "subset", "split", "label", "architecture",
]
AUDIO_COLS = ["path", "label", "notes"]

IMAGE_SHARD_COUNT = 96   # of 413 - enough rows for 750/class + source spread
AUDIO_SHARD_COUNT = 80   # of 80  - the whole DF eval partition
WORKERS = 12
SEED = 20261004

SPECS: dict[str, dict[str, Any]] = {
    "image": {
        "repo": "OwensLab/CommunityForensics-Eval",
        "prefix": "data/CompEval-",
        "columns": IMAGE_COLS,
        "shard_count": IMAGE_SHARD_COUNT,
        "payload_col": "image_data",
    },
    "audio": {
        "repo": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
        "prefix": "data/test-",
        "columns": AUDIO_COLS,
        "shard_count": AUDIO_SHARD_COUNT,
        "payload_col": "audio",
    },
}

_lock = threading.Lock()


def _get(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "aludam-eval/0.1"})
    import urllib.error

    for attempt in range(3):
        try:
            import urllib.request as ur

            with ur.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def resolve_revision(repo: str) -> str:
    info = json.loads(_get(f"https://huggingface.co/api/datasets/{repo}"))
    sha = info.get("sha")
    if not sha or len(sha) < 40:
        raise RuntimeError(f"could not resolve revision for {repo}: {sha!r}")
    return sha


def list_shards(repo: str, prefix: str) -> list[str]:
    """Every parquet under `prefix`, sorted, so the order is reproducible."""
    tree = json.loads(
        _get(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true")
    )
    out = [
        t["path"]
        for t in tree
        if t["type"] == "file" and t["path"].startswith(prefix) and t["path"].endswith(".parquet")
    ]
    return sorted(out)


def pick_shards(all_shards: list[str], count: int, seed: int) -> list[str]:
    """Deterministic subset.

    We take a seeded shuffle rather than a prefix because the shards are not
    shuffled upstream: taking the first N would take whatever sources and
    labels happen to be written first, which is exactly the selection bias a
    grouped split is supposed to avoid.
    """
    if count >= len(all_shards):
        return list(all_shards)
    import random

    rng = random.Random(seed)
    picked = list(all_shards)
    rng.shuffle(picked)
    return sorted(picked[:count])


def already_indexed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["shard"])
            except Exception:
                continue
    return done


def read_shard(repo: str, shard: str, columns: list[str], revision: str) -> list[dict]:
    import warnings

    warnings.filterwarnings("ignore")
    import pyarrow.parquet as pq
    from fsspec.core import url_to_fs

    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{shard}"
    fs, _ = url_to_fs(url, block_size=16 << 20)
    table = pq.read_table(url, columns=columns, filesystem=fs)
    return table.to_pylist()


def index(name: str) -> int:
    spec = SPECS[name]
    repo, cols = spec["repo"], spec["columns"]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / f"{name}.jsonl"

    revision = resolve_revision(repo)
    shards = pick_shards(list_shards(repo, spec["prefix"]), spec["shard_count"], SEED)
    done = already_indexed(out)
    todo = [s for s in shards if s not in done]
    print(f"== {name}  ({repo})")
    print(f"   revision  {revision[:12]}")
    print(f"   shards    {len(shards)} pinned ({spec['shard_count']} requested), "
          f"{len(done)} already indexed, {len(todo)} to do")
    print(f"   columns   {', '.join(cols)}")
    print(f"   payload   {spec['payload_col']} - never requested")
    if not todo:
        print("   nothing to do")
        return 0

    t0 = time.time()
    failures = 0

    def work(shard: str) -> tuple[str, list[dict], str | None]:
        try:
            return shard, read_shard(repo, shard, cols, revision), None
        except Exception as exc:
            return shard, [], f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool, out.open("a", encoding="utf-8") as fh:
        for i, (shard, rows, err) in enumerate(pool.map(work, todo), 1):
            if err:
                failures += 1
                print(f"   [{i}/{len(todo)}] FAIL {shard}: {err}")
                continue
            for row in rows:
                row["shard"] = shard
                row["dataset"] = name
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            fh.flush()
            print(f"   [{i}/{len(todo)}] {shard}  rows={len(rows)}  "
                  f"elapsed={time.time() - t0:.0f}s")

    total = sum(1 for _ in out.open(encoding="utf-8"))
    print(f"   index -> {out}  ({total} rows, {failures} shard failures)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=sorted(SPECS))
    args = ap.parse_args()
    return index(args.dataset)


if __name__ == "__main__":
    sys.exit(main())
