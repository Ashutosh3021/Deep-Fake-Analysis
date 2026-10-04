"""Fetch Phase 0 evaluation splits, pinned to a commit SHA.

Design decisions that matter for the numbers we will report:

* Only *evaluation* splits are fetched. MAGE's ``train.csv`` is 404 MB and we
  will never score on it - pulling it would only create the temptation, and
  any model we swap in later may have trained on exactly that file.
* Every fetch is pinned to an immutable commit SHA, then re-hashed. A
  "before/after" comparison is meaningless if the dataset silently changed
  underneath between runs.
* Network access must be opted into with DEEPFAKE_ALLOW_DOWNLOADS=1. After
  this script has run once, evaluation can proceed fully offline.

Run:  DEEPFAKE_ALLOW_DOWNLOADS=1 python eval/fetch.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MANIFEST = DATA_DIR / "manifest.json"

# label_convention: how to map the source's native labels onto our
# internal convention (1 = synthetic, 0 = authentic). Recorded per dataset
# because MAGE ships the *inverted* convention and silently flipping it
# would invert every metric we report.
DATASETS: dict[str, dict] = {
    "mage": {
        "repo": "yaful/MAGE",
        "license": "apache-2.0",
        "label_convention": "0=machine, 1=human (INVERTED - flip at ingest)",
        "label_field": "label",
        # splits we score on. train.csv deliberately omitted (see module docstring).
        "files": {
            "test.csv": "in-distribution testbed",
            "test_ood_set_gpt.csv": "unseen domain + GPT-4 (the 'wild' stratum)",
            "test_ood_set_gpt_para.csv": "paraphrase attack (hardest stratum)",
            "valid.csv": "calibration split - never used for reporting",
        },
    },
    "hc3": {
        "repo": "Hello-SimpleAI/HC3",
        "license": "cc-by-sa-4.0",
        "label_convention": "human_answers vs chatgpt_answers (verify at ingest)",
        "label_field": "human_answers/chatgpt_answers",
        "files": {
            "all.jsonl": "all five domains, human vs ChatGPT answers",
        },
    },
}


def _api(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "aludam-eval/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def resolve_revision(repo: str) -> tuple[str, str]:
    """Return (commit_sha, last_modified) so the fetch is pinned."""
    info = _api(f"https://huggingface.co/api/datasets/{repo}")
    sha = info.get("sha")
    if not sha or len(sha) < 40:
        raise RuntimeError(f"could not resolve commit sha for {repo}: {sha!r}")
    return sha, info.get("lastModified") or ""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    if os.environ.get("DEEPFAKE_ALLOW_DOWNLOADS") != "1":
        print("refusing: set DEEPFAKE_ALLOW_DOWNLOADS=1 to fetch datasets")
        return 2

    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    failures = 0
    for name, spec in DATASETS.items():
        repo = spec["repo"]
        print(f"\n== {name}  ({repo})")
        try:
            sha, lastmod = resolve_revision(repo)
        except Exception as exc:
            print(f"   [FAIL] resolve revision: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        print(f"   revision {sha[:12]}  ({lastmod})")

        entry = manifest.setdefault(name, {})
        entry.update(
            {
                "repo": repo,
                "revision": sha,
                "license": spec["license"],
                "label_convention": spec["label_convention"],
                "files": {},
            }
        )
        # `_prev_sha` survives the update above; it holds last run's digests so
        # we can tell "we fetched a new upstream revision" from "file changed".
        prev: dict[str, str] = entry.get("_prev_sha") or {}

        for fname, purpose in spec["files"].items():
            try:
                # repo_type="dataset" is mandatory: hf_hub_download defaults to
                # "model", which silently resolves a non-existent model repo and
                # reports a confusing 404/401.
                path = Path(
                    hf_hub_download(
                        repo_id=repo,
                        filename=fname,
                        revision=sha,
                        repo_type="dataset",
                    )
                )
            except Exception as exc:
                print(f"   [FAIL] {fname}: {type(exc).__name__}: {exc}")
                failures += 1
                continue

            digest = sha256_file(path)
            size = path.stat().st_size
            entry["files"][fname] = {
                "path": str(path),
                "purpose": purpose,
                "size_bytes": size,
                "sha256": digest,
            }
            prev_digest = prev.get(fname)
            mark = "   <-- CHANGED SINCE LAST FETCH" if (
                prev_digest and prev_digest != digest
            ) else ""
            print(
                f"   [ok] {fname:24} {size:>12,} B  {digest[:16]}{mark}"
            )

        entry["_prev_sha"] = {
            f: v["sha256"] for f, v in entry.get("files", {}).items()
        }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nmanifest -> {MANIFEST}")

    total = sum(
        f["size_bytes"]
        for d in manifest.values()
        for f in d.get("files", {}).values()
    )
    print(f"cached total: {total:,} bytes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
