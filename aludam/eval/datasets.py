"""Load Phase 0 text datasets into one shape, offline.

Every loader returns a DataFrame with exactly these columns:

    text    str
    label   int  (1 = synthetic / AI, 0 = authentic / human)
    group   str  (unit for grouped bootstrap)
    src     str  (origin, for stratum reports)

Two things make this module worth having rather than ad-hoc read_csv calls:

1. **Label-convention guards.** MAGE ships 0 = machine / 1 = human - the
   inverse of ours. Getting this wrong inverts every metric while still
   producing plausible-looking numbers. Each loader asserts a structural
   invariant against the *source* column, so an inversion cannot survive.
2. **Group id for grouped bootstrap.** Rows sharing an origin document must
   stay together when resampling, or near-duplicate text leaks across the
   resample and the confidence interval collapses.

Offline by design: reads only what ``fetch.py`` already cached.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "data" / "manifest.json"

REQUIRED_COLUMNS = ("text", "label", "group", "src")


class DatasetError(RuntimeError):
    """A dataset is missing, malformed, or violates its label invariant."""


def _manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise DatasetError(
            f"{MANIFEST_PATH} not found - run "
            "`DEEPFAKE_ALLOW_DOWNLOADS=1 python eval/fetch.py` first"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _file(dataset: str, filename: str) -> Path:
    man = _manifest()
    try:
        entry = man[dataset]["files"][filename]
    except KeyError as exc:
        raise DatasetError(
            f"{dataset}/{filename} missing from manifest - rerun fetch.py"
        ) from exc
    path = Path(entry["path"])
    if not path.exists():
        raise DatasetError(
            f"{path} no longer on disk (expected {entry['size_bytes']:,} B) "
            "- rerun fetch.py"
        )
    return path


def _finish(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"{name}: missing columns {missing}")
    if df["label"].isna().any():
        raise DatasetError(f"{name}: null labels present")
    labels = set(df["label"].unique().tolist())
    if not labels <= {0, 1}:
        raise DatasetError(f"{name}: labels must be 0/1, got {sorted(labels)}")
    if df["text"].isna().any():
        raise DatasetError(f"{name}: null text present")
    df = df[list(REQUIRED_COLUMNS)].reset_index(drop=True)
    if len(df) < 50:
        raise DatasetError(f"{name}: only {len(df)} rows - not usable")
    return df


def load_mage(split: str = "test") -> pd.DataFrame:
    """Load a MAGE split.

    split: "test" | "valid" | "ood_gpt" | "ood_gpt_para"
    """
    mapping = {
        "test": "test.csv",
        "valid": "valid.csv",
        "ood_gpt": "test_ood_set_gpt.csv",
        "ood_gpt_para": "test_ood_set_gpt_para.csv",
    }
    if split not in mapping:
        raise DatasetError(f"unknown MAGE split {split!r}; want {sorted(mapping)}")

    path = _file("mage", mapping[split])
    raw = pd.read_csv(path, usecols=["text", "label", "src"])
    if list(raw.columns) != ["text", "label", "src"]:
        raise DatasetError(
            f"MAGE {split}: unexpected columns {list(raw.columns)}"
        )

    # MAGE native: 1 = human-written, 0 = machine-generated. Ours is the
    # inverse. The guard below proves the inversion from the source string
    # rather than trusting the documented convention.
    native = raw["label"].astype(int)
    if set(native.unique()) - {0, 1}:
        raise DatasetError(f"MAGE {split}: unexpected native labels")

    src = raw["src"].astype(str)
    human_by_src = src.str.endswith("_human")

    # Structural invariant: every source literally named "*_human" is human,
    # and every other source (machine_*, gpt4*, *_para) is machine. Verified
    # against the full crosstab before this file was written.
    if not (native[human_by_src] == 1).all():
        bad = int((native[human_by_src] != 1).sum())
        raise DatasetError(
            f"MAGE {split}: {bad} rows have a '*_human' source but native "
            f"label != 1 - refusing to guess the label convention"
        )
    if not (native[~human_by_src] == 0).all():
        bad = int((native[~human_by_src] != 0).sum())
        raise DatasetError(
            f"MAGE {split}: {bad} non-human rows carry native label 1 - "
            "label convention appears to have changed upstream"
        )

    out = pd.DataFrame(
        {
            "text": raw["text"].astype(str),
            # our convention: 1 = synthetic
            "label": (1 - native).astype(int),
            # group by origin domain so resamples cannot mix domains
            "group": "mage:" + src,
            "src": src,
        }
    )
    return _finish(out, name=f"MAGE/{split}")


def load_hc3(max_per_source: int | None = None) -> pd.DataFrame:
    """Load HC3: human_answers -> 0, chatgpt_answers -> 1."""
    path = _file("hc3", "all.jsonl")

    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    srcs: list[str] = []
    seen_per_source: dict[str, int] = {}

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"HC3 line {lineno}: {exc}") from exc

            src = str(rec.get("source", "unknown"))
            if max_per_source is not None and seen_per_source.get(src, 0) >= max_per_source:
                continue

            # One group per question: all answers derived from the same
            # question must move together under the bootstrap.
            group = f"hc3:{src}:{lineno}"
            q = str(rec.get("question") or "")

            human = rec.get("human_answers") or []
            chat = rec.get("chatgpt_answers") or []

            for ans in human:
                t = (str(ans) or "").strip()
                if not t:
                    continue
                texts.append(q + "\n" + t)
                labels.append(0)
                groups.append(group)
                srcs.append(src)
            for ans in chat:
                t = (str(ans) or "").strip()
                if not t:
                    continue
                texts.append(q + "\n" + t)
                labels.append(1)
                groups.append(group)
                srcs.append(src)

            if max_per_source is not None:
                seen_per_source[src] = seen_per_source.get(src, 0) + 1

    if not texts:
        raise DatasetError("HC3 produced no rows")

    out = pd.DataFrame(
        {"text": texts, "label": labels, "group": groups, "src": srcs}
    )
    return _finish(out, name="HC3")


def drop_seen(df: pd.DataFrame, seen_texts) -> tuple[pd.DataFrame, int]:
    """Remove rows whose text already appeared in another split.

    Returns (filtered_df, n_removed).

    MAGE ships a handful of rows duplicated between ``valid.csv`` and
    ``test.csv`` (52 when first measured). Fitting thresholds on calibration
    data and then scoring those same strings on test is leakage. The number
    is small enough not to move AUROC, but it is cheap to remove and a
    protocol that knowingly leaks is not a protocol.
    """
    if "text" not in df.columns:
        raise DatasetError("drop_seen expects a frame with a 'text' column")
    seen = set(seen_texts)
    keep = ~df["text"].isin(seen)
    removed = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), removed


def load_split(name: str) -> pd.DataFrame:
    """Dispatcher used by the runner.

    Names are stable keys, so an eval record can say exactly what it scored.
    """
    if name.startswith("mage:"):
        return load_mage(name.split(":", 1)[1])
    if name == "hc3":
        return load_hc3()
    raise DatasetError(f"unknown dataset {name!r}")


def available_splits() -> tuple[str, ...]:
    return ("mage:test", "mage:valid", "mage:ood_gpt", "mage:ood_gpt_para", "hc3")


def describe(name: str) -> str:
    df = load_split(name)
    pos = int(df["label"].sum())
    return (
        f"{name}: n={len(df):,}  synthetic={pos:,}  authentic={len(df) - pos:,}  "
        f"groups={df['group'].nunique():,}  sources={df['src'].nunique():,}"
    )
