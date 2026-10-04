"""Phase 0 runner: score a split through a detector, then report metrics.

Three subcommands, designed so a multi-hour run can be interrupted and resumed:

  plan    show what would be scored, stratum counts, and a runtime estimate
          without loading the detector
  score   append one JSON record per input to a JSONL file, skipping rows
          already present (resume is automatic)
  report  compute AUROC + TPR@1%FPR + grouped-bootstrap CI from the JSONL

Why sampling is not optional: text_det spends ~95% of its time in
``_detectgpt_curvature`` (21 sequential GPT-2 forwards, see
``prof_text.py``), i.e. ~4 s/item on a 12-core CPU regardless of thread count.
Full ``mage:test`` is therefore ~63 h. Strata are sampled with a fixed seed so
the same command reproduces the same rows, and ``--n-per-class`` makes the
sample size an explicit, reportable choice rather than an accident.

Protocol notes (ALUDAM_PLAN.md decisions):
  * raw scores only - thresholds are unfitted, AUROC never consults them (D4)
  * ``drop-seen-split`` removes calibration rows before test scoring, so the
    52 MAGE valid/test duplicates cannot leak (finding 16)
  * ``--arm`` labels which arm of the comparison produced these numbers, so a
    "before/after" table can say whether a change was a port fix or a model
    swap (D2)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS = HERE / "runs"
ENV_SNAPSHOT = HERE / "data" / "env_snapshot.json"

# eval/ is deliberately NOT part of the installed package (it is Phase 0
# tooling, see eval/__init__.py). Support both `python -m eval.run` from the
# aludam/ directory and `python eval/run.py` from anywhere.
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))
    from eval import datasets as _datasets  # noqa: E402,F401
    from eval import metrics as _metrics  # noqa: E402,F401
else:
    from . import datasets as _datasets  # noqa: F401
    from . import metrics as _metrics  # noqa: F401

# Fixed, interpretable text-length buckets (characters). Fixed edges rather
# than quantiles so a stratum means the same thing across splits and arms.
LENGTH_BUCKETS = (
    ("short", 0, 400),
    ("medium", 400, 1500),
    ("long", 1500, None),
)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def length_bucket(text: str) -> str:
    n = len(text)
    for name, lo, hi in LENGTH_BUCKETS:
        if n >= lo and (hi is None or n < hi):
            return name
    return LENGTH_BUCKETS[-1][0]


def sample_per_class(
    df, n_per_class: int, seed: int
) -> Tuple[Any, Dict[str, int]]:
    """Deterministic stratified sample: up to ``n_per_class`` rows per label.

    Sampling is reproducible for a given (split, n, seed) triple, so the
    report can state exactly which rows a number came from.
    """
    import pandas as pd  # local: plan must run without pandas loaded

    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")

    counts = Counter(int(x) for x in df["label"].tolist())
    take: Dict[str, int] = {}
    rng = random.Random(seed)
    pieces = []
    for lab in sorted(counts):
        part = df[df["label"] == lab]
        idx = list(part.index)
        rng.shuffle(idx)
        keep = idx if len(idx) <= n_per_class else idx[:n_per_class]
        take[str(lab)] = len(keep)
        pieces.append(part.loc[keep])

    out = pd.concat(pieces)
    # Deterministic global shuffle. Sorting by index instead would hand back
    # rows in dataset order, so `--limit` smoke tests would draw only the
    # first class present in the file (observed: 8/8 rows label 0) and prove
    # nothing about the other class.
    out = out.sample(frac=1, random_state=seed)
    return out, take


def load_for_score(
    split: str,
    n_per_class: Optional[int],
    seed: int,
    drop_seen_split: Optional[str],
) -> Tuple[Any, Dict[str, Any]]:
    load_split, drop_seen = _datasets.load_split, _datasets.drop_seen
    df = load_split(split)
    # Stable dataset row id, captured BEFORE drop_seen (which resets the
    # index to a fresh 0..n-1 range). JSONL keys must identify the row in the
    # dataset, not the row's position in whatever sample happens to be under
    # evaluation - otherwise changing --n-per-class or --seed would silently
    # re-key a resumed run and every "already scored" check would be wrong.
    df = df.copy()
    df["_rowid"] = df.index
    meta: Dict[str, Any] = {
        "split": split,
        "rows_available": int(len(df)),
        "dropped_seen": 0,
        "drop_seen_split": drop_seen_split,
    }

    if drop_seen_split:
        guard = load_split(drop_seen_split)
        df, removed = drop_seen(df, guard["text"].tolist())
        meta["dropped_seen"] = removed

    meta["rows_after_drop"] = int(len(df))
    meta["n_per_class_requested"] = n_per_class

    if n_per_class is None:
        sampled = df
        meta["sampled"] = False
        take = {str(k): int(v) for k, v in Counter(
            int(x) for x in df["label"].tolist()).items()}
    else:
        sampled, take = sample_per_class(df, n_per_class, seed)
        meta["sampled"] = True
    meta["class_counts"] = take
    meta["seed"] = seed
    return sampled, meta


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def src_domain(src: str) -> str:
    """``cmv_machine_continuation_opt_125m`` -> ``cmv``.

    Raw ``src`` has ~320 distinct values in MAGE (domain x label x generator),
    which yields strata of n=1 and says nothing. The domain is the stratum
    that actually matters: it is what a grouped bootstrap should hold out and
    what "does this hold across writing domains?" is asking.
    """
    for sep in ("_human", "_machine"):
        i = src.find(sep)
        if i > 0:
            return src[:i]
    return src


def _key(split: str, row_id: Any) -> str:
    return f"{split}:{row_id}"


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def load_done(path: Path) -> Dict[str, Dict[str, Any]]:
    """Existing records keyed by row key, for resume."""
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A truncated last line (interrupted mid-write) is expected
                # after a crash; it is simply not done yet.
                continue
            if isinstance(rec, dict) and "key" in rec:
                done[rec["key"]] = rec
    _ = lineno
    return done


def score_split(
    split: str,
    detector_name: str,
    out_path: Path,
    n_per_class: Optional[int],
    seed: int,
    drop_seen_split: Optional[str],
    arm: str,
    limit: Optional[int],
    quiet: bool,
) -> Dict[str, Any]:
    from aludam import load_detector

    df, meta = load_for_score(split, n_per_class, seed, drop_seen_split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    rows = [(row["_rowid"], row) for _rid, row in df.iterrows()]
    if limit is not None:
        rows = rows[:limit]

    todo = [(rowid, row) for rowid, row in rows
            if _key(split, rowid) not in done
            or done[_key(split, rowid)].get("error")]

    if not quiet:
        print(f"split={split} available={meta['rows_available']} "
              f"after_drop={meta['rows_after_drop']} "
              f"sampled={meta['class_counts']}")
        print(f"already scored={len(rows) - len(todo)}  to score={len(todo)}")
        if not todo:
            print("nothing to do")
            return {"meta": meta, "scored": 0, "skipped": len(rows)}

    det = load_detector(detector_name)
    started = time.perf_counter()
    scored = 0
    errors = 0

    with out_path.open("a", encoding="utf-8") as fh:
        for i, (rowid, row) in enumerate(todo, 1):
            text = str(row["text"])
            key = _key(split, rowid)
            t0 = time.perf_counter()
            rec: Dict[str, Any] = {
                "key": key,
                "split": split,
                "arm": arm,
                "detector": detector_name,
                "row": int(rowid) if isinstance(rowid, int) else str(rowid),
                "label": int(row["label"]),
                "group": str(row["group"]),
                "src": str(row["src"]),
                "src_domain": src_domain(str(row["src"])),
                "text_sha256": _text_sha(text),
                "n_chars": len(text),
                "len_bucket": length_bucket(text),
            }
            try:
                r = det.predict(text)
                rec.update({
                    "score": float(r.score),
                    "upstream_label": r.upstream_label,
                    "label_out": r.label,
                    "kind": r.kind,
                    "calibrated": bool(r.runtime.get("calibrated")),
                    "confidence": float(r.confidence),
                    "seconds": round(time.perf_counter() - t0, 4),
                    "error": None,
                })
                scored += 1
            except Exception as exc:  # keep the run alive; resume retries
                rec.update({
                    "score": None,
                    "seconds": round(time.perf_counter() - t0, 4),
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                })
                errors += 1

            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
            fh.flush()

            if not quiet and (i % 25 == 0 or i == len(todo)):
                el = time.perf_counter() - started
                per = el / max(i, 1)
                left = per * (len(todo) - i)
                print(f"  [{i}/{len(todo)}] {per:5.2f}s/item "
                      f"elapsed={el/60:5.1f}m eta={left/60:5.1f}m "
                      f"errors={errors}", flush=True)

    elapsed = time.perf_counter() - started
    meta.update({
        "arm": arm,
        "detector": detector_name,
        "out_path": str(out_path),
        "scored_now": scored,
        "skipped": len(rows) - len(todo),
        "errors": errors,
        "seconds": round(elapsed, 1),
    })
    return {"meta": meta, "scored": scored, "skipped": len(rows) - len(todo),
            "errors": errors}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Scored rows, deduplicated by key (last write wins).

    The scoring loop skips rows already present, so duplicates should not
    occur; a stray one would double-count a row and bias every metric, so the
    reader is defensive rather than trusting the writer.
    """
    by_key: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("score") is not None:
                # Derive src_domain here rather than trusting the writer: an
                # in-flight run started before this field existed would
                # otherwise produce a file with mixed records and every
                # src_domain stratum would come back empty.
                if "src_domain" not in rec and rec.get("src"):
                    rec["src_domain"] = src_domain(str(rec["src"]))
                by_key[str(rec.get("key"))] = rec
    return list(by_key.values())


def _block(recs: Sequence[Dict[str, Any]], n_boot: int, seed: int) -> Dict[str, Any]:
    auroc = _metrics.auroc
    tpr_at_fpr = _metrics.tpr_at_fpr
    bootstrap_auroc = _metrics.bootstrap_auroc
    MIN_ROWS = _metrics.MIN_ROWS

    scores = [float(r["score"]) for r in recs]
    labels = [int(r["label"]) for r in recs]
    groups = [r.get("group") for r in recs]

    out: Dict[str, Any] = {
        "n": len(recs),
        "pos": int(sum(labels)),
        "neg": int(len(labels) - sum(labels)),
    }
    if len(recs) < MIN_ROWS or out["pos"] < 5 or out["neg"] < 5:
        out["skipped"] = "too few rows or one class absent"
        return out

    out["auroc"] = round(auroc(scores, labels), 4)
    out["tpr_at_1pct_fpr"] = round(tpr_at_fpr(scores, labels, 0.01), 4)
    try:
        # bootstrap_auroc returns (point, lo, hi).
        _point, lo, hi = bootstrap_auroc(scores, labels, groups,
                                         n_boot=n_boot, seed=seed)
        out["auroc_ci"] = [round(lo, 4), round(hi, 4)]
    except Exception as exc:
        out["auroc_ci_error"] = f"{type(exc).__name__}: {exc}"
    return out


def summarize(
    path: Path, n_boot: int, seed: int, by: Sequence[str]
) -> Dict[str, Any]:
    recs = read_jsonl(path)
    if not recs:
        return {"error": f"no scored rows in {path}"}

    by_label = Counter(r["arm"] for r in recs)
    report: Dict[str, Any] = {
        "file": str(path),
        "arms": dict(by_label),
        "detectors": sorted({r["detector"] for r in recs}),
        "split": sorted({r["split"] for r in recs}),
        "n_boot": n_boot,
        "seed": seed,
        "overall": _block(recs, n_boot, seed),
        "strata": {},
        "errors": sum(1 for r in read_jsonl_all(path) if r.get("error")),
    }

    for field in by:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recs:
            val = r.get(field)
            if val is None:
                val = "<none>"
            buckets[str(val)].append(r)
        report["strata"][field] = {
            k: _block(v, n_boot, seed)
            for k, v in sorted(buckets.items())
        }

    # Per-class performance of the current label rule is NOT a metric yet -
    # thresholds are unfitted (D4). Report the disagreement rate instead, so
    # nobody mistakes label accuracy for a calibrated result.
    disagree = sum(1 for r in recs
                   if r.get("label_out") in ("ai", "real")
                   and r.get("upstream_label") not in ("UNCERTAIN", None))
    report["label_rule"] = {
        "calibrated": any(r.get("calibrated") for r in recs),
        "note": ("labels follow the upstream verdict while uncalibrated (D4); "
                 "AUROC above uses raw score only"),
        "upstream_abstain_rate": round(
            sum(1 for r in recs if r.get("upstream_label") == "UNCERTAIN")
            / len(recs), 4),
        "upstream_abstained_non_uncertain_label": disagree,
    }
    return report


def read_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _env_head() -> Dict[str, Any]:
    if not ENV_SNAPSHOT.exists():
        return {"available": False}
    try:
        snap = json.loads(ENV_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "git_sha": (snap.get("git") or {}).get("sha"),
        "git_dirty": (snap.get("git") or {}).get("dirty"),
        "opencv_owner": (snap.get("opencv") or {}).get("owner"),
        "face_detector": ((snap.get("behaviour_env") or {})
                          .get("DEEPGUARD_FACE_DETECTOR") or {}).get("effective"),
    }


def plan(args: argparse.Namespace) -> int:
    load_split, drop_seen = _datasets.load_split, _datasets.drop_seen

    df = load_split(args.split)
    avail = len(df)
    if args.drop_seen_split:
        guard = load_split(args.drop_seen_split)
        df, removed = drop_seen(df, guard["text"].tolist())
        print(f"dropped {removed} rows already seen in {args.drop_seen_split}")

    if args.n_per_class is None:
        sampled = df
        label = "ALL rows (no subsampling)"
    else:
        sampled, take = sample_per_class(df, args.n_per_class, args.seed)
        label = f"n_per_class={args.n_per_class} -> {take}"

    pos = int(sampled["label"].sum())
    neg = int(len(sampled) - pos)

    print(f"split           : {args.split}")
    print(f"rows available  : {avail}")
    print(f"rows to score   : {len(sampled)}  (pos={pos} neg={neg})")
    print(f"sample          : {label}")
    print(f"detector        : {args.detector}")
    print(f"arm             : {args.arm}")
    print(f"drop_seen_split : {args.drop_seen_split or '-'}")
    print(f"seed            : {args.seed}")
    print(f"output          : {args.out}")

    buckets = Counter(length_bucket(str(t)) for t in sampled["text"])
    print("length strata   : " + ", ".join(
        f"{k}={v}" for k, v in sorted(buckets.items())))
    print(f"src strata      : {sampled['src'].nunique()} distinct")

    if args.assume_seconds:
        est = len(sampled) * args.assume_seconds
        print(f"\nprojected runtime at {args.assume_seconds:.1f}s/item "
              f"-> {est/3600:.1f} h  ({est/60:.0f} min)")
        print("  text_det measures ~4.0 s/item on this box "
              "(~95% in _detectgpt_curvature).")
    print("\nreproduce with:")
    print(f"  python eval/run.py score --split {args.split} "
          f"--detector {args.detector} --out {args.out}"
          + (f" --n-per-class {args.n_per_class}" if args.n_per_class else "")
          + f" --seed {args.seed} --arm {args.arm}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval.run", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split", default="mage:test",
                       help="mage:test|mage:valid|mage:ood_gpt|"
                            "mage:ood_gpt_para|hc3")
        p.add_argument("--detector", default="text_det")
        p.add_argument("--arm", default="baseline-legacy",
                       help="which arm produced these numbers (D2)")

    p = sub.add_parser("plan", help="estimate without scoring")
    common(p)
    p.add_argument("--n-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--drop-seen-split", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--assume-seconds", type=float, default=4.0)

    p = sub.add_parser("score", help="score rows, append JSONL (resumable)")
    common(p)
    p.add_argument("--n-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--drop-seen-split", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N new rows (smoke test)")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("report", help="metrics from a JSONL")
    p.add_argument("--file", required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--by", nargs="*", default=["src_domain", "len_bucket"],
                   help="stratum fields to break down (raw 'src' has ~320 "
                        "values in MAGE and yields n=1 strata)")
    p.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "plan":
        return plan(args)

    if args.cmd == "score":
        out = Path(args.out) if args.out else (
            DEFAULT_RUNS / f"{args.split.replace(':', '_')}."
            f"{args.detector}.{args.arm}.jsonl")
        res = score_split(
            split=args.split,
            detector_name=args.detector,
            out_path=out,
            n_per_class=args.n_per_class,
            seed=args.seed,
            drop_seen_split=args.drop_seen_split,
            arm=args.arm,
            limit=args.limit,
            quiet=args.quiet,
        )
        print(json.dumps(res["meta"], indent=2))
        return 0 if res.get("errors", 0) == 0 else 1

    if args.cmd == "report":
        rep = summarize(Path(args.file), args.n_boot, args.seed, args.by)
        rep["env"] = _env_head()
        if args.json:
            print(json.dumps(rep, indent=2, sort_keys=True))
        else:
            _print_report(rep)
        return 0 if "error" not in rep else 1

    return 2


def _print_report(rep: Dict[str, Any]) -> None:
    if "error" in rep:
        print(rep["error"])
        return
    env = rep.get("env") or {}
    print("=" * 74)
    print(f"split     : {', '.join(rep['split'])}")
    print(f"arms      : {rep['arms']}")
    print(f"detector  : {', '.join(rep['detectors'])}")
    print(f"git       : {str(env.get('git_sha'))[:12]} "
          f"dirty={env.get('git_dirty')}  opencv={env.get('opencv_owner')} "
          f"face={env.get('face_detector')}")
    print(f"bootstrap : {rep['n_boot']} grouped, seed={rep['seed']}")
    print("=" * 74)

    def show(name: str, b: Dict[str, Any]) -> None:
        if "skipped" in b:
            print(f"  {name:26} n={b['n']:<6} {b['skipped']}")
            return
        ci = b.get("auroc_ci")
        if ci:
            ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        elif b.get("auroc_ci_error"):
            ci_s = "[n/a: " + b["auroc_ci_error"] + "]"
        else:
            ci_s = "[n/a]"
        print(f"  {name:26} n={b['n']:<6} pos={b['pos']:<5} neg={b['neg']:<5} "
              f"AUROC={b['auroc']:.4f} {ci_s} "
              f"TPR@1%FPR={b['tpr_at_1pct_fpr']:.4f}")

    print("OVERALL")
    show(rep["split"][0] if rep["split"] else "all", rep["overall"])
    for field, blocks in rep.get("strata", {}).items():
        print(f"\nBY {field.upper()}")
        for k, b in blocks.items():
            show(k[:26], b)
    lr = rep.get("label_rule") or {}
    print(f"\nlabel rule   calibrated={lr.get('calibrated')}  "
          f"upstream abstain rate={lr.get('upstream_abstain_rate')}")
    print(f"             {lr.get('note')}")
    if rep.get("errors"):
        print(f"\nWARNING: {rep['errors']} rows had scoring errors")
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())
