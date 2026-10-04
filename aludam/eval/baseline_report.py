"""Produce the Phase 0 baseline report.

Consumes what Phase 0 already produced - the environment snapshot, one or
more run JSONLs, and optionally a threshold artifact - and writes a single
markdown document. Its job is to be the thing a gate can be decided from, so
it carries the environment next to the numbers: a metric without the box it
was measured on is not evidence (ALUDAM_PLAN.md sec 5.1).

Two-arm mode answers the question Phase 3 is gated on: *did the change help?*
It reports the paired-bootstrap delta (both arms scored the same rows, so the
comparison is within-row and far tighter than two independent CIs), and then
applies the stated gate rather than leaving the reader to eyeball it:

  * lower bound of the 95% CI on d(AUROC) > 0
  * TPR@1%FPR not worse than the baseline
  * no stratum regresses by more than ~3 points

Run:
  python eval/baseline_report.py --a run.jsonl [--b changed.jsonl] \\
      --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent

if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))
    from eval import metrics as _metrics  # noqa: E402,F401
    from eval.run import (  # noqa: E402,F401
        ENV_SNAPSHOT, _block, _env_head, read_jsonl, src_domain, summarize,
    )
else:
    from . import metrics as _metrics  # noqa: F401
    from .run import (  # noqa: F401
        ENV_SNAPSHOT, _block, _env_head, read_jsonl, src_domain, summarize,
    )

# Regression tolerance for a stratum, in AUROC points (plan sec 5.1).
STRATUM_REGRESSION_TOL = 0.03


def _load_env() -> Dict[str, Any]:
    if not ENV_SNAPSHOT.exists():
        return {"available": False}
    try:
        raw = json.loads(ENV_SNAPSHOT.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not isinstance(raw, dict):
        return {"available": False, "error": "snapshot is not an object"}
    # collect() does not stamp `available` on itself; add it here or the
    # snapshot is silently reported as missing and the report claims the
    # numbers are unaudited.
    raw.setdefault("available", True)
    return raw


def label_diagnostics(recs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the uncalibrated label rule actually did.

    Not a performance metric - the thresholds are unfitted (decision D4) -
    but it is the thing most likely to be mistaken for one, so it is stated
    explicitly: a detector that never emits `ai` has AUROC fine and is still
    useless to anyone consuming `label`.
    """
    if not recs:
        return {}
    upstream = Counter(r.get("upstream_label") for r in recs)
    out = Counter(r.get("label_out") for r in recs)
    pos = [r for r in recs if r["label"] == 1]
    neg = [r for r in recs if r["label"] == 0]
    return {
        "n": len(recs),
        "upstream": dict(upstream),
        "label_out": dict(out),
        "calibrated": any(r.get("calibrated") for r in recs),
        "abstain_rate": round(
            upstream.get("UNCERTAIN", 0) / len(recs), 4),
        "synthetic_called_ai": sum(1 for r in pos if r.get("label_out") == "ai"),
        "synthetic_total": len(pos),
        "authentic_called_real": sum(1 for r in neg if r.get("label_out") == "real"),
        "authentic_total": len(neg),
        "score_max": round(max((r["score"] for r in recs), default=0.0), 4),
        "score_min": round(min((r["score"] for r in recs), default=0.0), 4),
    }


def strata(recs: Sequence[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in recs:
        v = r.get(field)
        if v is None and field == "src_domain" and r.get("src"):
            v = src_domain(str(r["src"]))
        buckets[str(v if v is not None else "<none>")].append(r)
    return {k: _block(v, 1000, 0) for k, v in sorted(buckets.items())}


def compare(
    a: Sequence[Dict[str, Any]],
    b: Sequence[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired delta plus the plan's gate, on rows present in both arms."""
    idx_a = {r["key"]: r for r in a}
    idx_b = {r["key"]: r for r in b}
    shared = sorted(set(idx_a) & set(idx_b))

    out: Dict[str, Any] = {"shared_rows": len(shared)}
    if len(shared) < 30:
        out["error"] = (
            f"only {len(shared)} rows scored by both arms; a paired "
            "comparison needs both arms to have scored the same rows"
        )
        return out

    sa = [float(idx_a[k]["score"]) for k in shared]
    sb = [float(idx_b[k]["score"]) for k in shared]
    y = [int(idx_a[k]["label"]) for k in shared]
    groups = [idx_a[k].get("group") for k in shared]

    auroc_a = _metrics.auroc(sa, y)
    auroc_b = _metrics.auroc(sb, y)
    out["arm_a_auroc"] = round(auroc_a, 4)
    out["arm_b_auroc"] = round(auroc_b, 4)
    out["delta_auroc"] = round(auroc_b - auroc_a, 4)

    try:
        point, lo, hi = _metrics.paired_bootstrap_delta(
            sb, sa, y, groups, n_boot=n_boot, seed=seed)
        out["delta_ci"] = [round(lo, 4), round(hi, 4)]
        out["delta_point"] = round(point, 4)
    except Exception as exc:
        out["delta_ci_error"] = f"{type(exc).__name__}: {exc}"

    tpr_a = _metrics.tpr_at_fpr(sa, y, 0.01)
    tpr_b = _metrics.tpr_at_fpr(sb, y, 0.01)
    out["tpr_a"] = round(tpr_a, 4)
    out["tpr_b"] = round(tpr_b, 4)
    out["tpr_not_dropped"] = bool(tpr_b >= tpr_a - 1e-9)

    # Per-stratum regression check.
    regressions = []
    by_field: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list))
    for i, k in enumerate(shared):
        r = idx_a[k]
        by_field["src_domain"][r.get("src_domain")
                                or src_domain(str(r.get("src", "")))].append(i)
        by_field["len_bucket"][str(r.get("len_bucket", "<none>"))].append(i)
    stratum_rows = {}
    for field, groups_map in by_field.items():
        for name, idxs in groups_map.items():
            if len(idxs) < 30:
                continue
            ya = [y[i] for i in idxs]
            if len(set(ya)) < 2:
                continue
            a_ = _metrics.auroc([sa[i] for i in idxs], ya)
            b_ = _metrics.auroc([sb[i] for i in idxs], ya)
            d = b_ - a_
            stratum_rows[f"{field}:{name}"] = {
                "n": len(idxs), "a": round(a_, 4), "b": round(b_, 4),
                "delta": round(d, 4),
            }
            if d < -STRATUM_REGRESSION_TOL:
                regressions.append(
                    f"{field}:{name} {a_:.3f} -> {b_:.3f} ({d:+.3f})")
    out["strata"] = stratum_rows
    out["stratum_regressions"] = regressions

    # The gate itself, stated as pass/fail per clause so a reader can see
    # which clause failed rather than a single opaque verdict.
    ci = out.get("delta_ci")
    clauses = {
        "delta_ci_lower_gt_0": bool(ci and ci[0] > 0),
        "tpr_at_1pct_fpr_not_dropped": out["tpr_not_dropped"],
        "no_stratum_regresses_beyond_tol": not regressions,
        "enough_shared_rows": len(shared) >= 30,
    }
    if not ci:
        clauses["delta_ci_lower_gt_0"] = False
    out["gate"] = clauses
    out["gate_pass"] = all(clauses.values())
    return out


def _fmt_ci(b: Dict[str, Any]) -> str:
    ci = b.get("auroc_ci")
    if ci:
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    if b.get("auroc_ci_error"):
        return "[n/a]"
    return "[n/a]"


def render(
    runs: List[Dict[str, Any]],
    env: Dict[str, Any],
    cmp: Optional[Dict[str, Any]],
    notes: Sequence[str],
) -> str:
    L: List[str] = []
    w = L.append

    w("# Phase 0 baseline report")
    w("")

    # ---- environment -------------------------------------------------
    w("## Environment (the numbers below are only meaningful with this)")
    w("")
    if env.get("available"):
        git = env.get("git") or {}
        op = env.get("opencv") or {}
        beh = env.get("behaviour_env") or {}
        w("| field | value |")
        w("|---|---|")
        w(f"| git SHA | `{git.get('sha')}` |")
        w(f"| working tree | {'dirty' if git.get('dirty') else 'clean'} "
          f"({git.get('dirty_count', 0)} files) |")
        w(f"| python | {(env.get('python') or {}).get('version', '?').split()[0]} |")
        w(f"| cv2 active | {op.get('cv2_version')} "
          f"({op.get('owner')}) |")
        w(f"| OpenCV pip listing | {'<br>'.join(env.get('pip_opencv') or [])} |")
        face = beh.get("DEEPGUARD_FACE_DETECTOR") or {}
        w(f"| `DEEPGUARD_FACE_DETECTOR` | `{face.get('effective')}` "
          f"({'set' if face.get('set') else 'unset -> code default'}) |")
        w(f"| `HF_HUB_OFFLINE` | `{(beh.get('HF_HUB_OFFLINE') or {}).get('value')}` |")
        vers = env.get("library_versions") or {}
        w("| libraries | " + ", ".join(
            f"{k}={v}" for k, v in vers.items()) + " |")
        w("")
        if op.get("contrib_active"):
            w(f"> OpenCV: **two distributions installed**; the live one is "
              f"`{op.get('owner')}` (version match). The pinned "
              f"`opencv-python` files are shadowed. Per decision D1 this is "
              f"recorded, not deduped, before the baseline.")
            w("")
        if face.get("effective") == "haar":
            w("> Face detection: **Haar cascade**, not RetinaFace "
              "(`retina-face` pulls TensorFlow but is never imported by "
              "default). Any face-swap numbers here are Haar numbers.")
            w("")
    else:
        w("_No environment snapshot found - run `python eval/env_snapshot.py` "
          "first. Treat every number below as unaudited._")
        w("")

    # ---- runs --------------------------------------------------------
    for run in runs:
        rep = run["report"]
        w(f"## `{', '.join(rep['split'])}` - arm `{run['arm']}`")
        w("")
        o = rep["overall"]
        w(f"- rows scored: **{o.get('n', 0)}** "
          f"(pos={o.get('pos', 0)}, neg={o.get('neg', 0)})")
        w(f"- **AUROC {o.get('auroc', 'n/a')} {_fmt_ci(o)}**")
        w(f"- TPR@1%FPR {o.get('tpr_at_1pct_fpr', 'n/a')}")
        w(f"- file: `{rep.get('file')}`")
        w("")

        ld = label_diagnostics(run["recs"])
        if ld:
            w("Label rule while uncalibrated (decision D4) - "
              "**not** a performance metric:")
            w("")
            w(f"- upstream verdicts: {ld['upstream']}")
            w(f"- emitted labels: {ld['label_out']}")
            w(f"- abstain rate: {ld['abstain_rate']:.1%}")
            w(f"- synthetic rows labelled `ai`: "
              f"{ld['synthetic_called_ai']}/{ld['synthetic_total']}")
            w(f"- authentic rows labelled `real`: "
              f"{ld['authentic_called_real']}/{ld['authentic_total']}")
            w(f"- score range observed: {ld['score_min']} .. {ld['score_max']}")
            w("")
            if ld["synthetic_total"] and ld["synthetic_called_ai"] == 0:
                w("> **The detector never emitted `ai` on a single synthetic "
                  "row.** `label` is therefore not usable for rejection at "
                  "this operating point regardless of AUROC.")
                w("")

        for field in ("src_domain", "len_bucket"):
            blocks = rep.get("strata", {}).get(field)
            if not blocks:
                continue
            usable = {k: b for k, b in blocks.items() if "skipped" not in b}
            w(f"### Strata by `{field}`")
            w("")
            if not usable:
                w(f"_No `{field}` stratum reached the minimum row count yet "
                  f"({len(blocks)} strata present, all below the "
                  f"threshold)._")
                w("")
                continue
            w("| stratum | n | pos | neg | AUROC | 95% CI | TPR@1%FPR |")
            w("|---|---|---|---|---|---|---|")
            for k, b in usable.items():
                w(f"| {k} | {b['n']} | {b['pos']} | {b['neg']} | "
                  f"{b['auroc']:.4f} | {_fmt_ci(b)} | "
                  f"{b['tpr_at_1pct_fpr']:.4f} |")
            skipped = len(blocks) - len(usable)
            if skipped:
                w("")
                w(f"_{skipped} further stratum/strata omitted (below the "
                  f"minimum)._")
            w("")

    # ---- comparison / gate -------------------------------------------
    if cmp is not None:
        w("## Before / after")
        w("")
        if cmp.get("error"):
            w(f"_{cmp['error']}_")
            w("")
        else:
            w(f"- rows scored by both arms: **{cmp['shared_rows']}**")
            w(f"- AUROC: {cmp['arm_a_auroc']} -> {cmp['arm_b_auroc']} "
              f"(delta {cmp['delta_auroc']:+})")
            ci = cmp.get("delta_ci")
            if ci:
                w(f"- paired-bootstrap 95% CI on delta: "
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]")
            elif cmp.get("delta_ci_error"):
                w(f"- paired-bootstrap CI unavailable: {cmp['delta_ci_error']}")
            w(f"- TPR@1%FPR: {cmp['tpr_a']} -> {cmp['tpr_b']}")
            w("")
            w("### Gate")
            w("")
            w("| clause | result |")
            w("|---|---|")
            for k, v in (cmp.get("gate") or {}).items():
                w(f"| {k} | {'PASS' if v else 'FAIL'} |")
            verdict = "PASS - eligible to merge" if cmp.get("gate_pass") \
                else "FAIL - do not merge"
            w("")
            w(f"**Verdict: {verdict}**")
            w("")
            if cmp.get("stratum_regressions"):
                w("Regressing strata (beyond 3 points):")
                w("")
                for r in cmp["stratum_regressions"]:
                    w(f"- {r}")
                w("")

    if notes:
        w("## Notes")
        w("")
        for n in notes:
            w(f"- {n}")
        w("")

    w("---")
    w("_Generated by `eval/baseline_report.py`. Metrics from "
      "`eval/metrics.py`; bootstrap is grouped by source document._")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval.baseline_report", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline run JSONL")
    ap.add_argument("--b", default=None,
                    help="changed run JSONL, for a before/after comparison")
    ap.add_argument("--arm-a", default="baseline")
    ap.add_argument("--arm-b", default="candidate")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", action="append", default=[],
                    help="free-text note to append (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs = []
    for path, arm in ((args.a, args.arm_a),
                      (args.b, args.arm_b) if args.b else (None, None)):
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            print(f"missing run file: {p}")
            return 1
        recs = read_jsonl(p)
        if not recs:
            print(f"no scored rows in {p}")
            return 1
        rep = _run_report(p, args.n_boot, args.seed)
        runs.append({"arm": arm, "recs": recs, "report": rep})

    cmp = None
    if args.b and len(runs) == 2:
        cmp = compare(runs[0]["recs"], runs[1]["recs"],
                      n_boot=args.n_boot, seed=args.seed)
        cmp["arm_a"] = args.arm_a
        cmp["arm_b"] = args.arm_b

    env = _load_env()
    notes = list(args.note)
    if not env.get("available"):
        notes.insert(0, "NO ENVIRONMENT SNAPSHOT - numbers are unaudited.")

    md = render(runs, env, cmp, notes)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"report -> {args.out}")
    else:
        print(md)
    return 0


def _run_report(path: Path, n_boot: int, seed: int) -> Dict[str, Any]:
    return summarize(Path(path), n_boot, seed, ["src_domain", "len_bucket"])


if __name__ == "__main__":
    sys.exit(main())
