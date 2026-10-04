"""Metrics and calibration for Phase 0.

Dev tooling - deliberately NOT part of the installed `aludam` package so it
does not pull pandas/sklearn into the runtime dependency set.

Label convention used everywhere in this module:

    1 = synthetic / AI / fake / generated
    0 = authentic / human / real

MAGE ships the *opposite* convention (0 = machine, 1 = human). Conversion
happens at ingest, never here, so a silently inverted label cannot survive
into a reported number.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

__all__ = [
    "auroc",
    "tpr_at_fpr",
    "roc_curve_points",
    "bootstrap_auroc",
    "paired_bootstrap_delta",
    "fit_calibrator",
    "reliability_bins",
]

# Below this, an AUROC/TPR number is not worth printing - it means the run
# produced degenerate input (single class, or too few rows for the resamples).
MIN_ROWS = 30
MIN_POS = 5
MIN_NEG = 5


def _as_arrays(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} vs labels {y.shape}")
    if s.ndim != 1:
        raise ValueError("expected 1-D inputs")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN/inf")
    classes = np.unique(y)
    if not set(classes.tolist()) <= {0, 1}:
        raise ValueError(f"labels must be 0/1, got {sorted(classes.tolist())}")
    if len(classes) < 2:
        raise ValueError(
            "AUROC is undefined with a single class present "
            f"(only {classes.tolist()[0]!r})"
        )
    return s, y


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC with a higher score meaning "more likely synthetic"."""
    s, y = _as_arrays(scores, labels)
    if len(s) < MIN_ROWS or y.sum() < MIN_POS or (len(y) - y.sum()) < MIN_NEG:
        raise ValueError(
            f"too few rows for a stable AUROC: n={len(s)} "
            f"pos={int(y.sum())} neg={int(len(y) - y.sum())} "
            f"(need n>={MIN_ROWS}, pos>={MIN_POS}, neg>={MIN_NEG})"
        )
    return float(roc_auc_score(y, s))


def roc_curve_points(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fpr, tpr, thresholds) - thresholds are ascending in score."""
    s, y = _as_arrays(scores, labels)
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def tpr_at_fpr(
    scores: Sequence[float], labels: Sequence[int], target_fpr: float = 0.01
) -> float:
    """Maximum TPR achievable at FPR <= target_fpr.

    Uses the conservative reading: we may only *lower* the threshold relative
    to the operating point that first crosses target_fpr, so we take the best
    TPR among all thresholds whose empirical FPR is still <= target.
    """
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError(f"target_fpr must be in [0,1], got {target_fpr}")
    fpr, tpr, _ = roc_curve_points(scores, labels)
    eligible = tpr[fpr <= target_fpr]
    if eligible.size == 0:
        return 0.0
    return float(eligible.max())


def bootstrap_auroc(
    scores: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[object] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Grouped percentile bootstrap for AUROC.

    Returns (point_estimate, lo, hi).

    When `groups` is given, whole groups are resampled with replacement. This
    is the correct unit when rows share a source document - sampling rows
    independently would let near-duplicate text leak across the resample and
    shrink the interval artificially.
    """
    s, y = _as_arrays(scores, labels)
    point = auroc(s, y)
    rng = np.random.default_rng(seed)

    if groups is None:
        g = np.arange(len(s))
    else:
        g = np.asarray(groups)
        if g.shape != y.shape:
            raise ValueError(f"groups {g.shape} vs labels {y.shape}")

    uniq = np.unique(g)
    if len(uniq) < 5:
        raise ValueError(
            f"need >=5 distinct groups to bootstrap, got {len(uniq)}"
        )

    vals: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(g == p) for p in picked])
        yi = y[idx]
        if yi.min() == yi.max():
            continue  # resample lost a class - skip, as is standard
        vals.append(float(roc_auc_score(yi, s[idx])))

    if len(vals) < max(100, n_boot // 10):
        raise ValueError(
            f"only {len(vals)}/{n_boot} resamples were usable; "
            "the resample is too often degenerate to form an interval"
        )

    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return point, lo, hi


def paired_bootstrap_delta(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
    groups: Sequence[object] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap of (A - B) AUROC on the *same* rows.

    Returns (delta, lo, hi). A gain is real only if lo > 0: paired
    resampling cancels most row-level noise, which is why this interval is
    far tighter than comparing two independent intervals.
    """
    sa, y = _as_arrays(scores_a, labels)
    sb, y2 = _as_arrays(scores_b, labels)
    if not np.array_equal(y, y2):
        raise ValueError("models must be scored on identical labels")
    if sa.shape != sb.shape:
        raise ValueError("models must be scored on identical rows")

    delta = auroc(sa, y) - auroc(sb, y)
    rng = np.random.default_rng(seed)

    g = np.arange(len(y)) if groups is None else np.asarray(groups)
    if g.shape != y.shape:
        raise ValueError(f"groups {g.shape} vs labels {y.shape}")
    uniq = np.unique(g)
    if len(uniq) < 5:
        raise ValueError(f"need >=5 distinct groups, got {len(uniq)}")

    vals: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(g == p) for p in picked])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        vals.append(
            float(roc_auc_score(yi, sa[idx]) - roc_auc_score(yi, sb[idx]))
        )

    if len(vals) < max(100, n_boot // 10):
        raise ValueError(f"only {len(vals)}/{n_boot} resamples usable")

    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return float(delta), lo, hi


def fit_calibrator(
    scores: Sequence[float],
    labels: Sequence[int],
    method: str = "isotonic",
):
    """Fit a monotone map from raw score -> P(synthetic | score).

    Returns a callable. Isotonic is the default because the relationship
    between a detector's score and true probability is not sigmoid - it is
    usually flat at the extremes, which Platt sigmoid cannot represent.

    Never fit on the test split.
    """
    s, y = _as_arrays(scores, labels)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s, y)
        return lambda x: np.clip(np.atleast_1d(iso.predict(x)), 0.0, 1.0)
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        # Platt scaling is logistic regression on the raw score.
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(s.reshape(-1, 1), y)
        return lambda x: lr.predict_proba(
            np.asarray(x, dtype=np.float64).reshape(-1, 1)
        )[:, 1]
    raise ValueError(f"unknown method {method!r} (use isotonic|platt)")


def reliability_bins(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """(mean_predicted, observed_rate, count) per equal-width bin."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.shape != y.shape:
        raise ValueError("probs and labels must match")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            out.append((float((lo + hi) / 2), float("nan"), 0))
            continue
        out.append((float(p[m].mean()), float(y[m].mean()), n))
    return out


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected calibration error - weighted mean absolute bin gap."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    total = len(p)
    if total == 0:
        raise ValueError("empty input")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            m = (p >= lo) & (p < hi)
        else:
            m = (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        err += (n / total) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(err)
