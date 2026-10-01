"""
Runtime memory limits for optional heavy dependencies.

Some features pull in very large libraries (mediapipe -> full TensorFlow
~380MB RSS, ultralytics -> YOLO + torch ~180MB). On a small container (Render's
512MB-1GB plans) loading them takes the gunicorn worker past its memory limit
and the service goes into an OOM restart loop. These helpers let each feature
decide at runtime whether it can afford to load.

DEEPGUARD_HEAVY_DEPS=on|off forces the choice; the default (`auto`) allows a
dependency only when the instance limit leaves enough headroom above the
baseline working set.
"""
import os
from typing import Optional, Tuple

_GIB = 1024 ** 3

# Below this container limit the baseline worker is already close to the
# ceiling, so optional heavy dependencies stay off.
_AUTO_MIN_LIMIT = 1.5 * _GIB


def memory_limit_bytes() -> Optional[int]:
    """Container/cgroup memory limit in bytes, or None when unknown/unlimited."""
    for path in ("/sys/fs/cgroup/memory.max",                  # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 prints a huge number when no limit is configured.
        if 0 < value < (1 << 60):
            return value
    return None


def allow(feature: str, min_limit_gib: float = None) -> Tuple[bool, str]:
    """
    (allowed, reason) for an optional heavy dependency.

    Explicit DEEPGUARD_HEAVY_DEPS=on/off wins; otherwise `auto` allows the
    feature only when the container limit (if any) is at least
    min_limit_gib (default 1.5GB).
    """
    flag = os.environ.get("DEEPGUARD_HEAVY_DEPS", "auto").strip().lower()
    if flag in ("on", "1", "true", "yes"):
        return True, f"{feature}: enabled via DEEPGUARD_HEAVY_DEPS"
    if flag in ("off", "0", "false", "no"):
        return False, f"{feature}: disabled via DEEPGUARD_HEAVY_DEPS"

    limit = memory_limit_bytes()
    if limit is None:
        return True, f"{feature}: no container memory limit detected"

    threshold = (_AUTO_MIN_LIMIT if min_limit_gib is None else min_limit_gib * _GIB)
    if limit < threshold:
        return False, (f"{feature}: instance memory limit {limit / _GIB:.1f}GB is below the "
                       f"{threshold / _GIB:.1f}GB needed (set DEEPGUARD_HEAVY_DEPS=on to force)")
    return True, f"{feature}: instance memory limit {limit / _GIB:.1f}GB"
