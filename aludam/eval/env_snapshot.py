"""Capture the exact environment a baseline was measured on.

A number without its environment is not a measurement. This records
everything needed to tell "the model got better" apart from "the box changed":

* which ``cv2`` is actually importable, and which pip distribution owns it
  (this venv carries two OpenCV trees; whichever installed last owns the
  files, so the baseline must say which one is live);
* ``pip freeze`` plus a focused OpenCV listing;
* git SHA and whether the tree is dirty;
* pinned HF revisions for every cached snapshot;
* the flags that change detector behaviour: ``HF_HUB_OFFLINE`` and
  ``DEEPGUARD_FACE_DETECTOR`` (default ``haar`` - so a baseline taken now is
  a Haar baseline).

Run:  python eval/env_snapshot.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data" / "env_snapshot.json"

# Environment variables that change what the detectors do. Values recorded
# verbatim EXCEPT secrets, which are recorded as set/unset only.
BEHAVIOUR_VARS = (
    "DEEPGUARD_FACE_DETECTOR",
    "DEEPGUARD_HEAVY_DEPS",
    "HF_HUB_OFFLINE",
    "DEEPFAKE_ALLOW_DOWNLOADS",
    "ALUDAM_MODELS_DIR",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "IMAGE_DETECTOR_MODEL",
    "TEXT_DETECTOR_PROBE_MODEL",
    "AUDIO_DETECTOR_MODEL",
    "AUDIO_CLASSIFIER_PATH",
)

# Never print these - presence only.
SECRET_VARS = ("GEMINI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Defaults the legacy code applies when the variable is unset, so a report can
# say what actually ran rather than just "None". Source of truth: models/*.py.
BEHAVIOUR_DEFAULTS = {
    "DEEPGUARD_FACE_DETECTOR": "haar",
    "DEEPGUARD_HEAVY_DEPS": "auto",
    "IMAGE_DETECTOR_MODEL": "umm-maybe/AI-image-detector",
    "TEXT_DETECTOR_PROBE_MODEL": "gpt2",
    "AUDIO_DETECTOR_MODEL": "MelodyMachine/Deepfake-audio-detection-V2",
    "AUDIO_CLASSIFIER_PATH": "models/audio_rf_classifier.pkl",
}


def behaviour_env() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in BEHAVIOUR_VARS:
        raw = os.environ.get(key)
        default = BEHAVIOUR_DEFAULTS.get(key)
        out[key] = {
            "set": raw is not None,
            "value": raw,
            "code_default": default,
            "effective": raw if raw is not None else default,
        }
    for key in SECRET_VARS:
        out[key] = {"set": bool(os.environ.get(key)), "value": "<redacted>"}
    return out


def _run(argv: List[str], timeout: int = 120) -> str:
    try:
        out = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<failed: {type(exc).__name__}: {exc}>"
    if out.returncode != 0:
        err = (out.stderr or out.stdout or "").strip()
        return f"<exit {out.returncode}: {err[:400]}>"
    return out.stdout.strip()


def pip_freeze() -> str:
    return _run([sys.executable, "-m", "pip", "freeze"])


def pip_opencv() -> List[str]:
    """The `pip list | grep -i opencv` equivalent (Windows has no grep)."""
    raw = _run([sys.executable, "-m", "pip", "list", "--format=freeze"])
    if raw.startswith("<"):
        return [raw]
    return [ln for ln in raw.splitlines() if "opencv" in ln.lower()]


def opencv_details() -> Dict[str, Any]:
    """Which OpenCV is importable, and which distribution owns those files."""
    info: Dict[str, Any] = {
        "importable": False,
        "cv2_version": None,
        "cv2_file": None,
        "cv2_realpath": None,
        "owner": None,
        "candidates": {},
        "build_information": None,
    }
    try:
        import cv2  # noqa: PLC0415
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["importable"] = True
    info["cv2_version"] = getattr(cv2, "__version__", None)
    info["cv2_file"] = getattr(cv2, "__file__", None)
    real = Path(cv2.__file__).resolve() if cv2.__file__ else None
    info["cv2_realpath"] = str(real) if real else None

    # Each installed opencv distribution drops files into the same
    # site-packages/cv2, so path comparison alone cannot always say who won
    # the "installed last" race: locate_file('cv2') yields the directory while
    # cv2.__file__ is a file inside it. Two independent signals settle it:
    #   1. directory/file containment, and
    #   2. version match - cv2.__version__ is stamped by whichever wheel
    #      actually wrote the module, so it identifies the owner outright.
    try:
        from importlib import metadata
    except Exception:
        metadata = None  # type: ignore[assignment]

    active_versions: List[str] = []
    if metadata is not None:
        for dist_name in (
            "opencv-python",
            "opencv-python-headless",
            "opencv-contrib-python",
            "opencv-contrib-python-headless",
        ):
            try:
                dist = metadata.distribution(dist_name)
            except metadata.PackageNotFoundError:
                continue
            version = dist.version
            try:
                claimed = Path(dist.locate_file("cv2")).resolve()
            except Exception:
                claimed = None

            path_match = bool(
                real and claimed
                and (real == claimed or real.parent == claimed
                     or claimed in real.parents)
            )
            # Compare the wheel's x.y.z against the loaded cv2's x.y.z.
            ver_match = False
            if real and version and info["cv2_version"]:
                own = ".".join(str(info["cv2_version"]).split(".")[:3])
                theirs = ".".join(str(version).split(".")[:3])
                ver_match = own == theirs

            info["candidates"][dist_name] = {
                "version": version,
                "cv2_path": str(claimed) if claimed else None,
                "path_match": path_match,
                "version_match": ver_match,
                "is_active": ver_match,
            }
            if ver_match:
                active_versions.append(dist_name)

        if len(active_versions) == 1:
            info["owner"] = active_versions[0]
        elif len(active_versions) > 1:
            info["owner"] = f"ambiguous: version matches {active_versions}"
        else:
            info["owner"] = (
                "ambiguous: cv2.__version__ "
                f"{info['cv2_version']!r} matched no installed OpenCV wheel"
            )

    # Contrib builds expose modules the plain wheel does not. Presence of any
    # of these is direct evidence that the contrib tree is the live one.
    info["contrib_modules"] = sorted(
        m for m in ("ximgproc", "face", "tracking", "text", "dnn_superres")
        if hasattr(cv2, m)
    )
    info["contrib_active"] = bool(info["contrib_modules"])

    try:
        # Cheap (in-memory) and worth keeping: build flags differ between
        # opencv-python and opencv-contrib builds.
        bi = cv2.getBuildInformation()
        info["build_information"] = bi if isinstance(bi, str) else str(bi)
    except Exception as exc:
        info["build_information"] = f"<unavailable: {type(exc).__name__}>"

    return info


def git_state() -> Dict[str, Any]:
    repo = _run(["git", "rev-parse", "HEAD"])
    if repo.startswith("<"):
        return {"sha": None, "error": repo}
    status = _run(["git", "status", "--porcelain"])
    dirty_lines = [ln for ln in status.splitlines() if ln.strip()]
    return {
        "sha": repo.strip(),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "dirty": bool(dirty_lines),
        "dirty_files": dirty_lines[:200],
        "dirty_count": len(dirty_lines),
        "subject": _run(["git", "log", "-1", "--format=%s"]).strip(),
    }


def hf_revisions() -> Dict[str, Any]:
    """Every HF snapshot cached locally, keyed by repo, with its commit SHA.

    This is what makes a run reproducible offline: the weights were fetched
    at these revisions, and nothing re-resolves them afterwards.
    """
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    for env in ("HUGGINGFACE_HUB_CACHE",):
        if os.environ.get(env):
            roots.append(Path(os.environ[env]))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    out: Dict[str, Any] = {"cache_roots": [], "repos": {}}
    seen = set()
    for root in roots:
        root = root.expanduser()
        if not root.is_dir() or str(root) in seen:
            continue
        seen.add(str(root))
        out["cache_roots"].append(str(root))
        for repo_dir in root.iterdir():
            if not repo_dir.is_dir() or not repo_dir.name.startswith(
                ("models--", "datasets--")
            ):
                continue
            snaps = repo_dir / "snapshots"
            if not snaps.is_dir():
                continue
            shas = sorted(p.name for p in snaps.iterdir() if p.is_dir())
            if shas:
                out["repos"][repo_dir.name] = shas
    return out


def datasets_manifest() -> Dict[str, Any]:
    """Our own pinned dataset SHAs (from eval/fetch.py)."""
    path = HERE / "data" / "manifest.json"
    if not path.exists():
        return {"available": False}
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "datasets": {
            name: {
                "repo": d.get("repo"),
                "revision": d.get("revision"),
                "license": d.get("license"),
                "files": {
                    f: {
                        "sha256": v.get("sha256"),
                        "size_bytes": v.get("size_bytes"),
                    }
                    for f, v in (d.get("files") or {}).items()
                },
            }
            for name, d in man.items()
            if isinstance(d, dict)
        },
    }


def runtime_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    for mod in (
        "numpy", "pandas", "sklearn", "torch", "transformers",
        "PIL", "librosa", "onnxruntime", "cv2", "scipy",
    ):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "?")
        except Exception as exc:
            versions[mod] = f"MISSING ({type(exc).__name__})"
    return versions


def collect() -> Dict[str, Any]:
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "behaviour_env": behaviour_env(),
        "git": git_state(),
        "opencv": opencv_details(),
        "pip_opencv": pip_opencv(),
        "pip_freeze": pip_freeze(),
        "library_versions": runtime_versions(),
        "hf_revisions": hf_revisions(),
        "datasets": datasets_manifest(),
    }


def human_report(snap: Dict[str, Any]) -> str:
    cv = snap.get("opencv", {})
    lines = [
        "ENVIRONMENT SNAPSHOT",
        "=" * 72,
        f"git          : {snap.get('git', {}).get('sha')} "
        f"(dirty={snap.get('git', {}).get('dirty')})",
        f"python       : {snap.get('python', {}).get('version', '').split()[0]}",
        f"platform     : {snap.get('platform', {}).get('system')}",
        "",
        "-- OpenCV ---------------------------------------------------------",
        f"cv2.__version__: {cv.get('cv2_version')}",
        f"cv2.__file__   : {cv.get('cv2_file')}",
        f"active owner   : {cv.get('owner')}",
        f"contrib modules: {', '.join(cv.get('contrib_modules') or []) or 'none'}"
        f"  (contrib_active={cv.get('contrib_active')})",
        "pip -i opencv  :",
    ]
    for ln in snap.get("pip_opencv", []) or ["  (none)"]:
        lines.append(f"    {ln}")
    lines.append("  installed candidates:")
    for name, meta in (cv.get("candidates") or {}).items():
        mark = "  <-- ACTIVE" if meta.get("is_active") else ""
        lines.append(
            f"    {name:32} {meta.get('version'):<12}"
            f" path_match={meta.get('path_match')} "
            f"version_match={meta.get('version_match')}{mark}"
        )
    lines += [
        "",
        "-- behaviour flags ----------------------------------------------",
    ]
    for k, v in (snap.get("behaviour_env") or {}).items():
        if v.get("value") is not None:
            lines.append(f"    {k} = {v['value']!r}   (set)")
        elif v.get("code_default") is not None:
            lines.append(
                f"    {k} unset -> code default {v['code_default']!r}"
            )
        else:
            lines.append(f"    {k} = <unset>")
    lines += ["", "-- pinned HF revisions ------------------------------------------"]
    repos = (snap.get("hf_revisions") or {}).get("repos", {})
    if not repos:
        lines.append("    (no HF snapshots cached)")
    for repo, shas in sorted(repos.items()):
        lines.append(f"    {repo}")
        for s in shas:
            lines.append(f"        {s}")
    lines += ["", "-- pinned dataset revisions --------------------------------------"]
    ds = snap.get("datasets", {})
    if not ds.get("available"):
        lines.append("    (manifest not found - run eval/fetch.py)")
    for name, meta in (ds.get("datasets") or {}).items():
        lines.append(f"    {name:10} {meta.get('repo')} @ {str(meta.get('revision'))[:12]}")
    lines += ["", "-- key library versions ------------------------------------------"]
    for k, v in (snap.get("library_versions") or {}).items():
        lines.append(f"    {k:14} {v}")
    lines += ["=" * 72]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="write JSON here")
    ap.add_argument("--quiet", action="store_true", help="suppress the report")
    args = ap.parse_args()

    snap = collect()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not args.quiet:
        print(human_report(snap))
    print(f"\njson -> {out_path}")

    cv = snap.get("opencv", {})
    if isinstance(cv.get("owner"), str) and "ambiguous" in cv["owner"]:
        print(
            "\nWARNING: more than one OpenCV distribution is installed and "
            "none cleanly matched cv2.__file__. Record this alongside any "
            "face-detection numbers."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
