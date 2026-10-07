"""DeepGuard v2 scan response envelope.

Builds the response body:

    request_id, status, media_type, summary, ensemble_results,
    metadata_analysis, reason, explanation, runtime

plus the legacy keys the dashboard still reads (success, file_type, result).

reason      -- dense, technical, professional-grade derivation of the
               detector's own numbers (family scores, margins, intervals).
               When aludam starts emitting ``result["reason"]`` (Phase 3)
               that value is passed through untouched.

explanation -- plain-language, self-explanatory rewrite produced by an LLM.
               Provider is auto-detected from whichever key env var is set:

                   OPENAI_API_KEY   -> OpenAI-compatible (OPENAI_BASE_URL
                                       optional, works with Groq/OpenRouter/Ollama)
                   GEMINI_API_KEY   -> Google Gemini
                   ANTHROPIC_API_KEY-> Anthropic Claude

               Model name comes from LLM_MODEL (required to call an LLM).
               If no key/model is configured, or the call fails, a
               deterministic fallback explanation is generated so the field
               is never empty.  Set LLM_DISABLE=1 to force the fallback.
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Signal vocabulary (Docs/ALUDAM_PLAN.md §7 signal name mapping)
# ---------------------------------------------------------------------------
SIGNAL_MAP = {
    "face_swap": "FACE_SWAP",
    "fully_ai_generated": "FULLY_AI_GENERATED",
    "ai_edited_region": "SPLICE_EDIT",
    "splice": "SPLICE_EDIT",
    "temporal_inconsistency": "TEMPORAL_INCONSISTENCY",
    "audio_visual_sync": "AV_SYNC_MISMATCH",
    "av_sync": "AV_SYNC_MISMATCH",
    "provenance_ai_detected": "AI_PROVENANCE_TAG",
    "provenance": "AI_PROVENANCE_TAG",
    "exif_software": "GENERATOR_SIGNATURE",
    "software_signature": "GENERATOR_SIGNATURE",
    "stylometry_signal": "STYLOMETRY_ANOMALY",
    "stylometry": "STYLOMETRY_ANOMALY",
    "pattern_signal": "AI_PHRASE_PATTERN",
    "pattern": "AI_PHRASE_PATTERN",
    "watermark_signal": "WATERMARK_DETECTED",
    "watermark": "WATERMARK_DETECTED",
    "ai_generated": "AI_GENERATED",
    "bombek1": "AI_GENERATED",
    "t2v_generated": "T2V_GENERATED",
    "ai_music": "AI_MUSIC",
}

MANIP_FAMILIES = {
    "face_swap", "ai_edited_region", "splice",
    "temporal_inconsistency", "audio_visual_sync", "av_sync",
}
GEN_FAMILIES = {
    "fully_ai_generated", "provenance_ai_detected", "provenance",
    "ai_generated", "t2v_generated", "ai_music",
}

FAMILY_CN = {
    "provenance": "provenance",
    "fully_ai_generated": "fully_ai_generated",
    "ai_edited_region": "ai_edited_region",
    "face_swap": "face_swap",
    "temporal_inconsistency": "temporal_inconsistency",
    "audio_visual_sync": "audio_visual_sync",
}


def _signal(key: str) -> str:
    k = str(key).strip().lower()
    if k in SIGNAL_MAP:
        return SIGNAL_MAP[k]
    for stem, mapped in SIGNAL_MAP.items():
        if stem in k:
            return mapped
    return str(key).upper()


def _dedup(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fired(result: Dict[str, Any]) -> List[str]:
    ft = result.get("fake_type")
    if isinstance(ft, list):
        return [str(x) for x in ft]
    return []


def _confidence_pct(result: Dict[str, Any]) -> float:
    try:
        c = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        c = 0.0
    if 0.0 < c <= 1.0:
        c *= 100.0
    return round(min(max(c, 0.0), 100.0), 1)


def _confidence_label(pct: float) -> str:
    if pct >= 75:
        return "HIGH"
    if pct >= 55:
        return "MEDIUM"
    return "LOW"


def _verdict_and_basis(media: str, result: Dict[str, Any]) -> Tuple[str, str]:
    """Map a detector label to the plan's verdict vocabulary."""
    label = str(result.get("label", "")).upper()

    if "fused_fake_probability" in result or media == "FUSION":
        # fusion engine verdicts
        if label in ("LIKELY_SYNTHETIC",):
            return "SYNTHETIC", "generation"
        if label in ("LIKELY_AUTHENTIC",):
            return "AUTHENTIC", "none"
        if label in ("INDETERMINATE", "UNCERTAIN"):
            return "INCONCLUSIVE", "insufficient"

    if label in ("AUTHENTIC", "HUMAN_WRITTEN", "REAL"):
        return "AUTHENTIC", "none"
    if label in ("UNCERTAIN", "INDETERMINATE"):
        return "INCONCLUSIVE", "insufficient"

    # FAKE / SYNTHETIC / AI_GENERATED style labels: is the claim generation
    # or manipulation of real content? (ALUDAM_PLAN §7.1 precedence: if both
    # fired, the more specific claim -- ALTERED -- wins.)
    fired = {f.lower() for f in _fired(result)}
    manip = fired & MANIP_FAMILIES
    gen = fired & GEN_FAMILIES
    if media == "AUDIO":
        return "SYNTHETIC", "generation"
    if media == "TEXT":
        return "SYNTHETIC", "generation"
    if manip:
        return "ALTERED", "manipulation"
    if gen:
        return "SYNTHETIC", "generation"
    if label in ("FAKE", "SYNTHETIC", "AI_GENERATED"):
        # fired families missing (e.g. mock detectors): keep it honest
        return "SYNTHETIC", "generation"
    return "INCONCLUSIVE", "insufficient"


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _signals_for(media: str, result: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if media in ("IMAGE", "VIDEO", "FUSION"):
        out += [_signal(f) for f in _fired(result)]
    if media == "AUDIO":
        src = result.get("signal_source")
        if src and str(src).lower() not in ("none", "unknown", "heuristic"):
            out.append(_signal(str(src)))
        label = str(result.get("label", "")).upper()
        if label == "SYNTHETIC" and not out:
            out.append("SYNTHETIC_SPEECH_RESONANCE")
    if media == "TEXT":
        checks = (
            ("stylometry_signal", "STYLOMETRY_ANOMALY"),
            ("pattern_signal", "AI_PHRASE_PATTERN"),
            ("watermark_signal", "WATERMARK_DETECTED"),
            ("detectgpt_signal", "DETECTGPT_ANOMALY"),
        )
        for key, name in checks:
            v = _num(result.get(key))
            if v is not None and v >= 0.5:
                out.append(name)
        if result.get("perplexity_signal") is not None and _num(result.get("ai_probability") or 0) and float(result.get("ai_probability", 0)) >= 0.65:
            out.append("LOW_PERPLEXITY")
    return _dedup(out)


def _ensemble(media: str, result: Dict[str, Any],
              modality_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    def grp(score: Optional[float], signals: List[str]) -> Dict[str, Any]:
        g: Dict[str, Any] = {}
        if score is not None:
            g["score"] = round(float(score), 3)
        if signals:
            g["signals_detected"] = signals
        return g

    if media in ("IMAGE", "VIDEO"):
        fs = result.get("family_scores") or {}
        manip_parts = [_num(fs.get(k)) for k in
                       ("face_swap", "ai_edited_region", "temporal_inconsistency")]
        manip = max([p for p in manip_parts if p is not None], default=None)
        gen_parts = [_num(fs.get(k)) for k in ("fully_ai_generated", "provenance")]
        gen = max([p for p in gen_parts if p is not None], default=None)
        fired = _fired(result)
        out: Dict[str, Any] = {}
        if manip is not None:
            out["visual_manipulation"] = grp(
                manip, _dedup([_signal(f) for f in fired
                               if f.lower() in MANIP_FAMILIES]))
        if gen is not None:
            out["visual_generation"] = grp(
                gen, _dedup([_signal(f) for f in fired
                             if f.lower() in GEN_FAMILIES]))
        if media == "VIDEO":
            av = _num((result.get("family_scores") or {}).get("audio_visual_sync"))
            if av is not None:
                out["audio_manipulation"] = grp(
                    av, ["AV_SYNC_MISMATCH"] if av >= 0.5 else [])
        return out

    if media == "AUDIO":
        sig = _signals_for("AUDIO", result)
        return {"audio_manipulation": grp(result.get("fake_probability"), sig)}

    if media == "TEXT":
        return {"text_generation": grp(result.get("ai_probability"),
                                       _signals_for("TEXT", result))}

    if media == "FUSION" and modality_results:
        out = {}
        for mod, res in modality_results.items():
            m = str(mod).upper()
            if m in ("IMAGE", "VIDEO"):
                out.update(_ensemble(m, res))
            elif m == "AUDIO":
                out.update(_ensemble("AUDIO", res))
            elif m == "TEXT":
                out.update(_ensemble("TEXT", res))
        return out

    return {}


def _metadata(media: str, result: Dict[str, Any],
              modality_results: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if media == "IMAGE":
        fs = result.get("family_scores") or {}
        prov = _num(fs.get("provenance")) or 0.0
        return {
            "has_c2pa": prov >= 0.5 or "provenance_ai_detected" in _fired(result),
            "software_signature": "Unknown / Stripped",
        }
    if media == "VIDEO":
        # container-level C2PA is not probed for video yet -- stay honest
        return {"has_c2pa": None, "software_signature": "Unknown / Stripped"}
    if media == "FUSION" and modality_results:
        for mod in ("image", "IMAGE"):
            if mod in modality_results:
                return _metadata("IMAGE", modality_results[mod])
        return {"has_c2pa": None, "software_signature": "Unknown / Stripped"}
    return None


def _overall(media: str, result: Dict[str, Any]) -> Optional[float]:
    v = None
    if "fused_fake_probability" in result:
        v = _num(result["fused_fake_probability"])
    elif media == "AUDIO":
        v = _num(result.get("fake_probability"))
    elif media == "TEXT":
        v = _num(result.get("ai_probability"))
    else:
        v = _num(result.get("p_synthetic"))
        if v is None:
            v = _num((result.get("family_scores") or {}).get("fused_score"))
    if v is None:
        return None
    return round(min(max(v, 0.0), 1.0), 4)


def _fmt(v: Any, nd: int = 3) -> str:
    n = _num(v)
    if n is None:
        return str(v)
    return f"{n:.{nd}f}"


def build_reason(media: str, result: Dict[str, Any],
                 modality_results: Optional[Dict[str, Any]] = None) -> str:
    """Dense technical reasoning an analyst can audit.

    When aludam supplies ``result['reason']`` (Phase 3), it wins.
    """
    if isinstance(result.get("reason"), str) and result["reason"].strip():
        return result["reason"].strip()

    parts: List[str] = []
    fs = result.get("family_scores") or {}

    if media in ("IMAGE", "VIDEO"):
        fam = {k: fs[k] for k in FAMILY_CN if k in fs}
        if fam:
            parts.append("family scores: " + ", ".join(
                f"{k}={_fmt(v)}" for k, v in fam.items()))
        msr = _num(fs.get("model_signal_raw"))
        if msr is not None:
            parts.append(f"model_signal_raw={_fmt(msr)}")
        fused = _num(fs.get("fused_score")) or _num(result.get("p_synthetic"))
        if fused is not None:
            parts.append(f"fused_score={_fmt(fused)}")
        if result.get("fake_type"):
            parts.append("fired families: " + ", ".join(result["fake_type"]))
        if "faces_detected" in result:
            parts.append(f"faces_detected={result['faces_detected']}")
        if "frames_analyzed" in result:
            parts.append(f"frames_analyzed={result['frames_analyzed']}")
        if result.get("suspicious_intervals"):
            ivs = ", ".join(f"[{a:.1f}-{b:.1f}s]"
                            for a, b in result["suspicious_intervals"][:6])
            parts.append(f"suspicious_intervals={ivs}")
        if "tier" in result:
            parts.append(f"tier={result['tier']} ({result.get('tier_name', '')})")

    elif media == "AUDIO":
        parts.append(f"fake_probability={_fmt(result.get('fake_probability'), 4)}")
        if result.get("signal_source"):
            parts.append(f"signal_source={result['signal_source']}")
        fsu = result.get("feature_summary") or {}
        if isinstance(fsu, dict) and fsu:
            kv = ", ".join(f"{k}={_fmt(v) if _num(v) is not None else v}"
                           for k, v in list(fsu.items())[:8])
            parts.append(f"features: {kv}")
        segs = result.get("suspicious_segments") or []
        if segs:
            parts.append(f"suspicious_segments={len(segs)}")

    elif media == "TEXT":
        parts.append(f"ai_probability={_fmt(result.get('ai_probability'), 4)}")
        for key in ("perplexity_signal", "stylometry_signal",
                    "pattern_signal", "watermark_signal", "detectgpt_signal"):
            if result.get(key) is not None:
                parts.append(f"{key}={_fmt(result[key], 4)}")
        metrics = result.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            kv = ", ".join(f"{k}={_fmt(v) if _num(v) is not None else v}"
                           for k, v in list(metrics.items())[:6])
            parts.append(f"metrics: {kv}")
        if result.get("suspicious_spans"):
            parts.append(f"suspicious_spans={len(result['suspicious_spans'])}")

    elif media == "FUSION":
        if result.get("modality_scores"):
            parts.append("modality_scores: " + ", ".join(
                f"{k}={_fmt(v)}" for k, v in result["modality_scores"].items()))
        parts.append(f"fused_fake_probability={_fmt(result.get('fused_fake_probability'), 4)}")
        if result.get("quality_gates"):
            parts.append("quality_gates: " + ", ".join(
                f"{k}={_fmt(v)}" for k, v in result["quality_gates"].items()))
        parts.append(
            f"agreement: fake={result.get('fake_agreement_count', 0)}, "
            f"real={result.get('real_agreement_count', 0)}")
        if result.get("v2_gamed_veto") is not None:
            parts.append(f"gamed_veto={_fmt(result['v2_gamed_veto'])}")
        if result.get("reasoning"):
            parts.append(f"fusion reasoning: {result['reasoning']}")

    if result.get("notes"):
        parts.append(f"notes: {result['notes']}")

    reason = "; ".join(p for p in parts if p)
    if not reason:
        reason = f"verdict={result.get('label', 'UNKNOWN')}"
    return reason[:1500]


# ---------------------------------------------------------------------------
# Explanation: LLM (multi-provider) with deterministic fallback
# ---------------------------------------------------------------------------

def _llm_config() -> Optional[Dict[str, str]]:
    """Detect provider from which key env var is set. None -> no LLM."""
    if os.environ.get("LLM_DISABLE", "").strip() not in ("", "0", "false", "False"):
        return None
    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        return None
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return {
            "provider": "openai",
            "model": model,
            "key": os.environ["OPENAI_API_KEY"].strip(),
            "base": os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        }
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return {
            "provider": "gemini",
            "model": model,
            "key": os.environ["GEMINI_API_KEY"].strip(),
            "base": "https://generativelanguage.googleapis.com/v1beta",
        }
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return {
            "provider": "anthropic",
            "model": model,
            "key": os.environ["ANTHROPIC_API_KEY"].strip(),
            "base": "https://api.anthropic.com",
        }
    return None


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str],
               timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _llm_call(cfg: Dict[str, str], system: str, user: str,
              timeout: float) -> str:
    if cfg["provider"] == "openai":
        data = _post_json(
            f"{cfg['base']}/chat/completions",
            {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": 220,
            },
            {"Authorization": f"Bearer {cfg['key']}"},
            timeout,
        )
        return data["choices"][0]["message"]["content"].strip()
    if cfg["provider"] == "gemini":
        data = _post_json(
            f"{cfg['base']}/models/{cfg['model']}:generateContent",
            {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user",
                              "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.3,
                                     "maxOutputTokens": 220},
            },
            {"x-goog-api-key": cfg["key"]},
            timeout,
        )
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    if cfg["provider"] == "anthropic":
        data = _post_json(
            f"{cfg['base']}/v1/messages",
            {
                "model": cfg["model"],
                "max_tokens": 220,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            {"x-api-key": cfg["key"],
             "anthropic-version": "2023-06-01"},
            timeout,
        )
        return data["content"][0]["text"].strip()
    raise ValueError(f"unknown provider {cfg['provider']!r}")


def _clip(text: str, limit: int = 700) -> str:
    text = " ".join(str(text).split())
    text = text.encode("ascii", "replace").decode("ascii")
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "..."
    return text


def _fallback_explanation(verdict: str, conf_label: str,
                          score: Optional[float], signals: List[str]) -> str:
    """Deterministic, self-explanatory wording used when no LLM runs."""
    score_s = f" with a score of {score:.2f}" if score is not None else ""
    if verdict == "AUTHENTIC":
        base = (f"No strong evidence of manipulation was found, so the content "
                f"is judged authentic (confidence {conf_label}{score_s}).")
    elif verdict == "ALTERED":
        base = (f"The content appears to have been modified after capture "
                f"(verdict ALTERED, confidence {conf_label}{score_s}).")
    elif verdict == "SYNTHETIC":
        base = (f"The content appears to be machine-generated rather than "
                f"captured from the real world (verdict SYNTHETIC, confidence "
                f"{conf_label}{score_s}).")
    else:
        base = (f"The evidence is mixed or too weak for a firm call "
                f"(verdict INCONCLUSIVE, confidence {conf_label}{score_s}).")
    if signals:
        ev = "Evidence: " + ", ".join(signals[:5]).replace("_", " ").lower() + "."
    else:
        ev = "No individual detector signal crossed its firing threshold."
    caveat = ("Forensic detectors are probabilistic: treat this as a strong "
              "lead for review, not proof.")
    return _clip(f"{base} {ev} {caveat}", 600)


def explain(media: str, verdict: str, conf_label: str,
            score: Optional[float], signals: List[str], reason: str,
            *,
            timeout: Optional[float] = None) -> Tuple[str, str]:
    """Return (explanation, explainer_tag). Never raises, never empty."""
    fallback = _fallback_explanation(verdict, conf_label, score, signals)
    cfg = _llm_config()
    if not cfg:
        return fallback, "fallback"

    system = (
        "You explain deepfake-detection results to non-experts. Write 2-3 "
        "short sentences in plain language: what the verdict means, what the "
        "evidence was, and one practical caveat. No markdown, no headings, "
        "no bullet points, do not repeat the raw numbers verbatim."
    )
    user = (
        f"Media: {media}. Verdict: {verdict} "
        f"(confidence {conf_label}"
        + (f", score {score:.2f}" if score is not None else "") + "). "
        + (f"Signals: {', '.join(signals[:8])}. " if signals else "")
        + f"Technical analysis: {reason[:900]}"
    )
    to = float(timeout or os.environ.get("LLM_TIMEOUT", "20"))
    try:
        text = _llm_call(cfg, system, user, to)
        if text:
            return _clip(text, 700), f"llm:{cfg['provider']}/{cfg['model']}"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        # LLM trouble must never fail a scan -- log and fall back.
        print(f"[scan_response] llm explanation failed "
              f"({cfg['provider']}): {type(exc).__name__}: {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[scan_response] llm explanation failed: {exc}", flush=True)
    return fallback, "fallback"


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def build_scan_response(
    media: str,
    file_type: str,
    result: Dict[str, Any],
    *,
    modality_results: Optional[Dict[str, Any]] = None,
    file_hash: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    degraded: bool = False,
) -> Dict[str, Any]:
    """Assemble the full scan response (envelope + legacy dashboard keys)."""
    t_start = time.time()

    media = str(media).upper()
    verdict, basis = _verdict_and_basis(media, result)
    conf_pct = _confidence_pct(result)
    conf_label = _confidence_label(conf_pct)
    score = _overall(media, result)
    signals = _signals_for(media, result) or _signals_for("FUSION", result)
    reason = build_reason(media, result, modality_results)

    # FUSION: signals = union across modalities
    if media == "FUSION" and modality_results:
        signals = _dedup(
            [s for mod, res in modality_results.items()
             for s in _signals_for(str(mod).upper(), res)] or signals)

    explanation, explainer = explain(
        media, verdict, conf_label, score, signals, reason)

    metadata = _metadata(media, result, modality_results)

    summary: Dict[str, Any] = {
        "verdict": verdict,
        "overall_score": score,
        "confidence": conf_label,
        "confidence_pct": conf_pct,
        "claim_basis": basis,
    }

    runtime: Dict[str, Any] = {
        "backend": "light" if degraded else "neural",
        "degraded": bool(degraded),
        "explainer": explainer,
        "elapsed_ms": elapsed_ms,
    }
    if media in ("IMAGE", "VIDEO", "FUSION"):
        runtime["face_detector"] = os.environ.get(
            "DEEPGUARD_FACE_DETECTOR", "haar")

    envelope: Dict[str, Any] = {
        "request_id": f"rd_scan_{secrets.token_hex(6)}",
        "status": "COMPLETED",
        "media_type": media,
        "summary": summary,
        "ensemble_results": _ensemble(media, result, modality_results),
        "reason": reason,
        "explanation": explanation,
        "runtime": runtime,
    }
    if metadata is not None:
        envelope["metadata_analysis"] = metadata

    # Legacy keys the dashboard and older clients still read.
    envelope["success"] = True
    envelope["file_type"] = file_type
    envelope["result"] = result
    if modality_results is not None:
        envelope["modality_results"] = modality_results
    if file_hash:
        envelope["file_hash"] = file_hash

    envelope["_build_ms"] = int((time.time() - t_start) * 1000)
    return envelope
