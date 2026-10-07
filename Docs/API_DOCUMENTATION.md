# DeepGuard AI — API Documentation

**Base URL:** `http://localhost:5000`
**CORS:** Enabled for all routes
**Max upload:** 100 MB
**File types:** `png, jpg, jpeg, gif, mp4, avi, mov, mp3, wav, txt, pdf`

---

## 1. API Endpoint Documentation

### GET `/deep-guard/status`

Check API health and which models are loaded.

**Response `200`:**
```json
{
  "status": "running",
  "version": "3.0",
  "models": {
    "image": true,
    "audio": true,
    "video": true,
    "text": true,
    "query_assistant": true,
    "fusion_engine": true
  },
  "features": ["C2PA/EXIF provenance", "..."],
  "supported_types": ["image", "audio", "video", "text", "query", "fusion"]
}
```

---

### Common scan response envelope (v2)

All `/deep-guard/detect/*` endpoints (image, audio, video, text, auto, fusion)
return the same envelope. Example (video):

```json
{
  "request_id": "rd_scan_9847120aef",
  "status": "COMPLETED",
  "media_type": "VIDEO",
  "summary": {
    "verdict": "ALTERED",
    "overall_score": 0.66,
    "confidence": "MEDIUM",
    "confidence_pct": 62.0,
    "claim_basis": "manipulation"
  },
  "ensemble_results": {
    "visual_manipulation": { "score": 0.86, "signals_detected": ["TEMPORAL_INCONSISTENCY"] },
    "visual_generation":   { "score": 0.31 },
    "audio_manipulation":  { "score": 0.77, "signals_detected": ["AV_SYNC_MISMATCH"] }
  },
  "metadata_analysis": { "has_c2pa": null, "software_signature": "Unknown / Stripped" },
  "reason": "family scores: ...; fired families: temporal_inconsistency; frames_analyzed=42; suspicious_intervals=[3.2-5.0s], [11.0-12.4s]; tier=3 (Inconclusive)",
  "explanation": "The content appears to have been modified after capture (verdict ALTERED, confidence MEDIUM with a score of 0.66). Evidence: temporal inconsistency. Forensic detectors are probabilistic: treat this as a strong lead for review, not proof.",
  "runtime": { "backend": "neural", "degraded": false, "explainer": "fallback", "elapsed_ms": 9000, "face_detector": "haar" },
  "success": true,
  "file_type": "video",
  "result": { "label": "FAKE", "...": "full detector payload (unchanged)" }
}
```

| Field | Description |
|-------|-------------|
| `request_id` | Unique per scan: `rd_scan_` + 12 hex chars |
| `status` | `COMPLETED` for every successful scan (errors use the error format below) |
| `media_type` | `IMAGE` / `AUDIO` / `VIDEO` / `TEXT` / `FUSION` |
| `summary.verdict` | `AUTHENTIC` · `SYNTHETIC` · `ALTERED` · `INCONCLUSIVE` — manipulation claim takes precedence over generation when both fired |
| `summary.overall_score` | 0–1 fake probability (image `p_synthetic`, video fusion score, audio `fake_probability`, text `ai_probability`, fusion `fused_fake_probability`) |
| `summary.confidence` | `HIGH` ≥ 75 · `MEDIUM` ≥ 55 · `LOW` — plus exact `confidence_pct` |
| `ensemble_results` | Per-claim groups (`visual_manipulation`, `visual_generation`, `audio_manipulation`, `text_generation`) each `{score, signals_detected}` |
| `metadata_analysis` | Present for image/video/fusion; `has_c2pa` from provenance score (null = not probed) |
| `reason` | Dense technical derivation of the detector's own numbers (family scores, fired families, margins, intervals, segments). Passed through untouched once aludam Phase 3 emits its own `reason` |
| `explanation` | Plain-language rewrite for end users (see below) |
| `runtime` | `backend` neural/light, `degraded` (mock detector used), `explainer` tag, `elapsed_ms`, `face_detector` |
| `success`, `file_type`, `result`, `modality_results`, `file_hash` | Legacy keys, kept for the dashboard and older clients |

**`explanation` (LLM):** the backend rewrites `reason` for a non-expert audience
using a configured LLM. Provider is auto-detected from whichever key env var is
set; model comes from `LLM_MODEL` (both required):

| Env var | Provider |
|---------|----------|
| `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) | OpenAI-compatible chat completions (Groq / OpenRouter / Ollama work) |
| `GEMINI_API_KEY` | Google Gemini |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `LLM_MODEL` | Model name, e.g. `gpt-4o-mini`, `claude-3-5-haiku-latest` |
| `LLM_TIMEOUT` | Seconds (default 20) |
| `LLM_DISABLE=1` | Force the deterministic fallback |

No key/model configured, or the call fails → a deterministic fallback text is
returned and `runtime.explainer = "fallback"`; a scan never fails because of
the LLM. Successful calls report `runtime.explainer = "llm:<provider>/<model>"`.

---

### POST `/deep-guard/detect/image`

Detect deepfakes in an image.

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `file` | file (png/jpg/jpeg/gif) | Yes |

**Response `200`:**
```json
{
  "success": true,
  "file_type": "image",
  "result": {
    "label": "FAKE | AUTHENTIC | UNCERTAIN",
    "confidence": 92.5,
    "family_scores": { "fully_ai_generated": 0.85 },
    "reasons": [{ "signal": "...", "score": 0.9, "region": null, "description": "..." }],
    "faces_detected": 1,
    "notes": "..."
  }
}
```

**Errors:** `400` missing/invalid file · `500` server error

---

### POST `/deep-guard/detect/audio`

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `file` | file (mp3/wav) | Yes |

**Response `200`:**
```json
{
  "success": true,
  "file_type": "audio",
  "result": {
    "label": "UNCERTAIN",
    "confidence": 50.0,
    "fake_probability": 0.72,
    "signal_source": "trained_classifier",
    "feature_summary": {},
    "suspicious_segments": [],
    "notes": "..."
  }
}
```

**Errors:** `400` missing/invalid file · `500` server error

---

### POST `/deep-guard/detect/video`

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `file` | file (mp4/avi/mov) | Yes |

**Response `200`:**
```json
{
  "success": true,
  "file_type": "video",
  "result": {
    "label": "FAKE",
    "confidence": 88.0,
    "family_scores": {},
    "reasons": [],
    "frames_analyzed": 42,
    "suspicious_intervals": [],
    "notes": "..."
  }
}
```

**Errors:** `400` missing/invalid file · `500` server error

---

### POST `/deep-guard/detect/text`

Detect AI-generated text.

**Request:** `application/json`
```json
{ "text": "Your text to analyze here..." }
```

**Response `200`:**
```json
{
  "success": true,
  "file_type": "text",
  "result": {
    "label": "UNCERTAIN",
    "confidence": 50.0,
    "ai_probability": 0.61,
    "stylometry_signal": 0.3,
    "pattern_signal": 0.2,
    "watermark_signal": 0.0,
    "metrics": { "word_count": 240 },
    "suspicious_spans": [],
    "notes": "..."
  }
}
```

**Errors:** `400` missing/empty text · `500` server error

---

### POST `/deep-guard/detect/auto`

Auto-detect file type from extension and run the matching detector.

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `file` | file | Yes |

**Response `200`:**
```json
{
  "success": true,
  "file_type": "image | audio | video",
  "file_hash": "<sha256 hex>",
  "result": { }
}
```

**Errors:** `400` unsupported type · `500` server error

---

### POST `/deep-guard/detect/fusion`

Multi-modal calibrated fusion (Bayesian LLR + quality gating + 4-tier verdict).

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `files` | file(s) — image/audio/video | At least one of `files` or `text` |
| `text` | string (form field) | Optional |

**Response `200`:**
```json
{
  "success": true,
  "file_type": "image",
  "result": {
    "fused_fake_probability": 0.71,
    "verdict": "LIKELY_SYNTHETIC",
    "tier": 4,
    "tier_name": "Likely Synthetic",
    "confidence": 71.0,
    "quality_gates": { "image": 0.6, "text": 0.4 },
    "modality_scores": { "image": 0.72, "text": 0.55 },
    "fake_agreement_count": 2,
    "real_agreement_count": 0,
    "v2_gamed_veto": null,
    "reasoning": "image: synthetic-leaning (score=0.72, weight=0.60); ..."
  },
  "modality_results": { "image": { }, "text": { } }
}
```

**4-tier verdicts:**

| Tier | Verdict | Condition |
|------|---------|-----------|
| 1 | `LIKELY_SYNTHETIC` (Verified Provenance) | Valid C2PA provenance |
| 2 | `LIKELY_AUTHENTIC` | `p < 0.35` |
| 3 | `INDETERMINATE` | `0.35 ≤ p ≤ 0.65` or veto/conflict |
| 4 | `LIKELY_SYNTHETIC` | `p > 0.65` with ≥2 agreeing signals |

**Errors:** `400` no valid content · `500` server error

---

### POST `/deep-guard/query`

YOLOv8 + Gemini query assistant.

**Option A — image + query:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `file` | image file | Yes |
| `query` | string (form) | Optional |

**Option B — text-only query:**
- `application/json`: `{ "query": "What's in this image?" }`
- or `application/x-www-form-urlencoded` with `query`

**Response `200`:**
```json
{ "success": true, "...assistant_result": "..." }
```

**Errors:** `400` missing file/query · `500` assistant failed to load

---

### Error format (all endpoints)

```json
{ "error": "message" }
```

| Code | Meaning |
|------|---------|
| 400 | Missing/invalid input, unsupported file type |
| 500 | Detector/model/server failure |

---

### Run it

```bash
pip install -r requirements.txt
cd backend
python api.py
# → http://localhost:5000
```

---

## 2. Method, Request Body, Response — Tabular Form

### Endpoint Summary Table

| # | Method | Endpoint | Request Body / Content-Type | Success Response (200) | Error Responses |
|---|--------|----------|----------------------------|------------------------|-----------------|
| 1 | **GET** | `/deep-guard/status` | None | `{ status, version, models:{image,audio,video,text,query_assistant,fusion_engine}, features[], supported_types[] }` | — |
| 2 | **POST** | `/deep-guard/detect/image` | `multipart/form-data` · field `file` (image: png/jpg/jpeg/gif) | `{ success:true, file_type:"image", result:{ label, confidence, family_scores, reasons[], faces_detected, notes } }` | 400 no/invalid file · 500 server error |
| 3 | **POST** | `/deep-guard/detect/audio` | `multipart/form-data` · field `file` (mp3/wav) | `{ success:true, file_type:"audio", result:{ label, confidence, fake_probability, signal_source, feature_summary, suspicious_segments[], notes } }` | 400 no/invalid file · 500 server error |
| 4 | **POST** | `/deep-guard/detect/video` | `multipart/form-data` · field `file` (mp4/avi/mov) | `{ success:true, file_type:"video", result:{ label, confidence, family_scores, reasons[], frames_analyzed, suspicious_intervals[], notes } }` | 400 no/invalid file · 500 server error |
| 5 | **POST** | `/deep-guard/detect/text` | `application/json` · `{ "text": "string" }` | `{ success:true, file_type:"text", result:{ label, confidence, ai_probability, stylometry_signal, pattern_signal, watermark_signal, metrics, suspicious_spans[], notes } }` | 400 missing/empty text · 500 server error |
| 6 | **POST** | `/deep-guard/detect/auto` | `multipart/form-data` · field `file` (any allowed type) | `{ success:true, file_type, file_hash:"sha256", result:{ ...depends on type } }` | 400 missing/unsupported type · 500 server error |
| 7 | **POST** | `/deep-guard/detect/fusion` | `multipart/form-data` · `files[]` (multi file: image/audio/video) + optional `text` form field | `{ success:true, file_type, result:{ fused_fake_probability, verdict, tier, tier_name, confidence, quality_gates, modality_scores, fake_agreement_count, real_agreement_count, v2_gamed_veto, reasoning }, modality_results:{...} }` | 400 no valid content · 500 server error |
| 8 | **POST** | `/deep-guard/query` | **A)** `multipart/form-data` · `file` (image) + `query` (string, optional) **OR B)** `application/json` · `{ "query": "string" }` | `{ success:true, ...assistant_result }` | 400 missing file/query · 500 assistant load fail |

---

### Request Body Detail (POST only)

| Endpoint | Content-Type | Body Fields |
|----------|-------------|-------------|
| `/deep-guard/detect/image` | `multipart/form-data` | `file` = binary image |
| `/deep-guard/detect/audio` | `multipart/form-data` | `file` = binary audio |
| `/deep-guard/detect/video` | `multipart/form-data` | `file` = binary video |
| `/deep-guard/detect/text` | `application/json` | `text` (string, required) |
| `/deep-guard/detect/auto` | `multipart/form-data` | `file` = binary (image/audio/video) |
| `/deep-guard/detect/fusion` | `multipart/form-data` | `files` = binary[] (repeated field) OR `text` = string (at least one required) |
| `/deep-guard/query` (image) | `multipart/form-data` | `file` = binary image, `query` = string |
| `/deep-guard/query` (text) | `application/json` | `query` = string |
| `/deep-guard/status` | — | GET, no body |

---

### Response Fields Summary

| Endpoint | Top-level keys in 200 response | Key `result` fields |
|----------|-------------------------------|---------------------|
| `/deep-guard/status` | `status, version, models, features, supported_types` | — |
| `/deep-guard/detect/image` | `success, result, file_type` | `label, confidence, family_scores, reasons, faces_detected, notes` |
| `/deep-guard/detect/audio` | `success, result, file_type` | `label, confidence, fake_probability, signal_source, feature_summary, suspicious_segments, notes` |
| `/deep-guard/detect/video` | `success, result, file_type` | `label, confidence, family_scores, reasons, frames_analyzed, suspicious_intervals, notes` |
| `/deep-guard/detect/text` | `success, result, file_type` | `label, confidence, ai_probability, stylometry_signal, pattern_signal, watermark_signal, metrics, suspicious_spans, notes` |
| `/deep-guard/detect/auto` | `success, result, file_type, file_hash` | depends on detected type (image/audio/video schema) |
| `/deep-guard/detect/fusion` | `success, result, modality_results, file_type` | `fused_fake_probability, verdict, tier, tier_name, confidence, quality_gates, modality_scores, fake/real_agreement_count, v2_gamed_veto, reasoning` |
| `/deep-guard/query` | `success, ...assistant_result` | varies (Gemini/YOLO output) |
