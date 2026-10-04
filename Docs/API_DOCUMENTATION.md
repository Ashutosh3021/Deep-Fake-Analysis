# DeepGuard AI — API Documentation

**Base URL:** `http://localhost:5000`
**CORS:** Enabled for all routes
**Max upload:** 100 MB
**File types:** `png, jpg, jpeg, gif, mp4, avi, mov, mp3, wav, txt, pdf`

---

## 1. API Endpoint Documentation

### GET `/api/status`

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

### POST `/api/detect/image`

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

### POST `/api/detect/audio`

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

### POST `/api/detect/video`

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

### POST `/api/detect/text`

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

### POST `/api/detect/auto`

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

### POST `/api/detect/fusion`

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

### POST `/api/query`

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
| 1 | **GET** | `/api/status` | None | `{ status, version, models:{image,audio,video,text,query_assistant,fusion_engine}, features[], supported_types[] }` | — |
| 2 | **POST** | `/api/detect/image` | `multipart/form-data` · field `file` (image: png/jpg/jpeg/gif) | `{ success:true, file_type:"image", result:{ label, confidence, family_scores, reasons[], faces_detected, notes } }` | 400 no/invalid file · 500 server error |
| 3 | **POST** | `/api/detect/audio` | `multipart/form-data` · field `file` (mp3/wav) | `{ success:true, file_type:"audio", result:{ label, confidence, fake_probability, signal_source, feature_summary, suspicious_segments[], notes } }` | 400 no/invalid file · 500 server error |
| 4 | **POST** | `/api/detect/video` | `multipart/form-data` · field `file` (mp4/avi/mov) | `{ success:true, file_type:"video", result:{ label, confidence, family_scores, reasons[], frames_analyzed, suspicious_intervals[], notes } }` | 400 no/invalid file · 500 server error |
| 5 | **POST** | `/api/detect/text` | `application/json` · `{ "text": "string" }` | `{ success:true, file_type:"text", result:{ label, confidence, ai_probability, stylometry_signal, pattern_signal, watermark_signal, metrics, suspicious_spans[], notes } }` | 400 missing/empty text · 500 server error |
| 6 | **POST** | `/api/detect/auto` | `multipart/form-data` · field `file` (any allowed type) | `{ success:true, file_type, file_hash:"sha256", result:{ ...depends on type } }` | 400 missing/unsupported type · 500 server error |
| 7 | **POST** | `/api/detect/fusion` | `multipart/form-data` · `files[]` (multi file: image/audio/video) + optional `text` form field | `{ success:true, file_type, result:{ fused_fake_probability, verdict, tier, tier_name, confidence, quality_gates, modality_scores, fake_agreement_count, real_agreement_count, v2_gamed_veto, reasoning }, modality_results:{...} }` | 400 no valid content · 500 server error |
| 8 | **POST** | `/api/query` | **A)** `multipart/form-data` · `file` (image) + `query` (string, optional) **OR B)** `application/json` · `{ "query": "string" }` | `{ success:true, ...assistant_result }` | 400 missing file/query · 500 assistant load fail |

---

### Request Body Detail (POST only)

| Endpoint | Content-Type | Body Fields |
|----------|-------------|-------------|
| `/api/detect/image` | `multipart/form-data` | `file` = binary image |
| `/api/detect/audio` | `multipart/form-data` | `file` = binary audio |
| `/api/detect/video` | `multipart/form-data` | `file` = binary video |
| `/api/detect/text` | `application/json` | `text` (string, required) |
| `/api/detect/auto` | `multipart/form-data` | `file` = binary (image/audio/video) |
| `/api/detect/fusion` | `multipart/form-data` | `files` = binary[] (repeated field) OR `text` = string (at least one required) |
| `/api/query` (image) | `multipart/form-data` | `file` = binary image, `query` = string |
| `/api/query` (text) | `application/json` | `query` = string |
| `/api/status` | — | GET, no body |

---

### Response Fields Summary

| Endpoint | Top-level keys in 200 response | Key `result` fields |
|----------|-------------------------------|---------------------|
| `/api/status` | `status, version, models, features, supported_types` | — |
| `/api/detect/image` | `success, result, file_type` | `label, confidence, family_scores, reasons, faces_detected, notes` |
| `/api/detect/audio` | `success, result, file_type` | `label, confidence, fake_probability, signal_source, feature_summary, suspicious_segments, notes` |
| `/api/detect/video` | `success, result, file_type` | `label, confidence, family_scores, reasons, frames_analyzed, suspicious_intervals, notes` |
| `/api/detect/text` | `success, result, file_type` | `label, confidence, ai_probability, stylometry_signal, pattern_signal, watermark_signal, metrics, suspicious_spans, notes` |
| `/api/detect/auto` | `success, result, file_type, file_hash` | depends on detected type (image/audio/video schema) |
| `/api/detect/fusion` | `success, result, modality_results, file_type` | `fused_fake_probability, verdict, tier, tier_name, confidence, quality_gates, modality_scores, fake/real_agreement_count, v2_gamed_veto, reasoning` |
| `/api/query` | `success, ...assistant_result` | varies (Gemini/YOLO output) |
