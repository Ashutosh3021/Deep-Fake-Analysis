# DeepGuard AI — Working Principles (Complete System Reference)

This document describes **exactly how the DeepGuard AI system works**, end to
end, every module and every decision rule, as implemented in the codebase.

---

## 1. Core Thesis

**There is no single foolproof binary detector.** Unimodal models fail
catastrophically on out-of-distribution content, social-media compression,
paraphrasing, and adversarial evasion. Therefore DeepGuard AI uses a
**Layered Defense-in-Depth Architecture** and never issues a bare accusation —
every output is *screening evidence* with an explicit **Indeterminate** tier.

The five conceptual layers:

1. **Layer 1 — Active Provenance & Custody**: C2PA/JUMBF content credentials,
   EXIF/XMP software tags, SHA-256 file hash (chain of custody).
2. **Layer 2 — Classical Forensic Invariants**: 2D DFT spectral decay &
   periodicity, wavelet-denoised noise-residual consistency, Error Level
   Analysis (ELA), texture/histogram uniformities.
3. **Layer 3 — Modality-Specific Feature & Neural Estimators**: pretrained
   image classifier (multi-crop), face landmark kinematics, mediapipe face
   mesh, MFCC/HNR/pitch audio features, GPT-2 perplexity text probe.
4. **Layer 4 — Cross-Modal Consistency**: audio-visual sync (energy vs.
   mouth-aspect-ratio cross-correlation), identity-consistency tracking.
5. **Layer 5 — Calibrated Evidence Fusion & Policy Engine**: per-family
   weights, quality-gated weighting, Bayesian-style fusion, and a **4-tier
   verdict** that can abstain.

---

## 2. Project Layout

```
Deep-Fake-Analysis/
├── models/
│   ├── forensics_core.py          # Shared forensic primitives (all families)
│   ├── final_image_detector.py    # Image detector (production)
│   ├── final_video_detector.py    # Video detector (production)
│   ├── final_audio_detector.py    # Audio detector (production)
│   ├── final_text_detector.py     # Text detector (production)
│   ├── temporal_analyzer.py       # LSTM-based temporal analyzer (legacy/aux)
│   ├── video_detector.py          # Legacy video detector (references app.ml)
│   └── query_assistant.py         # YOLOv8 + Gemini 2.5 Flash assistant
├── backend/
│   └── api.py                     # Flask REST API + EvidenceFusionEngine
├── dashboard/
│   ├── index.html                 # Frontend dashboard
│   ├── styles.css                 # Styling
│   └── app.js                     # Frontend logic
├── plan.md                        # Implementation roadmap/spec
├── README.md / MODEL_CARD.md      # Docs
├── requirements.txt               # Python dependencies
└── yolov8n.pt                     # YOLOv8 weights (query assistant)
```

Key architectural rule: **`forensics_core.py` is the single shared source of
forensic signals**; both the image and video detectors import it so splice /
face logic is never duplicated or drifting.

---

## 3. Shared Forensic Core (`models/forensics_core.py`)

A `Finding` is the universal evidence unit:

```json
{"signal": "str", "score": 0.0-1.0, "region": (x,y,w,h) or null, "description": "str"}
```

All threshold numbers throughout the codebase are **unfit starting points**
encoding forensic literature principles — explicitly not fit on labeled data.

### 3.1 Provenance & Metadata Family (Layer 1)

- **`sha256_file_hash(path)`** — streams the file in 64 KB chunks and returns
  a SHA-256 hex digest for chain-of-custody integrity.
- **`c2pa_provenance_score(path)`** — scans the JPEG/raw bytes (up to 1 MB,
  then up to 4 MB for XMP) for:
  - the byte signature `jumb` and `c2pa` (JUMBF/C2PA boxes in JPEG),
  - `http://c2pa.org` or `GPC:CT` (XMP-encoded C2PA manifests).
  - Found → score `0.9`, strong evidence the file was authored by a
    C2PA-compliant tool. Not found → score `0.0` **with a caveat** that
    absence is NOT proof of fakeness (metadata can be stripped).
- **`exif_metadata_score(path)`** — via PIL:
  - EXIF `Software` tag matching known generators (`midjourney`, `dall-e`,
    `stable diffusion`, `flux`, `firefly`, `copilot`, `gemini`, `imagen`,
    `craiyon`, `nightcafe`, `lexica`, `playground`) → score `0.85`, direct
    evidence.
  - No EXIF at all → `0.3` (common for AI/screenshots/exports).
  - No camera Make/Model → `0.35`; valid camera metadata → `0.1`.

### 3.2 Global AI-Generation Family (Layer 2)

- **`fft_periodicity_score(gray)`** — 2D FFT, log-magnitude, radial rings
  every 5 px. Three fused signals:
  1. **Radial bumpiness** — std of the 2nd derivative of ring means
     (`bump_score = clip(bumpiness/0.15)`), catches GAN/diffusion upsampling
     and checkerboard artifacts.
  2. **Spectral decay deviation** — log-log slope fitted to radial power vs.
     radius; natural images follow ~`1/f^2` (slope ≈ −2). `alpha_deviation`
     from 2.0 (normalized by 1.5) scores higher for synthetic images.
  3. **Periodic peak detection** — count of rings above `mean + 2.5σ`
     (peaks/3). Checkerboard artifacts produce such spikes.
  - Combined: `0.4*bump + 0.35*decay + 0.25*peaks`.
- **`texture_uniformity_score(gray)`** — divides image into a 6×6 grid,
  computes local variance per cell and its coefficient of variation.
  **LOW variation across the frame is suspicious** (generative images look
  unnaturally uniform); score = `1 - clip(CV/1.2)`.
- **`histogram_smoothness_score(img)`** — per RGB channel, normalized 256-bin
  histogram, fraction of nearly-flat 2nd-difference bins. True sensor noise
  makes histograms jagged; over-smooth histograms (diffusion decodes) score
  higher (`mean_flatness * 1.5`).

### 3.3 Splice / Localized-Edit Family (Layer 2)

- **`error_level_analysis(img_path, quality=90)`** — re-encodes the image at
  JPEG quality 90 and diffs against the original. An 8×8 grid of cell-mean
  errors; cells whose error z-score is far from the frame mean (`worst_z`)
  indicate a region edited/composited **after** the last save.
  `score = clip((worst_z - 1.5)/3.0)`. Returns the `region` box when > 0.3.
  Caveat: false-positives on already-low-quality/PNG sources, hence only one
  signal among many.
- **`_wavelet_denoise(gray)`** — 2-level (in code: 1-level applied via Haar
  rows/columns) hard-threshold wavelet denoiser with VisuShrink universal
  threshold `σ·√(2·ln N)`, σ estimated via MAD on the HH sub-band; falls back
  to a bilateral filter if scipy/unexpected shapes occur.
- **`noise_residual_consistency(gray)`**:
  - Residual `R = I − L(I)` (wavelet-denoised estimate).
  - 6×6 patch grid; computes per-patch `Var(R_k)`.
  - Plan formula `V_R = (1/K) Σ_k (Var(R_k) − mean(Var(R)))²`. z-score of the
    worst patch (`(worst_z − 1.5)/3.0`) scores splice/insert.
  - **Absence signal**: total noise variance < 5.0 → suspicious ("no natural
    sensor noise"), adds up to `0.5`. Combined score clipped to [0,1].

### 3.4 Face-Region / Face-Swap Family (per detected face)

- **`detect_faces(img)`** — RetinaFace (RGB, multi-face, angle-robust); falls
  back to OpenCV Haar frontal cascade (flagged via `detector: "haar_fallback"`)
  if RetinaFace is unavailable. Returns `{box, landmarks, confidence, detector}`.
- **`face_blend_boundary_score(img, box)`** — three signals comparing inner
  face vs. outer ring:
  1. **Sharpness ratio** — Laplacian variance inner/outer; faces markedly
     smoother than surroundings → `clip((0.6 − ratio)/0.6)`.
  2. **Color divergence** — Bhattacharyya distance between inner/outer HSV
     2-D histograms → `clip((div − 0.15)/0.5)`.
  3. **Edge density ratio** — Canny edge density inner/outer → `clip((0.5 −
     ratio)/0.5)`.
  - Combined: `0.4*sharp + 0.35*color + 0.25*edge`.
- **`face_noise_mismatch_score(img, box)`** — median-blur residual variance of
  face vs. *masked-out* surrounding background (face pixels excluded). Either
  much higher or much lower noise than background is suspicious (different
  capture/generation source): `score = clip((|ln(ratio)| − 0.3)/1.2)`.
- **`analyze_face_region(img, face)`** — runs both checks and emits
  `face_swap_score = mean(scores)` with per-face `findings`.

### 3.5 Orchestration — `full_image_forensics(img_path)`

Runs ALL families and returns a single dict (`error` if unreadable):

```
file_hash        -> sha256
provenance_findings -> [c2pa, exif]
global_findings     -> [fft_periodicity, texture_uniformity, histogram_smoothness]
splice_findings     -> [error_level_analysis, noise_residual_consistency]
face_analyses       -> [analyze_face_region(...)] one per detected face
faces_detected      -> count
```

---

## 4. Image Detector (`models/final_image_detector.py`)

Covers three manipulation families you cannot know in advance:
**A) fully AI-generated**, **B) AI-edited/spliced region**,
**C) face-swap/reenactment**.

### 4.1 Config constants
- Fire thresholds: provenance `0.70`, global model `0.55`, splice `0.45`,
  face-swap `0.45`.
- Tier thresholds: likely-synthetic `0.65`, indeterminate `0.35–0.65`.
- `LOW_CONFIDENCE_BAND = 0.12`, `MULTI_CROP_COUNT = 4`.

### 4.2 Model & multi-crop inference
- Primary model: HuggingFace `umm-maybe/AI-image-detector`
  (`IMAGE_DETECTOR_MODEL` env override). Loaded lazily; if it fails, model
  signature is `None` and the system relies on forensic heuristics (a note is
  appended).
- **Multi-crop aggregation** (plan: `p_neural = (1/K)Σ p(y=1|Crop_k)`): takes 4
  crops — center, top-left, bottom-right, center-right, each
  `min(w,h)//2` square — and averages per-crop AI scores. If the image is too
  small (<32 px crops) it scores the full image once.
- Label mapping: any label containing fake/ai/synthetic/generated/artificial →
  its score; real/human/authentic/natural → `1 − score`.

### 4.3 Family scoring
- **Provenance**: `prov_score = max(provenance_findings scores)`.
- **Global (A)**: `global_score = 0.7*model_signal + 0.3*mean(global_findings)`
  (falls back to pure heuristics if no model).
- **Splice (B)**: `splice_score = max(splice_finding scores)`.
- **Face-swap (C)**: `face_swap_score = max(face_swap_score over faces)`.

### 4.4 Firing & evidence collection
Any family ≥ its fire threshold is appended to `fired_families`; every finding
above its threshold goes into `reasons`, then sorted by score descending.
Labels: `fully_ai_generated`, `ai_edited_region`, `face_swap`,
`provenance_ai_detected`.

### 4.5 Fusion & 4-tier verdict
Fixed family weights: provenance `0.20`, global `0.35`, splice `0.25`, face
`0.20`.

```
fused_score = Σ weights × family scores
non_zero_scores = scores > 0.05
has_conflict = (≥2 non-zero) and (max − min > 0.30)
overall_score = max(all family scores)
is_borderline = any |overall_score − fire_threshold| < 0.12
```

Verdict decision order:

1. `prov_score ≥ 0.70` and nothing else fired → **FAKE**, confidence
   `85 + prov*15` (provenance trusted).
2. Families fired and **no** conflict →
   - fused ≥ 0.65 → **FAKE**, confidence `50 + fused*50`;
   - else → **UNCERTAIN**, confidence `50`.
3. Conflict or borderline → **UNCERTAIN**, confidence `40 + fused*20`.
4. Otherwise → **AUTHENTIC**, confidence `50 + (1−fused)*50`.

Empty reasons get a placeholder `"none"` finding. Notes flag missing model /
no faces. Output dict: label, fake_type, confidence, reasons, family_scores,
faces_detected, file_hash, notes.

The module exposes a module-level singleton `final_image_detector`.

---

## 5. Video Detector (`models/final_video_detector.py`)

Reuses all four image families **per frame**, plus temporal layers D and E.

### 5.1 Capabilities & constants
- MediaPipe FaceMesh (5 faces, refined landmarks) if installed; librosa if
  installed — both optional with graceful degradation.
- `MAX_FRAMES_SAMPLED = 24`, `FULL_FORENSICS_FRAME_COUNT = 5`.
- Fire thresholds: global `0.55`, splice `0.45`, face `0.45`, temporal `0.55`,
  provenance `0.70`. AV-sync suspicion threshold `0.4`.
- Natural blink rate range: **8–30 blinks/min**.
- Spike σ for velocity and acceleration: `3.0`.

### 5.2 Frame sampling
- **`_extract_keyframes`**: uniform sampling (up to `max_frames`) combined
  with **scene-cut detection** — Bhattacharyya distance of 8×8×8 RGB
  histograms between candidate frames; if diff > 0.6, also captures the frame
  immediately before the cut. Falls back to uniform `_extract_frames`.

### 5.3 Per-frame full forensics
- 5 frames (`np.linspace` sampling across the keyframes) are written to
  temp files and run through the **image detector**; results tagged with
  `frame_index`.
- Frame family scores are aggregated by **max** across sampled frames:
  `global_score`, `splice_score`, `face_swap_score`.

### 5.4 Temporal analysis (`_temporal_analysis`)
Requires ≥4 frames with detected faces. Signals:

1. **Landmark kinematics** — tracks nose position across frames; computes
   `v_t = l_t − l_{t−1}`, `a_t = v_t − v_{t−1}`. Frames where velocity or
   acceleration exceed `mean + 3σ` are flagged as biomechanically implausible
   (`landmark_kinematics_spike`) and recorded with time_sec + velocity z.
   `kinematics_score = clip((vel_spikes + acc_spikes)/(len(valid)*0.3))`.
2. **Optical flow warp residual** — Farneback dense flow between up to 10
   consecutive frame pairs; warps frame₁ toward frame₂ and measures mean
   abs residual. Residuals > `mean + 2σ` produce
   `flow_score = clip(mean(those)/3.0)`.
3. **Blink rate** — eye-aspect-ratio (EAR) from `[33,160,158,133,153,144]`
   and `[263,387,385,362,380,373]`; threshold = `mean − 0.5*std`; counts
   downward EAR transitions as blinks. Blink rate in 8–30/min → natural
   (`blink_score = 0.15`), otherwise `0.6`.
4. **Identity consistency** — geometry-based cosine similarity of normalized
   landmark identity vectors (indices 1, 33, 263, 61, 291) between
   consecutive frames. `mean_sim` per pair; `identity_score = clip(1 − min_sim)`;
   pairs with `sim < 0.8` flagged `identity_drift` with their timestamps.

Votes combined with weights: kinematics `0.30`, optical-flow `0.25`, blink
`0.20`, identity `0.25` (renormalized by used weight).

Returns `score`, per-signal scores, `suspicious_frames` (frame index +
time + reason), `position_stability`, estimated blink rate, and natural-range
flag. If mediapipe is missing or too few faces → `available: False` with a
reason.

### 5.5 Audio-Visual cross-modal sync (`_audio_visual_sync`)
- Extracts the video's audio via `ffmpeg` (`pcm_s16le`, 16 kHz mono).
- Audio energy in 100 ms windows (50 ms hop); **mouth-aspect-ratio (MAR)**
  (upper-lower lip / mouth corners) extracted via FaceMesh over up to 100
  frames.
- Both sequences normalized (z-score) and cross-correlated (`mode='full'`).
  `max_corr` at best lag; `lag_penalty = |best_lag|/len`; 
  `sync_score = max_corr * (1 − lag_penalty*2)`.
- Inverted to **suspicion**: `score = 1 − sync_score`. High suspicion / poor
  sync → possible dubbing, audio cloning, or lip-sync manipulation.
- Fully optional (needs librosa + ffmpeg + face mesh); reports
  `available: False` otherwise.

### 5.6 Fusion & verdict
Family weights: global `0.25`, splice `0.20`, face `0.20`, temporal `0.20`,
AV-sync `0.15`.

- `overall_score = max(all five)`.
- Conflict check identical to image detector over the five scores.
- Decision order:
  1. families fired, no conflict, fused ≥ 0.65 → **FAKE** (`50 + fused*50`).
  2. conflict OR fused ∈ [0.30, 0.65] → **UNCERTAIN** (`40 + fused*20`).
  3. no per-frame results AND temporal unavailable → **UNCERTAIN** (`0`).
  4. families fired → **FAKE** (`50 + overall*50`).
  5. else → **AUTHENTIC** (`50 + (1−overall)*50`).

### 5.7 Temporal localization
`suspicious_frames` are converted to `[start_sec, end_sec]` intervals of
`[t, t+1.0]` (capped at video duration). Findings reaching score ≥ 0.4 are
promoted to `reasons` with frame/region annotations; temporal and AV-sync
reasons include descriptive text.

---

## 6. Audio Detector (`models/final_audio_detector.py`)

Three stacked strategies (first available wins per inference):

- **Strategy B — trained classifier** (preferred, if
  `models/audio_rf_classifier.pkl` exists): RandomForest `predict_proba`.
- **Strategy A — pretrained HF model** (`MelodyMachine/Deepfake-audio-detection-V2`,
  `AUDIO_DETECTOR_MODEL` env override): audio-classification pipeline with
  fake/spoof/synthetic/generated vs. real/bonafide/genuine/human label mapping.
- **Strategy C — heuristic fallback**: geometric mean of flatness, low-jitter,
  phase-discontinuity indicators. **Confidence intentionally suppressed**
  (distance-from-mid halved) because heuristics are unfit.

### 6.1 Preprocessing (`_preprocess_audio`)
- Load at 16 kHz mono via librosa; returns `None` if shorter than 0.3 s for
  feature extraction.
- **Silence trimming**: RMS per 2048/512 window; windows below `0.1×mean RMS`
  are cut except a mostly-silent file (<0.2 s voiced) which is kept as-is.

### 6.2 Enhanced feature vector (`extract_training_features`)
Implements plan formulas. Per file:
- 13 static MFCC mean/std (26), delta-MFCC mean/std (26), delta²-MFCC
  mean/std (26).
- Spectral centroid mean/std, spectral flatness mean/std, spectral rolloff
  mean, **high-frequency void** score above 8 kHz (STFT energy ratio → neural
  vocoders degrade HF; `clip(1 − hf_ratio*10)`).
- **HNR** `10·log10(P_harmonic/P_noise)` via HPSS (normalized /30), plus
  harmonic ratio.
- **F0 via `librosa.pyin`** (C2–C7): mean, std, **jitter**
  `mean|ΔT|/mean(T)`, and **pitch-jump score** (>50 Hz jumps ratio).
- **Phase coherence**: fraction of phase differences > π/4 (vocoder artifact).
- Skew and kurtosis of the raw waveform.

### 6.3 Quality-weighted segment aggregation (`_segment_scores`)
- Splits audio into overlapping 2 s windows (1 s hop).
- Per segment: SNR-based `quality = clip(snr/10, 0.1, 1.0)` (first 0.25 s of
  the segment as noise estimate), and a per-segment fake score via
  `_single_predict` on a temp WAV.
- Aggregates `p_audio = Σ q_i·p_i / Σ q_i`.
- Segments with `fake_score > 0.6` become `suspicious_segments`
  (start_sec, end_sec, fake_score, quality).

### 6.4 Verdict
- Effective score priority: trained classifier → HF model → heuristic.
- `fake_probability = clip(score)`; `distance_from_mid = |p − 0.5|·2`
  (halved for heuristic mode).
- `distance_from_mid < 0.40` (i.e. `1 − LOW_CONFIDENCE_THRESHOLD` with
  threshold 0.60) → **UNCERTAIN**; else `p > 0.5` → **SYNTHETIC**;
  `p < 0.5` → **AUTHENTIC**.
- `confidence = 50 + distance_from_mid·50`.
- `signal_source` records which strategy produced the score. Includes
  `feature_summary` (raw score, quality-weighted score, segment counts).

### 6.5 Offline training script
`train_from_dataset(bonafide_dir, spoof_dir)` extracts features, trains a
300-tree RandomForest (max_depth 20), reports a classification report, and
pickles the model to `models/audio_rf_classifier.pkl` for Strategy B.

---

## 7. Text Detector (`models/final_text_detector.py`)

Never a definitive accusation — always *Stylometric Screening Evidence*.

### 7.1 Signal weights
perplexity `0.40`, DetectGPT `0.15`, stylometry `0.25`, pattern `0.15`,
watermark `0.05`. Minimum 50 words for reliability; `LOW_CONFIDENCE_THRESHOLD
= 0.58`.

### 7.2 Perplexity + burstiness (GPT-2 probe)
- Probe: `gpt2` (`TEXT_DETECTOR_PROBE_MODEL`) via
  `GPT2LMHeadModel`/`GPT2TokenizerFast`, ≤1024 tokens; mean per-token NLL →
  `perplexity = exp(mean_nll)`.
- **Burstiness (token)**: std of per-token NLL.
- **Burstiness (formula)**: `B = (σ − μ)/(σ + μ)` over sentence word lengths
  (≥3 sentences required).
- To AI score: low perplexity (PPL 15–55 mapped to 1→0),
  low token burstiness (1.5–4.0 mapped to 1→0), and negative B (uniform
  cadence) all push AI probability up. Weighted `0.4/0.3/0.3`.

### 7.3 Stylometry (`_stylometry_signal`)
- **TTR** `|V_unique|/N_total`; **Hapax Legomena ratio** (words appearing
  once); **Shannon entropy** of word frequency distribution; **formulaic
  transition density**; **sentence-length uniformity** (low CV → AI-leaning).
- Combined: `0.3*TTR + 0.25*entropy + 0.2*hapax + 0.15*uniformity +
  0.1*transition`.

### 7.4 Watermark z-score (KGW-style)
- `z = (|s|_G − γ·T)/√(T·γ·(1−γ))` with γ = 0.5.
- Tries default seeds `[0, 42, 1337, 9999, 31415]`;
  green = `hash(word + str(seed)) % 2 == 0`; reports best |z|.
- `score = clip((|z| − 2.0)/4.0)`; `z > 4.0` → likely actively watermarked
  (with the caveat that absence ≠ human authorship). Requires ≥20 words.

### 7.5 DetectGPT curvature
- Estimates `d(x, q) = logP(x) − mean_k logP(x̃_k)` using 20 lightweight
  perturbations (random synonym swaps of ~15% of words from a closed list,
  seeded RNG, case preserved).
- Positive curvature (original scores above its perturbations) → AI-leaning;
  `score = clip(curvature/0.5)` when > 0. Returns interpretation text.
  Skipped if the probe LM is unavailable.

### 7.6 Pattern signal
- Regex dictionary of 20 weighted AI-discourse markers (`"as an AI"` 2.0,
  `"delve into"` 1.8, `"robust framework"` 1.5, …).
- Plus repeated-trigram ratio. `0.6*markers + 0.4*repetition`, with an explicit
  caveat about formal/academic/non-native writers.

### 7.7 Confidence & criteria
- `ai_probability` = weighted mean of available signals (weights renormalized).
- If probe LM missing → `distance_from_mid *= 0.4`; if <50 words → `*= 0.6`.
- `distance_from_mid < 0.42` → **UNCERTAIN**; `p > 0.5` → **AI_GENERATED**;
  else **HUMAN_WRITTEN**. `confidence = 50 + distance*50`.
- Full metrics dict + `suspicious_spans` (up to 10 sentences with reasons:
  formulaic phrase match, very short sentence).

---

## 8. Temporal Analyzer (`models/temporal_analyzer.py`) — legacy/aux

LSTM-based consistency analyzer (2-layer LSTM, 512→256→2) with a motion
feature extractor and `MotionAnalyzer`. References a helper module
`app.ml.temporal_analyzer` (via `video_detector.py`) and the FaceLandmarker
task file that is not shipped, so in practice the **production video pipeline
uses `final_video_detector.py`**, not this file. Its legacy `video_detector.py`
(IsolationForest + `TemporalConsistencyNet`) is also not wired into the API.

---

## 9. Query Assistant (`models/query_assistant.py`)

- **Image mode** — YOLOv8 (`yolov8n.pt`, lazy-loaded under a mutex, with a
  temporary `torch.load(weights_only=False)` patch for PyTorch ≥2.6 during
  weight load) detects objects with boxes/confidences; then Gemini 2.5 Flash
  (`GEMINI_API_KEY` from `.env`) describes the scene, optionally guided by a
  user query. Returns objects, object_summary, gemini_answer.
- **Text mode** — Gemini 2.5 Flash free-form QA with a DeepGuard system prompt.
- Lazy clients; raises `ValueError` if no API key.

---

## 10. Backend API + Fusion Engine (`backend/api.py`)

Flask app with CORS, 100 MB upload cap, allowed extensions
(png/jpg/jpeg/gif, mp4/avi/mov, mp3/wav, txt/pdf — note `mkv/ogg/flac/doc*`
are accepted by the file-type helper but not in `ALLOWED_EXTENSIONS`).
Models are **lazy-loaded once** into a shared `models` dict; on import failure
each route falls back to a **mock detector**.

### 10.1 Routes
- `GET /` and `GET /<path>` → serves the dashboard.
- `POST /api/detect/image|audio|video` — multipart file; saves to `uploads/`,
  runs the detector, deletes the file, returns `{success, result, file_type}`.
- `POST /api/detect/text` — JSON `{text}` → detector.predict(text).
- `POST /api/detect/auto` — infers type from extension, runs the right
  detector, returns file hash too.
- `POST /api/detect/fusion` — accepts multiple files **and/or** an
  `text` form field; runs each modality detector, then
  `EvidenceFusionEngine.fuse(results)`; returns fused result + per-modality
  results.
- `POST /api/query` — image (file) or text query for the assistant.
- `GET /api/status` — running status, model availability, feature list.

### 10.2 `EvidenceFusionEngine`
**Quality-aware gating** — per-modality `q`:
- image: base 0.5 + 0.2 if faces detected + 0.3 if a real model signal exists.
- video: base 0.5 + 0.2 if ≥5 frames + 0.3 if temporal analysis available.
- audio: base 0.5 + 0.4 trained-classifier / +0.3 HF / −0.2 heuristic.
- text: base 0.5 ± 0.3/0.2 by word count (≥200 → +0.3, <50 → −0.2), +0.2 if
  perplexity signal exists.
- Error results gate to `0.0`. Gates floor at `0.05` and are **normalized to
  sum to 1** (softmax-like normalization).

**`_modality_to_fake_score`** — maps each modality verdict to a [0,1] fake
probability: image/video FAKE → `0.5 + conf/200`, AUTHENTIC → `0.5 − conf/200`,
UNCERTAIN → 0.5; audio → `fake_probability`; text → `ai_probability`.

**Fusion** — `fused_score = Σ gate_m · fake_score_m`, clipped to [0,1].
Agreement counters: `fake > 0.6`, `real < 0.4`. Provenance flag if any
modality fired `provenance` in fake_type or `family_scores.provenance > 0.7`.

**4-tier verdict** (`has_provenance` wins):
1. Provenance → **Tier 1 / Verified Provenance / LIKELY_SYNTHETIC**,
   confidence 90.
2. fused < 0.35 → **Tier 2 / Likely Authentic / LIKELY_AUTHENTIC**,
   `50 + (0.35 − fused)*100`.
3. 0.35 ≤ fused ≤ 0.65 → **Tier 3 / Indeterminate / INDETERMINATE**,
   `30 + (1 − |fused − 0.5|·2)·20`.
4. fused > 0.65 **and** ≥2 fake agreements → **Tier 4 / Likely Synthetic /
   LIKELY_SYNTHETIC**, `50 + (fused − 0.65)*100`.
5. Fallback (fused > 0.65 but few agreements) → **Tier 3 / Indeterminate**,
   confidence 30.

Confidence capped at 99. Reasoning string lists each modality's direction,
score, and weight, sorted by distance from neutral.

### 10.3 Mock detectors
Used whenever a real detector fails to load. Image fires on
ai/generated/fake/midjourney/dalle/stable keywords with random confidence;
audio/video/text return bland UNCERTAIN results (50% confidence).

---

## 11. Dashboard (`dashboard/`)

- Drag-and-drop upload for images/video/audio, direct text analysis.
- 4-tier verdict display with indeterminate warnings; animated confidence
  meters; per-family evidence breakdown; suspicious region/time-slice
  highlighting; local-storage analysis history; responsive mobile layout.
- Talks to the same `backend/api.py` endpoints.

---

## 12. End-to-End Request Flow

```
Browser / client
  → POST /api/detect/{image|audio|video}  (file)
  → file saved to uploads/ (secure_filename)
  → lazy loader instantiates the detector (or mock fallback)
  → detector.predict(path):
        image  → forensics_core.full_image_forensics + multi-crop model
                 + family fusion + 4-tier verdict
        video  → keyframes (scene-cut aware) → per-frame image forensics
                 (5 frames) + temporal analysis + AV sync + temporal
                 localization + fusion + verdict
        audio  → 16kHz mono + silence trim → (classifier|HF|heuristic)
                 + quality-weighted segments + verdict
        text   → GPT-2 perplexity/burstiness + DetectGPT + stylometry
                 + watermarks + patterns + span highlight + verdict
  → file deleted from uploads/
  → JSON {success, result, file_type[, file_hash]} returned

POST /api/detect/fusion
  → each uploaded modality detected independently
  → EvidenceFusionEngine: quality gates → fake scores → weighted fusion
     → agreement counts → provenance check → 4-tier verdict
  → JSON {success, result(fused), modality_results}
```

---

## 13. Consolidated Threshold & Weight Tables

**Command-level fire thresholds (per-family):**

| Family | Image | Video |
|---|---|---|
| Provenance | 0.70 | 0.70 |
| Global / fully_ai_generated | 0.55 | 0.55 |
| Splice / ai_edited_region | 0.45 | 0.45 |
| Face-swap | 0.45 | 0.45 |
| Temporal inconsistency | — | 0.55 |
| Audio-visual desync | — | 0.40 |

**Fusion weights:**

| Modality | Image | Video |
|---|---|---|
| Provenance | 0.20 | — (not used in video fused score) |
| Global | 0.35 | 0.25 |
| Splice | 0.25 | 0.20 |
| Face-swap | 0.20 | 0.20 |
| Temporal | — | 0.20 |
| AV-sync | — | 0.15 |

**Text signal weights:** perplexity .40, DetectGPT .15, stylometry .25,
patterns .15, watermark .05.

**Temporal vote weights:** kinematics .30, optical-flow .25, identity .25,
blink .20.

**Tier boundaries:** authentic < 0.35, indeterminate 0.35–0.65, synthetic
> 0.65 (needs ≥2 agreeing signals at the fusion layer).

---

## 14. Safety & Correctness Principles

1. **Abstention by design** — UncERTAIN / Indeterminate is a first-class
   outcome, not a bug. Conflict, borderline, or missing model signals route
   there.
2. **Absence ≠ proof** — missing C2PA, metadata, watermark, or sensor noise is
   never treated as guilt by itself.
3. **Every score is explainable** — structured findings with signal name,
   score, region, and description; localization (regions, spans, time
   intervals) wherever possible.
4. **Graceful degradation** — missing heavy dependencies (mediapipe, librosa,
   transformers, ffmpeg, GPU) downgrades capability, never crashes; notes
   record what was skipped.
5. **No false accusations** — text detection is explicitly labeled screening
   evidence; audio heuristic mode suppresses its own confidence;
   provenance evidence outweighs everything at the fusion layer.
6. **Calibration disclaimer** — numeric thresholds encode literature
   principles but are not dataset-fit; they must be re-fit on labeled data
   from the actual use case.