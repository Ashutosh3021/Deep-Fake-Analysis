# DeepGuard AI: Comprehensive Multi-Layer Forensic Implementation Plan

## Executive Summary & Theoretical Foundations

This implementation plan translates the empirical and theoretical consensus from modern deepfake forensics research (synthesized across leading 2025–2026 surveys) into the **DeepGuard AI** production codebase.

### Core Theoretical Axiom
**There is no single foolproof binary detector.** Unimodal models trained on isolated benchmark datasets fail catastrophically under out-of-distribution conditions, social media compression, paraphrasing, and adversarial evasion (failing up to 80–88% of the time).

To achieve state-of-the-art resilience, DeepGuard AI implements a **Layered Defense-in-Depth Architecture**:
1. **Layer 1 — Active Provenance & Custody**: Cryptographic signatures (C2PA / Content Credentials), SHA-256 integrity, EXIF/XMP/ICC metadata consistency.
2. **Layer 2 — Classical Forensic Invariants**: Frequency-domain anomalies (DFT, DCT), noise-residual variance, Error Level Analysis (ELA), and sensor PRNU characteristics.
3. **Layer 3 — Modality-Specific Feature & Neural Estimators**: Deep representations, face landmark kinematics, and spectral feature modeling.
4. **Layer 4 — Cross-Modal Consistency**: Audio-visual phoneme-to-lip synchronization, scene acoustic matching, and text-image alignment.
5. **Layer 5 — Calibrated Evidence Fusion & Policy Engine**: Bayesian log-likelihood ratio fusion, quality-gated weighting, and a 4-tier decision verdict that incorporates an **Indeterminate** tier to prevent false-positive accusations.

---

## Modality Implementation Roadmap

```
  ┌────────────────────────────────────────────────────────────┐
  │                   DeepGuard AI Architecture                │
  └─────────────────────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 1. IMAGE FORENSICS      2. VIDEO FORENSICS      3. AUDIO FORENSICS
 • C2PA / EXIF Hash      • Frame Aggregation     • STFT / Mel / CQT
 • 2D DFT / DCT Decay    • Landmark Dynamics     • Glottal Flow / HNR
 • Noise Residual (ELA)  • Optical Flow Warping  • Spectral Descriptors
 • Face Blend & Meshes   • Audio-Visual SyncNet  • Quality-Weighted Segs
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                        4. TEXT FORENSICS
                        • Perplexity ($PP_T$) & Burstiness ($B$)
                        • Stylometry & Token Richness
                        • Watermark Verification (KGW / SynthID)
                        • Perturbation Curvature (DetectGPT)
                                │
                                ▼
                   5. CALIBRATED FUSION ENGINE
                   • Quality-Aware Gating ($\alpha_m$)
                   • Log-Likelihood Ratio Fusion
                   • Cost-Sensitive Decision Policy
```

---

# Phase 1: Image Detection Module (`models/final_image_detector.py` & `models/forensics_core.py`)

### 1.1 Mathematical Formulas & Principles

1. **Cryptographic Integrity & Metadata Verification**:
   - File Hash:
     $$h = \operatorname{SHA256}(x)$$
   - Provenance Score:
     $$P_{\text{prov}} = w_c C_{\text{c2pa}} + w_m M_{\text{exif}} + w_h H_{\text{hash}}$$
     Checks for C2PA JUMBF metadata boxes, camera sensor metadata plausibility, and software tags.

2. **2D Discrete Fourier Transform (DFT) & Spectral Anomaly**:
   - 2D DFT of image $I(x,y)$ of size $M \times N$:
     $$F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} I(x,y) \, e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
   - Power spectrum:
     $$S(u,v) = |F(u,v)|^2 = \operatorname{Re}(F)^2 + \operatorname{Im}(F)^2$$
   - Spectral Deviation from Natural Image Power-Law Decay ($1/f^\alpha$):
     $$k(f) = \left| \log |G_{\text{observed}}(f)|^2 - \log |G_{\text{natural}}(f)|^2 \right|$$
   - Periodic Peak Detection: Identifies high-frequency spikes $(\mu + 2.5\sigma)$ caused by convolutional upsampling and checkerboard GAN/diffusion artifacts.

3. **Noise Residual & Sensor Pattern Noise (PRNU)**:
   - Denoised residual estimation:
     $$R = I - L(I) \quad \text{where } L(I) \text{ is a high-pass / Wiener / Wavelet filter}$$
   - Local Noise Inconsistency across $K$ spatial patches:
     $$V_R = \frac{1}{K} \sum_{k=1}^K \left( \operatorname{Var}(R_k) - \overline{\operatorname{Var}(R)} \right)^2$$
   - Synthetic images either lack natural camera sensor noise or exhibit patch-inconsistent noise distributions.

4. **Error Level Analysis (ELA)**:
   - Compression error residual at quality factor $Q=90$:
     $$\Delta_{\text{ELA}}(x, y) = |I(x,y) - \operatorname{JPEG}_Q(I)(x,y)|$$
   - Measures compression divergence between spliced/inpainted inserts and native image regions.

5. **Face Splicing & Boundary Discontinuity**:
   - Gradient step across detected face boundary mask $\partial M_{\text{face}}$:
     $$\nabla_{\text{boundary}} = \frac{1}{|\partial M|} \oint_{\partial M} \|\nabla I(s)\| \, ds$$
   - Color distribution divergence (Wasserstein or KL distance) between face crop and surrounding background pixels.

6. **Multi-Crop Neural Classifier**:
   $$p_{\text{neural}} = \frac{1}{K} \sum_{k=1}^K p(y=1 \mid \operatorname{Crop}_k(I))$$

### 1.2 Implementation Tasks for Phase 1
- [ ] **Task 1.1**: Implement `C2PAExtractor` and EXIF parser in `forensics_core.py` to parse hardware signatures and software generators.
- [ ] **Task 1.2**: Refactor `fft_periodicity_score` to compute radial average spectral decay against theoretical natural $1/f^\alpha$ baselines.
- [ ] **Task 1.3**: Enhance `noise_inconsistency_score` to use localized patch variance and wavelet-denoised residual calculation.
- [ ] **Task 1.4**: Upgrade `detect_face_blend_discontinuity` with convex-hull boundary gradient analysis on YOLOv8/OpenCV face crops.
- [ ] **Task 1.5**: Implement multi-crop image inference aggregation in `final_image_detector.py`.
- [ ] **Task 1.6**: Add calibrated evidence weights and structured JSON findings with localized bounding boxes.

---

# Phase 2: Video Detection Module (`models/final_video_detector.py` & `models/temporal_analyzer.py`)

### 2.1 Mathematical Formulas & Principles

1. **Robust Frame-Level Aggregation**:
   - Rather than simple average or max (which false-alarms on single corrupted frames):
     $$p_{\text{frame}} = \operatorname{median}(p_1, \dots, p_T) + \lambda \cdot Q_{0.90}(p_1, \dots, p_T)$$
     where $Q_{0.90}$ is the 90th percentile anomaly score.

2. **Landmark Kinematics & Facial Dynamics**:
   - For tracked facial landmark $l_t = (x_t, y_t)$ at frame $t$:
     - Velocity:
       $$v_t = l_t - l_{t-1}$$
     - Acceleration:
       $$a_t = v_t - v_{t-1} = l_t - 2l_{t-1} + l_{t-2}$$
   - Jitter metric: Spikes exceeding biomechanical human limits ($> 3\sigma$) flag synthetic frame-by-frame reenactment or face-swap jitter.

3. **Optical Flow Motion Warp Residual**:
   - Dense optical flow motion field $\mathbf{u}_t = (u, v)$ between consecutive frames $F_t$ and $F_{t+1}$.
   - Motion compensation error:
     $$E_{\text{flow}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{\|F_{t+1} - \operatorname{warp}(F_t, \mathbf{u}_t)\|_1}{\operatorname{Area}(F)}$$
   - Discrepancy between face motion flow and background camera flow flags localized face swaps.

4. **Biometric Continuity & Identity Drift**:
   - Face embedding vector $e_t$ computed per frame.
   - Cosine identity distance across temporal window $W$:
     $$D_{\text{identity}}(t) = 1 - \frac{e_t \cdot e_{t-\tau}}{\|e_t\| \|e_{t-\tau}\|}$$
   - Unnatural identity fluctuation indicates generative morphing.
   - Eye-blink rate and eye-closure duration dynamics analysis.

5. **Audio-Visual Cross-Modal Synchronization (SyncNet Principle)**:
   - For speech audio embedding $a_t$ and mouth opening visual embedding $v_t$:
     $$S_{\text{sync}} = \max_{\delta \in [-\Delta, \Delta]} \frac{a_t \cdot v_{t+\delta}}{\|a_t\| \|v_{t+\delta}\|}$$
   - Asynchrony or persistent offset $\delta \ne 0$ indicates dubbing, audio cloning, or visual lip-sync manipulation.

6. **Unified Spatio-Temporal Loss/Score**:
   $$\mathcal{L}_{st} = \lambda_{\text{spatial}} \mathcal{L}_{\text{spatial}} + \lambda_{\text{temporal}} \mathcal{L}_{\text{temporal}}$$

### 2.2 Implementation Tasks for Phase 2
- [ ] **Task 2.1**: Implement keyframe selection algorithm (uniform spacing + scene-cut detection via frame histogram differences).
- [ ] **Task 2.2**: Upgrade `temporal_analyzer.py` to calculate landmark velocity ($v_t$) and acceleration ($a_t$) variance across frames.
- [ ] **Task 2.3**: Implement Farneback dense optical flow warp residual scoring across facial bounding boxes.
- [ ] **Task 2.4**: Implement face identity consistency tracker using embedding cosine distance across sampled frames.
- [ ] **Task 2.5**: Extract audio track from video container and implement cross-modal audio energy vs mouth-aspect-ratio (MAR) correlation.
- [ ] **Task 2.6**: Add temporal localization returning exact suspicious time intervals `[start_sec, end_sec]`.

---

# Phase 3: Audio Detection Module (`models/final_audio_detector.py`)

### 3.1 Mathematical Formulas & Principles

1. **Short-Time Fourier Transform (STFT) & Cepstral Features**:
   - STFT of discrete audio signal $x[n]$ with window $w[n]$:
     $$X(t, f) = \sum_n x[n] w[n-t] e^{-j 2\pi f n}$$
   - Mel-Frequency Cepstral Coefficients (MFCC):
     $$c_n = \sum_{m=1}^M \log(E_m) \cos\left( \frac{\pi n (m - 0.5)}{M} \right)$$
   - Delta ($\Delta$) and Delta-Delta ($\Delta^2$) Dynamic Transitions:
     $$\Delta X(t, n) = \frac{\sum_{r=1}^R r \cdot (X(t+r, n) - X(t-r, n))}{2 \sum_{r=1}^R r^2}$$

2. **Spectral Moments & Timbral Discontinuities**:
   - **Spectral Centroid** (brightness):
     $$C_t = \frac{\sum_f f |X(t, f)|}{\sum_f |X(t, f)|}$$
   - **Spectral Flatness** (Wiener entropy / noise vs harmonic tonality):
     $$F_t = \frac{\exp\left(\frac{1}{N} \sum_f \ln P_t(f)\right)}{\frac{1}{N} \sum_f P_t(f)}$$
   - **Harmonic-to-Noise Ratio (HNR)**:
     $$\operatorname{HNR} = 10 \log_{10}\left( \frac{P_{\text{harmonic}}}{P_{\text{noise}}} \right)$$
     Neural vocoders (HiFi-GAN, WaveNet) typically exhibit anomalous HNR degradation in unvoiced phonemes and frequency bands above 8 kHz.

3. **Glottal Flow & Vocal Physiology**:
   - Fundamental frequency ($F_0$) pitch continuity and jitter (pitch perturbation):
     $$\operatorname{Jitter} = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N} \sum_{i=1}^N T_i}$$
   - Amplitude shimmer (loudness perturbation across pitch periods).
   - Unnatural micro-prosody and lack of physiological breath acoustic transitions.

4. **Quality-Weighted Segment Aggregation**:
   - Chunk audio into short overlapping analysis windows $i \in \{1, \dots, N\}$:
     $$p_{\text{audio}} = \frac{\sum_{i=1}^N q_i \cdot p_i}{\sum_{i=1}^N q_i}$$
     where $q_i$ represents segment SNR / clarity to suppress false alarms from silent or background noise intervals.

### 3.2 Implementation Tasks for Phase 3
- [ ] **Task 3.1**: Standardize audio preprocessing (16 kHz mono conversion, silence trimming via RMS thresholding).
- [ ] **Task 3.2**: Implement delta & delta-delta feature extraction ($\Delta \text{MFCC}, \Delta^2 \text{MFCC}$) in `final_audio_detector.py`.
- [ ] **Task 3.3**: Compute spectral centroid, spectral flatness, and high-frequency roll-off (void cutoffs above 8kHz/16kHz).
- [ ] **Task 3.4**: Implement $F_0$ pitch tracking with pitch-jump anomaly detection (using librosa pyin).
- [ ] **Task 3.5**: Integrate HNR (Harmonic-to-Noise Ratio) and phase continuity estimators.
- [ ] **Task 3.6**: Implement quality-weighted segment aggregation and suspicious audio time-slice reporting.

---

# Phase 4: Text Detection Module (`models/final_text_detector.py`)

### 4.1 Mathematical Formulas & Principles

1. **Perplexity Analysis ($PP_T$)**:
   - For token sequence $w_1, \dots, w_t$ under language model probability $P_M$:
     $$\operatorname{PPL}(W) = \exp\left( -\frac{1}{t} \sum_{i=1}^t \log P_M(w_i \mid w_{<i}) \right)$$
   - AI-generated text exhibits characteristically low and uniform perplexity compared to natural human writing.

2. **Burstiness ($B$) — Sentence Complexity Variance**:
   - Perplexity variance formulation:
     $$\operatorname{Burst}(W) = \operatorname{Var}\{\operatorname{PPL}(s) \mid s \in \operatorname{Sentences}(W)\}$$
   - Inter-arrival / sentence length dynamic formulation:
     $$B = \frac{\sigma - \mu}{\sigma + \mu} \in [-1, 1]$$
     where $\sigma$ and $\mu$ are the standard deviation and mean of sentence lengths.
     - Human text: $B > 0$ (bursty, rhythmic variation).
     - AI text: $B \le 0$ (uniform, formulaic sentence cadence).

3. **Stylometric Footprinting & Vocabulary Richness**:
   - **Type-Token Ratio (TTR)** & Hapax Legomena Ratio ($H$):
     $$\operatorname{TTR} = \frac{|V_{\text{unique}}|}{N_{\text{total}}}, \quad H = \frac{N_{\text{words appearing once}}}{N_{\text{total}}}$$
   - Function word frequency distribution entropy (stop-word predictability).
   - Syntactic formulaic transitions density (e.g., "In conclusion", "It is crucial to consider", "Delve into").

4. **Active Watermarking Verification (Kirchenbauer / SynthID Principles)**:
   - Green-list / Red-list token classification using hashing rule:
     $$z = \frac{|s|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}$$
     where $|s|_G$ is green token count, $\gamma = 0.5$, and $T$ is total tokens.
   - Significant statistical excess ($z > 4.0$) indicates active token-level watermarking.

5. **DetectGPT Log-Curvature Principle**:
   - Measure sensitivity of log-probability under local sentence/word perturbations $\tilde{x} \sim q(\cdot \mid x)$:
     $$\mathbf{d}(x, q) = \log P_M(x) - \frac{1}{K} \sum_{k=1}^K \log P_M(\tilde{x}_k)$$
   - AI text lies in local negative log-probability curvature maxima.

6. **Screening Policy Guardrail**:
   - Text detectors **never** issue definitive accusations alone. Output is strictly designated as *Stylometric Screening Evidence* with clear limitations noted.

### 4.2 Implementation Tasks for Phase 4
- [ ] **Task 4.1**: Implement sentence tokenizer and mathematical burstiness calculation ($B = \frac{\sigma - \mu}{\sigma + \mu}$).
- [ ] **Task 4.2**: Implement vocabulary richness metrics (TTR, Hapax Legomena ratio, Shannon entropy of word frequencies).
- [ ] **Task 4.3**: Implement token watermark z-score tester supporting configurable hash seeds.
- [ ] **Task 4.4**: Enhance perplexity proxy scoring using n-gram language model or lightweight distilled transformer.
- [ ] **Task 4.5**: Integrate formulaic AI discourse markers and hedging phrase density checks.
- [ ] **Task 4.6**: Add localized span highlighting returning suspicious sentences with explanatory justifications.

---

# Phase 5: Calibrated Multi-Modal Fusion & Policy Engine (`backend/api.py` & `models/`)

### 5.1 Mathematical Formulas & Principles

1. **Quality-Aware Gating**:
   - Dynamic modality reliability weights:
     $$\alpha_m = \frac{\exp(g_m(q_m))}{\sum_j \exp(g_j(q_j))}$$
     where $q_m$ reflects quality factor (e.g., low resolution or heavy compression decreases visual weight; high background noise decreases audio weight).

2. **Bayesian Log-Likelihood Ratio Fusion**:
   $$\log \frac{P(\text{Fake} \mid E)}{P(\text{Real} \mid E)} = \log \frac{P(\text{Fake})}{P(\text{Real})} + \sum_{m} \alpha_m \log \frac{P(E_m \mid \text{Fake})}{P(E_m \mid \text{Real})}$$

3. **Cost-Sensitive Decision Thresholds**:
   - For cost of false positive $C_{FP}$ and cost of false negative $C_{FN}$:
     $$\tau^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$$
   - If false accusation cost is high ($C_{FP} \gg C_{FN}$), the threshold for "Likely Synthetic" shifts upward to prevent false accusations.

4. **Four-Tier Forensic Verdict Classification**:
   - **Tier 1: Verified Provenance** (Valid C2PA cryptographic signature confirms authenticity or AI tool creation).
   - **Tier 2: Likely Authentic** ($p_{\text{synthetic}} < 0.35$ and no strong anomalies).
   - **Tier 3: Indeterminate / Inconclusive** ($0.35 \le p_{\text{synthetic}} \le 0.65$ or conflicting modalities).
   - **Tier 4: Likely Synthetic / Manipulated** ($p_{\text{synthetic}} > 0.65$ with $\ge 2$ agreeing forensic signals).

### 5.2 Implementation Tasks for Phase 5
- [ ] **Task 5.1**: Build unified `EvidenceFusionEngine` in `backend/api.py` implementing quality gating and Bayesian likelihood fusion.
- [ ] **Task 5.2**: Update REST response schema to return structured findings, localized regions/timestamps, and confidence bounds.
- [ ] **Task 5.3**: Update frontend dashboard (`dashboard/app.js` and `dashboard/index.html`) to display the 4-tier verdict, indeterminate warnings, and evidence breakdowns.

---

## Execution Phasing Schedule

```
Phase 1: Image Forensics Enhancement (DFT, PRNU noise, ELA, face boundaries)
   │
Phase 2: Video Forensics Enhancement (Landmarks velocity/accel, optical flow, sync)
   │
Phase 3: Audio Forensics Enhancement (STFT, MFCC deltas, HNR, glottal dynamics)
   │
Phase 4: Text Forensics Enhancement (Burstiness B, perplexity, watermarks, stylometry)
   │
Phase 5: Unified Fusion, REST API & Dashboard Upgrades
```
