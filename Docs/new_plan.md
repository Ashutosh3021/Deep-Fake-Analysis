# DeepGuard AI - V2 Architectural Upgrade Plan

## Baseline (Current State)

The existing codebase implements a 5-layer defense-in-depth pipeline:

- **Image**: FFT radial spectral decay, ELA, noise residual, face blend boundary, HuggingFace ViT classifier (6 classes)
- **Video**: Per-frame image forensics, MediaPipe landmark kinematics, optical flow warp residual, identity cosine distance, audio-visual energy/MAR cross-correlation
- **Audio**: MFCC+delta+delta2, spectral centroid/flatness/rolloff/contrast, HNR, F0 jitter, phase coherence, quality-weighted segments, HuggingFace wav2vec2 classifier (4 classes)
- **Text**: GPT-2 perplexity, DetectGPT log-curvature, stylometry (TTR, hapax, entropy, transitions, hedging), watermark z-score, regex patterns
- **Fusion**: Fixed weighted sum, quality-gated scaling, 4-tier verdict (Verified / Likely Authentic / Indeterminate / Likely Synthetic)

All thresholds are uncalibrated literature starting points. No adversarial robustness. No continual learning. No cross-attention between modalities.

---

## Gap Analysis: Upgrade Targets

| Dimension | Current | Gap | V2 Target |
|---|---|---|---|
| Image frequency | Radial FFT spectral decay | No scale-aware frequency features | MSCA-FFT multi-scale channel attention; FreqNet DCT plugin (~1.9M params) |
| Image spatial-freq | None | Haar wavelet residuals and SPSL missing | Haar wavelet high-freq residual; SPSL autoencoder (~1.2M params) |
| Audio backbones | wav2vec2-base (95M) single | Single-SSL, no adversarial defense | Dual-SSL: HuBERT-Large (304M) + WavLM-Large (316M); AASIST2 with LoRA + LearnT |
| Audio temporal | MFCC + delta + HNR + F0 jitter | No neural temporal aggregation | AMFF + NeXt-TDNN with ECA |
| Fusion architecture | Fixed weighted sum | No MoE, no cross-attention, no veto | FuseMoE Laplace gating; GAMED veto voting; CAST cross-attention |
| Physiological | No rPPG, no lip-sync | Missing physiological deepening | AV-HuBERT lip-sync; localized facial rPPG maps |
| Adversarial robustness | None | Models vulnerable to FGSM/PGD | FGSM/PGD adversarial training; feature squeezing; tanh-bounded cosine classifiers |
| Continual learning | Static pretrained models | No drift detection or evolving training | Evolving training pipeline with forgetting detection |

---

## Phase 1: Multi-Scale Spatial-Frequency Learning

**Files touched**: models/forensics_core.py, models/final_image_detector.py

### 1A: MSCA-FFT (Multi-Scale Channel Attention on FFT Features)

Replace single-scale radial FFT spectral decay with multi-scale channel attention.

Architecture:
`
Input: 256x256 RGB
  -> Scale 1: 3x3 avg pool -> 85x85 -> FFT -> 85x85x2
  -> Scale 2: 7x7 avg pool -> 37x37 -> FFT -> 37x37x2
  -> Scale 3: 15x15 avg pool -> 17x17 -> FFT -> 17x17x2
  -> SE attention per scale -> adaptive avg pool to 32x32
  -> Concat -> Conv 1x1 -> 32x32x32
  -> Global avg pool -> Linear(32) -> MSCA_FFT_feature (32-dim)
`

Parameters: ~280K
Expected: FFT AUC 0.85 -> 0.92+ on FaceForensics++ cross-manipulation

### 1B: FreqNet Frequency-Domain Plugin

DCT-based detection head:
`
Input: 256x256 grayscale -> 8x8 block DCT -> 32x32x64
  -> Conv 3x3 64->64 -> BN -> ReLU
  -> Conv 3x3 64->128 stride 2 -> BN -> ReLU
  -> Conv 3x3 128->128 -> BN -> ReLU
  -> GAP -> Linear(128->32) -> ReLU -> Linear(32->8) -> FreqNet_feature
`

Parameters: ~1.9M
Expected: ~94% on ProGAN/StyleGAN2 standalone; 96%+ combined

### 1C: Haar Wavelet Residual and SPSL

Haar wavelet 3-level decomposition for high-frequency residual map.
SPSL autoencoder (~1.2M params) with spectral bottleneck of 64-dim DCT eigenvectors.

Expected: +3-5% detection rate on lightly compressed (Q>70) GAN images

### Phase 1 Feature Vector Expansion

`
Current:  13 features
V2:       118 features (adds msca_fft_32d, freqnet_8d, haar_3d, spsl_64d)
`

New ImageFeatureFusion MLP: Linear(118->64)->ReLU->Dropout(0.3)->Linear(64->32)->ReLU->Linear(32->1)->Sigmoid
Parameters: ~8.5K

---

## Phase 2: Next-Generation Audio Forensics

**Files touched**: models/final_audio_detector.py, models/forensics_core.py

### 2A: Dual-SSL Backbone (HuBERT-Large + WavLM-Large)

| Component | Model | Params | Output dim | Role |
|---|---|---|---|---|
| SSL Stream 1 | HuBERT-Large | 304M | 1024 | Phoneme-level, noise-robust |
| SSL Stream 2 | WavLM-Large | 316M | 1024 | Speaker/environment-aware |

Fusion: Cross-attention (Q=WavLM, KV=HuBERT) -> Linear(1024->256) -> 256-dim embedding

Freeze strategy: Backbones frozen. LoRA adapters (rank 8, alpha 16, dropout 0.05) on final 4 transformer layers of each backbone. Trainable per backbone: ~3.2M. Total LoRA: ~6.4M.

### 2B: AMFF + NeXt-TDNN with ECA

AMFF (Attention-based Multi-scale Feature Fusion):
`
256d embedding
  -> Scale 1: Conv 1x1 -> 64d (frame-level, ~25ms)
  -> Scale 2: Conv 3x3 -> 64d (phrase-level, ~150ms)
  -> Scale 3: Conv 5x5 -> 64d (utterance-level, ~400ms)
  -> Multi-head attention (4 heads) across scales
  -> Concat -> Linear(192->128) -> amff_features_128d
`

NeXt-TDNN with ECA:
`
128d -> TDNN Conv1d 128->128 k=5 dil=1 -> BN -> PReLU
     -> TDNN Conv1d 128->128 k=3 dil=2 -> BN -> PReLU
     -> TDNN Conv1d 128->128 k=3 dil=3 -> BN -> PReLU
     -> ECA (k=5, adaptive avg pool -> sigmoid -> scale)
     -> Stat pool: mean+std -> 256d
     -> Linear(256->64) -> ReLU -> Dropout(0.3) -> Linear(64->1) -> Sigmoid
`

Parameters: AMFF ~120K, NeXt-TDNN ~210K, ECA ~0.8K

### 2C: AASIST2 with LoRA + LearnT

Graph attention network for anti-spoofing:
- AASIST2 backbone: Graph Attention Layer (GAL) operating on spectro-temporal features
- Input: 64-dim mel spectrogram frames -> graph nodes
- 3 GAL layers with 4 attention heads each
- LearnT: learnable temperature parameter for softmax attention (init=1.0, learnable)
- LoRA rank 4 on final 2 GAL layers
- Parameters: ~12M base + ~800K LoRA
- Output: 32-dim AASIST2 embedding

### 2D: Existing Classical Features (Retained)

Keep MFCC+delta+delta2, spectral centroid/flatness, HNR, F0 jitter, phase coherence as supplementary features. Append 48-dim classical feature vector to neural embeddings.

### Phase 2 Final Audio Feature Vector

`
Current:  wav2vec2 embeddings + 48 classical = ~768 + 48 = 816 features -> MLP -> p_audio
V2:       dual_ssl_256d + amff_128d + tdnn_64d + aasist2_32d + classical_48d = 528 features
          -> AudioFeatureFusion: Linear(528->128)->ReLU->Dropout(0.3)->Linear(128->32)->ReLU->Linear(32->1)->Sigmoid -> p_audio
`

Parameters for fusion MLP: ~69K
Target: EER < 1% on ASVspoof 2021 LA; AUC > 99% on In-the-Wild

---

## Phase 3: Advanced Multi-Modal Fusion

**Files touched**: bbackend/api.py, models/forensics_core.py

### 3A: FuseMoE (Mixture of Experts with Laplace Distance Gating)

Replace fixed weighted sum with learnable expert routing:

Architecture:
`
Inputs: p_image, p_video, p_audio, p_text (each 1-dim from modality classifiers)
  -> Quality scores: q_image, q_video, q_audio, q_text (from each detector's confidence)
  ->
  Gate network:
    input = [p_image, p_video, p_audio, p_text, q_image, q_video, q_audio, q_text] (8-dim)
    -> Linear(8 -> 4) -> Laplace distance weighting -> expert_weights (4-dim, sum to 1)
  ->
  Expert networks (4 experts, each a 2-layer MLP):
    Expert 1 (image-specialist): Linear(4->16)->ReLU->Linear(16->1)
    Expert 2 (audio-specialist): Linear(4->16)->ReLU->Linear(16->1)
    Expert 3 (cross-modal):     Linear(4->16)->ReLU->Linear(16->1)
    Expert 4 (conservative):    Linear(4->16)->ReLU->Linear(16->1)
  ->
  Fused score = sum(weight_i * expert_i_output)
  -> Sigmoid -> p_fused
`

**Laplace distance gating** (instead of softmax):
`
gate_i = exp(-|x - mu_i| / b_i) / sum(exp(-|x - mu_j| / b_j))
`
Where mu_i = learned expert centroid, b_i = learned bandwidth per expert.

Parameters: Gate ~36K, 4 experts ~1.2K each = ~4.8K. Total FuseMoE: ~41K.

### 3B: GAMED Veto Voting

Guard Against Manipulation with Evidential Deletion:

`
For each modality m in {image, video, audio, text}:
    if p_m exists AND quality_m > threshold (0.3):
        votes.append(p_m)

if len(votes) >= 2:
    majority = sum(1 for v in votes if v > 0.5) / len(votes)
    if majority > 0.5:
        verdict = "Likely Synthetic"
    elif majority < 0.3:
        verdict = "Likely Authentic"
    else:
        verdict = "Indeterminate"

# VETO rule: if any single modality has p_m < 0.1 (strongly real) AND
# quality_m > 0.7 (high confidence), apply veto:
for m in modalities:
    if p_m < 0.1 and quality_m > 0.7:
        verdict = max(verdict, "Indeterminate")  # cannot accuse
`

### 3C: CAST Cross-Attention Between Modalities

Cross-modal attention to learn inter-modality dependencies:

`
For video analysis (image + audio fusion):
    Q = video_frame_features (from image stream, 32-dim)
    K = V = audio_segment_features (from audio stream, 32-dim)
    -> Multi-head cross-attention (2 heads, dim=16 per head)
    -> Cross-modal consistency score: cos_sim(video_attended, audio_attended)
    -> If consistency < 0.3 -> flag as suspicious cross-modal mismatch
`

Parameters: ~24K per cross-attention module. One for image-audio, one for image-text. Total: ~48K.

### 3D: Quality-Aware Gating (Upgraded)

Current: Simple quality thresholding.
V2: Learnable quality gate:

`
alpha_m = sigmoid(quality_linear(q_m))
p_m_weighted = alpha_m * p_m
`

Where quality_linear: Linear(1->1) per modality, initialized to identity.
Parameters: ~4 total (one scalar per modality)

### Phase 3 Fusion Output

`
Fused features for final verdict:
  [p_fused, p_image, p_video, p_audio, p_text, q_image, q_video, q_audio, q_text,
   cross_modal_consistency_ia, cross_modal_consistency_it, expert_weights_4d]
  = 21-dim

Final verdict MLP: Linear(21->16)->ReLU->Dropout(0.2)->Linear(16->1)->Sigmoid -> p_final
`

Parameters: ~360
Target: Fusion AUC > 99.5% when >= 2 modalities available; graceful degradation to best-single-modality when only 1 available.

---

## Phase 4: Physiological and Biomechanical Deepening

**Files touched**: models/ffinal_video_detector.py, models/forensics_core.py

### 4A: AV-HuBERT Lip-Sync Verification

Replace the current energy/MAR cross-correlation with a neural lip-sync model:

- Model: AV-HuBERT (facebook/avhubert-large-ls960) fine-tuned for deepfake detection
- Input: Audio waveform + video mouth ROI (96x96)
- Output: Sync confidence score in [-1, 1]
- LoRA rank 8 on final 2 transformer layers
- Parameters: ~900M frozen + ~1.6M LoRA
- Output: 128-dim lip-sync embedding

Threshold: sync_score < 0.3 -> flag as lip-sync manipulation
Expected: EER < 5% on FaceForensics++ c23 lip-sync attacks

### 4B: Localized Facial rPPG Maps

Remote photoplethysmography for physiological signal verification:

- Extract facial ROI (forehead + cheeks) from detected face landmarks
- Apply CHROM (chrominance-based) rPPG method:
  `
  X_t = 3*R_t - 2*G_t
  Y_t = 1.5*R_t + G_t - 1.5*B_t
  rPPG_signal = alpha * X_t + beta * Y_t
  where alpha, beta are derived from temporal statistics
  `
- Compute pulse spectrum via FFT on rPPG signal
- Heart rate estimation and pulse regularity index
- rPPG spatial map: 32x32 heatmap of pulse signal amplitude across face
- Deepfake faces typically show flat/irregular rPPG vs natural pulsatile pattern

Parameters: ~85K (CHROM processing + 1x1 conv for spatial map)
Output: 128-dim rPPG embedding + pulse_rate + regularity_index

Expected: rPPG consistency adds 8-12% detection rate on face-swap deepfakes

### Phase 4 Video Feature Vector Expansion

`
Current:  landmark_jitter_6d + flow_residual_1d + identity_dist_1d + av_sync_1d + per_frame_p_3d = 12 features
V2:       landmark_jitter_6d + flow_residual_1d + identity_dist_1d + av_hubert_128d + rppg_128d + pulse_3d + per_frame_p_3d = 268 features
`

VideoFeatureFusion MLP: Linear(268->64)->ReLU->Dropout(0.3)->Linear(64->32)->ReLU->Linear(32->1)->Sigmoid -> p_video
Parameters: ~18K

---

## Phase 5: Adversarial Hardening and Tanh-Bounded Classifiers

**Files touched**: models/forensics_core.py, all detector files

### 5A: Tanh-Bounded Cosine Classifiers

Replace standard Sigmoid output with tanh-bounded cosine classifier for better calibration:

Current (all detectors):
`
score = sigmoid(W @ features + b)  # unbounded logits -> sigmoid
`

V2:
`
raw = cosine_similarity(features, W_real) - cosine_similarity(features, W_fake)
score = (tanh(k * raw) + 1) / 2  # bounded in (0, 1), k controls sharpness
`

Where:
- W_real, W_fake: learnable prototype vectors (dim = feature_dim)
- k: learnable sharpness parameter (init=2.0, min=0.5, max=10.0)
- tanh ensures bounded output even under adversarial perturbation
- Cosine similarity is inherently normalized, adding robustness

Parameters: 2 * feature_dim + 1 scalar per detector. Total: ~1.2K across all 4 detectors.

Expected improvement: Calibration ECE reduction from ~0.15 to ~0.05

### 5B: FGSM/PGD Adversarial Training

Input-space adversarial hardening during training:

**FGSM (Fast Gradient Sign Method)**:
`
x_adv = x + epsilon * sign(nabla_x L(f(x), y))
epsilon: 8/255 for images, 0.002 for audio, 0.01 for text embeddings
`

**PGD (Projected Gradient Descent)** - stronger:
`
x_{t+1} = clip(x_t + alpha * sign(nabla_x L(f(x_t), y)), x-epsilon, x+epsilon)
K=7 iterations, alpha=epsilon/(2*K)
`

**Training procedure**:
`
For each batch:
    x_clean, y = batch
    x_adv_1 = FGSM(model, x_clean, y, eps)
    x_adv_2 = PGD(model, x_clean, y, eps, K=7)
    x_adv_3 = feature_squeeze(x_clean)  # bit-depth reduction + spatial smoothing
    loss = 0.33 * CE(model(x_clean), y) + 0.33 * CE(model(x_adv_1), y) + 0.34 * CE(model(x_adv_2), y)
`

Expected robustness: Clean accuracy maintains within 1% of baseline; FGSM attack success rate drops from ~85% to ~15%; PGD attack success rate drops from ~70% to ~25%.

### 5C: Feature Squeezing (Inference-Time Defense)

Apply at inference to reduce adversarial perturbation impact:

`
def feature_squeeze(x):
    # 1. Spatial smoothing: median filter 3x3
    x_smooth = cv2.medianBlur(x, 3)
    # 2. Bit depth reduction: 8-bit -> 5-bit -> back to 8-bit
    x_quantized = (x_smooth // 8) * 8
    # 3. Averaging: mean of original + smoothed + quantized
    return (x + x_smooth + x_quantized) / 3
`

Parameters: 0 (heuristic, no learnable params)
Overhead: ~2ms per image, ~1ms per audio frame

### 5D: Continual Learning Pipeline

Evolving training with forgetting detection:

`
Continual Learning Cycle (monthly):
  1. Collect new deepfake samples from honeypot + web crawl
  2. Run current model on new samples -> identify hard negatives
  3. Compute forgetting metric: F_i = accuracy_old(i) - accuracy_current(i)
     For each class/domain i, if F_i > 0.05 (5% forgetting threshold)
  4. Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention:
     L_total = L_new + lambda * sum(F_i * (theta_i - theta_old_i)^2)
     lambda = 1000 (importance weight)
  5. Fine-tune for 3 epochs with EWC loss on combined old+new data
  6. Validate on held-out test set from ALL previously seen domains
  7. If any domain accuracy drops > 2%, rollback to previous checkpoint
  8. Deploy updated model, archive old model as backup
`

**Training data sources**:
- FaceForensics++ (1,000 real + 1,000 fake per manipulation type)
- Celeb-DF v2 (590 real + 5,639 fake)
- WildDeepfake (3,805 real + 3,805 fake)
- ASVspoof 2021 LA (real + spoof audio)
- In-the-Wild (in-the-wild deepfake videos)

**Schedule**: Fine-tune monthly with 20% new data + 80% rehearsal buffer from previous months.

---

## Implementation Priority and Timeline

| Phase | Priority | Effort | Impact | Dependencies |
|---|---|---|---|---|
| 5A: Tanh-bounded classifiers | 1 (do first) | 1 day | Medium - better calibration | None |
| 1A: MSCA-FFT | 2 | 3 days | High - core image improvement | None |
| 1B: FreqNet | 3 | 2 days | High - frequency detection | 1A |
| 1C: Haar + SPSL | 4 | 2 days | Medium - complementary | None |
| Phase 1 integration | 5 | 1 day | High - feature fusion | 1A, 1B, 1C |
| 2A: Dual-SSL | 6 | 3 days | Very High - audio backbone | None |
| 2B: AMFF + NeXt-TDNN | 7 | 2 days | High - temporal modeling | 2A |
| 2C: AASIST2 | 8 | 2 days | High - anti-spoofing | None |
| Phase 2 integration | 9 | 1 day | High - audio fusion | 2A, 2B, 2C |
| 3A: FuseMoE | 10 | 3 days | Very High - fusion intelligence | Phase 1 + 2 |
| 3B: GAMED veto | 11 | 1 day | High - safety guardrail | 3A |
| 3C: CAST cross-attention | 12 | 2 days | Medium - cross-modal | 3A |
| 3D: Quality gating upgrade | 13 | 0.5 day | Low - incremental | 3A |
| Phase 3 integration | 14 | 1 day | High - end-to-end fusion | 3A, 3B, 3C |
| 4A: AV-HuBERT lip-sync | 15 | 3 days | High - video physiology | None |
| 4B: rPPG maps | 16 | 2 days | High - physiological signals | None |
| Phase 4 integration | 17 | 1 day | High - video feature expansion | 4A, 4B |
| 5B: Adversarial training | 18 | 3 days | High - robustness | 5A |
| 5C: Feature squeezing | 19 | 0.5 day | Medium - inference defense | None |
| 5D: Continual learning | 20 | 3 days | High - long-term maintenance | All phases |

**Total estimated effort**: ~35-40 engineering days

---

## Parameter Budget Summary

| Module | Parameters | Trainable |
|---|---|---|
| MSCA-FFT | 280K | 280K |
| FreqNet | 1.9M | 1.9M |
| Haar + SPSL | 1.2M | 1.2M |
| Image Fusion MLP | 8.5K | 8.5K |
| Dual-SSL (frozen) | 620M | 0 |
| LoRA adapters (audio) | 6.4M | 6.4M |
| Cross-attn + projection | 520K | 520K |
| AMFF | 120K | 120K |
| NeXt-TDNN + ECA | 211K | 211K |
| AASIST2 | 12.8M | 12.8M |
| Audio Fusion MLP | 69K | 69K |
| FuseMoE | 41K | 41K |
| CAST modules | 48K | 48K |
| Final verdict MLP | 360 | 360 |
| AV-HuBERT (frozen) | 900M | 0 |
| AV-HuBERT LoRA | 1.6M | 1.6M |
| rPPG module | 85K | 85K |
| Video Fusion MLP | 18K | 18K |
| Tanh classifiers | 1.2K | 1.2K |
| **Total (frozen)** | **~1.52B** | - |
| **Total (trainable)** | - | **~26.8M** |

---

## Expected Performance Targets

| Metric | Current (estimated) | V2 Target | Dataset |
|---|---|---|---|
| Image AUC (cross-method) | ~88% | >95% | FaceForensics++ cross-manipulation |
| Audio EER | ~5-8% | <1% | ASVspoof 2021 LA |
| Video lip-sync detection | ~75% | >95% | FaceForensics++ c23 lip-sync |
| Text detection accuracy | ~82% | >90% | AI-generated text benchmarks |
| Fusion AUC (multi-modal) | ~92% | >99% | WildDeepfake + In-the-Wild |
| Adversarial robustness | ~15% (FGSM) | >85% | FGSM + PGD attacks |
| Calibration ECE | ~0.15 | <0.05 | All datasets |

---

## Risk Mitigation

1. **GPU Memory**: Dual-SSL (620M) + AV-HuBERT (900M) = ~1.5B frozen params. Use CPU offloading or model parallelism for inference. Batch size 1 for audio/video analysis.
2. **Inference Latency**: Current ~3s/image. V2 target ~5-8s/image with all modules. Mitigation: async processing, module caching, optional fast-path for high-confidence cases.
3. **Windows Path Bug**: Fix hardcoded /tmp/ in final_video_detector.py:150 to use tempfile.gettempdir().
4. **YOLO Weights**: Migrate yolov8n.pt (6.23 MB) to Git LFS.
5. **Dependency Conflicts**: Dual-SSL requires PyTorch 2.0+ with FlashAttention support. Verify `requirements.txt` compatibility.



?
??
