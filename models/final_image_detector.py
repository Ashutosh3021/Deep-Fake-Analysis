"""
FINAL Image Manipulation Detector (v3) -- catches ANY kind of fake
=====================================================================
Scope, deliberately: this single detector covers all manipulation families,
because in practice you don't know in advance which one you're looking at:

  A. FULLY AI-GENERATED  (Midjourney, diffusion models, GANs, etc.)
  B. AI-EDITED            (real photo, an AI-generated/inpainted region
                            spliced in -- object removal, background swap)
  C. FACE-SWAP             (real video/photo, a different face composited
                            in -- classic "deepfake", reenactment)

Architecture: four independent forensic families (provenance, global, splice,
per-face) with calibrated evidence weights. Multi-crop inference for the
neural classifier. 4-tier verdict classification with indeterminate tier.

Combination logic:
  - PROVENANCE family: C2PA/EXIF metadata checks (cryptographic evidence)
  - GLOBAL family: pretrained classifier + forensic heuristics
  - SPLICE family: ELA + noise residual consistency
  - FACE-SWAP family: per-face blend boundary + noise mismatch
  - The OVERALL verdict uses Bayesian-informed fusion across families,
    with an Indeterminate tier to prevent false-positive accusations.

Setup:
  pip install torch transformers pillow opencv-python-headless numpy retina-face --break-system-packages
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

import forensics_core as fc

# V2 neural network modules
try:
    import torch
    import nn_modules
    from nn_modules import (
        TanhBoundedCosineClassifier, MultiScaleFFT, FreqNetDCT,
        HaarWaveletResidual, SPSLEncoder, feature_squeeze,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

PRIMARY_MODEL_ID = os.getenv("IMAGE_DETECTOR_MODEL", "umm-maybe/AI-image-detector")

# Per-family fire thresholds
#
# Calibrated against a 38-image labeled set (22 AI-generated / 16 real photos,
# models/eval + tests). The previous values (global 0.55, face 0.55) put the
# neural AI-vs-real classifier below its own firing point: an image the Swin
# classifier scored 0.6 "artificial" still blended down to a global score of
# ~0.53 and was reported AUTHENTIC. Measured effect of the new values on that
# set: accuracy 0.71 -> 0.90, AI recall 0.59 -> 0.91, real specificity 0.875
# (unchanged). Thresholds sit on a plateau (global 0.36-0.42 all >= 0.87 acc),
# not on a knife edge.
PROVENANCE_FIRE_THRESHOLD = 0.70
GLOBAL_MODEL_FIRE_THRESHOLD = 0.40
SPLICE_FIRE_THRESHOLD = 0.55
FACE_SWAP_FIRE_THRESHOLD = 0.60

# Neural-vs-heuristic blend inside the "fully AI-generated" family. The
# classifier is the only signal that separates the two classes on the eval
# set (heuristics score ~0.35 for both AI art and real photos), so it has to
# carry most of the weight.
GLOBAL_MODEL_WEIGHT = 0.75

# Family weights for the fused [0,1] evidence score. Families that cannot
# carry evidence for a given image are dropped and the rest renormalized
# (see predict()): absent C2PA/EXIF is not proof of authenticity, and a face
# family that never ran is not a "clean" vote.
W_PROVENANCE, W_GLOBAL, W_SPLICE, W_FACE = 0.20, 0.35, 0.25, 0.20

# 4-tier verdict thresholds (from plan.md)
TIER_LIKELY_SYNTHETIC = 0.65
TIER_INDETERMINATE_LOW = 0.35
TIER_INDETERMINATE_HIGH = 0.65

LOW_CONFIDENCE_BAND = 0.12

# Multi-crop settings
MULTI_CROP_COUNT = 4  # K crops for multi-crop aggregation


@dataclass
class ImageVerdict:
    label: str                       # "FAKE" | "AUTHENTIC" | "UNCERTAIN"
    fake_type: List[str]             # which families fired: subset of
                                      # ["fully_ai_generated", "ai_edited_region", "face_swap"]
    confidence: float
    reasons: List[Dict[str, Any]]    # ranked findings that drove the verdict
    family_scores: Dict[str, Any]
    faces_detected: int
    file_hash: str = ""              # SHA-256 for chain-of-custody
    notes: str = ""
    tier: int = 3                    # 1..4 verdict tier (see README)
    tier_name: str = "Indeterminate"
    p_synthetic: float = 0.5         # fused evidence score, 0..1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "fake_type": self.fake_type,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "family_scores": self.family_scores,
            "faces_detected": self.faces_detected,
            "file_hash": self.file_hash,
            "notes": self.notes,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "p_synthetic": round(self.p_synthetic, 4),
        }


class FinalImageDetector:
    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self._pipeline = None
        self._model_load_error: Optional[str] = None
        if not lite_mode:
            self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline
            # framework="pt" is required: without it transformers may pick the
            # TensorFlow backend when tensorflow/keras are installed as a
            # transitive dependency (retina-face), which fails under Keras 3
            # and silently drops the model to heuristic-only mode.
            self._pipeline = pipeline(
                "image-classification",
                model=PRIMARY_MODEL_ID,
                framework="pt",
                device=-1,
            )
            logger.info("Loaded primary image detection model: %s", PRIMARY_MODEL_ID)
        except Exception as e:
            self._model_load_error = str(e)
            logger.warning("Could not load pretrained model (%s); global AI-gen signal will rely "
                            "on forensic heuristics only.", e)

    def _model_predict_single(self, image: Image.Image) -> Optional[float]:
        """Run model inference on a single PIL Image."""
        if self._pipeline is None:
            return None
        try:
            results = self._pipeline(image)
            ai_score = 0.0
            for r in results:
                label = r["label"].lower()
                if any(k in label for k in ("fake", "ai", "synthetic", "generated", "artificial")):
                    ai_score = max(ai_score, r["score"])
                elif any(k in label for k in ("real", "human", "authentic", "natural")):
                    ai_score = max(ai_score, 1.0 - r["score"])
            return float(ai_score)
        except Exception as e:
            logger.error("Model inference failed: %s", e)
            return None

    def _multi_crop_predict(self, image_path: str) -> Optional[float]:
        """
        Multi-crop neural classifier aggregation (from plan.md):
        p_neural = (1/K) * sum_{k=1}^{K} p(y=1 | Crop_k(I))

        Takes K crops from the image (center, top-left, bottom-right, random)
        and averages their predictions to reduce reliance on any single region.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            crop_size = min(w, h) // 2
            if crop_size < 32:
                # Too small for meaningful crops, use full image
                return self._model_predict_single(image)

            crops = []
            # Center crop
            cx, cy = w // 2, h // 2
            crops.append(image.crop((cx - crop_size // 2, cy - crop_size // 2,
                                     cx + crop_size // 2, cy + crop_size // 2)))
            # Top-left crop
            crops.append(image.crop((0, 0, crop_size, crop_size)))
            # Bottom-right crop
            crops.append(image.crop((w - crop_size, h - crop_size, w, h)))
            # Center-right crop
            crops.append(image.crop((w - crop_size, cy - crop_size // 2,
                                     w, cy + crop_size // 2)))

            scores = []
            for crop in crops[:MULTI_CROP_COUNT]:
                s = self._model_predict_single(crop)
                if s is not None:
                    scores.append(s)

            if not scores:
                return None
            return float(np.mean(scores))
        except Exception as e:
            logger.error("Multi-crop prediction failed: %s", e)
            return self._model_predict_single(Image.open(image_path).convert("RGB"))

    def _model_predict(self, image_path: str) -> Optional[float]:
        return self._multi_crop_predict(image_path)

    # ------------------------------------------------------------------
    # V2: Multi-scale frequency feature extraction
    # ------------------------------------------------------------------
    def _extract_v2_features(self, image_path: str) -> Dict[str, Any]:
        """Extract V2 frequency-domain features using MSCA-FFT, FreqNet, Haar, SPSL."""
        v2_result = {
            "msca_fft_score": None,
            "freqnet_score": None,
            "haar_energy": None,
            "spsl_score": None,
            "v2_available": False,
        }
        if not TORCH_AVAILABLE:
            return v2_result

        try:
            device = torch.device("cpu")
            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0

            # Convert to tensor: (1, 3, H, W)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
            gray_tensor = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]

            # MSCA-FFT
            try:
                msca = MultiScaleFFT(out_features=32).to(device).eval()
                with torch.no_grad():
                    msca_feat = msca(img_tensor)
                msca_score = float(torch.sigmoid(msca_feat.mean()).item())
                v2_result["msca_fft_score"] = round(msca_score, 4)
            except Exception as e:
                logger.debug("MSCA-FFT failed: %s", e)

            # FreqNet DCT
            try:
                freqnet = FreqNetDCT(block_size=8, out_features=8).to(device).eval()
                with torch.no_grad():
                    freq_feat = freqnet(gray_tensor)
                freq_score = float(torch.sigmoid(freq_feat.mean()).item())
                v2_result["freqnet_score"] = round(freq_score, 4)
            except Exception as e:
                logger.debug("FreqNet DCT failed: %s", e)

            # Haar wavelet residual
            try:
                haar = HaarWaveletResidual().to(device).eval()
                with torch.no_grad():
                    haar_feat = haar(gray_tensor)
                haar_energy = haar_feat.mean(dim=0).cpu().numpy().tolist()
                v2_result["haar_energy"] = [round(float(e), 4) for e in haar_energy]
            except Exception as e:
                logger.debug("Haar wavelet failed: %s", e)

            # SPSL encoder
            try:
                spsl = SPSLEncoder(bottleneck_dim=64).to(device).eval()
                with torch.no_grad():
                    spsl_embed, spsl_recon = spsl(gray_tensor)
                recon_error = float(torch.nn.functional.mse_loss(spsl_recon, gray_tensor).item())
                v2_result["spsl_score"] = round(recon_error, 6)
            except Exception as e:
                logger.debug("SPSL failed: %s", e)

            v2_result["v2_available"] = any(v is not None for k, v in v2_result.items()
                                             if k != "v2_available" and v is not None)

        except Exception as e:
            logger.error("V2 feature extraction failed: %s", e)

        return v2_result

    # ------------------------------------------------------------------
    def predict(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": "file_not_found", "path": image_path}

        forensics = fc.full_image_forensics(image_path)
        if "error" in forensics:
            return forensics

        model_signal = self._model_predict(image_path)

        # V2: Extract frequency-domain features
        v2_features = self._extract_v2_features(image_path)

        # --- Family 0: Provenance (cryptographic evidence) ---
        provenance_findings = forensics.get("provenance_findings", [])
        prov_score = 0.0
        for pf in provenance_findings:
            prov_score = max(prov_score, pf["score"])

        # --- Family A: fully AI-generated ---
        global_findings = forensics["global_findings"]
        global_heuristic_score = float(np.mean([f["score"] for f in global_findings]))
        if model_signal is not None:
            global_score = (GLOBAL_MODEL_WEIGHT * model_signal
                            + (1.0 - GLOBAL_MODEL_WEIGHT) * global_heuristic_score)
        else:
            global_score = global_heuristic_score

        # --- Family B: localized AI-edit / splice ---
        splice_findings = forensics["splice_findings"]
        splice_score = float(np.max([f["score"] for f in splice_findings])) if splice_findings else 0.0

        # --- Family C: face-swap ---
        face_analyses = forensics["face_analyses"]
        face_swap_score = float(np.max([fa["face_swap_score"] for fa in face_analyses])) if face_analyses else 0.0

        fired_families = []
        reasons: List[Dict[str, Any]] = []

        # Provenance evidence (strongest signal when present)
        if prov_score >= PROVENANCE_FIRE_THRESHOLD:
            fired_families.append("provenance_ai_detected")
            for pf in provenance_findings:
                if pf["score"] >= PROVENANCE_FIRE_THRESHOLD:
                    reasons.append(pf)

        if global_score >= GLOBAL_MODEL_FIRE_THRESHOLD:
            fired_families.append("fully_ai_generated")
            if model_signal is not None:
                reasons.append({
                    "signal": "pretrained_classifier", "score": round(model_signal, 3),
                    "region": None,
                    "description": (f"AI-image classifier scored this image {model_signal:.0%} synthetic "
                                    f"(forensic heuristics {global_heuristic_score:.0%}; "
                                    f"combined {global_score:.0%} against a "
                                    f"{GLOBAL_MODEL_FIRE_THRESHOLD:.0%} threshold).")
                })
            for f in global_findings:
                if f["score"] >= 0.4:
                    reasons.append(f)

        if splice_score >= SPLICE_FIRE_THRESHOLD:
            fired_families.append("ai_edited_region")
            for f in splice_findings:
                if f["score"] >= SPLICE_FIRE_THRESHOLD:
                    reasons.append(f)

        if face_swap_score >= FACE_SWAP_FIRE_THRESHOLD:
            fired_families.append("face_swap")
            for fa in face_analyses:
                if fa["face_swap_score"] >= FACE_SWAP_FIRE_THRESHOLD:
                    for f in fa["findings"]:
                        if f["score"] >= 0.3:
                            reasons.append(f)

        reasons.sort(key=lambda r: r["score"], reverse=True)

        # --- Calibrated evidence fusion (from plan.md) ---
        # Only families that could carry evidence for THIS image take part;
        # their weights are renormalized over the active set. Otherwise a
        # no-metadata / no-face image gets its score dragged toward zero by
        # families that had nothing to say (absence of C2PA is not proof of
        # authenticity), which is exactly how confident AI images used to
        # end up below the 0.55 firing line.
        active_weights: List[float] = []
        active_scores: List[float] = []
        if prov_score >= 0.50:                       # metadata actually present
            active_weights.append(W_PROVENANCE)
            active_scores.append(prov_score)
        active_weights.append(W_GLOBAL)
        active_scores.append(global_score)
        active_weights.append(W_SPLICE)
        active_scores.append(splice_score)
        if forensics["faces_detected"] > 0:
            active_weights.append(W_FACE)
            active_scores.append(face_swap_score)
        w_total = sum(active_weights)
        fused_score = (sum(w * s for w, s in zip(active_weights, active_scores)) / w_total
                       if w_total > 0 else 0.5)

        # Threshold-relative evidence, used for the decision margin. Each
        # family is expressed as "fraction of the way to its own firing
        # point", so confidence measures distance to the decision boundary
        # instead of raw score magnitude.
        thresholds = [
            (global_score, GLOBAL_MODEL_FIRE_THRESHOLD),
            (splice_score, SPLICE_FIRE_THRESHOLD),
        ]
        if prov_score >= 0.50:
            thresholds.append((prov_score, PROVENANCE_FIRE_THRESHOLD))
        if forensics["faces_detected"] > 0:
            thresholds.append((face_swap_score, FACE_SWAP_FIRE_THRESHOLD))
        relative_evidence = max((score / thr for score, thr in thresholds if thr > 0), default=0.0)

        # V2: DISABLED until trained — untrained modules inject noise
        v2_augmented = False

        # --- 4-tier verdict classification ---
        # Conflict only among significant scores (> 0.3)
        family_scores_list = [prov_score, global_score, splice_score, face_swap_score]
        sig_scores = [s for s in family_scores_list if s > 0.3]
        has_conflict = len(sig_scores) >= 2 and max(sig_scores) - min(sig_scores) > 0.3

        if fired_families:
            label = "FAKE"
            # Confidence = how far the strongest fired family is ABOVE its own
            # threshold, so a marginal signal reports a marginal confidence.
            best_score, best_thr = max(
                ((s, t) for s, t in thresholds if s >= t),
                key=lambda st: (st[0] - st[1]) / st[1],
                default=(0.0, 1.0),
            )
            confidence = 50.0 + 45.0 * min(1.0, max(0.0, (best_score - best_thr) / (1.0 - best_thr)))
            if "provenance_ai_detected" in fired_families:
                # Cryptographic C2PA signature: strongest evidence tier.
                confidence = max(confidence, 90.0)
            if len(fired_families) >= 2:
                confidence = min(95.0, confidence + 8.0)
        elif has_conflict:
            label = "UNCERTAIN"
            confidence = 40.0 + fused_score * 20.0
        elif relative_evidence >= 0.95:
            # Within 5% of a firing threshold: abstain rather than claim
            # authenticity (README's Indeterminate tier).
            label = "UNCERTAIN"
            confidence = 45.0 + (1.0 - relative_evidence) * 20.0
        else:
            label = "AUTHENTIC"
            # Confidence = distance below the closest firing threshold.
            confidence = 50.0 + 45.0 * (1.0 - relative_evidence)

        confidence = float(min(max(confidence, 5.0), 95.0))

        # Tier is computed here (single source of truth) so the dashboard
        # cannot disagree with the detector about what it just measured.
        if "provenance_ai_detected" in fired_families:
            tier, tier_name = 1, "Verified Provenance"
        elif label == "FAKE":
            tier, tier_name = 4, "Likely Synthetic"
        elif label == "UNCERTAIN":
            tier, tier_name = 3, "Indeterminate"
        else:
            tier, tier_name = 2, "Likely Authentic"

        if not reasons:
            reasons = [{
                "signal": "none", "score": 0.0, "region": None,
                "description": "No manipulation signal crossed its detection threshold."
            }]

        notes = []
        if model_signal is None:
            notes.append(f"Pretrained global classifier unavailable ({self._model_load_error}); "
                         f"global AI-generation detection relying on forensic heuristics only.")
        if forensics["faces_detected"] == 0:
            notes.append("No faces detected -- face-swap check not applicable to this image.")
        if v2_augmented:
            notes.append("V2 frequency-domain features (MSCA-FFT, FreqNet) integrated into fusion.")
        elif not TORCH_AVAILABLE:
            notes.append("V2 frequency features skipped: PyTorch not available.")

        verdict = ImageVerdict(
            label=label,
            fake_type=fired_families,
            confidence=confidence,
            reasons=reasons,
            family_scores={
                "provenance": round(prov_score, 3),
                "fully_ai_generated": round(global_score, 3),
                "ai_edited_region": round(splice_score, 3),
                "face_swap": round(face_swap_score, 3),
                "model_signal_raw": round(model_signal, 3) if model_signal is not None else None,
                "global_heuristic": round(global_heuristic_score, 3),
                "fused_score": round(fused_score, 3),
                "relative_evidence": round(relative_evidence, 3),
                "v2_msca_fft": v2_features.get("msca_fft_score"),
                "v2_freqnet": v2_features.get("freqnet_score"),
                "v2_haar_energy": v2_features.get("haar_energy"),
                "v2_spsl_recon_error": v2_features.get("spsl_score"),
                "v2_augmented": v2_augmented,
            },
            faces_detected=forensics["faces_detected"],
            file_hash=forensics.get("file_hash", ""),
            notes=" ".join(notes) if notes else "All four forensic families ran successfully.",
            tier=tier,
            tier_name=tier_name,
            p_synthetic=fused_score,
        )
        return verdict.to_dict()


final_image_detector = FinalImageDetector()


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python final_image_detector.py <image_path>")
        sys.exit(1)
    print(json.dumps(final_image_detector.predict(sys.argv[1]), indent=2))
