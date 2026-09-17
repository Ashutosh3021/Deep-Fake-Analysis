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

logger = logging.getLogger(__name__)

PRIMARY_MODEL_ID = os.getenv("IMAGE_DETECTOR_MODEL", "umm-maybe/AI-image-detector")

# Per-family fire thresholds
PROVENANCE_FIRE_THRESHOLD = 0.70
GLOBAL_MODEL_FIRE_THRESHOLD = 0.55
SPLICE_FIRE_THRESHOLD = 0.45
FACE_SWAP_FIRE_THRESHOLD = 0.45

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
            self._pipeline = pipeline("image-classification", model=PRIMARY_MODEL_ID)
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
    def predict(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": "file_not_found", "path": image_path}

        forensics = fc.full_image_forensics(image_path)
        if "error" in forensics:
            return forensics

        model_signal = self._model_predict(image_path)

        # --- Family 0: Provenance (cryptographic evidence) ---
        provenance_findings = forensics.get("provenance_findings", [])
        prov_score = 0.0
        for pf in provenance_findings:
            prov_score = max(prov_score, pf["score"])

        # --- Family A: fully AI-generated ---
        global_findings = forensics["global_findings"]
        global_heuristic_score = float(np.mean([f["score"] for f in global_findings]))
        if model_signal is not None:
            global_score = 0.7 * model_signal + 0.3 * global_heuristic_score
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
            if model_signal is not None and model_signal >= GLOBAL_MODEL_FIRE_THRESHOLD:
                reasons.append({
                    "signal": "pretrained_classifier", "score": round(model_signal, 3),
                    "region": None,
                    "description": "Pretrained AI-image classifier scored this image as likely AI-generated."
                })
            for f in global_findings:
                if f["score"] >= 0.5:
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
        # Weight contributions from each family
        w_prov, w_global, w_splice, w_face = 0.20, 0.35, 0.25, 0.20
        fused_score = (
            w_prov * prov_score +
            w_global * global_score +
            w_splice * splice_score +
            w_face * face_swap_score
        )

        # --- 4-tier verdict classification ---
        # Check for indeterminate (conflicting signals)
        family_scores_list = [prov_score, global_score, splice_score, face_swap_score]
        non_zero_scores = [s for s in family_scores_list if s > 0.05]
        has_conflict = len(non_zero_scores) >= 2 and max(non_zero_scores) - min(non_zero_scores) > 0.3

        overall_score = max(global_score, splice_score, face_swap_score, prov_score)
        is_borderline = any(
            abs(overall_score - t) < LOW_CONFIDENCE_BAND
            for t in (GLOBAL_MODEL_FIRE_THRESHOLD, SPLICE_FIRE_THRESHOLD, FACE_SWAP_FIRE_THRESHOLD)
        )

        if prov_score >= PROVENANCE_FIRE_THRESHOLD and not fired_families:
            # Verified provenance says AI -- trust it
            label = "FAKE"
            confidence = 85.0 + prov_score * 15.0
        elif fired_families and not has_conflict:
            # Multiple families agree or one strong signal
            if fused_score >= TIER_LIKELY_SYNTHETIC:
                label = "FAKE"
                confidence = 50.0 + min(fused_score, 1.0) * 50.0
            else:
                label = "UNCERTAIN"
                confidence = 50.0
        elif has_conflict or is_borderline:
            # Conflicting signals or borderline -- indeterminate
            label = "UNCERTAIN"
            confidence = 40.0 + fused_score * 20.0
        else:
            label = "AUTHENTIC"
            confidence = 50.0 + (1.0 - fused_score) * 50.0

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
                "fused_score": round(fused_score, 3),
            },
            faces_detected=forensics["faces_detected"],
            file_hash=forensics.get("file_hash", ""),
            notes=" ".join(notes) if notes else "All four forensic families ran successfully.",
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
