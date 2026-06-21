"""
FINAL Image Manipulation Detector (v2) -- catches ANY kind of fake
======================================================================
Scope, deliberately: this single detector now covers all three manipulation
families, because in practice you don't know in advance which one you're
looking at:

  A. FULLY AI-GENERATED  (Midjourney, diffusion models, GANs, etc.)
  B. AI-EDITED            (real photo, an AI-generated/inpainted region
                            spliced in -- object removal, background swap)
  C. FACE-SWAP             (real video/photo, a different face composited
                            in -- classic "deepfake", reenactment)

Architecture: rather than one global score, this runs THREE independent
forensic families (see forensics_core.py) and reports a verdict PLUS the
ranked list of findings that drove it -- "fake because X, in region Y" --
not just a number. If nothing fires, it says so plainly instead of forcing
a guess.

Combination logic:
  - Each family (global / splice / per-face) produces its own sub-score.
  - The OVERALL verdict is FAKE if *any* family crosses its threshold --
    a photo can be 95% real and have one small AI-inpainted region; a
    global average would wash that out, so we deliberately do NOT average
    across families. We take the max, with the global pretrained-model
    signal as an independent strong vote.
  - The "reason" is the full list of individual findings that crossed
    their own threshold, sorted by score.

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

# Per-family thresholds. A family "fires" (contributes to a FAKE verdict)
# if its score exceeds this. These are intentionally distinct from the
# 0.5 used for a coin-flip-style global score, because each family here
# is asking a more specific question ("is there a localized anomaly") and
# can be more conservative individually while still being decisive when
# it does fire.
GLOBAL_MODEL_FIRE_THRESHOLD = 0.55
SPLICE_FIRE_THRESHOLD = 0.45
FACE_SWAP_FIRE_THRESHOLD = 0.45

LOW_CONFIDENCE_BAND = 0.12  # how close to a threshold counts as "borderline"


@dataclass
class ImageVerdict:
    label: str                       # "FAKE" | "AUTHENTIC" | "UNCERTAIN"
    fake_type: List[str]             # which families fired: subset of
                                      # ["fully_ai_generated", "ai_edited_region", "face_swap"]
    confidence: float
    reasons: List[Dict[str, Any]]    # ranked findings that drove the verdict
    family_scores: Dict[str, Any]
    faces_detected: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "fake_type": self.fake_type,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "family_scores": self.family_scores,
            "faces_detected": self.faces_detected,
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

    def _model_predict(self, image_path: str) -> Optional[float]:
        if self._pipeline is None:
            return None
        try:
            image = Image.open(image_path).convert("RGB")
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

    # ------------------------------------------------------------------
    def predict(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": "file_not_found", "path": image_path}

        forensics = fc.full_image_forensics(image_path)
        if "error" in forensics:
            return forensics

        model_signal = self._model_predict(image_path)

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

        # --- Family C: face-swap (max across all detected faces -- one
        #     fake face among several real ones should still trigger) ---
        face_analyses = forensics["face_analyses"]
        face_swap_score = float(np.max([fa["face_swap_score"] for fa in face_analyses])) if face_analyses else 0.0

        fired_families = []
        reasons: List[Dict[str, Any]] = []

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

        overall_score = max(global_score, splice_score, face_swap_score)
        is_borderline = any(
            abs(overall_score - t) < LOW_CONFIDENCE_BAND
            for t in (GLOBAL_MODEL_FIRE_THRESHOLD, SPLICE_FIRE_THRESHOLD, FACE_SWAP_FIRE_THRESHOLD)
        )

        if fired_families:
            label = "FAKE"
            confidence = 50.0 + min(overall_score, 1.0) * 50.0
        elif is_borderline:
            label = "UNCERTAIN"
            confidence = 50.0
        else:
            label = "AUTHENTIC"
            confidence = 50.0 + (1.0 - overall_score) * 50.0

        if not reasons:
            reasons = [{
                "signal": "none", "score": 0.0, "region": None,
                "description": "No manipulation signal crossed its detection threshold."
            }]

        notes = []
        if model_signal is None:
            notes.append(f"Pretrained global classifier unavailable ({self._model_load_error}); "
                         f"global AI-generation detection relying on forensic heuristics only, "
                         f"lower reliability for that family specifically.")
        if forensics["faces_detected"] == 0:
            notes.append("No faces detected -- face-swap check not applicable to this image.")

        verdict = ImageVerdict(
            label=label,
            fake_type=fired_families,
            confidence=confidence,
            reasons=reasons,
            family_scores={
                "fully_ai_generated": round(global_score, 3),
                "ai_edited_region": round(splice_score, 3),
                "face_swap": round(face_swap_score, 3),
                "model_signal_raw": round(model_signal, 3) if model_signal is not None else None,
            },
            faces_detected=forensics["faces_detected"],
            notes=" ".join(notes) if notes else "All three forensic families ran successfully.",
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
