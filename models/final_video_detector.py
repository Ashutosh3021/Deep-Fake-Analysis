"""
FINAL Video Manipulation Detector (v2) -- catches ANY kind of fake
======================================================================
Same three-family coverage as final_image_detector.py, applied across
sampled frames, PLUS a temporal layer that's swap-agnostic and only
possible in video:

  A. FULLY AI-GENERATED video (Sora-style, fully synthesized)
  B. AI-EDITED video (object/background replaced in part of the frame,
     across some or all frames)
  C. FACE-SWAP / reenactment (per-face, per-frame, robust to multiple
     people / angles / partial occlusion via RetinaFace)
  D. TEMPORAL inconsistency (blink rate, head-pose jitter, landmark
     stability) -- this signal doesn't care WHICH technique made the
     fake, it just measures whether facial motion looks physically
     plausible across time. Genuinely complementary to A-C.

Verdict logic mirrors the image detector: any family firing -> FAKE,
with the full ranked list of findings (now annotated with WHICH sampled
frame they came from) as the "reason".

Setup:
  pip install opencv-python-headless mediapipe numpy retina-face --break-system-packages
  (uses final_image_detector.py's forensics via forensics_core.py)
"""

import os
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import cv2

import forensics_core as fc

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed -- temporal/landmark analysis will be skipped.")

MAX_FRAMES_SAMPLED = 24
FULL_FORENSICS_FRAME_COUNT = 5     # how many of the sampled frames get the
                                     # expensive full image-forensics pass
                                     # (model + FFT + ELA + per-face checks)

GLOBAL_FIRE_THRESHOLD = 0.55
SPLICE_FIRE_THRESHOLD = 0.45
FACE_SWAP_FIRE_THRESHOLD = 0.45
TEMPORAL_FIRE_THRESHOLD = 0.55

NATURAL_BLINK_RATE_RANGE = (8, 30)  # blinks/min, wide natural human range


@dataclass
class VideoVerdict:
    label: str
    fake_type: List[str]
    confidence: float
    reasons: List[Dict[str, Any]]
    family_scores: Dict[str, Any]
    frames_analyzed: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "fake_type": self.fake_type,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "family_scores": self.family_scores,
            "frames_analyzed": self.frames_analyzed,
            "notes": self.notes,
        }


class FinalVideoDetector:
    def __init__(self):
        self._face_mesh = None
        if MEDIAPIPE_AVAILABLE:
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=5,
                refine_landmarks=True, min_detection_confidence=0.5
            )

        self._image_detector = None
        try:
            from final_image_detector import FinalImageDetector
            self._image_detector = FinalImageDetector()
        except Exception as e:
            logger.warning("Could not load FinalImageDetector for per-frame scoring: %s", e)

    # ------------------------------------------------------------------
    def _extract_frames(self, video_path: str, max_frames: int = MAX_FRAMES_SAMPLED) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames) if total > 0 else 1

        frames = []
        idx = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        return frames

    # ------------------------------------------------------------------
    # Per-frame full forensics (model + FFT/ELA/noise + per-face), run on
    # a subset of sampled frames since this is the expensive pass.
    # ------------------------------------------------------------------
    def _per_frame_forensics(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        if self._image_detector is None or not frames:
            return []

        results = []
        tmp_dir = "/tmp/deepguard_video_frames"
        os.makedirs(tmp_dir, exist_ok=True)
        sample_indices = np.linspace(0, len(frames) - 1, min(FULL_FORENSICS_FRAME_COUNT, len(frames))).astype(int)

        for i in sample_indices:
            frame_path = os.path.join(tmp_dir, f"frame_{i}.jpg")
            try:
                cv2.imwrite(frame_path, frames[i])
                result = self._image_detector.predict(frame_path)
                result["frame_index"] = int(i)
                results.append(result)
            except Exception as e:
                logger.error("Per-frame forensics failed on frame %d: %s", i, e)
            finally:
                if os.path.exists(frame_path):
                    os.remove(frame_path)

        return results

    # ------------------------------------------------------------------
    # Temporal / geometric consistency (swap-agnostic, video-only signal)
    # ------------------------------------------------------------------
    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray, eye_indices: List[int]) -> float:
        p = pts[eye_indices]
        vertical1 = np.linalg.norm(p[1] - p[5])
        vertical2 = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        if horizontal < 1e-6:
            return 0.3
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def _landmark_sequence(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        if self._face_mesh is None:
            return []
        sequence = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)
            if results.multi_face_landmarks:
                # Track the largest/first face for the temporal sequence.
                # (Per-face temporal tracking across multiple people is a
                # natural extension but needs identity association across
                # frames, which is out of scope for this lightweight pass.)
                lm = results.multi_face_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm])
                sequence.append(pts)
            else:
                sequence.append(None)
        return sequence

    def _temporal_analysis(self, frames: List[np.ndarray], fps: float) -> Dict[str, Any]:
        if self._face_mesh is None:
            return {"available": False, "reason": "mediapipe not installed", "score": 0.0}

        sequence = self._landmark_sequence(frames)
        valid = [(i, p) for i, p in enumerate(sequence) if p is not None]

        if len(valid) < 4:
            return {"available": False, "reason": "insufficient face detections across frames",
                    "frames_with_face": len(valid), "score": 0.0}

        nose_positions = np.array([p[1][:2] for _, p in valid])
        face_scales = np.array([max(np.linalg.norm(p[33][:2] - p[263][:2]), 1e-4) for _, p in valid])
        frame_jitter = np.linalg.norm(np.diff(nose_positions, axis=0), axis=1)
        normalized_jitter = frame_jitter / face_scales[1:]
        position_stability = float(1.0 / (1.0 + np.std(normalized_jitter) * 50))

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [263, 387, 385, 362, 380, 373]
        ear_sequence = []
        for _, pts in valid:
            try:
                l = self._eye_aspect_ratio(pts, LEFT_EYE)
                r = self._eye_aspect_ratio(pts, RIGHT_EYE)
                ear_sequence.append((l + r) / 2.0)
            except Exception:
                ear_sequence.append(0.3)
        ear_sequence = np.array(ear_sequence)

        blink_threshold = (np.mean(ear_sequence) - 0.5 * np.std(ear_sequence)) if len(ear_sequence) > 1 else 0.2
        below = ear_sequence < blink_threshold
        blink_events = int(np.sum(np.diff(below.astype(int)) == 1))
        analyzed_seconds = len(frames) / fps if fps > 0 else len(frames) / 24.0
        blink_rate = (blink_events / max(analyzed_seconds, 1e-3)) * 60.0
        blink_natural = NATURAL_BLINK_RATE_RANGE[0] <= blink_rate <= NATURAL_BLINK_RATE_RANGE[1]

        all_pts = np.array([p for _, p in valid])
        mean_landmark_std = float(np.mean(np.std(all_pts[:, :, :2], axis=0)))

        votes = [
            1.0 - position_stability,
            0.6 if not blink_natural else 0.15,
            float(np.clip(1.0 - mean_landmark_std / 0.004, 0.0, 1.0)),
        ]
        score = float(np.clip(np.mean(votes), 0.0, 1.0))

        return {
            "available": True,
            "frames_with_face": len(valid),
            "position_stability": round(position_stability, 3),
            "estimated_blink_rate_per_min": round(blink_rate, 1),
            "blink_rate_in_natural_range": blink_natural,
            "score": round(score, 3),
        }

    # ------------------------------------------------------------------
    def predict(self, video_path: str) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            return {"error": "file_not_found", "path": video_path}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()

        frames = self._extract_frames(video_path)
        if not frames:
            return {"error": "no_frames_extracted", "path": video_path}

        per_frame_results = self._per_frame_forensics(frames)
        temporal = self._temporal_analysis(frames, fps)

        # --- Aggregate per-frame family scores across sampled frames.
        #     Use MAX, not mean: a fake is a fake even if it only shows in
        #     a few frames (e.g. a face-swap that glitches intermittently,
        #     or an edit confined to part of the timeline). ---
        global_scores, splice_scores, face_scores = [], [], []
        reasons: List[Dict[str, Any]] = []

        for fr in per_frame_results:
            if "family_scores" not in fr:
                continue
            fs = fr["family_scores"]
            global_scores.append(fs.get("fully_ai_generated", 0.0))
            splice_scores.append(fs.get("ai_edited_region", 0.0))
            face_scores.append(fs.get("face_swap", 0.0))

            for reason in fr.get("reasons", []):
                if reason.get("score", 0.0) >= 0.4 and reason.get("signal") != "none":
                    annotated = dict(reason)
                    annotated["frame_index"] = fr.get("frame_index")
                    reasons.append(annotated)

        global_score = float(np.max(global_scores)) if global_scores else 0.0
        splice_score = float(np.max(splice_scores)) if splice_scores else 0.0
        face_swap_score = float(np.max(face_scores)) if face_scores else 0.0
        temporal_score = temporal.get("score", 0.0) if temporal.get("available") else 0.0

        if temporal.get("available") and temporal_score >= TEMPORAL_FIRE_THRESHOLD:
            reasons.append({
                "signal": "temporal_consistency", "score": temporal_score, "region": None,
                "frame_index": None,
                "description": (
                    f"Facial motion across frames is physically implausible "
                    f"(blink rate {temporal.get('estimated_blink_rate_per_min')}/min, "
                    f"natural range {NATURAL_BLINK_RATE_RANGE[0]}-{NATURAL_BLINK_RATE_RANGE[1]}; "
                    f"position stability {temporal.get('position_stability')})."
                ),
            })

        reasons.sort(key=lambda r: r["score"], reverse=True)

        fired_families = []
        if global_score >= GLOBAL_FIRE_THRESHOLD:
            fired_families.append("fully_ai_generated")
        if splice_score >= SPLICE_FIRE_THRESHOLD:
            fired_families.append("ai_edited_region")
        if face_swap_score >= FACE_SWAP_FIRE_THRESHOLD:
            fired_families.append("face_swap")
        if temporal.get("available") and temporal_score >= TEMPORAL_FIRE_THRESHOLD:
            fired_families.append("temporal_inconsistency")

        overall_score = max(global_score, splice_score, face_swap_score, temporal_score)

        if fired_families:
            label = "FAKE"
            confidence = 50.0 + min(overall_score, 1.0) * 50.0
        elif not per_frame_results and not temporal.get("available"):
            label = "UNCERTAIN"
            confidence = 0.0
        else:
            label = "AUTHENTIC"
            confidence = 50.0 + (1.0 - overall_score) * 50.0

        if not reasons:
            reasons = [{
                "signal": "none", "score": 0.0, "region": None, "frame_index": None,
                "description": "No manipulation signal crossed its detection threshold in any sampled frame."
            }]

        notes = []
        if self._image_detector is None:
            notes.append("Per-frame image forensics unavailable; relying on temporal analysis only.")
        if not temporal.get("available"):
            notes.append(f"Temporal analysis unavailable: {temporal.get('reason', 'unknown')}.")

        verdict = VideoVerdict(
            label=label,
            fake_type=fired_families,
            confidence=confidence,
            reasons=reasons,
            family_scores={
                "fully_ai_generated": round(global_score, 3),
                "ai_edited_region": round(splice_score, 3),
                "face_swap": round(face_swap_score, 3),
                "temporal_inconsistency": round(temporal_score, 3),
                "temporal_details": temporal,
            },
            frames_analyzed=len(frames),
            notes=" ".join(notes) if notes else
                  f"Full forensics run on {len(per_frame_results)} sampled frames; "
                  f"temporal analysis across {len(frames)} frames.",
        )
        return verdict.to_dict()


final_video_detector = FinalVideoDetector()


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python final_video_detector.py <video_path>")
        sys.exit(1)
    print(json.dumps(final_video_detector.predict(sys.argv[1]), indent=2))
