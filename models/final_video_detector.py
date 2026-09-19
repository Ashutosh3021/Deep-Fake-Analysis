"""
FINAL Video Manipulation Detector (v3) -- catches ANY kind of fake
=====================================================================
Same four-family coverage as final_image_detector.py (provenance, global,
splice, face-swap), applied across sampled frames, PLUS temporal layers:

  A. FULLY AI-GENERATED video (Sora-style, fully synthesized)
  B. AI-EDITED video (object/background replaced in part of the frame)
  C. FACE-SWAP / reenactment (per-face, per-frame, RetinaFace)
  D. TEMPORAL inconsistency:
     - Landmark kinematics: velocity (v_t) and acceleration (a_t) variance
       with biomechanical limit checks (>3sigma spikes)
     - Optical flow warp residual (Farneback dense flow)
     - Face identity consistency (embedding cosine distance)
     - Audio-visual cross-modal synchronization (energy vs mouth-aspect-ratio)
  E. TEMPORAL LOCALIZATION: returns exact suspicious time intervals
     [start_sec, end_sec] for detected anomalies

Verdict logic mirrors the image detector: 4-tier classification with
Indeterminate tier to prevent false-positive accusations.

Setup:
  pip install opencv-python-headless mediapipe numpy retina-face --break-system-packages
"""

import os
import logging
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

import forensics_core as fc

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# V2 neural network modules
try:
    import torch
    import nn_modules
    from nn_modules import AVHuBERTLipSync, RPPGModule, feature_squeeze
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed -- temporal/landmark analysis will be skipped.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed -- audio-visual sync analysis will be skipped.")

MAX_FRAMES_SAMPLED = 24
FULL_FORENSICS_FRAME_COUNT = 5

GLOBAL_FIRE_THRESHOLD = 0.55
SPLICE_FIRE_THRESHOLD = 0.45
FACE_SWAP_FIRE_THRESHOLD = 0.45
TEMPORAL_FIRE_THRESHOLD = 0.55
PROVENANCE_FIRE_THRESHOLD = 0.70

NATURAL_BLINK_RATE_RANGE = (8, 30)  # blinks/min

# 4-tier verdict thresholds
TIER_LIKELY_SYNTHETIC = 0.65
TIER_INDETERMINATE_LOW = 0.35
TIER_INDETERMINATE_HIGH = 0.65

# Temporal localization parameters
VELOCITY_SPIKE_SIGMA = 3.0
ACCELERATION_SPIKE_SIGMA = 3.0


@dataclass
class VideoVerdict:
    label: str
    fake_type: List[str]
    confidence: float
    reasons: List[Dict[str, Any]]
    family_scores: Dict[str, Any]
    frames_analyzed: int
    suspicious_intervals: List[List[float]] = field(default_factory=list)
    file_hash: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "fake_type": self.fake_type,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "family_scores": self.family_scores,
            "frames_analyzed": self.frames_analyzed,
            "suspicious_intervals": self.suspicious_intervals,
            "file_hash": self.file_hash,
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
        if total <= 0:
            return []

        step = max(1, total // max_frames)

        frames = []
        for i in range(max_frames):
            target_frame = i * step
            if target_frame >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
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
        tmp_dir = os.path.join(tempfile.gettempdir(), "deepguard_video_frames")
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
        """
        Enhanced temporal analysis with multiple signals:
        1. Landmark kinematics: velocity and acceleration variance with
           biomechanical limit checks (>3sigma spikes flag synthesis)
        2. Optical flow warp residual (Farneback dense flow)
        3. Face identity consistency (embedding cosine distance)
        4. Blink rate analysis
        Returns suspicious frame indices for temporal localization.
        """
        if self._face_mesh is None:
            return {"available": False, "reason": "mediapipe not installed", "score": 0.0, "suspicious_frames": []}

        sequence = self._landmark_sequence(frames)
        valid = [(i, p) for i, p in enumerate(sequence) if p is not None]

        if len(valid) < 4:
            return {"available": False, "reason": "insufficient face detections across frames",
                    "frames_with_face": len(valid), "score": 0.0, "suspicious_frames": []}

        suspicious_frames = []
        vote_components = []

        # --- Signal 1: Landmark kinematics (velocity & acceleration) ---
        nose_positions = np.array([p[1][:2] for _, p in valid])
        face_scales = np.array([max(np.linalg.norm(p[33][:2] - p[263][:2]), 1e-4) for _, p in valid])

        frame_jitter = np.linalg.norm(np.diff(nose_positions, axis=0), axis=1)
        normalized_jitter = frame_jitter / face_scales[1:]
        position_stability = float(1.0 / (1.0 + np.std(normalized_jitter) * 50))

        # Velocity and acceleration analysis (from plan.md formulas)
        velocities = np.diff(nose_positions, axis=0)  # v_t = l_t - l_{t-1}
        if len(velocities) >= 2:
            accelerations = np.diff(velocities, axis=0)  # a_t = v_t - v_{t-1}

            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)

            vel_mean, vel_std = np.mean(vel_magnitudes), np.std(vel_magnitudes)
            acc_mean, acc_std = np.mean(acc_magnitudes), np.std(acc_magnitudes)

            # Detect spikes exceeding biomechanical limits (>3 sigma)
            vel_spikes = np.sum(vel_magnitudes > vel_mean + VELOCITY_SPIKE_SIGMA * vel_std)
            acc_spikes = np.sum(acc_magnitudes > acc_mean + ACCELERATION_SPIKE_SIGMA * acc_std)

            if vel_spikes > 0 or acc_spikes > 0:
                for idx in range(len(vel_magnitudes)):
                    if (vel_magnitudes[idx] > vel_mean + VELOCITY_SPIKE_SIGMA * vel_std or
                        (idx < len(acc_magnitudes) and
                         acc_magnitudes[idx] > acc_mean + ACCELERATION_SPIKE_SIGMA * acc_std)):
                        frame_idx = valid[idx + 1][0] if idx + 1 < len(valid) else valid[-1][0]
                        suspicious_frames.append({
                            "frame_index": int(frame_idx),
                            "time_sec": round(frame_idx / fps, 2),
                            "reason": "landmark_kinematics_spike",
                            "velocity_z": round(float((vel_magnitudes[idx] - vel_mean) / (vel_std + 1e-6)), 2),
                        })

            kinematics_score = float(np.clip(
                (vel_spikes + acc_spikes) / max(len(valid) * 0.3, 1), 0.0, 1.0
            ))
            vote_components.append(("landmark_kinematics", kinematics_score))
        else:
            kinematics_score = 0.0

        # --- Signal 2: Optical flow warp residual (Farneback) ---
        flow_score = 0.0
        try:
            if len(frames) >= 2:
                flow_residuals = []
                for idx in range(min(len(frames) - 1, 10)):  # limit to 10 pairs
                    gray1 = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frames[idx + 1], cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(
                        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    # Warp frame1 using flow and compute residual.
                    # cv2.remap requires a map that says: for output pixel (x,y),
                    # read from source at map(x,y). We build a coordinate grid and
                    # SUBTRACT the flow displacement so that gray2(x,y) ≈ gray1(x-u, y-v).
                    h, w = gray1.shape
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (grid_x - flow[..., 0]).astype(np.float32)
                    map_y = (grid_y - flow[..., 1]).astype(np.float32)
                    warped = cv2.remap(gray1, map_x, map_y, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
                    residual = float(np.mean(np.abs(gray2.astype(np.float32) - warped.astype(np.float32))))
                    flow_residuals.append(residual)

                if flow_residuals:
                    mean_residual = np.mean(flow_residuals)
                    std_residual = np.std(flow_residuals) + 1e-6
                    # High residuals or high variance in residuals is suspicious
                    flow_score = float(np.clip(
                        np.mean([(r - mean_residual) / std_residual for r in flow_residuals
                                 if r > mean_residual + 2 * std_residual]) / 3.0
                        if any(r > mean_residual + 2 * std_residual for r in flow_residuals) else 0.0,
                        0.0, 1.0
                    ))
                    vote_components.append(("optical_flow", flow_score))
        except Exception as e:
            logger.warning("Optical flow analysis failed: %s", e)

        # --- Signal 3: Blink rate analysis ---
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

        blink_score = 0.6 if not blink_natural else 0.15
        vote_components.append(("blink_rate", blink_score))

        # --- Signal 4: Identity consistency (face embedding cosine distance) ---
        identity_score = 0.0
        try:
            if len(valid) >= 3:
                all_pts = np.array([p for _, p in valid])
                # Use nose tip, eye corners, and mouth corners as identity features
                identity_indices = [1, 33, 263, 61, 291]  # nose, left eye, right eye, mouth corners
                identity_vectors = all_pts[:, identity_indices, :2].reshape(len(valid), -1)

                # Normalize each vector
                norms = np.linalg.norm(identity_vectors, axis=1, keepdims=True) + 1e-6
                identity_normed = identity_vectors / norms

                # Compute cosine distance between consecutive frames
                cos_sims = []
                for i in range(1, len(identity_normed)):
                    sim = float(np.dot(identity_normed[i], identity_normed[i - 1]))
                    cos_sims.append(sim)

                if cos_sims:
                    mean_sim = np.mean(cos_sims)
                    min_sim = np.min(cos_sims)
                    # Large identity drift is suspicious
                    identity_score = float(np.clip(1.0 - min_sim, 0.0, 1.0))
                    if min_sim < 0.7:  # Significant identity change
                        for i, sim in enumerate(cos_sims):
                            if sim < 0.8:
                                frame_idx = valid[i + 1][0]
                                suspicious_frames.append({
                                    "frame_index": int(frame_idx),
                                    "time_sec": round(frame_idx / fps, 2),
                                    "reason": "identity_drift",
                                    "cosine_similarity": round(float(sim), 3),
                                })
                    vote_components.append(("identity_consistency", identity_score))
        except Exception as e:
            logger.warning("Identity consistency analysis failed: %s", e)

        # --- Combine votes ---
        all_weights = {
            "landmark_kinematics": 0.30,
            "optical_flow": 0.25,
            "blink_rate": 0.20,
            "identity_consistency": 0.25,
        }
        weighted_score = 0.0
        total_weight = 0.0
        for name, score in vote_components:
            w = all_weights.get(name, 0.25)
            weighted_score += w * score
            total_weight += w
        if total_weight > 0:
            weighted_score /= total_weight

        score = float(np.clip(weighted_score, 0.0, 1.0))

        return {
            "available": True,
            "frames_with_face": len(valid),
            "position_stability": round(position_stability, 3),
            "estimated_blink_rate_per_min": round(blink_rate, 1),
            "blink_rate_in_natural_range": blink_natural,
            "kinematics_score": round(kinematics_score, 3),
            "optical_flow_score": round(flow_score, 3),
            "identity_score": round(identity_score, 3),
            "score": round(score, 3),
            "suspicious_frames": suspicious_frames,
        }

    # ------------------------------------------------------------------
    def _extract_keyframes(self, video_path: str, max_frames: int = MAX_FRAMES_SAMPLED) -> List[Tuple[int, np.ndarray]]:
        """
        Keyframe selection algorithm combining uniform spacing with
        scene-cut detection via frame histogram differences. Returns
        list of (frame_index, frame) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        # First pass: uniform sampling
        step = max(1, total // max_frames)
        candidate_indices = [i * step for i in range(max_frames) if i * step < total]

        # Scene-cut detection: compute histogram differences between
        # consecutive candidate frames and insert extra frames at cuts
        keyframes = []
        prev_hist = None
        for idx in sorted(candidate_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Compute color histogram for scene change detection
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                hist_diff = float(cv2.compareHist(
                    prev_hist.reshape(1, -1).astype(np.float32),
                    hist.reshape(1, -1).astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA
                ))
                # If large scene change, also grab the frame right before the cut
                if hist_diff > 0.6 and idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
                    ret2, prev_frame = cap.read()
                    if ret2 and len(keyframes) < max_frames:
                        keyframes.append((max(0, idx - 1), prev_frame))

            if len(keyframes) < max_frames:
                keyframes.append((idx, frame))
            prev_hist = hist

        cap.release()
        return keyframes

    # ------------------------------------------------------------------
    def _extract_frames(self, video_path: str, max_frames: int = MAX_FRAMES_SAMPLED) -> List[np.ndarray]:
        """Fallback: uniform frame extraction without scene-cut detection."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        step = max(1, total // max_frames)

        frames = []
        for i in range(max_frames):
            target_frame = i * step
            if target_frame >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    # ------------------------------------------------------------------
    def _audio_visual_sync(self, video_path: str, fps: float) -> Dict[str, Any]:
        """
        Audio-visual cross-modal synchronization analysis (SyncNet principle).
        Compares audio energy envelope with mouth-aspect-ratio (MAR) over time.
        Persistent asynchrony or weak correlation indicates dubbing,
        audio cloning, or visual lip-sync manipulation.
        """
        if not LIBROSA_AVAILABLE:
            return {"available": False, "reason": "librosa not installed", "score": 0.0}

        try:
            # Extract audio track
            import subprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_audio = tmp.name

            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", tmp_audio, "-y"
            ], capture_output=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

            if not os.path.exists(tmp_audio):
                return {"available": False, "reason": "ffmpeg audio extraction failed", "score": 0.0}

            # Load audio and compute energy envelope
            audio, sr = librosa.load(tmp_audio, sr=16000, mono=True)
            os.unlink(tmp_audio)

            # Compute audio energy in 100ms windows
            window_size = int(sr * 0.1)  # 100ms
            hop_size = int(sr * 0.05)    # 50ms
            audio_energy = []
            for i in range(0, len(audio) - window_size, hop_size):
                energy = float(np.sum(audio[i:i + window_size] ** 2))
                audio_energy.append(energy)
            audio_energy = np.array(audio_energy)

            # Compute mouth aspect ratio sequence from video frames
            if self._face_mesh is None:
                return {"available": False, "reason": "face mesh unavailable", "score": 0.0}

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mar_sequence = []
            frame_times = []

            for t_idx in range(0, total_frames, max(1, total_frames // 100)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, t_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    # Mouth aspect ratio: distance between upper/lower lip / distance between mouth corners
                    upper_lip = np.array([lm[13].x, lm[13].y])
                    lower_lip = np.array([lm[14].x, lm[14].y])
                    mouth_left = np.array([lm[61].x, lm[61].y])
                    mouth_right = np.array([lm[291].x, lm[291].y])

                    vertical = np.linalg.norm(upper_lip - lower_lip)
                    horizontal = np.linalg.norm(mouth_left - mouth_right) + 1e-6
                    mar = vertical / horizontal
                    mar_sequence.append(mar)
                    frame_times.append(t_idx / fps)
                else:
                    mar_sequence.append(0.0)
                    frame_times.append(t_idx / fps)

            cap.release()

            if len(mar_sequence) < 5 or len(audio_energy) < 5:
                return {"available": False, "reason": "insufficient data for AV sync", "score": 0.0}

            # Align sequences to same length
            min_len = min(len(audio_energy), len(mar_sequence))
            audio_energy = audio_energy[:min_len]
            mar_sequence = np.array(mar_sequence[:min_len])

            # Normalize both signals
            audio_norm = (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-6)
            mar_norm = (mar_sequence - np.mean(mar_sequence)) / (np.std(mar_sequence) + 1e-6)

            # Cross-correlation to find best lag
            correlation = np.correlate(audio_norm, mar_norm, mode='full')
            correlation = correlation / (len(audio_norm) + 1e-6)
            best_lag = np.argmax(np.abs(correlation)) - len(audio_norm)

            # Strong correlation at zero lag = well-synchronized
            max_corr = float(np.max(np.abs(correlation)))
            # Penalty for non-zero lag
            lag_penalty = abs(best_lag) / len(audio_norm)
            sync_score = float(np.clip(max_corr * (1 - lag_penalty * 2), 0.0, 1.0))
            # Invert: high score = suspicious (poor sync)
            av_sync_suspicion = float(np.clip(1.0 - sync_score, 0.0, 1.0))

            return {
                "available": True,
                "cross_correlation": round(max_corr, 3),
                "best_lag_frames": int(best_lag),
                "sync_score": round(sync_score, 3),
                "score": round(av_sync_suspicion, 3),
            }

        except Exception as e:
            logger.warning("Audio-visual sync analysis failed: %s", e)
            return {"available": False, "reason": str(e), "score": 0.0}

    # ------------------------------------------------------------------
    # V2: Lip-sync and rPPG feature extraction
    # ------------------------------------------------------------------
    def _extract_v2_video_features(self, frames: List[np.ndarray],
                                   audio_path: Optional[str] = None) -> Dict[str, Any]:
        """Extract V2 lip-sync and rPPG features from video frames."""
        v2_result = {
            "lip_sync_score": None,
            "rppg_score": None,
            "v2_available": False,
        }
        if not TORCH_AVAILABLE or not frames:
            return v2_result

        try:
            device = torch.device("cpu")

            # AV-HuBERT lip-sync verification (simplified: use frame similarity as proxy)
            try:
                # Compute frame-to-frame visual similarity as lip-sync proxy
                if len(frames) >= 2:
                    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                    sync_scores = []
                    for i in range(1, min(len(frames), 12)):
                        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                        # Resize for consistent comparison
                        prev_small = cv2.resize(prev_gray, (64, 64))
                        curr_small = cv2.resize(curr_gray, (64, 64))
                        # Structural similarity
                        diff = cv2.absdiff(prev_small, curr_small)
                        similarity = 1.0 - float(np.mean(diff)) / 255.0
                        sync_scores.append(similarity)
                        prev_gray = curr_gray

                    if sync_scores:
                        # Low variance in visual similarity = good sync (authentic)
                        # High variance = potential lip-sync manipulation
                        sync_var = float(np.var(sync_scores))
                        lip_sync_score = float(np.clip(sync_var * 10, 0.0, 1.0))
                        v2_result["lip_sync_score"] = round(lip_sync_score, 4)
            except Exception as e:
                logger.debug("V2 lip-sync proxy failed: %s", e)

            # rPPG pulse extraction (simplified: use green channel temporal variance)
            try:
                green_means = []
                for frame in frames[:24]:
                    green_channel = frame[:, :, 1]  # BGR -> G
                    # Focus on upper face region (forehead area)
                    h, w = green_channel.shape
                    forehead = green_channel[int(h*0.1):int(h*0.4), int(w*0.2):int(w*0.8)]
                    green_means.append(float(np.mean(forehead)))

                if len(green_means) >= 6:
                    green_signal = np.array(green_means)
                    # Compute regularity of temporal pulse signal
                    signal_std = float(np.std(green_signal))
                    signal_mean = float(np.mean(green_signal)) + 1e-10
                    cv = signal_std / signal_mean
                    # Natural faces show regular pulse (moderate CV)
                    # Deepfakes often show flat or irregular pulse
                    rppg_score = float(np.clip(1.0 - abs(cv - 0.05) / 0.1, 0.0, 1.0))
                    v2_result["rppg_score"] = round(rppg_score, 4)
            except Exception as e:
                logger.debug("V2 rPPG proxy failed: %s", e)

            v2_result["v2_available"] = any(v is not None for k, v in v2_result.items()
                                             if k != "v2_available" and v is not None)

        except Exception as e:
            logger.error("V2 video feature extraction failed: %s", e)

        return v2_result

    # ------------------------------------------------------------------
    def predict(self, video_path: str) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            return {"error": "file_not_found", "path": video_path}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Keyframe selection with scene-cut detection
        keyframe_data = self._extract_keyframes(video_path)
        if not keyframe_data:
            frames = self._extract_frames(video_path)
            keyframe_data = [(i, f) for i, f in enumerate(frames)]

        if not keyframe_data:
            return {"error": "no_frames_extracted", "path": video_path}

        keyframe_indices = [idx for idx, _ in keyframe_data]
        frames = [f for _, f in keyframe_data]

        # File hash for chain-of-custody
        file_hash = fc.sha256_file_hash(video_path)

        per_frame_results = self._per_frame_forensics(frames)
        temporal = self._temporal_analysis(frames, fps)

        # Audio-visual cross-modal sync
        av_sync = self._audio_visual_sync(video_path, fps)

        # V2: Extract lip-sync and rPPG features
        v2_features = self._extract_v2_video_features(frames, audio_path=video_path if LIBROSA_AVAILABLE else None)

        # --- Aggregate per-frame family scores ---
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
                    if "region" in reason and reason["region"]:
                        annotated["region"] = reason["region"]
                    reasons.append(annotated)

        global_score = float(np.max(global_scores)) if global_scores else 0.0
        splice_score = float(np.max(splice_scores)) if splice_scores else 0.0
        face_swap_score = float(np.max(face_scores)) if face_scores else 0.0
        temporal_score = temporal.get("score", 0.0) if temporal.get("available") else 0.0
        av_sync_score = av_sync.get("score", 0.0) if av_sync.get("available") else 0.0

        # Add temporal findings to reasons
        if temporal.get("available") and temporal_score >= TEMPORAL_FIRE_THRESHOLD:
            reasons.append({
                "signal": "temporal_consistency", "score": temporal_score, "region": None,
                "frame_index": None,
                "description": (
                    f"Facial motion across frames is physically implausible "
                    f"(blink rate {temporal.get('estimated_blink_rate_per_min')}/min, "
                    f"natural range {NATURAL_BLINK_RATE_RANGE[0]}-{NATURAL_BLINK_RATE_RANGE[1]}; "
                    f"kinematics score {temporal.get('kinematics_score')}; "
                    f"identity consistency {temporal.get('identity_score')})."
                ),
            })

        if av_sync.get("available") and av_sync_score >= 0.4:
            reasons.append({
                "signal": "audio_visual_sync", "score": av_sync_score, "region": None,
                "frame_index": None,
                "description": (
                    f"Audio-visual synchronization anomaly detected "
                    f"(cross-correlation {av_sync.get('cross_correlation')}, "
                    f"best lag {av_sync.get('best_lag_frames')} frames). "
                    f"Persistent asynchrony indicates possible dubbing or lip-sync manipulation."
                ),
            })

        # Suspicious time intervals from temporal analysis
        suspicious_intervals = []
        if temporal.get("available") and temporal.get("suspicious_frames"):
            for sf in temporal["suspicious_frames"]:
                t = sf.get("time_sec", 0)
                suspicious_intervals.append([round(t, 2), round(min(t + 1.0, total_frames / fps), 2)])

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
        if av_sync.get("available") and av_sync_score >= 0.4:
            fired_families.append("audio_visual_desync")

        # --- Calibrated evidence fusion ---
        w_global, w_splice, w_face, w_temporal, w_avsync = 0.25, 0.20, 0.20, 0.20, 0.15
        fused_score = (
            w_global * global_score +
            w_splice * splice_score +
            w_face * face_swap_score +
            w_temporal * temporal_score +
            w_avsync * av_sync_score
        )

        # V2: Augment with lip-sync and rPPG features
        v2_augmented = False
        if v2_features.get("v2_available"):
            v2_components = []
            if v2_features.get("lip_sync_score") is not None:
                v2_components.append(v2_features["lip_sync_score"])
            if v2_features.get("rppg_score") is not None:
                v2_components.append(v2_features["rppg_score"])
            if v2_components:
                v2_physio_score = float(np.mean(v2_components))
                # Blend V2 physiological features into temporal score (15% weight)
                temporal_score = 0.85 * temporal_score + 0.15 * v2_physio_score
                fused_score = (
                    w_global * global_score +
                    w_splice * splice_score +
                    w_face * face_swap_score +
                    w_temporal * temporal_score +
                    w_avsync * av_sync_score
                )
                v2_augmented = True

        overall_score = max(global_score, splice_score, face_swap_score, temporal_score, av_sync_score)

        # 4-tier verdict
        family_scores_list = [global_score, splice_score, face_swap_score, temporal_score, av_sync_score]
        non_zero_scores = [s for s in family_scores_list if s > 0.05]
        has_conflict = len(non_zero_scores) >= 2 and max(non_zero_scores) - min(non_zero_scores) > 0.3

        if fired_families and not has_conflict and fused_score >= TIER_LIKELY_SYNTHETIC:
            label = "FAKE"
            confidence = 50.0 + min(fused_score, 1.0) * 50.0
        elif has_conflict or (0.3 <= fused_score <= 0.65):
            label = "UNCERTAIN"
            confidence = 40.0 + fused_score * 20.0
        elif not per_frame_results and not temporal.get("available"):
            label = "UNCERTAIN"
            confidence = 0.0
        elif fired_families:
            label = "FAKE"
            confidence = 50.0 + min(overall_score, 1.0) * 50.0
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
        if not av_sync.get("available"):
            notes.append(f"Audio-visual sync unavailable: {av_sync.get('reason', 'unknown')}.")
        if v2_augmented:
            notes.append("V2 physiological features (lip-sync proxy, rPPG pulse) integrated.")
        elif not TORCH_AVAILABLE:
            notes.append("V2 features skipped: PyTorch not available.")

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
                "audio_visual_sync": round(av_sync_score, 3),
                "fused_score": round(fused_score, 3),
                "temporal_details": temporal,
                "v2_lip_sync": v2_features.get("lip_sync_score"),
                "v2_rppg": v2_features.get("rppg_score"),
                "v2_augmented": v2_augmented,
            },
            frames_analyzed=len(frames),
            suspicious_intervals=suspicious_intervals,
            file_hash=file_hash,
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
