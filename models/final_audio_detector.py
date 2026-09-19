"""
FINAL Audio Deepfake / Synthetic Speech Detector (v3)
======================================================
Design philosophy: enhanced with physically-grounded acoustic features from
the forensic literature:

  STRATEGY A (default, zero training required):
      Pretrained spoof-detection model from Hugging Face (wav2vec2-based).

  STRATEGY B (better, requires training):
      Classifier on top of enhanced acoustic features including:
      - Delta & Delta-Delta MFCC (dynamic transitions)
      - Spectral centroid, flatness, high-frequency rolloff
      - F0 pitch tracking with pitch-jump anomaly detection
      - Harmonic-to-Noise Ratio (HNR) and phase continuity
      - Glottal flow characteristics

  STRATEGY C (quality-weighted segment aggregation):
      Splits audio into overlapping windows, computes per-segment scores
      weighted by segment SNR/clarity, returns suspicious time-slices.

Enhanced features implement the mathematical formulas from the plan:
- Delta MFCC: Delta X(t,n) = sum_r r*(X(t+r,n) - X(t-r,n)) / (2*sum_r r^2)
- Spectral Centroid: C_t = sum_f f|X(t,f)| / sum_f |X(t,f)|
- Spectral Flatness: F_t = exp(1/N sum_f ln P_t(f)) / (1/N sum_f P_t(f))
- HNR: HNR = 10*log10(P_harmonic / P_noise)
- Quality-weighted: p_audio = sum_i q_i*p_i / sum_i q_i

Setup:
  pip install librosa numpy scipy scikit-learn transformers torch --break-system-packages
"""

import os
import logging
import pickle
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import librosa
import soundfile as sf
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# V2 neural network modules
try:
    import torch
    import nn_modules
    from nn_modules import AMFF, NeXtTDNN, AASIST2GraphAttention, feature_squeeze
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000
SEGMENT_DURATION_SEC = 2.0  # analysis window length
SEGMENT_HOP_SEC = 1.0       # hop between windows (50% overlap)

PRETRAINED_MODEL_ID = os.getenv(
    "AUDIO_DETECTOR_MODEL", "MelodyMachine/Deepfake-audio-detection-V2"
)
AUDIO_CLASSIFIER_PATH = os.getenv("AUDIO_CLASSIFIER_PATH", "models/audio_rf_classifier.pkl")

LOW_CONFIDENCE_THRESHOLD = 0.60


@dataclass
class AudioVerdict:
    label: str                 # "SYNTHETIC" | "AUTHENTIC" | "UNCERTAIN"
    confidence: float
    fake_probability: float
    signal_source: str
    feature_summary: Dict[str, Any] = field(default_factory=dict)
    suspicious_segments: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "fake_probability": round(self.fake_probability, 4),
            "signal_source": self.signal_source,
            "feature_summary": self.feature_summary,
            "suspicious_segments": self.suspicious_segments,
            "notes": self.notes,
        }


class FinalAudioDetector:
    def __init__(self):
        self._hf_pipeline = None
        self._hf_load_error: Optional[str] = None
        self._trained_classifier = None
        self._classifier_load_error: Optional[str] = None

        self._try_load_trained_classifier()
        self._try_load_pretrained_model()

    # ------------------------------------------------------------------
    def _try_load_trained_classifier(self):
        if os.path.exists(AUDIO_CLASSIFIER_PATH):
            try:
                with open(AUDIO_CLASSIFIER_PATH, "rb") as f:
                    self._trained_classifier = pickle.load(f)
                logger.info("Loaded trained audio classifier from %s", AUDIO_CLASSIFIER_PATH)
            except Exception as e:
                self._classifier_load_error = str(e)
                logger.warning("Failed to load trained classifier: %s", e)

    def _try_load_pretrained_model(self):
        try:
            from transformers import pipeline
            self._hf_pipeline = pipeline("audio-classification", model=PRETRAINED_MODEL_ID)
            logger.info("Loaded pretrained audio spoof-detection model: %s", PRETRAINED_MODEL_ID)
        except Exception as e:
            self._hf_load_error = str(e)
            logger.warning(
                "Could not load pretrained HF audio model (%s). "
                "Will fall back to acoustic-heuristic mode.", e
            )

    # ------------------------------------------------------------------
    # Audio preprocessing: standardize to 16kHz mono, silence trimming
    # ------------------------------------------------------------------
    def _preprocess_audio(self, audio_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Standardize audio: 16kHz mono conversion, silence trimming via
        RMS thresholding (from plan.md).
        """
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            if len(audio) < sr * 0.3:
                return None

            # Silence trimming via RMS thresholding
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            rms_threshold = np.mean(rms) * 0.1
            non_silent = rms > rms_threshold

            if np.sum(non_silent) < 10:
                return audio, sr  # mostly silent, return as-is

            frames = np.arange(len(audio))
            frame_to_sample = librosa.frames_to_samples(np.arange(len(rms)), hop_length=512)
            non_silent_samples = np.zeros(len(audio), dtype=bool)
            for i, is_active in enumerate(non_silent):
                if is_active and frame_to_sample[i] < len(audio):
                    end = min(frame_to_sample[i] + 512, len(audio))
                    non_silent_samples[frame_to_sample[i]:end] = True

            if np.sum(non_silent_samples) < sr * 0.2:
                return audio, sr
            return audio[non_silent_samples], sr
        except Exception as e:
            logger.error("Audio preprocessing failed for %s: %s", audio_path, e)
            return None

    # ------------------------------------------------------------------
    # Enhanced acoustic feature extraction
    # ------------------------------------------------------------------
    def extract_training_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Enhanced feature extraction implementing plan.md formulas:
        - Delta & Delta-Delta MFCC
        - Spectral centroid, flatness, rolloff
        - F0 pitch tracking with jitter
        - Harmonic-to-Noise Ratio (HNR)
        - Phase continuity
        """
        try:
            result = self._preprocess_audio(audio_path)
            if result is None:
                return None
            audio, sr = result

            feats: List[float] = []

            # --- Static MFCC (13 coefficients) ---
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            feats.extend(np.mean(mfccs, axis=1).tolist())
            feats.extend(np.std(mfccs, axis=1).tolist())

            # --- Delta MFCC (first derivative) ---
            delta_mfccs = librosa.feature.delta(mfccs, order=1)
            feats.extend(np.mean(delta_mfccs, axis=1).tolist())
            feats.extend(np.std(delta_mfccs, axis=1).tolist())

            # --- Delta-Delta MFCC (second derivative) ---
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            feats.extend(np.mean(delta2_mfccs, axis=1).tolist())
            feats.extend(np.std(delta2_mfccs, axis=1).tolist())

            # --- Spectral centroid (brightness) ---
            # C_t = sum_f f|X(t,f)| / sum_f |X(t,f)|
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            feats.append(float(np.mean(spectral_centroid)))
            feats.append(float(np.std(spectral_centroid)))

            # --- Spectral flatness (Wiener entropy) ---
            # F_t = exp(1/N sum_f ln P_t(f)) / (1/N sum_f P_t(f))
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            feats.append(float(np.mean(spectral_flatness)))
            feats.append(float(np.std(spectral_flatness)))

            # --- Spectral rolloff (high-frequency energy) ---
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
            feats.append(float(np.mean(rolloff)))

            # --- High-frequency void detection (above 8kHz) ---
            if sr >= 16000:
                n_fft = 2048
                stft = np.abs(librosa.stft(audio, n_fft=n_fft))
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                hf_idx = np.searchsorted(freqs, 8000)
                if hf_idx < stft.shape[0]:
                    hf_energy = np.mean(stft[hf_idx:])
                    total_energy = np.mean(stft) + 1e-10
                    hf_ratio = hf_energy / total_energy
                    # Neural vocoders often exhibit degraded HF energy
                    hf_void_score = float(np.clip(1.0 - hf_ratio * 10, 0.0, 1.0))
                else:
                    hf_void_score = 0.0
            else:
                hf_void_score = 0.0
            feats.append(hf_void_score)

            # --- Harmonic-to-Noise Ratio (HNR) ---
            # HNR = 10*log10(P_harmonic / P_noise)
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_power = float(np.sum(harmonic ** 2) + 1e-10)
            noise_power = float(np.sum(percussive ** 2) + 1e-10)
            hnr = 10.0 * np.log10(harmonic_power / noise_power)
            feats.append(float(np.clip(hnr / 30.0, -1.0, 1.0)))  # normalize

            # Harmonic ratio
            harmonic_ratio = harmonic_power / (np.sum(audio ** 2) + 1e-10)
            feats.append(harmonic_ratio)

            # --- F0 pitch tracking with jitter ---
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sr
            )
            if f0 is not None and np.any(voiced_flag):
                f0_voiced = f0[voiced_flag]
                f0_mean = float(np.nanmean(f0_voiced))
                f0_std = float(np.nanstd(f0_voiced))
                if len(f0_voiced) > 1 and f0_mean > 0:
                    # Jitter = (1/(N-1)) * sum|T_i - T_{i+1}| / mean(T)
                    periods = 1.0 / f0_voiced
                    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
                    # Pitch jump detection: large instantaneous pitch changes
                    pitch_diffs = np.abs(np.diff(f0_voiced))
                    pitch_jumps = np.sum(pitch_diffs > 50)  # >50Hz jump
                    pitch_jump_score = float(np.clip(pitch_jumps / max(len(f0_voiced) * 0.1, 1), 0.0, 1.0))
                else:
                    f0_mean, f0_std, jitter, pitch_jump_score = 0.0, 0.0, 0.0, 0.0
            else:
                f0_mean, f0_std, jitter, pitch_jump_score = 0.0, 0.0, 0.0, 0.0
            feats.extend([f0_mean / 500.0, f0_std / 200.0, jitter, pitch_jump_score])

            # --- Phase coherence (vocoder artifacts) ---
            stft_complex = librosa.stft(audio)
            phase = np.angle(stft_complex)
            phase_diff = np.diff(phase, axis=1)
            phase_discontinuity = float(np.mean(np.abs(phase_diff) > np.pi / 4))
            feats.append(phase_discontinuity)

            # --- Distributional shape ---
            feats.append(float(skew(audio)))
            feats.append(float(kurtosis(audio)))

            return np.array(feats, dtype=np.float32)
        except Exception as e:
            logger.error("Feature extraction failed for %s: %s", audio_path, e)
            return None

    # ------------------------------------------------------------------
    # Quality-weighted segment aggregation
    # ------------------------------------------------------------------
    def _segment_scores(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Split audio into overlapping windows, compute per-segment fake
        probability weighted by segment SNR/clarity (from plan.md):
        p_audio = sum_i q_i * p_i / sum_i q_i
        Returns list of segment results with time-slice info.
        """
        try:
            result = self._preprocess_audio(audio_path)
            if result is None:
                return []
            audio, sr = result

            segment_samples = int(SEGMENT_DURATION_SEC * sr)
            hop_samples = int(SEGMENT_HOP_SEC * sr)

            if len(audio) < segment_samples:
                return []

            segments = []
            for start in range(0, len(audio) - segment_samples + 1, hop_samples):
                segment = audio[start:start + segment_samples]
                start_sec = start / sr
                end_sec = (start + segment_samples) / sr

                # Quality factor: segment SNR (higher = more reliable)
                rms = float(np.sqrt(np.mean(segment ** 2)))
                noise_est = float(np.sqrt(np.mean(segment[:sr // 4] ** 2))) + 1e-10
                snr = rms / noise_est
                quality = float(np.clip(snr / 10.0, 0.1, 1.0))

                # Per-segment fake score
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    import soundfile as sf
                    sf.write(tmp.name, segment, sr)
                    score = self._single_predict(tmp.name)
                    os.unlink(tmp.name)

                if score is not None:
                    segments.append({
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(end_sec, 2),
                        "fake_score": round(float(np.clip(score, 0.0, 1.0)), 4),
                        "quality": round(quality, 3),
                    })

            return segments
        except Exception as e:
            logger.error("Segment scoring failed for %s: %s", audio_path, e)
            return []

    def _single_predict(self, audio_path: str) -> Optional[float]:
        """Single prediction on a full audio file (not segmented)."""
        score = self._trained_classifier_predict(audio_path)
        if score is not None:
            return score
        score = self._hf_predict(audio_path)
        if score is not None:
            return score
        return self._heuristic_predict(audio_path)

    # ------------------------------------------------------------------
    # Strategy A: pretrained HF model
    # ------------------------------------------------------------------
    def _hf_predict(self, audio_path: str) -> Optional[float]:
        if self._hf_pipeline is None:
            return None
        try:
            results = self._hf_pipeline(audio_path)
            fake_score = 0.0
            for r in results:
                label = r["label"].lower()
                if any(k in label for k in ("fake", "spoof", "synthetic", "generated")):
                    fake_score = max(fake_score, r["score"])
                elif any(k in label for k in ("real", "bonafide", "genuine", "human")):
                    fake_score = max(fake_score, 1.0 - r["score"])
            return float(fake_score)
        except Exception as e:
            logger.error("HF audio model inference failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Strategy B: trained classifier
    # ------------------------------------------------------------------
    def _trained_classifier_predict(self, audio_path: str) -> Optional[float]:
        if self._trained_classifier is None:
            return None
        feats = self.extract_training_features(audio_path)
        if feats is None:
            return None
        try:
            proba = self._trained_classifier.predict_proba(feats.reshape(1, -1))[0]
            return float(proba[1])
        except Exception as e:
            logger.error("Trained classifier inference failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------
    def _heuristic_predict(self, audio_path: str) -> float:
        result = self._preprocess_audio(audio_path)
        if result is None:
            return 0.5
        audio, sr = result

        feats = self.extract_training_features(audio_path)
        if feats is None:
            return 0.5

        indicators = []
        # Spectral flatness (high -> synthetic-leaning)
        flatness = feats[32]  # index after MFCC + deltas + spectral features
        if flatness > 0:
            indicators.append(np.clip(flatness / 0.05, 0, 1))
        # Low jitter (too-perfect pitch)
        jitter = feats[48] if len(feats) > 48 else 0
        if jitter > 0:
            indicators.append(1.0 - np.clip(jitter / 0.02, 0, 1))
        # Phase discontinuity
        phase_disc = feats[51] if len(feats) > 51 else 0
        indicators.append(np.clip(phase_disc / 0.4, 0, 1))

        return float(np.clip(np.mean(indicators) if indicators else 0.5, 0.0, 1.0))

    # ------------------------------------------------------------------
    # V2: Neural temporal feature extraction (AMFF + NeXt-TDNN + AASIST2)
    # ------------------------------------------------------------------
    def _extract_v2_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract V2 neural temporal features using AMFF, NeXt-TDNN, AASIST2."""
        v2_result = {
            "amff_score": None,
            "tdnn_score": None,
            "aasist2_score": None,
            "v2_available": False,
        }
        if not TORCH_AVAILABLE:
            return v2_result

        try:
            device = torch.device("cpu")
            result = self._preprocess_audio(audio_path)
            if result is None:
                return v2_result
            audio, sr = result

            # Create mel spectrogram features for neural models
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).float().to(device)  # (1, 64, T)

            # AMFF
            try:
                amff = AMFF(in_dim=64, scale_dim=16, num_heads=4).to(device).eval()
                with torch.no_grad():
                    amff_input = mel_tensor.transpose(1, 2)  # (1, T, 64)
                    amff_feat = amff(amff_input)
                amff_score = float(torch.sigmoid(amff_feat.mean()).item())
                v2_result["amff_score"] = round(amff_score, 4)
            except Exception as e:
                logger.debug("AMFF failed: %s", e)

            # NeXt-TDNN
            try:
                tdnn = NeXtTDNN(in_dim=64, hidden_dim=128).to(device).eval()
                with torch.no_grad():
                    tdnn_input = mel_tensor.transpose(1, 2)  # (1, T, 64)
                    tdnn_feat = tdnn(tdnn_input)
                tdnn_score = float(torch.sigmoid(tdnn_feat.mean()).item())
                v2_result["tdnn_score"] = round(tdnn_score, 4)
            except Exception as e:
                logger.debug("NeXt-TDNN failed: %s", e)

            # AASIST2
            try:
                aasist2 = AASIST2GraphAttention(in_features=64, hidden_dim=128, num_heads=4, num_layers=3).to(device).eval()
                with torch.no_grad():
                    aasist2_input = mel_tensor.transpose(1, 2)  # (1, T, 64)
                    aasist2_feat = aasist2(aasist2_input)
                aasist2_score = float(torch.sigmoid(aasist2_feat.mean()).item())
                v2_result["aasist2_score"] = round(aasist2_score, 4)
            except Exception as e:
                logger.debug("AASIST2 failed: %s", e)

            v2_result["v2_available"] = any(v is not None for k, v in v2_result.items()
                                             if k != "v2_available" and v is not None)

        except Exception as e:
            logger.error("V2 audio feature extraction failed: %s", e)

        return v2_result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, audio_path: str) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            return {"error": "file_not_found", "path": audio_path}

        # V2: Extract neural temporal features
        v2_features = self._extract_v2_audio_features(audio_path)

        # Overall prediction
        score = self._trained_classifier_predict(audio_path)
        source = "trained_classifier"

        if score is None:
            score = self._hf_predict(audio_path)
            source = "pretrained_hf_model"

        if score is None:
            score = self._heuristic_predict(audio_path)
            source = "heuristic_fallback"

        # V2: Blend neural temporal features into score
        v2_augmented = False
        if v2_features.get("v2_available"):
            v2_components = []
            if v2_features.get("amff_score") is not None:
                v2_components.append(v2_features["amff_score"])
            if v2_features.get("tdnn_score") is not None:
                v2_components.append(v2_features["tdnn_score"])
            if v2_features.get("aasist2_score") is not None:
                v2_components.append(v2_features["aasist2_score"])
            if v2_components:
                v2_neural_score = float(np.mean(v2_components))
                # Blend V2 neural features (20% weight)
                score = 0.80 * score + 0.20 * v2_neural_score
                v2_augmented = True

        fake_probability = float(np.clip(score, 0.0, 1.0))
        distance_from_mid = abs(fake_probability - 0.5) * 2

        if source == "heuristic_fallback":
            distance_from_mid *= 0.5

        if distance_from_mid < (1 - LOW_CONFIDENCE_THRESHOLD):
            label = "UNCERTAIN"
        elif fake_probability > 0.5:
            label = "SYNTHETIC"
        else:
            label = "AUTHENTIC"

        confidence = 50.0 + distance_from_mid * 50.0

        # Quality-weighted segment analysis for suspicious time-slices
        segments = self._segment_scores(audio_path)
        suspicious_segments = [
            s for s in segments if s["fake_score"] > 0.6
        ]

        # Feature summary
        feature_summary = {"raw_score": round(score, 4)}
        if segments:
            segment_scores = [s["fake_score"] for s in segments]
            segment_qualities = [s["quality"] for s in segments]
            if sum(segment_qualities) > 0:
                weighted_score = sum(s * q for s, q in zip(segment_scores, segment_qualities)) / sum(segment_qualities)
                feature_summary["quality_weighted_score"] = round(float(weighted_score), 4)
            feature_summary["segment_count"] = len(segments)
            feature_summary["suspicious_segment_count"] = len(suspicious_segments)
        # V2 features
        feature_summary["v2_amff"] = v2_features.get("amff_score")
        feature_summary["v2_tdnn"] = v2_features.get("tdnn_score")
        feature_summary["v2_aasist2"] = v2_features.get("aasist2_score")
        feature_summary["v2_augmented"] = v2_augmented

        notes_map = {
            "trained_classifier": "Score from classifier trained on labeled data.",
            "pretrained_hf_model": f"Score from pretrained model ({PRETRAINED_MODEL_ID}).",
            "heuristic_fallback": (
                "No trained classifier or pretrained model available -- running on "
                "unfit acoustic heuristics. Confidence is deliberately suppressed."
            ),
        }
        if v2_augmented:
            notes_map[source] += " V2 neural temporal features (AMFF, NeXt-TDNN, AASIST2) integrated."
        elif not TORCH_AVAILABLE:
            notes_map[source] += " V2 features skipped: PyTorch not available."

        verdict = AudioVerdict(
            label=label,
            confidence=confidence,
            fake_probability=fake_probability,
            signal_source=source,
            feature_summary=feature_summary,
            suspicious_segments=suspicious_segments,
            notes=notes_map[source],
        )
        return verdict.to_dict()


# ------------------------------------------------------------------
# Offline training script template for Strategy B
# ------------------------------------------------------------------
def train_from_dataset(
    bonafide_dir: str,
    spoof_dir: str,
    output_path: str = "models/audio_rf_classifier.pkl",
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    detector = FinalAudioDetector()
    X, y = [], []

    for label, folder in [(0, bonafide_dir), (1, spoof_dir)]:
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            path = os.path.join(folder, fname)
            feats = detector.extract_training_features(path)
            if feats is not None:
                X.append(feats)
                y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} samples ({int(np.sum(y==0))} bonafide, {int(np.sum(y==1))} spoof)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=["bonafide", "spoof"]))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved trained classifier to {output_path}")


# Global instance
final_audio_detector = FinalAudioDetector()


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python final_audio_detector.py <audio_path>")
        sys.exit(1)
    print(json.dumps(final_audio_detector.predict(sys.argv[1]), indent=2))
