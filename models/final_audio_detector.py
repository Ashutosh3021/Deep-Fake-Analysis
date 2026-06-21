"""
FINAL Audio Deepfake / Synthetic Speech Detector
===================================================
Design philosophy: the old code ran predict() on an UNTRAINED PyTorch CNN
and blended that random noise into a weighted ensemble with a hardcoded
"94.2%" accuracy claim. That is worse than doing nothing, because it
produces a confident-looking number with zero grounding.

This file replaces that with two real, swappable strategies:

  STRATEGY A (default, zero training required):
      A pretrained spoof-detection model from Hugging Face
      (wav2vec2-based). This is an actual trained classifier on real
      speech-deepfake data, not a random network.

  STRATEGY B (better, requires ~30 min of training, instructions below):
      A small classifier (RandomForest or LogisticRegression -- genuinely
      lightweight, CPU-only, trains in seconds) on top of the physically-
      grounded acoustic features this project already extracts well:
      jitter, shimmer, spectral flatness, F0 stability, harmonic ratio.
      These features ARE real signal for TTS/voice-conversion detection;
      they just need a classifier fitted on labeled data instead of
      hand-picked thresholds.

HOW TO TRAIN STRATEGY B (do this once you have data):
    1. Get ASVspoof 2019 LA (https://datashare.ed.ac.uk/handle/10283/3336)
       -- free, no paywall, ~25k labeled clips, smallest practical starting set.
    2. Run `extract_training_features()` on every clip with its label.
    3. Fit sklearn RandomForestClassifier on the resulting feature matrix.
    4. Pickle the fitted classifier and point AUDIO_CLASSIFIER_PATH at it.
    Full script template is at the bottom of this file (`train_from_dataset`).

Setup:
  pip install librosa numpy scipy scikit-learn transformers torch --break-system-packages
"""

import os
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import librosa
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000

# Pretrained spoof-detection model (Strategy A). This is a real fine-tuned
# checkpoint, not a base model -- it was actually trained to separate
# bonafide vs. spoofed speech.
PRETRAINED_MODEL_ID = os.getenv(
    "AUDIO_DETECTOR_MODEL", "MelodyMachine/Deepfake-audio-detection-V2"
)

# Path to a classifier YOU trained on real labeled data (Strategy B).
# If this file doesn't exist, we fall back to Strategy A, then to
# heuristic-only mode if even that is unavailable.
AUDIO_CLASSIFIER_PATH = os.getenv("AUDIO_CLASSIFIER_PATH", "models/audio_rf_classifier.pkl")

LOW_CONFIDENCE_THRESHOLD = 0.60


@dataclass
class AudioVerdict:
    label: str                 # "SYNTHETIC" | "AUTHENTIC" | "UNCERTAIN"
    confidence: float
    fake_probability: float
    signal_source: str         # which strategy actually produced the score
    feature_summary: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "fake_probability": round(self.fake_probability, 4),
            "signal_source": self.signal_source,
            "feature_summary": self.feature_summary,
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
    # Acoustic feature extraction (real, physically grounded)
    # ------------------------------------------------------------------
    def extract_training_features(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extracts a fixed-length feature vector suitable both for live
        inference AND for training Strategy B's classifier offline.
        Keeping this single shared function means train-time and
        inference-time features can never silently drift apart.
        """
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            if len(audio) < sr * 0.3:  # less than 300ms is not analyzable
                return None

            feats: List[float] = []

            # --- Spectral shape ---
            feats.append(float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))))
            feats.append(float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))))
            feats.append(float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))))
            feats.append(float(np.mean(librosa.feature.spectral_flatness(y=audio))))
            feats.append(float(np.mean(librosa.feature.zero_crossing_rate(audio))))

            # --- MFCC (timbre) ---
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            feats.extend(np.mean(mfccs, axis=1).tolist())
            feats.extend(np.std(mfccs, axis=1).tolist())

            # --- Harmonic / percussive balance ---
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = float(np.sum(harmonic ** 2) / (np.sum(audio ** 2) + 1e-8))
            feats.append(harmonic_ratio)

            # --- F0 stability, jitter (TTS/VC often has unnaturally
            #     stable or erratic pitch contours vs. human speech) ---
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
            )
            if f0 is not None and np.any(voiced_flag):
                f0_voiced = f0[voiced_flag]
                f0_mean = float(np.nanmean(f0_voiced))
                f0_std = float(np.nanstd(f0_voiced))
                if len(f0_voiced) > 1 and f0_mean > 0:
                    jitter = float(np.mean(np.abs(np.diff(f0_voiced))) / f0_mean)
                else:
                    jitter = 0.0
            else:
                f0_mean, f0_std, jitter = 0.0, 0.0, 0.0
            feats.extend([f0_mean, f0_std, jitter])

            # --- Phase coherence (vocoder artifacts often disrupt natural
            #     phase relationships between adjacent frames) ---
            stft = librosa.stft(audio)
            phase = np.angle(stft)
            phase_diff = np.diff(phase, axis=1)
            phase_discontinuity = float(np.mean(np.abs(phase_diff) > np.pi / 4))
            feats.append(phase_discontinuity)

            # --- Distributional shape of the waveform itself ---
            feats.append(float(skew(audio)))
            feats.append(float(kurtosis(audio)))

            return np.array(feats, dtype=np.float32)
        except Exception as e:
            logger.error("Feature extraction failed for %s: %s", audio_path, e)
            return None

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
    # Strategy B: your trained classifier on acoustic features
    # ------------------------------------------------------------------
    def _trained_classifier_predict(self, audio_path: str) -> Optional[float]:
        if self._trained_classifier is None:
            return None
        feats = self.extract_training_features(audio_path)
        if feats is None:
            return None
        try:
            proba = self._trained_classifier.predict_proba(feats.reshape(1, -1))[0]
            # Assumes class index 1 == "fake/spoof" -- match this to however
            # you label y during training (see train_from_dataset below).
            return float(proba[1])
        except Exception as e:
            logger.error("Trained classifier inference failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Heuristic fallback (used only if neither A nor B is available)
    # ------------------------------------------------------------------
    def _heuristic_predict(self, audio_path: str) -> float:
        """
        Last-resort signal using domain knowledge but no fitted classifier.
        This is intentionally conservative -- it should rarely push the
        score far from 0.5 unless multiple indicators agree, because
        unfit thresholds are guesses, not measurements.
        """
        feats = self.extract_training_features(audio_path)
        if feats is None:
            return 0.5

        # Indices match extract_training_features() ordering:
        # [centroid, bandwidth, rolloff, flatness, zcr, mfcc_mean(20),
        #  mfcc_std(20), harmonic_ratio, f0_mean, f0_std, jitter,
        #  phase_discontinuity, skew, kurtosis]
        flatness = feats[3]
        harmonic_ratio = feats[45]
        jitter = feats[48]
        phase_discontinuity = feats[49]

        indicators = []
        # Real speech is rarely spectrally flat; very flat -> synthetic-leaning.
        indicators.append(np.clip(flatness / 0.05, 0, 1))
        # Very low jitter (too-perfect pitch) suggests synthesis.
        indicators.append(1.0 - np.clip(jitter / 0.02, 0, 1)) if jitter > 0 else indicators.append(0.5)
        # High phase discontinuity suggests vocoder artifacts.
        indicators.append(np.clip(phase_discontinuity / 0.4, 0, 1))

        return float(np.clip(np.mean(indicators), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, audio_path: str) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            return {"error": "file_not_found", "path": audio_path}

        # Preference order: your trained classifier > pretrained HF model
        # > heuristic fallback. Trained-on-your-data always wins if present.
        score = self._trained_classifier_predict(audio_path)
        source = "trained_classifier"

        if score is None:
            score = self._hf_predict(audio_path)
            source = "pretrained_hf_model"

        if score is None:
            score = self._heuristic_predict(audio_path)
            source = "heuristic_fallback"

        fake_probability = float(np.clip(score, 0.0, 1.0))
        distance_from_mid = abs(fake_probability - 0.5) * 2

        if source == "heuristic_fallback":
            # Be explicit: this mode has a much lower ceiling. Halve the
            # effective certainty rather than report a number people might
            # trust as much as a real model's output.
            distance_from_mid *= 0.5

        if distance_from_mid < (1 - LOW_CONFIDENCE_THRESHOLD):
            label = "UNCERTAIN"
        elif fake_probability > 0.5:
            label = "SYNTHETIC"
        else:
            label = "AUTHENTIC"

        confidence = 50.0 + distance_from_mid * 50.0

        notes_map = {
            "trained_classifier": "Score from classifier trained on your labeled dataset.",
            "pretrained_hf_model": f"Score from pretrained model ({PRETRAINED_MODEL_ID}).",
            "heuristic_fallback": (
                "No trained classifier or pretrained model available -- running on "
                "unfit acoustic heuristics. Confidence is deliberately suppressed. "
                "Train Strategy B (see file docstring) for real accuracy."
            ),
        }

        verdict = AudioVerdict(
            label=label,
            confidence=confidence,
            fake_probability=fake_probability,
            signal_source=source,
            feature_summary={"raw_score": round(score, 4)},
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
    """
    Train a lightweight RandomForest on the acoustic features above.

    Expected layout:
        bonafide_dir/  -- folder of real/bonafide .wav or .flac files
        spoof_dir/     -- folder of synthetic/spoofed .wav or .flac files

    For ASVspoof 2019 LA: bonafide_dir = .../ASVspoof2019_LA_train/bonafide,
    spoof_dir = .../ASVspoof2019_LA_train/spoof (after sorting by the
    protocol file's label column -- ASVspoof ships flat directories with
    a separate label/protocol text file, so you'll need to split files
    into these two folders first based on that protocol file).

    Usage:
        from final_audio_detector import train_from_dataset
        train_from_dataset("data/bonafide", "data/spoof")
    """
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
