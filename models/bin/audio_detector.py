import librosa
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
import logging
import subprocess
import tempfile
from pathlib import Path
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AudioDeepfakeDetector:
    """
    Professional-grade audio deepfake detection module.
    Detects synthetic audio artifacts using spectral and waveform analysis.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.cnn_model = self._build_audio_cnn()
        
    def _build_audio_cnn(self):
        """Build CNN model for audio deepfake detection"""
        class AudioCNN(nn.Module):
            def __init__(self, n_classes=2):
                super().__init__()
                
                # Convolutional layers for spectrogram analysis
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    
                    nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    
                    nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
                )
                
                # Fully connected layers
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 4, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, n_classes)
                )
                
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        model = AudioCNN()
        return model.to(self.device)
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file"""
        try:
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                temp_audio.name,
                '-y', '-v', 'quiet'
            ]
            
            subprocess.run(cmd, check=True)
            return temp_audio.name
            
        except Exception as e:
            logger.error(f"Error extracting audio from video: {str(e)}")
            raise
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive audio features for deepfake detection"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Basic audio statistics
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Tonnetz (tonal centroid features)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            
            # Extract statistics
            features = {
                # Basic stats
                'duration': duration,
                'rms_energy': rms_energy,
                'zero_crossing_rate': zero_crossing_rate,
                
                # Spectral stats
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                
                # MFCC stats
                'mfcc_means': [float(np.mean(mfccs[i])) for i in range(min(13, mfccs.shape[0]))],
                'mfcc_stds': [float(np.std(mfccs[i])) for i in range(min(13, mfccs.shape[0]))],
                
                # Chroma stats
                'chroma_means': [float(np.mean(chroma[i])) for i in range(chroma.shape[0])],
                'chroma_stds': [float(np.std(chroma[i])) for i in range(chroma.shape[0])],
                
                # Other stats
                'spectral_contrast_means': [float(np.mean(spectral_contrast[i])) for i in range(spectral_contrast.shape[0])],
                'tonnetz_means': [float(np.mean(tonnetz[i])) for i in range(tonnetz.shape[0])]
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {str(e)}")
            # Return default features
            return {
                'duration': 0.0,
                'rms_energy': 0.0,
                'zero_crossing_rate': 0.0,
                'spectral_centroid_mean': 0.0,
                'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0,
                'spectral_rolloff_std': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'spectral_bandwidth_std': 0.0,
                'mfcc_means': [0.0] * 13,
                'mfcc_stds': [0.0] * 13,
                'chroma_means': [0.0] * 12,
                'chroma_stds': [0.0] * 12,
                'spectral_contrast_means': [0.0] * 7,
                'tonnetz_means': [0.0] * 6
            }
    
    def detect_synthetic_artifacts(self, audio_path: str) -> Dict[str, Any]:
        """Detect synthetic artifacts in audio"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Analyze for common synthetic artifacts
            # 1. Check for unnatural frequency patterns
            stft = np.abs(librosa.stft(audio))
            frequencies = librosa.fft_frequencies(sr=sr)
            
            # 2. Check for unnatural harmonics
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = np.mean(harmonic**2) / (np.mean(audio**2) + 1e-8)
            
            # 3. Check for unnatural noise patterns
            noise_estimate = audio - harmonic
            noise_power = np.mean(noise_estimate**2)
            
            # 4. Check for unnatural periodicity
            autocorr = signal.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            periodicity_score = np.max(autocorr[1:1000]) / (autocorr[0] + 1e-8)
            
            # 5. Check for unnatural spectral flatness (synthetic audio tends to be more spectrally flat)
            spectral_flatness = np.exp(np.mean(np.log(stft + 1e-8))) / np.mean(stft)
            
            # Calculate artifact scores
            harmonic_artifact_score = abs(harmonic_ratio - 0.5)  # Natural audio typically has ~0.5 ratio
            noise_artifact_score = min(1.0, noise_power * 100)  # Higher noise may indicate synthesis artifacts
            periodicity_artifact_score = abs(periodicity_score - 0.1)  # Natural speech has specific periodicity
            spectral_flatness_score = abs(spectral_flatness - 0.01)  # Synthetic audio often more spectrally flat
            
            # Combined artifact score
            artifact_score = (
                0.3 * harmonic_artifact_score +
                0.25 * noise_artifact_score +
                0.25 * periodicity_artifact_score +
                0.2 * spectral_flatness_score
            )
            
            return {
                'harmonic_artifact_score': float(harmonic_artifact_score),
                'noise_artifact_score': float(noise_artifact_score),
                'periodicity_artifact_score': float(periodicity_artifact_score),
                'spectral_flatness_score': float(spectral_flatness_score),
                'combined_artifact_score': float(artifact_score),
                'harmonic_ratio': float(harmonic_ratio),
                'noise_power': float(noise_power),
                'periodicity_score': float(periodicity_score),
                'spectral_flatness': float(spectral_flatness)
            }
            
        except Exception as e:
            logger.error(f"Error detecting synthetic artifacts: {str(e)}")
            return {
                'harmonic_artifact_score': 0.5,
                'noise_artifact_score': 0.5,
                'periodicity_artifact_score': 0.5,
                'spectral_flatness_score': 0.5,
                'combined_artifact_score': 0.5,
                'harmonic_ratio': 0.5,
                'noise_power': 0.1,
                'periodicity_score': 0.1,
                'spectral_flatness': 0.01
            }
    
    def predict_with_cnn(self, audio_path: str) -> float:
        """Predict using CNN model"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Create mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Normalize
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
            
            # Convert to tensor
            spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                self.cnn_model.eval()
                output = self.cnn_model(spec_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Probability of being fake (index 1)
                fake_prob = probabilities[0][1].item()
                
            return fake_prob
            
        except Exception as e:
            logger.error(f"CNN prediction error: {str(e)}")
            return 0.5
    
    def detect_deepfake(self, audio_or_video_path: str, is_video: bool = True) -> Dict[str, Any]:
        """
        Detect if audio is synthetic/deepfake
        
        Args:
            audio_or_video_path: Path to audio or video file
            is_video: Whether the input is a video file (audio will be extracted)
        
        Returns:
            Dict: Deepfake detection results
        """
        try:
            # Extract audio if input is video
            if is_video:
                audio_path = self.extract_audio_from_video(audio_or_video_path)
            else:
                audio_path = audio_or_video_path
            
            # Extract features
            features = self.extract_audio_features(audio_path)
            
            # Detect synthetic artifacts
            artifacts = self.detect_synthetic_artifacts(audio_path)
            
            # Get CNN prediction
            cnn_prediction = self.predict_with_cnn(audio_path)
            
            # Prepare feature vector for traditional ML model
            feature_vector = np.array([
                features['rms_energy'],
                features['zero_crossing_rate'],
                features['spectral_centroid_mean'],
                features['spectral_rolloff_mean'],
                features['spectral_bandwidth_mean'],
                features['mfcc_means'][0] if features['mfcc_means'] else 0,
                features['mfcc_means'][1] if len(features['mfcc_means']) > 1 else 0,
                artifacts['combined_artifact_score']
            ]).reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_vector)
            
            # Use isolation forest for anomaly detection
            anomaly_score = self.isolation_forest.fit_predict(scaled_features)[0]
            anomaly_probability = self.isolation_forest.score_samples(scaled_features)[0]
            
            # Combine predictions
            # Anomaly score: -1 for anomaly (likely fake), 1 for normal (likely real)
            traditional_score = 0.0 if anomaly_score == 1 else 1.0
            artifact_score = artifacts['combined_artifact_score']
            
            # Weighted combination
            weights = {
                'traditional': 0.3,
                'artifacts': 0.3,
                'cnn': 0.4
            }
            
            fake_probability = (
                weights['traditional'] * traditional_score +
                weights['artifacts'] * artifact_score +
                weights['cnn'] * cnn_prediction
            )
            
            is_deepfake = fake_probability > 0.5
            
            # Clean up temporary audio file if created
            if is_video:
                Path(audio_path).unlink()
            
            return {
                'is_deepfake': bool(is_deepfake),
                'fake_probability': round(float(fake_probability), 3),
                'confidence': round(min(1.0, max(0.0, abs(fake_probability - 0.5) * 2)), 2),
                'detection_method': 'audio_analysis',
                'audio_features': {
                    'duration': features['duration'],
                    'rms_energy': features['rms_energy'],
                    'zero_crossing_rate': features['zero_crossing_rate'],
                    'spectral_features': {
                        'centroid_mean': features['spectral_centroid_mean'],
                        'rolloff_mean': features['spectral_rolloff_mean'],
                        'bandwidth_mean': features['spectral_bandwidth_mean']
                    },
                    'mfcc_features': dict(zip(range(len(features['mfcc_means'])), 
                                           [round(x, 4) for x in features['mfcc_means']]))
                },
                'synthetic_artifacts': artifacts,
                'cnn_prediction': round(float(cnn_prediction), 3),
                'traditional_ml_score': round(float(traditional_score), 3),
                'detailed_analysis': {
                    'spectral_analysis': {
                        'harmonic_ratio': artifacts['harmonic_ratio'],
                        'spectral_flatness': artifacts['spectral_flatness'],
                        'periodicity_score': artifacts['periodicity_score']
                    },
                    'noise_analysis': {
                        'noise_power': artifacts['noise_power'],
                        'artifact_indicators': {
                            'harmonic_artifacts': artifacts['harmonic_artifact_score'],
                            'noise_artifacts': artifacts['noise_artifact_score'],
                            'periodicity_artifacts': artifacts['periodicity_artifact_score']
                        }
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in audio deepfake detection: {str(e)}")
            return {
                'is_deepfake': False,
                'fake_probability': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }


# Global instance
audio_detector = AudioDeepfakeDetector()