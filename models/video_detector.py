"""
Video Deepfake Detector Module
Implements ML algorithms for detecting deepfakes in videos using temporal consistency analysis
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import tempfile
import subprocess
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from app.ml.temporal_analyzer import temporal_analyzer
from app.ml.lipsync_detector import lipsync_detector

class VideoDeepfakeDetector:
    def __init__(self):
        """
        Initialize the Video Deepfake Detector with ML models for video analysis
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh for facial landmark tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Initialize models for temporal consistency
        self.temporal_consistency_model = TemporalConsistencyNet()
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # Feature names for temporal analysis
        self.feature_names = [
            'face_position_stability',
            'eye_aspect_ratio_consistency',
            'mouth_aspect_ratio_consistency',
            'head_pose_stability',
            'facial_landmark_variance',
            'motion_smoothness',
            'compression_artifacts',
            'temporal_coherence'
        ]
        
        self.is_trained = False

    def extract_video_features(self, video_path: str, max_frames: int = 100) -> Dict[str, any]:
        """
        Extract features from video for deepfake detection
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Dict: Extracted features for analysis
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly throughout the video
        step_size = max(1, frame_count // max_frames)
        
        face_landmarks_sequence = []
        head_poses = []
        eye_ratios = []
        mouth_ratios = []
        frame_positions = []
        
        frame_idx = 0
        processed_frame_count = 0
        
        while cap.isOpened() and processed_frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step_size == 0:
                # Process this frame
                landmarks, head_pose = self.extract_face_landmarks(frame)
                
                if landmarks is not None and head_pose is not None:
                    face_landmarks_sequence.append(landmarks)
                    head_poses.append(head_pose)
                    
                    # Calculate eye aspect ratio (EAR)
                    ear = self.calculate_eye_aspect_ratio(landmarks)
                    eye_ratios.append(ear)
                    
                    # Calculate mouth aspect ratio (MAR)
                    mar = self.calculate_mouth_aspect_ratio(landmarks)
                    mouth_ratios.append(mar)
                    
                    # Store face position
                    face_center = self.get_face_center(landmarks)
                    frame_positions.append(face_center)
                    
                    processed_frame_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate temporal consistency features
        features = self.calculate_temporal_features(
            face_landmarks_sequence, 
            head_poses, 
            eye_ratios, 
            mouth_ratios, 
            frame_positions
        )
        
        return {
            'frame_count': processed_frame_count,
            'face_landmarks_sequence': face_landmarks_sequence,
            'head_poses': head_poses,
            'eye_ratios': eye_ratios,
            'mouth_ratios': mouth_ratios,
            'frame_positions': frame_positions,
            'temporal_features': features
        }

    def extract_face_landmarks(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[Dict]]:
        """
        Extract facial landmarks and head pose from a frame
        
        Args:
            frame: Input frame from video
            
        Returns:
            Tuple of (landmarks, head_pose) or (None, None) if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Get the first face
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert landmarks to list of coordinates
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z
                    landmarks.append((x, y, z))
                
                # Estimate head pose
                head_pose = self.estimate_head_pose(landmarks, frame.shape)
                
                return landmarks, head_pose
            else:
                return None, None
        except Exception:
            return None, None

    def estimate_head_pose(self, landmarks: List[Tuple[int, int, float]], frame_shape: Tuple) -> Dict:
        """
        Estimate head pose from facial landmarks
        """
        if len(landmarks) < 468:
            return {'pitch': 0, 'yaw': 0, 'roll': 0, 'confidence': 0}
        
        # Simplified head pose estimation using specific landmarks
        # In a real implementation, this would use solvePnP with a 3D face model
        try:
            # Get specific landmarks for pose estimation
            nose_tip = np.array(landmarks[1][:2])  # Nose tip
            chin = np.array(landmarks[152][:2])    # Chin
            left_eye = np.array(landmarks[159][:2])  # Left eye
            right_eye = np.array(landmarks[386][:2]) # Right eye
            left_mouth = np.array(landmarks[61][:2])  # Left mouth
            right_mouth = np.array(landmarks[291][:2]) # Right mouth
            
            # Calculate relative positions
            nose_to_chin = np.linalg.norm(nose_tip - chin)
            eye_distance = np.linalg.norm(left_eye - right_eye)
            mouth_width = np.linalg.norm(left_mouth - right_mouth)
            
            # Estimate angles (simplified)
            pitch = (nose_to_chin / frame_shape[0]) * 30  # Rough estimation
            yaw = (abs(left_eye[0] - right_eye[0]) / eye_distance - 1) * 45
            roll = (left_eye[1] - right_eye[1]) / eye_distance * 30
            
            return {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'confidence': min(1.0, len(landmarks) / 468.0)
            }
        except:
            return {'pitch': 0, 'yaw': 0, 'roll': 0, 'confidence': 0}

    def calculate_eye_aspect_ratio(self, landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        """
        try:
            # Indices for eye landmarks (MediaPipe face mesh)
            left_eye_indices = [362, 385, 387, 263, 373, 380]  # Left eye
            right_eye_indices = [33, 160, 158, 133, 153, 144]  # Right eye
            
            if len(landmarks) < max(max(left_eye_indices), max(right_eye_indices)):
                return 0.0
            
            # Calculate EAR for both eyes
            def calculate_single_ear(indices):
                # Vertical landmarks
                vertical_1 = np.array(landmarks[indices[1]][:2])
                vertical_2 = np.array(landmarks[indices[2]][:2])
                vertical_3 = np.array(landmarks[indices[5]][:2])
                vertical_4 = np.array(landmarks[indices[4]][:2])
                
                # Horizontal landmarks
                horizontal_1 = np.array(landmarks[indices[0]][:2])
                horizontal_2 = np.array(landmarks[indices[3]][:2])
                
                # Calculate distances
                vertical_dist = (np.linalg.norm(vertical_1 - vertical_2) + 
                               np.linalg.norm(vertical_3 - vertical_4)) / 2.0
                horizontal_dist = np.linalg.norm(horizontal_1 - horizontal_2)
                
                if horizontal_dist == 0:
                    return 0.0
                
                return vertical_dist / horizontal_dist
            
            left_ear = calculate_single_ear(left_eye_indices)
            right_ear = calculate_single_ear(right_eye_indices)
            
            return (left_ear + right_ear) / 2.0
        except:
            return 0.0

    def calculate_mouth_aspect_ratio(self, landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR)
        """
        try:
            # Indices for mouth landmarks (MediaPipe face mesh)
            mouth_indices = [61, 39, 0, 269, 291, 405]  # Mouth corners and top/bottom
            
            if len(landmarks) < max(mouth_indices):
                return 0.0
            
            # Vertical landmarks (top to bottom of mouth)
            vertical_1 = np.array(landmarks[mouth_indices[2]][:2])  # Top lip
            vertical_2 = np.array(landmarks[mouth_indices[3]][:2])  # Bottom lip
            
            # Horizontal landmarks (left to right corners)
            horizontal_1 = np.array(landmarks[mouth_indices[0]][:2])  # Left corner
            horizontal_2 = np.array(landmarks[mouth_indices[4]][:2])  # Right corner
            
            vertical_dist = np.linalg.norm(vertical_1 - vertical_2)
            horizontal_dist = np.linalg.norm(horizontal_1 - horizontal_2)
            
            if horizontal_dist == 0:
                return 0.0
            
            return vertical_dist / horizontal_dist
        except:
            return 0.0

    def get_face_center(self, landmarks: List[Tuple[int, int, float]]) -> Tuple[float, float]:
        """
        Get the center position of the face
        """
        try:
            xs = [pt[0] for pt in landmarks]
            ys = [pt[1] for pt in landmarks]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        except:
            return (0, 0)

    def calculate_temporal_features(self, face_landmarks_seq: List, head_poses: List, 
                                  eye_ratios: List, mouth_ratios: List, 
                                  frame_positions: List) -> Dict:
        """
        Calculate temporal consistency features
        """
        if len(face_landmarks_seq) < 2:
            return {
                'position_stability': 0,
                'pose_stability': 0,
                'eye_ratio_consistency': 0,
                'mouth_ratio_consistency': 0,
                'smoothness_score': 0,
                'variance_score': 0
            }
        
        # Calculate face position stability
        positions = np.array(frame_positions)
        position_changes = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        position_stability = 1.0 / (1.0 + np.std(position_changes)) if len(position_changes) > 0 else 1.0
        
        # Calculate head pose stability
        poses = np.array([[hp['pitch'], hp['yaw'], hp['roll']] for hp in head_poses])
        pose_variance = np.mean(np.var(poses, axis=0)) if len(poses) > 0 else 1.0
        pose_stability = 1.0 / (1.0 + pose_variance)
        
        # Calculate eye ratio consistency
        eye_ratios_array = np.array(eye_ratios)
        eye_consistency = 1.0 / (1.0 + np.std(eye_ratios_array)) if len(eye_ratios_array) > 0 else 1.0
        
        # Calculate mouth ratio consistency
        mouth_ratios_array = np.array(mouth_ratios)
        mouth_consistency = 1.0 / (1.0 + np.std(mouth_ratios_array)) if len(mouth_ratios_array) > 0 else 1.0
        
        # Calculate motion smoothness
        if len(position_changes) > 1:
            acceleration = np.diff(position_changes)
            smoothness_score = 1.0 / (1.0 + np.std(acceleration)) if len(acceleration) > 0 else 1.0
        else:
            smoothness_score = 1.0
        
        # Calculate overall variance
        all_ratios = np.concatenate([eye_ratios_array, mouth_ratios_array]) if len(eye_ratios_array) > 0 and len(mouth_ratios_array) > 0 else np.array([])
        variance_score = 1.0 / (1.0 + np.var(all_ratios)) if len(all_ratios) > 0 else 1.0
        
        return {
            'position_stability': position_stability,
            'pose_stability': pose_stability,
            'eye_ratio_consistency': eye_consistency,
            'mouth_ratio_consistency': mouth_consistency,
            'smoothness_score': smoothness_score,
            'variance_score': variance_score
        }

    def detect_deepfake(self, video_path: str) -> Dict[str, any]:
        """
        Detect if a video is a deepfake using temporal consistency analysis
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dict: Deepfake detection results
        """
        try:
            # Extract features from video
            features_data = self.extract_video_features(video_path)
            
            # Perform advanced temporal analysis using LSTM
            advanced_temporal_results = temporal_analyzer.analyze_temporal_consistency(video_path)
            
            # Perform lip-sync analysis
            lipsync_results = lipsync_detector.detect_lip_sync_inconsistency(video_path)
            
            # Prepare feature vector for ML model
            temporal_features = features_data['temporal_features']
            feature_vector = np.array([
                temporal_features['position_stability'],
                temporal_features['pose_stability'],
                temporal_features['eye_ratio_consistency'],
                temporal_features['mouth_ratio_consistency'],
                temporal_features['smoothness_score'],
                temporal_features['variance_score'],
                self.estimate_compression_artifacts(video_path),
                self.calculate_temporal_coherence(features_data['face_landmarks_sequence'])
            ]).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.fit_transform(feature_vector)
            
            # Use isolation forest for anomaly detection
            anomaly_score = self.isolation_forest.fit_predict(feature_vector_scaled)[0]
            anomaly_probability = self.isolation_forest.score_samples(feature_vector_scaled)[0]
            
            # Calculate temporal consistency score
            consistency_score = self.calculate_consistency_score(temporal_features)
            
            # Incorporate advanced temporal and lip-sync analysis
            advanced_temporal_score = advanced_temporal_results['consistency_score']
            lipsync_score = lipsync_results['sync_confidence'] if lipsync_results['is_synced'] else (1 - lipsync_results['inconsistency_score'])
            
            # Weighted combination of all scores
            basic_score = max(0, min(1, (1 - consistency_score + abs(anomaly_probability))/2))
            advanced_score = 1 - advanced_temporal_results['consistency_score']
            lipsync_factor = lipsync_results['inconsistency_score'] if not lipsync_results['is_synced'] else (1 - lipsync_results['sync_confidence'])
            
            # Apply weights
            weights = {
                'basic': 0.3,
                'advanced_temporal': 0.4,
                'lipsync': 0.3
            }
            
            # Calculate final probability
            fake_probability = (
                weights['basic'] * basic_score + 
                weights['advanced_temporal'] * advanced_score + 
                weights['lipsync'] * lipsync_factor
            )
            
            # Final prediction
            is_deepfake = fake_probability > 0.5
            
            return {
                'is_deepfake': bool(is_deepfake),
                'fake_probability': round(float(fake_probability), 3),
                'consistency_score': round(consistency_score, 3),
                'anomaly_score': round(float(anomaly_probability), 3),
                'confidence': round(min(1.0, len(features_data['face_landmarks_sequence']) / 10.0), 2),  # Based on number of analyzed frames
                'frame_analysis_count': len(features_data['face_landmarks_sequence']),
                'temporal_features': temporal_features,
                'detailed_analysis': {
                    'basic_temporal_analysis': {
                        'position_stability': round(temporal_features['position_stability'], 3),
                        'pose_stability': round(temporal_features['pose_stability'], 3),
                        'eye_consistency': round(temporal_features['eye_ratio_consistency'], 3),
                        'mouth_consistency': round(temporal_features['mouth_ratio_consistency'], 3),
                        'motion_smoothness': round(temporal_features['smoothness_score'], 3)
                    },
                    'advanced_temporal_analysis': advanced_temporal_results,
                    'lip_sync_analysis': lipsync_results
                },
                'explanation': self.generate_explanation(temporal_features, is_deepfake, advanced_temporal_results, lipsync_results)
            }
        except Exception as e:
            return {
                'is_deepfake': False,
                'fake_probability': 0.0,
                'error': str(e),
                'confidence': 0.0
            }

    def estimate_compression_artifacts(self, video_path: str) -> float:
        """
        Estimate compression artifacts in video
        """
        try:
            # Use FFmpeg to analyze video quality
            cmd = [
                'ffprobe', 
                '-v', 'quiet', 
                '-show_streams', 
                '-select_streams', 'v:0', 
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse stream info to estimate quality
                # This is a simplified approach - real implementation would be more complex
                return 0.7  # Default medium quality indicator
            else:
                return 0.5  # Unknown quality
        except:
            return 0.5  # Default if ffprobe unavailable

    def calculate_temporal_coherence(self, face_landmarks_seq: List) -> float:
        """
        Calculate temporal coherence of facial landmarks
        """
        if len(face_landmarks_seq) < 2:
            return 0.5
        
        try:
            # Calculate landmark displacement between consecutive frames
            total_displacement = 0
            displacement_count = 0
            
            for i in range(1, len(face_landmarks_seq)):
                prev_landmarks = np.array(face_landmarks_seq[i-1])
                curr_landmarks = np.array(face_landmarks_seq[i])
                
                # Calculate average displacement of key landmarks
                key_indices = [1, 152, 362, 133, 61, 291]  # Nose, chin, eyes, mouth corners
                for idx in key_indices:
                    if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                        disp = np.linalg.norm(
                            np.array(prev_landmarks[idx][:2]) - 
                            np.array(curr_landmarks[idx][:2])
                        )
                        total_displacement += disp
                        displacement_count += 1
            
            if displacement_count > 0:
                avg_displacement = total_displacement / displacement_count
                # Lower displacement indicates higher coherence
                coherence = 1.0 / (1.0 + avg_displacement * 0.01)  # Adjust scaling factor as needed
            else:
                coherence = 0.5
            
            return min(1.0, coherence)
        except:
            return 0.5

    def calculate_consistency_score(self, temporal_features: Dict) -> float:
        """
        Calculate overall temporal consistency score
        """
        weights = {
            'position_stability': 0.15,
            'pose_stability': 0.2,
            'eye_ratio_consistency': 0.15,
            'mouth_ratio_consistency': 0.15,
            'smoothness_score': 0.2,
            'variance_score': 0.15
        }
        
        score = 0
        for feature, weight in weights.items():
            score += temporal_features.get(feature, 0.5) * weight
        
        return min(1.0, max(0.0, score))

    def generate_explanation(self, temporal_features: Dict, is_deepfake: bool, advanced_temporal_results: Dict, lipsync_results: Dict) -> str:
        """
        Generate explanation for the deepfake detection result
        """
        if is_deepfake:
            reasons = []
            
            if temporal_features['position_stability'] < 0.3:
                reasons.append("unstable face positioning")
            if temporal_features['pose_stability'] < 0.3:
                reasons.append("inconsistent head movements")
            if temporal_features['eye_ratio_consistency'] < 0.3:
                reasons.append("unnatural eye movements")
            if temporal_features['mouth_ratio_consistency'] < 0.3:
                reasons.append("inconsistent mouth movements")
            if temporal_features['smoothness_score'] < 0.3:
                reasons.append("jerky or unnatural motion patterns")
            
            # Add reasons from advanced analysis
            if advanced_temporal_results.get('temporal_artifacts_detected', False):
                reasons.append("detected temporal artifacts")
            if advanced_temporal_results.get('consistency_score', 1.0) < 0.5:
                reasons.append("low temporal consistency")
            if lipsync_results.get('inconsistency_score', 0.0) > 0.5:
                reasons.append("audio-visual lip-sync inconsistencies")
            
            if reasons:
                return f"Video flagged as deepfake due to: {', '.join(reasons)}. Temporal consistency analysis detected anomalies in the facial movements and positioning."
            else:
                return "Video flagged as deepfake based on subtle temporal inconsistencies that may indicate synthetic generation."
        else:
            return "Video verified as authentic with consistent temporal patterns in facial movements, positioning, and natural motion dynamics."

    def __del__(self):
        """
        Cleanup resources
        """
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class TemporalConsistencyNet(nn.Module):
    """
    Neural network for temporal consistency analysis
    """
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=1):
        super(TemporalConsistencyNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# Global instance
video_detector = VideoDeepfakeDetector()