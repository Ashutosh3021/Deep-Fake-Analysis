"""
Risk Analyzer Module for OSINT Deepfake Monitoring System
Implements ML algorithms for threat assessment and risk scoring
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
import pickle
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class RiskAnalyzer:
    def __init__(self):
        """
        Initialize the Risk Analyzer with ML models for threat assessment
        """
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Feature names for risk assessment
        self.feature_names = [
            'activity_frequency', 'platform_diversity', 'content_similarity',
            'posting_patterns', 'geolocation_variability', 'account_age',
            'profile_completeness', 'connection_density', 'behavioral_consistency'
        ]
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': (0.0, 0.3),
            'medium': (0.3, 0.6),
            'high': (0.6, 0.8),
            'critical': (0.8, 1.0)
        }
        
        self.is_trained = False

    def extract_features(self, digital_footprint: Dict[str, Any]) -> np.ndarray:
        """
        Extract risk assessment features from digital footprint data
        
        Args:
            digital_footprint: Dictionary containing footprint data
            
        Returns:
            np.ndarray: Feature vector for risk analysis
        """
        features = []
        
        # Activity frequency (posts per day)
        total_activities = len(digital_footprint.get('digital_footprint', []))
        time_span = self.calculate_time_span(digital_footprint)
        activity_frequency = total_activities / max(time_span, 1)  # Avoid division by zero
        features.append(activity_frequency)
        
        # Platform diversity (number of different platforms)
        platforms = set()
        for item in digital_footprint.get('digital_footprint', []):
            platforms.add(item.get('platform', ''))
        platform_diversity = len(platforms) / 10.0  # Normalize assuming max 10 platforms
        features.append(platform_diversity)
        
        # Content similarity (using TF-IDF and cosine similarity)
        content_texts = [item.get('content', '') for item in digital_footprint.get('digital_footprint', [])]
        content_similarity = self.calculate_content_similarity(content_texts)
        features.append(content_similarity)
        
        # Posting patterns (irregular posting times might indicate bot activity)
        posting_patterns = self.analyze_posting_patterns(digital_footprint)
        features.append(posting_patterns)
        
        # Geolocation variability (if location data available)
        geolocation_variability = self.calculate_geolocation_variability(digital_footprint)
        features.append(geolocation_variability)
        
        # Account age (normalized)
        account_age = self.estimate_account_age(digital_footprint)
        features.append(account_age)
        
        # Profile completeness (based on available fields)
        profile_completeness = self.assess_profile_completeness(digital_footprint)
        features.append(profile_completeness)
        
        # Connection density (estimated based on platform activity)
        connection_density = self.estimate_connection_density(digital_footprint)
        features.append(connection_density)
        
        # Behavioral consistency (consistency of content themes)
        behavioral_consistency = self.assess_behavioral_consistency(content_texts)
        features.append(behavioral_consistency)
        
        # Convert to numpy array and ensure correct shape
        features = np.array(features).reshape(1, -1)
        
        # Handle cases where we have fewer features than expected
        if features.shape[1] < len(self.feature_names):
            padding = np.zeros((1, len(self.feature_names) - features.shape[1]))
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > len(self.feature_names):
            features = features[:, :len(self.feature_names)]
        
        return features

    def calculate_time_span(self, digital_footprint: Dict[str, Any]) -> float:
        """Calculate the time span of digital activity in days"""
        timestamps = []
        for item in digital_footprint.get('digital_footprint', []):
            ts_str = item.get('timestamp', '')
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    timestamps.append(dt)
                except ValueError:
                    continue
        
        if len(timestamps) < 2:
            return 1.0  # Assume 1 day if insufficient data
        
        time_diff = max(timestamps) - min(timestamps)
        return max(time_diff.days, 1)

    def calculate_content_similarity(self, content_texts: List[str]) -> float:
        """Calculate content similarity using TF-IDF"""
        if len(content_texts) < 2:
            return 0.0
        
        try:
            # Vectorize the content
            vectors = self.text_vectorizer.fit_transform(content_texts)
            
            # Calculate average cosine similarity
            if vectors.shape[0] > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(vectors)
                # Get upper triangle of similarity matrix (excluding diagonal)
                triu_indices = np.triu_indices_from(similarities, k=1)
                if len(triu_indices[0]) > 0:
                    avg_similarity = np.mean(similarities[triu_indices])
                    return min(avg_similarity, 1.0)
        except:
            pass
        
        return 0.0

    def analyze_posting_patterns(self, digital_footprint: Dict[str, Any]) -> float:
        """Analyze posting patterns for irregularities"""
        timestamps = []
        for item in digital_footprint.get('digital_footprint', []):
            ts_str = item.get('timestamp', '')
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    timestamps.append(dt)
                except ValueError:
                    continue
        
        if len(timestamps) < 2:
            return 0.5  # Neutral value if insufficient data
        
        # Calculate time differences between consecutive posts
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                      for i in range(len(timestamps)-1)]
        
        if not time_diffs:
            return 0.5
        
        # Calculate coefficient of variation (higher CV = more irregular)
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        
        if mean_diff == 0:
            return 1.0 if std_diff > 0 else 0.0
        
        cv = std_diff / abs(mean_diff)
        # Normalize to 0-1 range (higher CV = higher risk)
        return min(cv / 5.0, 1.0)  # Cap at 1.0

    def calculate_geolocation_variability(self, digital_footprint: Dict[str, Any]) -> float:
        """Calculate geolocation variability"""
        locations = []
        for item in digital_footprint.get('digital_footprint', []):
            location = item.get('location', '')
            if location and location.lower() != 'unknown':
                locations.append(location.lower())
        
        if not locations:
            return 0.0  # No location data
        
        unique_locations = len(set(locations))
        total_locations = len(locations)
        
        # Higher ratio = more diverse locations = higher risk
        if total_locations == 0:
            return 0.0
        
        return min(unique_locations / total_locations, 1.0)

    def estimate_account_age(self, digital_footprint: Dict[str, Any]) -> float:
        """Estimate account age based on earliest activity"""
        timestamps = []
        for item in digital_footprint.get('digital_footprint', []):
            ts_str = item.get('timestamp', '')
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    timestamps.append(dt)
                except ValueError:
                    continue
        
        if not timestamps:
            return 0.1  # Low value if no timestamps
        
        oldest_date = min(timestamps)
        days_old = (datetime.now() - oldest_date).days
        
        # Normalize to 0-1 range (newer accounts = higher risk)
        # Assuming max account age of 10 years (3650 days)
        return min(days_old / 3650.0, 1.0)

    def assess_profile_completeness(self, digital_footprint: Dict[str, Any]) -> float:
        """Assess profile completeness based on available information"""
        # Count available profile fields across all platforms
        completeness_score = 0
        total_fields = 0
        
        for item in digital_footprint.get('digital_footprint', []):
            # Count non-empty fields
            for field in ['content', 'location', 'url']:
                if item.get(field, ''):
                    completeness_score += 1
                total_fields += 1
        
        if total_fields == 0:
            return 0.3  # Neutral value if no data
        
        return min(completeness_score / total_fields, 1.0)

    def estimate_connection_density(self, digital_footprint: Dict[str, Any]) -> float:
        """Estimate connection density based on activity patterns"""
        # This is a simplified estimation
        # In a real system, this would use actual connection data
        total_activities = len(digital_footprint.get('digital_footprint', []))
        
        # Higher activity density might indicate higher risk
        # Normalize based on expected activity levels
        expected_max_activities = 100  # Adjust based on use case
        return min(total_activities / expected_max_activities, 1.0)

    def assess_behavioral_consistency(self, content_texts: List[str]) -> float:
        """Assess behavioral consistency based on content themes"""
        if not content_texts:
            return 0.5  # Neutral value
        
        # Analyze content for consistency using simple keyword analysis
        # In a real system, this would use more sophisticated NLP
        keywords = set()
        total_words = 0
        
        for text in content_texts:
            words = re.findall(r'\w+', text.lower())
            keywords.update(words)
            total_words += len(words)
        
        if total_words == 0:
            return 0.5
        
        # Consistency measure (simplified)
        # More repeated words = higher consistency
        unique_words = len(keywords)
        if total_words == 0:
            return 0.5
        
        repetition_rate = 1 - (unique_words / max(total_words, 1))
        return repetition_rate

    def train_models(self, training_data: List[Tuple[Dict[str, Any], float]] = None):
        """
        Train the risk assessment models
        
        Args:
            training_data: List of tuples (digital_footprint, risk_score)
        """
        if training_data is None:
            # Use synthetic training data if none provided
            print("Using synthetic training data for risk models...")
            X_train, y_train = self.generate_synthetic_training_data()
        else:
            X_train = []
            y_train = []
            for footprint, risk_score in training_data:
                features = self.extract_features(footprint)[0]
                X_train.append(features)
                y_train.append(risk_score)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train isolation forest for anomaly detection
        self.isolation_forest.fit(X_train_scaled)
        
        # Train risk classifier
        self.risk_classifier.fit(X_train_scaled, (y_train * 100).astype(int))  # Scale risk scores
        
        self.is_trained = True
        print("Risk assessment models trained successfully.")

    def generate_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for risk models"""
        num_samples = 1000
        X = []
        y = []
        
        np.random.seed(42)
        
        for _ in range(num_samples):
            # Generate random features
            features = np.random.rand(len(self.feature_names))
            
            # Generate risk score based on feature combinations
            risk_score = 0.3  # Base risk
            
            # Increase risk based on certain feature combinations
            if features[0] > 0.8:  # High activity frequency
                risk_score += 0.2
            if features[1] < 0.3:  # Low platform diversity
                risk_score += 0.1
            if features[2] > 0.7:  # High content similarity
                risk_score += 0.2
            if features[3] > 0.8:  # Irregular posting patterns
                risk_score += 0.3
            if features[4] > 0.6:  # High geolocation variability
                risk_score += 0.2
            if features[5] < 0.1:  # Very new account
                risk_score += 0.2
            if features[6] < 0.3:  # Low profile completeness
                risk_score += 0.1
            if features[7] > 0.7:  # High connection density
                risk_score += 0.1
            if features[8] < 0.3:  # Low behavioral consistency
                risk_score += 0.2
            
            # Cap risk score between 0 and 1
            risk_score = min(risk_score, 1.0)
            
            X.append(features)
            y.append(risk_score)
        
        return np.array(X), np.array(y)

    def assess_risk(self, digital_footprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level for a given digital footprint
        
        Args:
            digital_footprint: Dictionary containing digital footprint data
            
        Returns:
            Dict: Risk assessment results
        """
        if not self.is_trained:
            print("Warning: Models not trained. Training with synthetic data...")
            self.train_models()
        
        # Extract features
        features = self.extract_features(digital_footprint)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from both models
        anomaly_score = self.isolation_forest.decision_function(features_scaled)[0]
        risk_prediction = self.risk_classifier.predict_proba(features_scaled)[0]
        
        # Combine predictions to get final risk score
        # Anomaly score is normalized to 0-1 range
        normalized_anomaly = (anomaly_score + 0.5) / 1.0  # Adjust range based on isolation forest output
        normalized_anomaly = max(0, min(1, normalized_anomaly))  # Clamp to 0-1
        
        # Average the risk predictions
        avg_risk = np.mean(risk_prediction * np.arange(len(risk_prediction))) / 100.0
        avg_risk = max(0, min(1, avg_risk))
        
        # Final risk score (weighted combination)
        final_risk_score = 0.6 * avg_risk + 0.4 * normalized_anomaly
        final_risk_score = max(0, min(1, final_risk_score))
        
        # Determine risk level
        risk_level = self.determine_risk_level(final_risk_score)
        
        # Generate risk factors
        risk_factors = self.identify_risk_factors(digital_footprint, features[0])
        
        return {
            'risk_score': round(float(final_risk_score), 3),
            'risk_level': risk_level,
            'anomaly_score': round(float(normalized_anomaly), 3),
            'confidence': round(0.85, 2),  # Fixed confidence for demo
            'risk_factors': risk_factors,
            'recommendations': self.generate_recommendations(risk_level, risk_factors),
            'timestamp': datetime.now().isoformat()
        }

    def determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        for level, (min_val, max_val) in self.risk_thresholds.items():
            if min_val <= risk_score <= max_val:
                return level
        return 'high'  # Default to high if outside ranges

    def identify_risk_factors(self, digital_footprint: Dict[str, Any], features: np.ndarray) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        # Check each feature against thresholds
        if features[0] > 0.7:  # High activity frequency
            risk_factors.append("High posting frequency")
        if features[1] < 0.3:  # Low platform diversity
            risk_factors.append("Limited platform presence")
        if features[2] > 0.6:  # High content similarity
            risk_factors.append("Content duplication detected")
        if features[3] > 0.7:  # Irregular posting patterns
            risk_factors.append("Irregular posting schedule")
        if features[4] > 0.6:  # High geolocation variability
            risk_factors.append("Multiple geographic locations")
        if features[5] < 0.1:  # Very new account
            risk_factors.append("Recently created account")
        if features[6] < 0.4:  # Low profile completeness
            risk_factors.append("Incomplete profile information")
        if features[7] > 0.7:  # High connection density
            risk_factors.append("Unusual connection patterns")
        if features[8] < 0.4:  # Low behavioral consistency
            risk_factors.append("Behavioral inconsistency")
        
        # Additional checks based on footprint content
        for item in digital_footprint.get('digital_footprint', []):
            content = item.get('content', '').lower()
            if any(keyword in content for keyword in ['urgent', 'limited time', 'act now', 'click here']):
                risk_factors.append("Suspicious promotional content")
            if any(keyword in content for keyword in ['verify account', 'confirm identity', 'login now']):
                risk_factors.append("Phishing indicators detected")
        
        return risk_factors if risk_factors else ["No significant risk factors identified"]

    def generate_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk level and factors"""
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Verify account authenticity manually")
            if 'Phishing indicators detected' in risk_factors:
                recommendations.append("Block suspicious content immediately")
        elif risk_level == 'medium':
            recommendations.append("Continue monitoring")
            recommendations.append("Review account periodically")
        
        if 'Recently created account' in risk_factors:
            recommendations.append("Monitor for rapid growth patterns")
        
        if 'Multiple geographic locations' in risk_factors:
            recommendations.append("Investigate potential account compromise")
        
        if not recommendations:
            recommendations.append("Maintain current monitoring level")
        
        return recommendations


# Global instance
risk_analyzer = RiskAnalyzer()