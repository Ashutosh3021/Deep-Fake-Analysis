"""
Digital Footprint Analyzer Module
Implements ML algorithms for analyzing and categorizing digital footprints
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import re
import warnings
from typing import Dict, List, Any, Tuple
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


class DigitalFootprintAnalyzer:
    def __init__(self):
        """
        Initialize the Digital Footprint Analyzer with ML models for pattern recognition
        """
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        self.cluster_model = None
        self.activity_patterns = {}
        self.platform_patterns = {}
        self.is_trained = False

    def extract_features(self, digital_footprint: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from digital footprint data for ML analysis
        
        Args:
            digital_footprint: Dictionary containing footprint data
            
        Returns:
            np.ndarray: Feature matrix for analysis
        """
        # Initialize feature lists
        features = {
            'activity_count': [],
            'platform_diversity': [],
            'content_length_avg': [],
            'posting_frequency': [],
            'content_uniqueness': [],
            'platform_engagement': [],
            'temporal_patterns': [],
            'content_similarity': [],
            'geographic_spread': [],
            'activity_type_diversity': []
        }
        
        footprint_items = digital_footprint.get('digital_footprint', [])
        
        if not footprint_items:
            # Return a default feature vector if no data
            return np.array([[0.0] * 10])
        
        # Calculate features
        features['activity_count'] = [len(footprint_items)]
        
        # Platform diversity
        platforms = [item.get('platform', 'unknown') for item in footprint_items]
        unique_platforms = len(set(platforms))
        features['platform_diversity'] = [unique_platforms / 10.0]  # Normalize
        
        # Content length average
        content_lengths = [len(item.get('content', '')) for item in footprint_items]
        avg_content_length = np.mean(content_lengths) if content_lengths else 0
        features['content_length_avg'] = [avg_content_length / 1000.0]  # Normalize
        
        # Posting frequency (posts per day)
        timestamps = []
        for item in footprint_items:
            ts_str = item.get('timestamp', '')
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    timestamps.append(dt)
                except ValueError:
                    continue
        
        if timestamps:
            time_span = (max(timestamps) - min(timestamps)).days
            time_span = max(time_span, 1)  # Avoid division by zero
            posting_freq = len(timestamps) / time_span
        else:
            posting_freq = 0
        features['posting_frequency'] = [posting_freq / 10.0]  # Normalize
        
        # Content uniqueness
        unique_content_ratio = len(set(content_lengths)) / max(len(content_lengths), 1)
        features['content_uniqueness'] = [unique_content_ratio]
        
        # Platform engagement
        platform_counts = Counter(platforms)
        engagement_score = np.std(list(platform_counts.values()))  # Higher std = more uneven engagement
        features['platform_engagement'] = [engagement_score / 10.0]  # Normalize
        
        # Temporal patterns (posting time distribution)
        if timestamps:
            hours = [dt.hour for dt in timestamps]
            hour_distribution = np.bincount(hours, minlength=24)
            temporal_pattern = np.std(hour_distribution) / 10.0  # Normalize
        else:
            temporal_pattern = 0
        features['temporal_patterns'] = [temporal_pattern]
        
        # Content similarity
        content_texts = [item.get('content', '') for item in footprint_items if item.get('content')]
        if len(content_texts) > 1:
            try:
                tfidf_matrix = self.text_vectorizer.fit_transform(content_texts)
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(tfidf_matrix)
                # Calculate average similarity
                triu_indices = np.triu_indices_from(similarities, k=1)
                if len(triu_indices[0]) > 0:
                    avg_similarity = np.mean(similarities[triu_indices])
                else:
                    avg_similarity = 0
            except:
                avg_similarity = 0
        else:
            avg_similarity = 0
        features['content_similarity'] = [avg_similarity]
        
        # Geographic spread
        locations = [item.get('location', 'unknown') for item in footprint_items if item.get('location')]
        unique_locations = len(set(loc for loc in locations if loc.lower() != 'unknown'))
        features['geographic_spread'] = [unique_locations / 20.0]  # Normalize
        
        # Activity type diversity
        activity_types = [item.get('type', 'unknown') for item in footprint_items]
        unique_types = len(set(activity_types))
        features['activity_type_diversity'] = [unique_types / 10.0]  # Normalize
        
        # Stack all features into a single array
        feature_matrix = np.column_stack([
            features['activity_count'],
            features['platform_diversity'],
            features['content_length_avg'],
            features['posting_frequency'],
            features['content_uniqueness'],
            features['platform_engagement'],
            features['temporal_patterns'],
            features['content_similarity'],
            features['geographic_spread'],
            features['activity_type_diversity']
        ])
        
        return feature_matrix

    def analyze_patterns(self, digital_footprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in digital footprint using ML algorithms
        
        Args:
            digital_footprint: Dictionary containing footprint data
            
        Returns:
            Dict: Pattern analysis results
        """
        # Extract features
        features = self.extract_features(digital_footprint)
        
        # Scale features
        if features.shape[0] > 1:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Perform clustering to identify patterns
        n_samples = features_scaled.shape[0]
        
        if n_samples > 1:
            # Determine optimal number of clusters using silhouette analysis
            best_k = 2
            best_score = -1
            
            for k in range(2, min(n_samples, 6)):  # Test 2-5 clusters
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(features_scaled)
                score = silhouette_score(features_scaled, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Apply clustering with optimal number of clusters
            self.cluster_model = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = self.cluster_model.fit_predict(features_scaled)
        else:
            cluster_labels = np.array([0])  # Single cluster for single sample
        
        # Analyze content patterns
        content_texts = [item.get('content', '') for item in digital_footprint.get('digital_footprint', [])]
        content_analysis = self.analyze_content_patterns(content_texts)
        
        # Analyze temporal patterns
        temporal_analysis = self.analyze_temporal_patterns(digital_footprint)
        
        # Analyze platform patterns
        platform_analysis = self.analyze_platform_patterns(digital_footprint)
        
        # Identify anomalies
        anomalies = self.detect_anomalies(features_scaled)
        
        return {
            'pattern_summary': self.generate_pattern_summary(
                digital_footprint, 
                cluster_labels, 
                content_analysis, 
                temporal_analysis
            ),
            'clusters': {
                'count': len(set(cluster_labels)) if len(cluster_labels) > 0 else 1,
                'labels': cluster_labels.tolist(),
                'silhouette_score': best_score if n_samples > 1 else 0
            },
            'content_analysis': content_analysis,
            'temporal_analysis': temporal_analysis,
            'platform_analysis': platform_analysis,
            'anomalies': anomalies,
            'risk_indicators': self.identify_risk_indicators(digital_footprint, anomalies),
            'timestamp': datetime.now().isoformat()
        }

    def analyze_content_patterns(self, content_texts: List[str]) -> Dict[str, Any]:
        """Analyze content patterns using NLP techniques"""
        if not content_texts:
            return {
                'dominant_topics': [],
                'sentiment_indicators': [],
                'language_patterns': [],
                'content_categories': []
            }
        
        # Use TF-IDF to identify dominant topics
        try:
            tfidf_matrix = self.text_vectorizer.fit_transform(content_texts)
            feature_names = self.text_vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores across all documents
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-10:][::-1]  # Top 10 features
            
            dominant_topics = [(feature_names[i], mean_scores[i]) for i in top_indices[:5]]
        except:
            dominant_topics = []
        
        # Identify sentiment indicators
        sentiment_indicators = []
        for text in content_texts:
            if any(word in text.lower() for word in ['urgent', 'important', 'critical', 'alert']):
                sentiment_indicators.append('Urgency indicators')
            if any(word in text.lower() for word in ['scam', 'fraud', 'fake', 'suspicious']):
                sentiment_indicators.append('Risk indicators')
            if any(word in text.lower() for word in ['love', 'happy', 'great', 'amazing']):
                sentiment_indicators.append('Positive sentiment')
        
        # Identify content categories
        content_categories = []
        for text in content_texts:
            if any(word in text.lower() for word in ['buy', 'sale', 'offer', 'discount']):
                content_categories.append('Commercial')
            elif any(word in text.lower() for word in ['news', 'update', 'information', 'report']):
                content_categories.append('Informational')
            elif any(word in text.lower() for word in ['personal', 'me', 'my', 'family']):
                content_categories.append('Personal')
        
        return {
            'dominant_topics': dominant_topics,
            'sentiment_indicators': list(set(sentiment_indicators)),
            'language_patterns': self.identify_language_patterns(content_texts),
            'content_categories': list(set(content_categories))
        }

    def identify_language_patterns(self, content_texts: List[str]) -> List[str]:
        """Identify language patterns in content"""
        patterns = []
        
        all_text = ' '.join(content_texts).lower()
        
        # Check for specific patterns
        if re.search(r'(click here|act now|limited time|offer expires)', all_text):
            patterns.append('Urgent call-to-action language')
        
        if re.search(r'(free|win|prize|congratulations)', all_text):
            patterns.append('Promotional language')
        
        if re.search(r'(verify|confirm|urgent|immediate)', all_text):
            patterns.append('Verification requests')
        
        if len(set(all_text.split())) / len(all_text.split()) < 0.5:  # High repetition
            patterns.append('Repetitive content')
        
        return patterns

    def analyze_temporal_patterns(self, digital_footprint: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in activity"""
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
            return {
                'posting_frequency': 'unknown',
                'active_hours': [],
                'suspicious_patterns': []
            }
        
        # Calculate posting frequency
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                          for i in range(len(timestamps)-1)]
            avg_interval = np.mean(time_diffs) / 3600  # Convert to hours
        else:
            avg_interval = 0
        
        # Identify active hours
        hours = [dt.hour for dt in timestamps]
        hour_counts = Counter(hours)
        active_hours = [hour for hour, count in hour_counts.most_common(3)]
        
        # Identify suspicious patterns
        suspicious_patterns = []
        if avg_interval < 3600:  # Less than 1 hour average
            suspicious_patterns.append('High frequency posting')
        
        if len(set(hours)) > 20:  # Active across most hours
            suspicious_patterns.append('24/7 activity pattern')
        
        # Check for burst activity
        if len(time_diffs) > 0:
            burst_threshold = np.percentile(time_diffs, 25)  # Lower quartile
            burst_count = sum(1 for diff in time_diffs if diff < burst_threshold/2)
            if burst_count / len(time_diffs) > 0.3:  # More than 30% are bursts
                suspicious_patterns.append('Activity burst patterns')
        
        return {
            'posting_frequency': f'{avg_interval:.2f} hours avg interval' if avg_interval > 0 else 'single post',
            'active_hours': active_hours,
            'suspicious_patterns': suspicious_patterns
        }

    def analyze_platform_patterns(self, digital_footprint: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across different platforms"""
        platform_data = {}
        all_items = digital_footprint.get('digital_footprint', [])
        
        for item in all_items:
            platform = item.get('platform', 'unknown')
            if platform not in platform_data:
                platform_data[platform] = {
                    'count': 0,
                    'content_types': [],
                    'locations': [],
                    'risk_tags': []
                }
            
            platform_data[platform]['count'] += 1
            platform_data[platform]['content_types'].append(item.get('type', 'unknown'))
            location = item.get('location', 'unknown')
            if location != 'unknown':
                platform_data[platform]['locations'].append(location)
            
            # Identify risk tags
            content = item.get('content', '').lower()
            risk_tags = []
            if any(word in content for word in ['urgent', 'limited', 'act now']):
                risk_tags.append('promotional')
            if any(word in content for word in ['verify', 'confirm', 'login']):
                risk_tags.append('verification')
            if any(word in content for word in ['click', 'link', 'website']):
                risk_tags.append('clickbait')
            
            platform_data[platform]['risk_tags'].extend(risk_tags)
        
        # Calculate platform diversity metrics
        total_platforms = len(platform_data)
        total_activities = len(all_items)
        
        platform_diversity = total_platforms / max(total_activities, 1)
        
        return {
            'platform_breakdown': platform_data,
            'diversity_score': platform_diversity,
            'dominant_platforms': sorted(
                [(p, data['count']) for p, data in platform_data.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
        }

    def detect_anomalies(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalies in the footprint data"""
        if features.shape[0] < 2:
            return []
        
        # Use Isolation Forest for anomaly detection
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)
        
        anomalies = []
        for i, label in enumerate(anomaly_labels):
            if label == -1:  # Anomaly detected
                anomalies.append({
                    'index': i,
                    'type': 'behavioral_anomaly',
                    'severity': 'high' if i < len(anomaly_labels) * 0.1 else 'medium'
                })
        
        return anomalies

    def identify_risk_indicators(self, digital_footprint: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> List[str]:
        """Identify risk indicators in the footprint"""
        risk_indicators = []
        
        # Check for anomalies
        if anomalies:
            risk_indicators.append(f"{len(anomalies)} behavioral anomalies detected")
        
        # Check for suspicious content
        for item in digital_footprint.get('digital_footprint', []):
            content = item.get('content', '').lower()
            if any(keyword in content for keyword in ['urgent', 'limited time', 'act now', 'click here']):
                risk_indicators.append("Urgent call-to-action content detected")
            if any(keyword in content for keyword in ['verify account', 'confirm identity', 'login now']):
                risk_indicators.append("Account verification requests detected")
            if any(keyword in content for keyword in ['free money', 'easy cash', 'make money fast']):
                risk_indicators.append("Financial scam indicators detected")
        
        # Check for temporal anomalies
        temporal_analysis = self.analyze_temporal_patterns(digital_footprint)
        if 'High frequency posting' in temporal_analysis.get('suspicious_patterns', []):
            risk_indicators.append("Unusually high posting frequency")
        
        if '24/7 activity pattern' in temporal_analysis.get('suspicious_patterns', []):
            risk_indicators.append("Non-human activity pattern")
        
        return list(set(risk_indicators))  # Remove duplicates

    def generate_pattern_summary(self, digital_footprint: Dict[str, Any], 
                                cluster_labels: np.ndarray, 
                                content_analysis: Dict[str, Any], 
                                temporal_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate a summary of identified patterns"""
        summary = {
            'profile_type': self.classify_profile_type(digital_footprint, content_analysis),
            'activity_level': self.assess_activity_level(digital_footprint),
            'content_nature': self.assess_content_nature(content_analysis),
            'temporal_characteristics': self.assess_temporal_characteristics(temporal_analysis),
            'pattern_confidence': self.calculate_pattern_confidence(digital_footprint)
        }
        return summary

    def classify_profile_type(self, digital_footprint: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """Classify the type of profile based on footprint"""
        content_categories = content_analysis.get('content_categories', [])
        platform_diversity = len(set(item.get('platform', '') for item in digital_footprint.get('digital_footprint', [])))
        
        if 'Commercial' in content_categories and platform_diversity >= 3:
            return "Commercial/Business"
        elif 'Personal' in content_categories and len(content_categories) <= 1:
            return "Personal/Individual"
        elif 'Informational' in content_categories:
            return "Informational/News"
        else:
            return "Mixed/Unknown"

    def assess_activity_level(self, digital_footprint: Dict[str, Any]) -> str:
        """Assess the level of activity"""
        activity_count = len(digital_footprint.get('digital_footprint', []))
        
        if activity_count == 0:
            return "Inactive"
        elif activity_count < 10:
            return "Low Activity"
        elif activity_count < 50:
            return "Moderate Activity"
        else:
            return "High Activity"

    def assess_content_nature(self, content_analysis: Dict[str, Any]) -> str:
        """Assess the nature of content"""
        sentiment_indicators = content_analysis.get('sentiment_indicators', [])
        content_categories = content_analysis.get('content_categories', [])
        
        if 'Risk indicators' in sentiment_indicators:
            return "Potentially Risky"
        elif 'Commercial' in content_categories:
            return "Commercial/Promotional"
        elif 'Informational' in content_categories:
            return "Informational/Educational"
        else:
            return "Mixed Nature"

    def assess_temporal_characteristics(self, temporal_analysis: Dict[str, Any]) -> str:
        """Assess temporal characteristics"""
        suspicious_patterns = temporal_analysis.get('suspicious_patterns', [])
        
        if 'High frequency posting' in suspicious_patterns:
            return "High-Frequency Automated"
        elif '24/7 activity pattern' in suspicious_patterns:
            return "Non-Standard Human Pattern"
        else:
            return "Standard Human Pattern"

    def calculate_pattern_confidence(self, digital_footprint: Dict[str, Any]) -> float:
        """Calculate confidence in pattern analysis"""
        activity_count = len(digital_footprint.get('digital_footprint', []))
        
        # Confidence increases with more data
        if activity_count == 0:
            return 0.1
        elif activity_count < 5:
            return 0.3
        elif activity_count < 20:
            return 0.6
        else:
            return 0.8


# Global instance
footprint_analyzer = DigitalFootprintAnalyzer()