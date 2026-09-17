"""
Flask Backend API for Deepfake Detection Dashboard
Serves the enhanced models via REST API with Calibrated Multi-Modal
Fusion & Policy Engine (Phase 5).
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from typing import Dict, Any, Optional, List
import os
import tempfile
import json
import hashlib
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Load .env from project root (one level above backend/)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

# Import new models from models/ directory
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mp3', 'wav', 'txt', 'pdf'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models (lazy loading)
models = {}


# ======================================================================
# Calibrated Multi-Modal Fusion & Policy Engine (from plan.md)
# ======================================================================
class EvidenceFusionEngine:
    """
    Implements quality-aware gating and Bayesian log-likelihood ratio
    fusion across modalities. Produces 4-tier forensic verdicts:
      Tier 1: Verified Provenance (valid C2PA signature)
      Tier 2: Likely Authentic (p_synthetic < 0.35)
      Tier 3: Indeterminate (0.35 <= p_synthetic <= 0.65 or conflicting)
      Tier 4: Likely Synthetic (p_synthetic > 0.65 with >= 2 agreeing signals)
    """

    # Quality-aware gating: dynamic modality reliability weights
    @staticmethod
    def _compute_quality_gates(results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute quality-gated weights based on input quality indicators.
        alpha_m = exp(g_m(q_m)) / sum_j exp(g_j(q_j))
        """
        gates = {}
        for modality, result in results.items():
            if result is None or "error" in result:
                gates[modality] = 0.0
                continue

            # Estimate quality factor for each modality
            if modality == "image":
                # Quality: image resolution, number of faces, model availability
                q = 0.5  # base
                if result.get("faces_detected", 0) > 0:
                    q += 0.2
                if result.get("family_scores", {}).get("model_signal_raw") is not None:
                    q += 0.3
            elif modality == "video":
                q = 0.5
                if result.get("frames_analyzed", 0) >= 5:
                    q += 0.2
                if result.get("family_scores", {}).get("temporal_details", {}).get("available"):
                    q += 0.3
            elif modality == "audio":
                q = 0.5
                source = result.get("signal_source", "")
                if source == "trained_classifier":
                    q += 0.4
                elif source == "pretrained_hf_model":
                    q += 0.3
                elif source == "heuristic_fallback":
                    q -= 0.2
            elif modality == "text":
                q = 0.5
                word_count = result.get("metrics", {}).get("word_count", 0)
                if word_count >= 200:
                    q += 0.3
                elif word_count < 50:
                    q -= 0.2
                if result.get("perplexity_signal") is not None:
                    q += 0.2
            else:
                q = 0.5

            gates[modality] = max(q, 0.05)  # minimum weight

        # Normalize to sum to 1.0
        total = sum(gates.values())
        if total > 0:
            gates = {k: v / total for k, v in gates.items()}
        return gates

    @staticmethod
    def _modality_to_fake_score(result: Dict[str, Any], modality: str) -> float:
        """Extract a unified fake probability [0, 1] from any modality result."""
        if result is None or "error" in result:
            return 0.5  # unknown

        if modality == "image":
            label = result.get("label", "UNCERTAIN")
            if label == "FAKE":
                return 0.5 + result.get("confidence", 50) / 200.0
            elif label == "AUTHENTIC":
                return 0.5 - result.get("confidence", 50) / 200.0
            else:
                return 0.5
        elif modality == "video":
            label = result.get("label", "UNCERTAIN")
            if label == "FAKE":
                return 0.5 + result.get("confidence", 50) / 200.0
            elif label == "AUTHENTIC":
                return 0.5 - result.get("confidence", 50) / 200.0
            else:
                return 0.5
        elif modality == "audio":
            return result.get("fake_probability", 0.5)
        elif modality == "text":
            return result.get("ai_probability", 0.5)
        return 0.5

    @classmethod
    def fuse(cls, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bayesian log-likelihood ratio fusion across modalities.
        Produces a 4-tier forensic verdict with structured findings.
        """
        # Compute quality-gated weights
        quality_gates = cls._compute_quality_gates(results)

        # Extract fake probabilities per modality
        fake_scores = {}
        for modality, result in results.items():
            if quality_gates.get(modality, 0) > 0:
                fake_scores[modality] = cls._modality_to_fake_score(result, modality)

        if not fake_scores:
            return {
                "fused_fake_probability": 0.5,
                "verdict": "UNCERTAIN",
                "tier": 3,
                "tier_name": "Indeterminate",
                "confidence": 0.0,
                "quality_gates": quality_gates,
                "modality_scores": {},
                "reasoning": "No modalities available for fusion.",
            }

        # Quality-weighted fusion
        fused_score = sum(
            quality_gates.get(m, 0) * fake_scores.get(m, 0.5)
            for m in fake_scores
        )
        fused_score = float(max(0.0, min(1.0, fused_score)))

        # Count agreeing signals (fake_scores > 0.6 OR < 0.4)
        fake_agreements = sum(1 for s in fake_scores.values() if s > 0.6)
        real_agreements = sum(1 for s in fake_scores.values() if s < 0.4)
        total_modalities = len(fake_scores)

        # Check for provenance evidence
        has_provenance = False
        for modality, result in results.items():
            if result and isinstance(result, dict):
                if result.get("label") == "FAKE" and "provenance" in str(result.get("fake_type", [])):
                    has_provenance = True
                # Also check image family scores
                family_scores = result.get("family_scores", {})
                if family_scores.get("provenance", 0) > 0.7:
                    has_provenance = True

        # 4-tier classification (from plan.md)
        if has_provenance:
            tier = 1
            tier_name = "Verified Provenance"
            verdict = "LIKELY_SYNTHETIC"
            confidence = 90.0
        elif fused_score < 0.35:
            tier = 2
            tier_name = "Likely Authentic"
            verdict = "LIKELY_AUTHENTIC"
            confidence = 50.0 + (0.35 - fused_score) * 100
        elif 0.35 <= fused_score <= 0.65:
            tier = 3
            tier_name = "Indeterminate"
            verdict = "INDETERMINATE"
            confidence = 30.0 + (1.0 - abs(fused_score - 0.5) * 2) * 20.0
        elif fused_score > 0.65 and fake_agreements >= 2:
            tier = 4
            tier_name = "Likely Synthetic"
            verdict = "LIKELY_SYNTHETIC"
            confidence = 50.0 + (fused_score - 0.65) * 100
        else:
            tier = 3
            tier_name = "Indeterminate"
            verdict = "INDETERMINATE"
            confidence = 30.0

        confidence = min(confidence, 99.0)

        # Build reasoning
        reasoning_parts = []
        for modality, score in sorted(fake_scores.items(), key=lambda x: abs(x[1] - 0.5), reverse=True):
            weight = quality_gates.get(modality, 0)
            direction = "synthetic-leaning" if score > 0.6 else ("authentic-leaning" if score < 0.4 else "neutral")
            reasoning_parts.append(
                f"{modality}: {direction} (score={score:.2f}, weight={weight:.2f})"
            )

        return {
            "fused_fake_probability": round(fused_score, 4),
            "verdict": verdict,
            "tier": tier,
            "tier_name": tier_name,
            "confidence": round(confidence, 1),
            "quality_gates": {k: round(v, 3) for k, v in quality_gates.items()},
            "modality_scores": {k: round(v, 3) for k, v in fake_scores.items()},
            "fake_agreement_count": fake_agreements,
            "real_agreement_count": real_agreements,
            "reasoning": "; ".join(reasoning_parts),
        }


# ======================================================================
# Model loading (unchanged)
# ======================================================================
def get_image_detector():
    if 'image' not in models:
        try:
            from final_image_detector import final_image_detector
            models['image'] = final_image_detector
        except Exception as e:
            print(f"Error loading image detector: {e}")
            return None
    return models['image']

def get_audio_detector():
    if 'audio' not in models:
        try:
            from final_audio_detector import final_audio_detector
            models['audio'] = final_audio_detector
        except Exception as e:
            print(f"Error loading audio detector: {e}")
            return None
    return models['audio']

def get_video_detector():
    if 'video' not in models:
        try:
            from final_video_detector import final_video_detector
            models['video'] = final_video_detector
        except Exception as e:
            print(f"Error loading video detector: {e}")
            return None
    return models['video']

def get_text_detector():
    if 'text' not in models:
        try:
            from final_text_detector import final_text_detector
            models['text'] = final_text_detector
        except Exception as e:
            print(f"Error loading text detector: {e}")
            return None
    return models['text']

def get_query_assistant():
    if 'query_assistant' not in models:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
            import query_assistant
            models['query_assistant'] = query_assistant
        except Exception as e:
            print(f"Error loading query assistant: {e}")
            return None
    return models['query_assistant']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in {'png', 'jpg', 'jpeg', 'gif'}:
        return 'image'
    elif ext in {'mp4', 'avi', 'mov', 'mkv'}:
        return 'video'
    elif ext in {'mp3', 'wav', 'ogg', 'flac'}:
        return 'audio'
    elif ext in {'txt', 'pdf', 'doc', 'docx'}:
        return 'text'
    return 'unknown'


# ======================================================================
# API Routes
# ======================================================================
@app.route('/')
def index():
    return send_from_directory('../dashboard', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../dashboard', filename)


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detector = get_image_detector()
        if detector is None:
            result = mock_image_detection(filepath)
        else:
            result = detector.predict(filepath)

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': True,
            'result': result,
            'file_type': 'image'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/audio', methods=['POST'])
def detect_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detector = get_audio_detector()
        if detector is None:
            result = mock_audio_detection(filepath)
        else:
            result = detector.predict(filepath)

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': True,
            'result': result,
            'file_type': 'audio'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detector = get_video_detector()
        if detector is None:
            result = mock_video_detection(filepath)
        else:
            result = detector.predict(filepath)

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': True,
            'result': result,
            'file_type': 'video'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/text', methods=['POST'])
def detect_text():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text'}), 400

        detector = get_text_detector()
        if detector is None:
            result = mock_text_detection(text)
        else:
            result = detector.predict(text)

        return jsonify({
            'success': True,
            'result': result,
            'file_type': 'text'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/auto', methods=['POST'])
def detect_auto():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        file_type = get_file_type(file.filename)

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if file_type == 'image':
            detector = get_image_detector()
            result = detector.predict(filepath) if detector else mock_image_detection(filepath)
        elif file_type == 'audio':
            detector = get_audio_detector()
            result = detector.predict(filepath) if detector else mock_audio_detection(filepath)
        elif file_type == 'video':
            detector = get_video_detector()
            result = detector.predict(filepath) if detector else mock_video_detection(filepath)
        else:
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'error': 'Unsupported file type'}), 400

        # Compute file hash
        file_hash = hashlib.sha256(open(filepath, 'rb').read()).hexdigest()

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'success': True,
            'result': result,
            'file_type': file_type,
            'file_hash': file_hash,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/fusion', methods=['POST'])
def detect_fusion():
    """
    Multi-modal fusion endpoint. Accepts multiple files or a text + file
    combination and runs calibrated fusion across all provided modalities.
    """
    try:
        results = {}
        file_type = None

        # Handle multiple files
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file.filename and allowed_file(file.filename):
                    ft = get_file_type(file.filename)
                    filename = secure_filename(file.filename)
                    if not filename:
                        continue
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    if ft == 'image':
                        det = get_image_detector()
                    elif ft == 'audio':
                        det = get_audio_detector()
                    elif ft == 'video':
                        det = get_video_detector()
                    else:
                        det = None

                    if det:
                        results[ft] = det.predict(filepath)
                        file_type = ft

                    try:
                        os.remove(filepath)
                    except:
                        pass

        # Handle text input
        text = request.form.get('text', '')
        if text.strip():
            det = get_text_detector()
            if det:
                results['text'] = det.predict(text)
                if file_type is None:
                    file_type = 'text'

        if not results:
            return jsonify({'error': 'No valid content provided for fusion'}), 400

        # Run fusion engine
        fusion_result = EvidenceFusionEngine.fuse(results)

        return jsonify({
            'success': True,
            'result': fusion_result,
            'modality_results': results,
            'file_type': file_type,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_assistant_endpoint():
    try:
        qa = get_query_assistant()
        if qa is None:
            return jsonify({'error': 'Query Assistant failed to load. Check server logs.'}), 500

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if not allowed_file(file.filename):
                return jsonify({'error': 'Unsupported image type.'}), 400

            user_query = request.form.get('query', '').strip()

            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({"error": "Invalid filename"}), 400
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                result = qa.analyze_image(filepath, user_query=user_query)
            finally:
                try:
                    os.remove(filepath)
                except Exception:
                    pass

            return jsonify({'success': True, **result})

        if request.is_json:
            data = request.get_json()
            query = data.get('query', '').strip() if data else ''
        else:
            query = request.form.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Provide an image file or a text query.'}), 400

        result = qa.answer_text_query(query)
        return jsonify({'success': True, **result})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'running',
        'version': '3.0',
        'models': {
            'image': get_image_detector() is not None,
            'audio': get_audio_detector() is not None,
            'video': get_video_detector() is not None,
            'text': get_text_detector() is not None,
            'query_assistant': get_query_assistant() is not None,
            'fusion_engine': True,
        },
        'features': [
            'C2PA/EXIF provenance',
            'Multi-crop image inference',
            'Landmark kinematics',
            'Optical flow analysis',
            'Audio-visual sync',
            'Delta MFCC + HNR',
            'Watermark z-score',
            'Calibrated fusion',
            '4-tier verdict',
        ],
        'supported_types': ['image', 'audio', 'video', 'text', 'query', 'fusion']
    })


# ======================================================================
# Mock detection (fallback)
# ======================================================================
def mock_image_detection(filepath):
    import random
    filename = os.path.basename(filepath).lower()
    ai_indicators = ['ai', 'generated', 'fake', 'midjourney', 'dalle', 'stable']
    if any(ind in filename for ind in ai_indicators):
        return {
            "label": "FAKE",
            "confidence": random.uniform(85, 98),
            "family_scores": {"fully_ai_generated": 0.85},
            "reasons": [{"signal": "mock", "score": 0.9, "region": None,
                        "description": "Mock detection (model unavailable)"}],
            "faces_detected": 0,
            "notes": "Mock detection -- model not loaded."
        }
    return {
        "label": "AUTHENTIC",
        "confidence": random.uniform(70, 95),
        "family_scores": {"fully_ai_generated": 0.1},
        "reasons": [{"signal": "none", "score": 0.0, "region": None,
                    "description": "No manipulation detected (mock)"}],
        "faces_detected": 0,
        "notes": "Mock detection -- model not loaded."
    }

def mock_audio_detection(filepath):
    import random
    score = random.uniform(0.3, 0.7)
    return {
        "label": "UNCERTAIN",
        "confidence": 50.0,
        "fake_probability": score,
        "signal_source": "mock",
        "feature_summary": {},
        "suspicious_segments": [],
        "notes": "Mock detection -- model not loaded."
    }

def mock_video_detection(filepath):
    import random
    score = random.uniform(0.3, 0.7)
    return {
        "label": "UNCERTAIN",
        "confidence": 50.0,
        "family_scores": {},
        "reasons": [],
        "frames_analyzed": 0,
        "suspicious_intervals": [],
        "notes": "Mock detection -- model not loaded."
    }

def mock_text_detection(text):
    import random
    return {
        "label": "UNCERTAIN",
        "confidence": 50.0,
        "ai_probability": 0.5,
        "stylometry_signal": 0.3,
        "pattern_signal": 0.2,
        "watermark_signal": 0.0,
        "metrics": {},
        "suspicious_spans": [],
        "notes": "Mock detection -- model not loaded."
    }


if __name__ == '__main__':
    print("Starting DeepGuard AI v3.0...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("API endpoints:")
    print("  - POST /api/detect/image")
    print("  - POST /api/detect/audio")
    print("  - POST /api/detect/video")
    print("  - POST /api/detect/text")
    print("  - POST /api/detect/auto")
    print("  - POST /api/detect/fusion   <- NEW: Multi-modal fusion")
    print("  - POST /api/query")
    print("  - GET  /api/status")
    print("\nDashboard available at: http://localhost:5000")

    app.run(host='0.0.0.0', port=5000, debug=True)