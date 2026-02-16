"""
Model Loader for Deepfake Detection Project - ACTUAL MODEL FILES ONLY

This module loads the REAL model files used by the project for AI-generated content detection.
"""

import os
import sys
from pathlib import Path
import json

def get_actual_models_status():
    """Check the status of actual model files in the project"""
    models_dir = Path("models")
    
    # Check for actual model files that would be used
    expected_model_files = {
        "cvt_model": {
            "path": models_dir / "cvt_13_deepfake_weights.h5",
            "exists": False,
            "description": "CvT-13 model weights for image deepfake detection"
        },
        "image_classifier": {
            "path": None,  # Would be downloaded from Hugging Face
            "exists": False,
            "description": "Pre-trained image classification models from Hugging Face"
        },
        "audio_model": {
            "path": None,  # Would be trained or downloaded
            "exists": False,
            "description": "AudioCNN model for audio deepfake detection"
        },
        "video_models": {
            "path": None,  # Would be trained
            "exists": False,
            "description": "LSTM and SyncNet models for video analysis"
        }
    }
    
    # Check if CvT weights file exists
    cvt_weights_path = models_dir / "cvt_13_deepfake_weights.h5"
    expected_model_files["cvt_model"]["exists"] = cvt_weights_path.exists()
    expected_model_files["cvt_model"]["path"] = cvt_weights_path if cvt_weights_path.exists() else None
    
    return expected_model_files

def check_project_models_availability():
    """Check what models are actually available in the project"""
    print("Checking actual model availability...")
    print("=" * 50)
    
    # Check for the specific CvT model weights file mentioned in the code
    cvt_weights_path = Path("models") / "cvt_13_deepfake_weights.h5"
    
    if cvt_weights_path.exists():
        print(f"✓ Found CvT model weights: {cvt_weights_path}")
        print("  This is the actual trained model file referenced in cvt_model.py")
        return {"cvt_model": str(cvt_weights_path)}
    else:
        print("✗ No actual trained model files found in the project")
        print("  The project uses model architectures that need to be trained or downloaded")
        print("\nWhat was found:")
        print("- Model architectures defined in code (backend/app/ml/)")
        print("- References to external pre-trained models (Hugging Face)")
        print("- No locally stored trained model weights")
        return {}

def load_available_models():
    """Attempt to load any available actual models"""
    available_models = check_project_models_availability()
    
    loaded_models = {}
    
    # Try to load CvT model if weights exist
    if "cvt_model" in available_models:
        try:
            # This would load the actual TensorFlow/Keras model
            import tensorflow as tf
            model = tf.keras.models.load_model(available_models["cvt_model"])
            loaded_models["cvt_model"] = {
                "model": model,
                "type": "Convolutional Vision Transformer",
                "source": "Local weights file"
            }
            print("✓ Successfully loaded CvT model")
        except Exception as e:
            print(f"✗ Failed to load CvT model: {e}")
    
    return loaded_models

def get_model_requirements():
    """Get information about what models need to be obtained"""
    requirements = {
        "image_detection": {
            "models_needed": [
                "capcheck/ai-image-detection",
                "jacoballessio/ai-image-detect-distilled", 
                "umm-maybe/AI-image-detector"
            ],
            "source": "Hugging Face Transformers",
            "usage": "Used by image_detector.py for AI-generated image detection"
        },
        "cvt_model": {
            "models_needed": ["cvt_13_deepfake_weights.h5"],
            "source": "Local training or external source",
            "usage": "Used by cvt_model.py for deepfake detection"
        },
        "audio_model": {
            "models_needed": ["AudioCNN trained weights"],
            "source": "Requires training on audio deepfake datasets",
            "usage": "Used by audio_detector.py for synthetic audio detection"
        },
        "video_models": {
            "models_needed": ["TemporalLSTM weights", "SyncNet weights"],
            "source": "Requires training on video deepfake datasets", 
            "usage": "Used by video_detector.py and related modules"
        }
    }
    
    return requirements

if __name__ == "__main__":
    print("Deepfake Detection Project - ACTUAL MODEL FILES CHECK")
    print("=" * 60)
    
    # Check what actual models exist
    available_models = check_project_models_availability()
    
    print(f"\nSummary:")
    if available_models:
        print(f"Found {len(available_models)} actual model file(s)")
        for model_name, model_path in available_models.items():
            print(f"  - {model_name}: {model_path}")
    else:
        print("No actual trained model files found in the project.")
        print("The project contains model architectures that need training or external model downloads.")
    
    print(f"\nModel Requirements:")
    requirements = get_model_requirements()
    for category, req_info in requirements.items():
        print(f"\n{category.upper()}:")
        print(f"  Needed: {', '.join(req_info['models_needed'])}")
        print(f"  Source: {req_info['source']}")
        print(f"  Usage: {req_info['usage']}")