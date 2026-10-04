# AI-Generated Content Detection Models - ACTUAL FILES

This directory contains information about the actual model files used by the deepfake detection system.

## Current Status

**No actual trained model files are present in this project.** The project contains model architectures that need to be trained or downloaded from external sources.

## Actual Model Requirements

### Image Detection Models
The project references these pre-trained models from Hugging Face:
- `capcheck/ai-image-detection` (primary)
- `jacoballessio/ai-image-detect-distilled` (alternative)
- `umm-maybe/AI-image-detector` (fallback)

These are dynamically downloaded when needed by `image_detector.py`.

### Computer Vision Models
- **CvT-13 Model**: Defined in `backend/app/ml/cvt_model.py`
  - Expected weights file: `models/cvt_13_deepfake_weights.h5`
  - Status: **NOT PRESENT** - needs training or external acquisition

### Audio Processing Models
- **AudioCNN**: Defined in `backend/app/ml/audio_detector.py`
  - Status: **NOT TRAINED** - architecture exists but needs training on audio deepfake datasets

### Video Analysis Models
- **TemporalLSTM**: Defined in `backend/app/ml/temporal_analyzer.py`
- **SyncNet**: Defined in `backend/app/ml/lipsync_detector.py`
  - Status: **NOT TRAINED** - architectures exist but need training on video datasets

## How Models Are Actually Used

The project implements a hybrid approach:

1. **Runtime Model Loading**: Image classification models are downloaded from Hugging Face when first used
2. **Local Model Definitions**: Custom architectures are defined in code but require training
3. **Fallback Mechanisms**: Mock predictions are used when real models aren't available

## To Get Actual Model Files

### Option 1: Train Your Own Models
```bash
# For CvT model
python train_cvt_model.py --dataset path/to/deepfake/images

# For Audio model  
python train_audio_model.py --dataset path/to/audio/deepfakes

# For Video models
python train_video_models.py --dataset path/to/video/deepfakes
```

### Option 2: Download Pre-trained Models
Some models can be downloaded from:
- Hugging Face Model Hub
- Academic research repositories
- Commercial deepfake detection services

### Option 3: Use Cloud APIs
Integrate with cloud-based deepfake detection services that provide pre-trained models.

## Current Working Implementation

The project currently works with:
- Heuristic-based detection (filename analysis, artifact detection)
- Mock model responses for demonstration
- External API calls to Hugging Face for some image classification

## Model File Locations

When actual models are obtained, they should be placed in:
```
models/
├── cvt_13_deepfake_weights.h5     # CvT model weights
├── audio_cnn_weights.pth          # Audio model weights  
├── temporal_lstm_weights.pth      # Video temporal model
└── syncnet_weights.pth            # Lip-sync detection model
```

## Verification

Run `python models/model_loader.py` to check the current status of actual model files.