# DeepGuard AI - Multi-Layer Forensic Deepfake Detection

A production-grade, **defense-in-depth** system for analyzing AI-generated and
manipulated content across images, video, audio, and text.

> **Important:** No single detector is foolproof. DeepGuard AI reports
> **evidence scores with an abstaining (Indeterminate) tier**, never a bare
> accusation. Treat every output as screening evidence, not authorship proof.

## Architecture

DeepGuard AI implements a **Layered Defense-in-Depth Architecture**:

1. **Layer 1 — Active Provenance & Custody**: C2PA / Content Credentials,
   SHA-256 integrity, EXIF/XMP/ICC metadata consistency.
2. **Layer 2 — Classical Forensic Invariants**: 2D DFT/DCT spectral anomalies,
   noise-residual variance, Error Level Analysis (ELA), sensor PRNU.
3. **Layer 3 — Modality-Specific Feature & Neural Estimators**: deep
   representations, face landmark kinematics, spectral feature modeling.
4. **Layer 4 — Cross-Modal Consistency**: audio-visual phoneme-to-lip
   synchronization, scene acoustic matching, text-image alignment.
5. **Layer 5 — Calibrated Evidence Fusion & Policy Engine**: Bayesian
   log-likelihood ratio fusion, quality-gated weighting, 4-tier verdict with an
   **Indeterminate** tier to prevent false-positive accusations.

```
                      DeepGuard AI
                           │
   ┌───────────┬───────────┼───────────┬───────────┐
   ▼           ▼           ▼           ▼           ▼
 IMAGE      VIDEO       AUDIO       TEXT       FUSION
 FORENSICS  FORENSICS   FORENSICS   FORENSICS   ENGINE
```

## Project Structure

```
Deep-Fake-Analysis/
├── models/
│   ├── forensics_core.py          # Shared forensic primitives
│   ├── final_image_detector.py    # Image manipulation detector
│   ├── final_video_detector.py    # Video manipulation detector
│   ├── final_audio_detector.py    # Audio deepfake detector
│   ├── final_text_detector.py     # AI-text detector
│   ├── temporal_analyzer.py       # Temporal consistency (LSTM/CNN)
│   ├── video_detector.py          # Legacy video detector
│   └── query_assistant.py         # YOLOv8 + Gemini query assistant
├── backend/
│   ├── api.py                     # Flask REST API + fusion engine
│   └── uploads/                   # Temporary upload directory
├── dashboard/
│   ├── index.html                 # Dashboard HTML
│   ├── styles.css                 # Dashboard styles
│   └── app.js                     # Dashboard JavaScript
├── plan.md                        # Implementation roadmap
├── session.log                    # Implementation session log
├── requirements.txt               # Python dependencies
├── start.bat / start.sh           # Startup scripts
└── yolov8n.pt                     # YOLOv8 weights (query assistant)
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Configure the Gemini query assistant:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

3. Navigate to the backend directory and run the Flask API:
```bash
cd backend
python api.py
```

4. Open your browser and go to:
```
http://localhost:5000
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/detect/image` | Detect deepfakes in images |
| `POST` | `/api/detect/audio` | Detect deepfakes in audio |
| `POST` | `/api/detect/video` | Detect deepfakes in videos |
| `POST` | `/api/detect/text` | Detect AI-generated text |
| `POST` | `/api/detect/auto` | Auto-detect file type and analyze |
| `POST` | `/api/detect/fusion` | Multi-modal calibrated fusion |
| `POST` | `/api/query` | YOLOv8 + Gemini query assistant |
| `GET`  | `/api/status` | Check API status and loaded models |

## 4-Tier Verdicts

| Tier | Name | Meaning |
|------|------|---------|
| 1 | Verified Provenance | Valid C2PA cryptographic signature confirms origin |
| 2 | Likely Authentic | `p_synthetic < 0.35` and no strong anomalies |
| 3 | Indeterminate | `0.35 <= p_synthetic <= 0.65` or conflicting signals |
| 4 | Likely Synthetic | `p_synthetic > 0.65` with ≥ 2 agreeing forensic signals |

The **Indeterminate** tier is essential: a system that always chooses real or
fake will appear decisive but will be unsafe.

## Dashboard Features

- Drag-and-drop file upload (images, video, audio)
- Direct text analysis
- 4-tier verdict display with indeterminate warnings
- Confidence visualization with animated meters
- Evidence breakdown per forensic family
- Suspicious region / time-slice localization
- Analysis history with local storage
- Query Assistant (YOLOv8 object detection + Gemini 2.5 Flash)
- Responsive design for mobile devices

## Important Caveats

- **No detector is foolproof.** Detectors fail on unseen generators,
  compression, re-encoding, paraphrasing, and adversarial attacks.
- **Treat scores as evidence, not proof.** Never make a high-impact decision
  from a single detector output.
- **Text detection is the weakest modality.** Short text, paraphrasing, and
  AI-assisted human writing are fundamentally ambiguous.
- **Absence of a watermark is not proof of human authorship.** The generator
  may not support watermarking, or the text may have been edited.
- **Provenance is stronger evidence than missing metadata.** A valid C2PA
  signature carries real weight; its absence does not.

## Requirements

- Python 3.8+
- Flask, Flask-CORS, python-dotenv
- OpenCV, Pillow, scikit-learn, numpy, scipy
- PyTorch, torchvision, transformers
- librosa, soundfile
- ultralytics (YOLOv8), google-generativeai (Gemini)
- mediapipe, retina-face (optional, for face analysis)
- gunicorn, eventlet (production deployment)

## License

MIT License
