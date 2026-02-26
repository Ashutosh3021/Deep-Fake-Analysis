"""
Query Assistant Module for DeepGuard AI
- Image input: YOLOv8 object detection + Gemini 2.5 Flash scene description
- Text input: Gemini 2.5 Flash free-form Q&A
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ─────────────────────────────────────────────────────────────
# YOLO loader (lazy – avoids import at module level)
# ─────────────────────────────────────────────────────────────
_yolo_model = None

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        import torch
        import functools
        from ultralytics import YOLO

        # PyTorch 2.6+ defaults torch.load to weights_only=True, which blocks
        # ultralytics custom classes. We patch torch.load to force weights_only=False
        # only while the official yolov8n.pt weights are being loaded.
        # This is safe: the file is from Ultralytics and we downloaded it ourselves.
        _original_load = torch.load

        @functools.wraps(_original_load)
        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        torch.load = _patched_load
        try:
            _yolo_model = YOLO("yolov8n.pt")
        finally:
            torch.load = _original_load   # always restore original

    return _yolo_model


# ─────────────────────────────────────────────────────────────
# Gemini client (lazy)
# ─────────────────────────────────────────────────────────────
_gemini_client = None

def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to the .env file in the project root."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai.GenerativeModel("gemini-2.5-flash")
    return _gemini_client


def _image_to_base64(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_type(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def analyze_image(filepath: str, user_query: str = "") -> dict:
    """
    Run YOLOv8 object detection on the image, then ask Gemini 2.5 Flash
    to describe the scene (optionally guided by the user's query).

    Returns:
        {
          "objects": [{"name": str, "confidence": float, "box": [x1,y1,x2,y2]}, …],
          "object_summary": str,        # comma-separated unique labels
          "gemini_answer": str,
          "mode": "image"
        }
    """
    # -- YOLO detection --
    yolo = _get_yolo()
    results = yolo(filepath, verbose=False)

    detected = []
    seen_labels = {}
    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls[0])]
            conf  = round(float(box.conf[0]) * 100, 1)
            coords = [round(v, 1) for v in box.xyxy[0].tolist()]
            detected.append({"name": label, "confidence": conf, "box": coords})
            seen_labels[label] = max(seen_labels.get(label, 0), conf)

    object_summary = ", ".join(
        f"{k} ({v}%)" for k, v in sorted(seen_labels.items(), key=lambda x: -x[1])
    ) if seen_labels else "no objects detected"

    # -- Gemini description --
    gemini = _get_gemini()
    import google.generativeai as genai

    img_data = _image_to_base64(filepath)
    mime     = _mime_type(filepath)

    if user_query.strip():
        prompt = (
            f"The user uploaded an image. "
            f"Object detection found: {object_summary}.\n\n"
            f"User's question: {user_query}\n\n"
            "Please answer the user's question about this image, "
            "referencing the detected objects where relevant."
        )
    else:
        prompt = (
            f"Object detection found these objects in this image: {object_summary}.\n\n"
            "Provide a concise, informative description of what is shown in this image."
        )

    response = gemini.generate_content([
        {"mime_type": mime, "data": img_data},
        prompt,
    ])

    return {
        "objects": detected,
        "object_summary": object_summary,
        "gemini_answer": response.text,
        "mode": "image",
    }


def answer_text_query(query: str) -> dict:
    """
    Send a free-form text query to Gemini 2.5 Flash and return the answer.

    Returns:
        {
          "gemini_answer": str,
          "mode": "text"
        }
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    gemini  = _get_gemini()
    system  = (
        "You are a helpful AI assistant integrated into DeepGuard AI, "
        "an advanced deepfake and AI-content detection platform. "
        "Answer user questions clearly and concisely."
    )
    response = gemini.generate_content(f"{system}\n\nUser: {query}")

    return {
        "gemini_answer": response.text,
        "mode": "text",
    }
