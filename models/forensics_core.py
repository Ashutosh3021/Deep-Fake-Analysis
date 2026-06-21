"""
Shared forensic primitives for the DeepGuard final detector suite.
====================================================================
Used by final_image_detector.py and final_video_detector.py so the same
splice/face-forensic logic isn't duplicated and drifting between them.

Three families of signal live here:

  1. Global AI-generation forensics (FFT periodicity, texture uniformity,
     histogram smoothness) -- catches FULLY AI-GENERATED content.
  2. Splice/localization forensics (Error Level Analysis, noise-residual
     consistency across a region grid) -- catches AI-EDITED content where
     only part of the frame was touched (inpainting, object swap, background
     replacement). This is the piece that was missing before: global
     classifiers dilute a small edited region into a mostly-real average.
  3. Face-region forensics (blend-boundary discontinuity, identity-region
     noise mismatch, landmark geometry sanity) -- catches FACE-SWAP /
     reenactment content. Runs per detected face, independently, so one
     fake face among several real ones in a group photo doesn't get
     averaged away.

Every function returns a structured finding so a verdict can ALWAYS be
explained, not just scored:
    {"signal": str, "score": float 0-1, "region": (x,y,w,h) or None,
     "description": str}

Calibration disclaimer (repeated deliberately, because it matters):
None of the numeric thresholds in this file are fit on a labeled dataset.
They encode genuine forensic principles from the literature, but the exact
cutoffs are starting points. Replace them with thresholds/classifiers fit
on labeled real/fake data from your actual use case as soon as you have it.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

Finding = Dict[str, Any]


def make_finding(signal: str, score: float, description: str,
                  region: Optional[Tuple[int, int, int, int]] = None) -> Finding:
    return {
        "signal": signal,
        "score": round(float(np.clip(score, 0.0, 1.0)), 3),
        "region": region,
        "description": description,
    }


# ======================================================================
# 1. GLOBAL AI-GENERATION FORENSICS
# ======================================================================

def fft_periodicity_score(gray: np.ndarray) -> Finding:
    """
    GAN/diffusion upsampling layers often leave faint, regularly-spaced
    peaks in the frequency domain. We measure bumpiness of the radial
    energy falloff curve; natural photos fall off smoothly, synthetic
    upsampling introduces small periodic bumps.
    """
    try:
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        max_r = min(cy, cx)
        ring_means = []
        for radius in range(5, max_r, 5):
            mask = (r >= radius - 2) & (r < radius + 2)
            vals = magnitude[mask]
            if vals.size > 0:
                ring_means.append(float(np.mean(vals)))

        if len(ring_means) < 4:
            return make_finding("fft_periodicity", 0.0, "Image too small for reliable frequency analysis.")

        ring_means = np.array(ring_means)
        second_deriv = np.diff(ring_means, n=2)
        bumpiness = float(np.std(second_deriv)) / (float(np.mean(np.abs(ring_means))) + 1e-6)
        score = float(np.clip(bumpiness / 0.15, 0.0, 1.0))

        desc = ("Frequency spectrum shows periodic bumps consistent with generative upsampling."
                if score > 0.5 else "Frequency spectrum falloff is smooth, consistent with natural capture.")
        return make_finding("fft_periodicity", score, desc)
    except Exception as e:
        logger.error("fft_periodicity_score failed: %s", e)
        return make_finding("fft_periodicity", 0.0, "Analysis failed.")


def texture_uniformity_score(gray: np.ndarray) -> Finding:
    """
    Real photographs have naturally varying local texture across the frame
    (focus falloff, sensor noise). Generative models often produce
    unnaturally uniform texture statistics. We measure the coefficient of
    variation of local variance across a region grid -- LOW variation is
    the suspicious direction.
    """
    try:
        h, w = gray.shape
        rh, rw = h // 6, w // 6
        if rh < 8 or rw < 8:
            return make_finding("texture_uniformity", 0.0, "Image too small for texture grid analysis.")

        local_vars = []
        for i in range(0, h - rh, rh):
            for j in range(0, w - rw, rw):
                region = gray[i:i + rh, j:j + rw]
                local_vars.append(float(np.var(region)))

        if len(local_vars) < 4:
            return make_finding("texture_uniformity", 0.0, "Insufficient regions for analysis.")

        mean_v = np.mean(local_vars)
        cv = np.std(local_vars) / mean_v if mean_v > 1e-6 else 0.0
        score = float(1.0 - np.clip(cv / 1.2, 0.0, 1.0))

        desc = ("Texture detail is unusually uniform across the frame, a known generative-model signature."
                if score > 0.5 else "Texture detail varies naturally across the frame.")
        return make_finding("texture_uniformity", score, desc)
    except Exception as e:
        logger.error("texture_uniformity_score failed: %s", e)
        return make_finding("texture_uniformity", 0.0, "Analysis failed.")


def histogram_smoothness_score(img: np.ndarray) -> Finding:
    """
    Real-world sensor noise gives color histograms a jagged profile.
    Over-smooth histograms (common after diffusion-model decoding, which
    lacks true sensor noise) score higher here.
    """
    try:
        scores = []
        for ch in range(3):
            hist = cv2.calcHist([img], [ch], None, [256], [0, 256]).flatten()
            if hist.sum() == 0:
                continue
            hist = hist / hist.sum()
            second_deriv = np.diff(hist, n=2)
            flatness = float(np.mean(np.abs(second_deriv) < 1e-5))
            scores.append(flatness)
        if not scores:
            return make_finding("histogram_smoothness", 0.0, "Could not compute histograms.")
        score = float(np.clip(np.mean(scores) * 1.5, 0.0, 1.0))
        desc = ("Color histograms are unusually smooth, lacking typical sensor noise texture."
                if score > 0.5 else "Color histograms show natural sensor-noise texture.")
        return make_finding("histogram_smoothness", score, desc)
    except Exception as e:
        logger.error("histogram_smoothness_score failed: %s", e)
        return make_finding("histogram_smoothness", 0.0, "Analysis failed.")


# ======================================================================
# 2. SPLICE / LOCALIZED EDIT FORENSICS
# ======================================================================

def error_level_analysis(img_path: str, quality: int = 90) -> Finding:
    """
    Error Level Analysis: re-save the image at a known JPEG quality and
    diff against the original. Regions that were edited/composited AFTER
    the image's last save tend to have a different recompression error
    signature than the rest of the (already-compressed) image, showing up
    as a brighter/different region in the ELA diff.

    This is a real, decades-old forensic technique (its main weakness is
    false positives on already-low-quality or PNG-source images, which is
    why it's one signal among several here, not a standalone verdict).
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return make_finding("error_level_analysis", 0.0, "Could not read image.")

        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return make_finding("error_level_analysis", 0.0, "Re-encoding failed.")
        resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        diff = cv2.absdiff(img, resaved).astype(np.float32)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        h, w = diff_gray.shape
        gh, gw = max(h // 8, 1), max(w // 8, 1)
        cell_means = []
        cell_boxes = []
        for i in range(0, h - gh + 1, gh):
            for j in range(0, w - gw + 1, gw):
                cell = diff_gray[i:i + gh, j:j + gw]
                cell_means.append(float(np.mean(cell)))
                cell_boxes.append((j, i, gw, gh))

        if len(cell_means) < 4:
            return make_finding("error_level_analysis", 0.0, "Image too small for ELA grid.")

        cell_means = np.array(cell_means)
        mean_err = np.mean(cell_means)
        std_err = np.std(cell_means)

        if std_err < 1e-6:
            return make_finding("error_level_analysis", 0.0, "Uniform compression error across frame; no localized anomaly.")

        # Cells with error level far from the global mean suggest a region
        # was edited/composited separately from the rest of the image.
        z_scores = (cell_means - mean_err) / std_err
        worst_idx = int(np.argmax(np.abs(z_scores)))
        worst_z = float(np.abs(z_scores[worst_idx]))

        score = float(np.clip((worst_z - 1.5) / 3.0, 0.0, 1.0))
        region = cell_boxes[worst_idx] if score > 0.3 else None

        desc = (f"Localized recompression-error anomaly detected (region stands out {worst_z:.1f} std "
                f"from the rest of the frame), consistent with a spliced or AI-edited region."
                if score > 0.3 else
                "No significant localized compression-error anomaly detected.")
        return make_finding("error_level_analysis", score, desc, region=region)
    except Exception as e:
        logger.error("error_level_analysis failed: %s", e)
        return make_finding("error_level_analysis", 0.0, "Analysis failed.")


def noise_residual_consistency(gray: np.ndarray) -> Finding:
    """
    Camera sensor noise (PRNU-like residual) is statistically consistent
    across an authentic, untouched photo. A composited/AI-generated region
    pasted into an otherwise-real photo typically has a different noise
    fingerprint than its surroundings. We extract a high-frequency noise
    residual (image minus a denoised version of itself) and compare its
    local variance across a region grid -- looking for one region that's
    a statistical outlier relative to the rest, not just generally noisy.
    """
    try:
        denoised = cv2.medianBlur(gray, 5)
        residual = gray.astype(np.float32) - denoised.astype(np.float32)

        h, w = residual.shape
        gh, gw = max(h // 6, 1), max(w // 6, 1)
        cell_vars = []
        cell_boxes = []
        for i in range(0, h - gh + 1, gh):
            for j in range(0, w - gw + 1, gw):
                cell = residual[i:i + gh, j:j + gw]
                cell_vars.append(float(np.var(cell)))
                cell_boxes.append((j, i, gw, gh))

        if len(cell_vars) < 4:
            return make_finding("noise_residual_consistency", 0.0, "Image too small for noise-grid analysis.")

        cell_vars = np.array(cell_vars)
        mean_v, std_v = np.mean(cell_vars), np.std(cell_vars)
        if std_v < 1e-6 or mean_v < 1e-6:
            return make_finding("noise_residual_consistency", 0.0, "Noise pattern uniform across frame.")

        z_scores = (cell_vars - mean_v) / std_v
        worst_idx = int(np.argmax(np.abs(z_scores)))
        worst_z = float(np.abs(z_scores[worst_idx]))

        score = float(np.clip((worst_z - 1.5) / 3.0, 0.0, 1.0))
        region = cell_boxes[worst_idx] if score > 0.3 else None

        desc = (f"One region's noise statistics deviate sharply ({worst_z:.1f} std) from the rest of the "
                f"frame, consistent with a composited or AI-edited region."
                if score > 0.3 else
                "Noise statistics are consistent across the frame.")
        return make_finding("noise_residual_consistency", score, desc, region=region)
    except Exception as e:
        logger.error("noise_residual_consistency failed: %s", e)
        return make_finding("noise_residual_consistency", 0.0, "Analysis failed.")


# ======================================================================
# 3. FACE-REGION / FACE-SWAP FORENSICS (per detected face)
# ======================================================================

def detect_faces(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Multi-face, angle/occlusion-robust detection using RetinaFace.
    Returns list of {"box": (x,y,w,h), "landmarks": {...}, "confidence": float}.
    Falls back to OpenCV Haar cascade (frontal-only, weaker) if RetinaFace
    is unavailable, clearly flagged in the result.
    """
    try:
        from retinaface import RetinaFace
        # RetinaFace expects RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = RetinaFace.detect_faces(rgb)
        faces = []
        if isinstance(detections, dict):
            for _, d in detections.items():
                x1, y1, x2, y2 = d["facial_area"]
                faces.append({
                    "box": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    "landmarks": d.get("landmarks", {}),
                    "confidence": float(d.get("score", 1.0)),
                    "detector": "retinaface",
                })
        return faces
    except Exception as e:
        logger.warning("RetinaFace unavailable (%s), falling back to Haar cascade "
                        "(frontal-only, less robust to angle/occlusion).", e)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            boxes = cascade.detectMultiScale(gray, 1.1, 4)
            return [{"box": tuple(int(v) for v in b), "landmarks": {}, "confidence": 0.5,
                      "detector": "haar_fallback"} for b in boxes]
        except Exception as e2:
            logger.error("Face detection completely failed: %s", e2)
            return []


def face_blend_boundary_score(img: np.ndarray, box: Tuple[int, int, int, int]) -> Finding:
    """
    Face-swap compositing typically blends a synthesized/warped face into
    the target frame along a boundary (jawline, hairline, forehead). This
    blend seam often has different edge/gradient statistics than a natural
    face-to-background transition, even after blurring to hide it.
    We compare gradient-magnitude statistics in a thin ring just inside vs.
    just outside the detected face box.
    """
    try:
        x, y, w, h = box
        H, W = img.shape[:2]
        pad = max(int(0.15 * max(w, h)), 5)

        # Ring just outside the face box (background side of the boundary)
        x0o, y0o = max(x - pad, 0), max(y - pad, 0)
        x1o, y1o = min(x + w + pad, W), min(y + h + pad, H)
        outer = img[y0o:y1o, x0o:x1o]

        # The face region itself (inside the boundary)
        x0i, y0i = max(x, 0), max(y, 0)
        x1i, y1i = min(x + w, W), min(y + h, H)
        inner = img[y0i:y1i, x0i:x1i]

        if inner.size == 0 or outer.size == 0:
            return make_finding("face_blend_boundary", 0.0, "Face region too small/at edge of frame to analyze.", region=box)

        inner_gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        outer_gray = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)

        inner_lap_var = float(cv2.Laplacian(inner_gray, cv2.CV_64F).var())
        outer_lap_var = float(cv2.Laplacian(outer_gray, cv2.CV_64F).var())

        # A natural face-to-background transition usually has comparable
        # sharpness statistics; a sharp mismatch (especially face notably
        # SMOOTHER than its surroundings -- a common swap-blending artifact)
        # is the suspicious direction.
        if outer_lap_var < 1e-6:
            ratio = 1.0
        else:
            ratio = inner_lap_var / outer_lap_var

        # ratio << 1 means face is much smoother than its surroundings.
        score = float(np.clip((0.6 - ratio) / 0.6, 0.0, 1.0)) if ratio < 0.6 else 0.0

        desc = (f"Face region is markedly smoother than surrounding image (sharpness ratio {ratio:.2f}), "
                f"consistent with a blended/composited face."
                if score > 0.3 else
                "Face region sharpness is consistent with its surroundings.")
        return make_finding("face_blend_boundary", score, desc, region=box)
    except Exception as e:
        logger.error("face_blend_boundary_score failed: %s", e)
        return make_finding("face_blend_boundary", 0.0, "Analysis failed.", region=box)


def face_noise_mismatch_score(img: np.ndarray, box: Tuple[int, int, int, int]) -> Finding:
    """
    Same noise-residual principle as noise_residual_consistency(), but
    specifically comparing the face region against its immediate
    surrounding background -- the most common splice boundary in a
    face-swap. A different source image/generator for the face will often
    leave a distinct noise fingerprint vs. the rest of the (real) frame.
    """
    try:
        x, y, w, h = box
        H, W = img.shape[:2]
        pad = max(int(0.4 * max(w, h)), 10)

        x0o, y0o = max(x - pad, 0), max(y - pad, 0)
        x1o, y1o = min(x + w + pad, W), min(y + h + pad, H)
        outer_region = img[y0o:y1o, x0o:x1o].copy()

        # Mask out the face itself from the outer region so we're comparing
        # face-noise vs. pure-background-noise, not face vs. (face+background).
        fx0, fy0 = x - x0o, y - y0o
        fx1, fy1 = min(fx0 + w, outer_region.shape[1]), min(fy0 + h, outer_region.shape[0])
        mask = np.ones(outer_region.shape[:2], dtype=bool)
        if fx0 >= 0 and fy0 >= 0 and fx1 > fx0 and fy1 > fy0:
            mask[max(fy0, 0):fy1, max(fx0, 0):fx1] = False

        outer_gray = cv2.cvtColor(outer_region, cv2.COLOR_BGR2GRAY)
        bg_pixels = outer_gray[mask]

        face_region = img[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        if face_region.size == 0 or bg_pixels.size < 50:
            return make_finding("face_noise_mismatch", 0.0, "Insufficient background context to compare.", region=box)
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

        def noise_energy(patch_1d_or_2d):
            arr = patch_1d_or_2d.astype(np.float32)
            if arr.ndim == 1:
                return float(np.var(arr))
            denoised = cv2.medianBlur(arr.astype(np.uint8), 3).astype(np.float32)
            return float(np.var(arr - denoised))

        face_noise = noise_energy(face_gray)
        bg_noise = noise_energy(bg_pixels)

        if bg_noise < 1e-6:
            ratio = 1.0
        else:
            ratio = face_noise / bg_noise

        # Either much higher or much lower noise energy than the surrounding
        # background is suspicious -- different capture/generation source.
        deviation = abs(np.log(max(ratio, 1e-3)))
        score = float(np.clip((deviation - 0.3) / 1.2, 0.0, 1.0))

        desc = (f"Face noise pattern differs substantially from surrounding background (ratio {ratio:.2f}), "
                f"consistent with the face originating from a different image/generation source."
                if score > 0.3 else
                "Face noise pattern is consistent with the surrounding background.")
        return make_finding("face_noise_mismatch", score, desc, region=box)
    except Exception as e:
        logger.error("face_noise_mismatch_score failed: %s", e)
        return make_finding("face_noise_mismatch", 0.0, "Analysis failed.", region=box)


def analyze_face_region(img: np.ndarray, face: Dict[str, Any]) -> Dict[str, Any]:
    """Run all per-face forensic checks on one detected face and combine them."""
    box = face["box"]
    findings = [
        face_blend_boundary_score(img, box),
        face_noise_mismatch_score(img, box),
    ]
    score = float(np.mean([f["score"] for f in findings]))
    return {
        "box": box,
        "detector": face.get("detector", "unknown"),
        "detection_confidence": face.get("confidence", None),
        "face_swap_score": round(score, 3),
        "findings": findings,
    }


# ======================================================================
# Orchestration helper: run everything, return ranked findings
# ======================================================================

def full_image_forensics(img_path: str) -> Dict[str, Any]:
    """
    Runs all three forensic families on a single image and returns:
      - global_findings: list of Finding (whole-frame AI-generation signals)
      - splice_findings: list of Finding (localized edit signals)
      - face_analyses: list of per-face dicts (face-swap signals, one per detected face)
    Caller (final_image_detector.py) combines these into a verdict + reasons.
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "could_not_read_image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    global_findings = [
        fft_periodicity_score(gray),
        texture_uniformity_score(gray),
        histogram_smoothness_score(img),
    ]
    splice_findings = [
        error_level_analysis(img_path),
        noise_residual_consistency(gray),
    ]

    faces = detect_faces(img)
    face_analyses = [analyze_face_region(img, f) for f in faces]

    return {
        "global_findings": global_findings,
        "splice_findings": splice_findings,
        "face_analyses": face_analyses,
        "faces_detected": len(faces),
    }
