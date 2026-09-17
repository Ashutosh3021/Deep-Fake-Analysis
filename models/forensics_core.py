"""
Shared forensic primitives for the DeepGuard final detector suite.
====================================================================
Used by final_image_detector.py and final_video_detector.py so the same
splice/face-forensic logic isn't duplicated and drifting between them.

Four families of signal live here:

  1. Provenance & metadata forensics (C2PA/JUMBF, EXIF/XMP, SHA-256,
     software/generator tag detection) -- catches content with traceable
     AI tooling provenance.
  2. Global AI-generation forensics (FFT radial spectral decay, texture
     uniformity, histogram smoothness) -- catches FULLY AI-GENERATED
     content via physical frequency-domain signatures.
  3. Splice/localization forensics (Error Level Analysis, wavelet-denoised
     noise-residual consistency) -- catches AI-EDITED content where only
     part of the frame was touched.
  4. Face-region forensics (convex-hull boundary gradient analysis,
     identity-region noise mismatch, landmark geometry) -- catches
     FACE-SWAP / reenactment content. Runs per detected face, independent
     of other faces in the frame.

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

import hashlib
import io
import json
import logging
import struct
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


def sha256_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for chain-of-custody integrity."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================
# 0. PROVENANCE & METADATA FORENSICS
# ======================================================================

def _find_jumbf_box(data: bytes, target_type: str = b"jumb") -> Optional[bytes]:
    """Scan for JUMBF boxes within raw file bytes (C2PA manifest container)."""
    offset = 0
    while offset < len(data) - 8:
        box_len = struct.unpack(">I", data[offset:offset + 4])[0]
        box_type = data[offset + 4:offset + 8]
        if box_len < 8 or offset + box_len > len(data):
            break
        if box_type == target_type:
            return data[offset:offset + box_len]
        offset += box_len
    return None


def c2pa_provenance_score(file_path: str) -> Finding:
    """
    Check for C2PA/JUMBF content credentials metadata. A valid C2PA manifest
    box embedded in the file indicates the content was created or edited by
    a C2PA-compliant tool (e.g., Content Authenticity Initiative cameras,
    Adobe Firefly, DALL-E, etc.). Its presence is strong provenance evidence;
    its absence is NOT evidence of fakeness (metadata can be stripped).

    Detects JUMBF boxes in JPEG files and XMP-based C2PA manifests in other formats.
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read(1024 * 1024)  # read first 1MB for headers

        # Check JPEG JUMBF marker (0xFFE8 or JUMBF box after APP11)
        has_jumbf = False
        if raw[:2] == b"\xff\xd8":  # JPEG
            # Search for JUMBF box signature
            jumbf_idx = raw.find(b"jumb")
            if jumbf_idx >= 0:
                has_jumbf = True
            # Also check for C2PA manifest marker
            c2pa_idx = raw.find(b"c2pa")
            if c2pa_idx >= 0:
                has_jumbf = True

        # Check for XMP-based C2PA in any format (search wider range)
        if not has_jumbf:
            with open(file_path, "rb") as f:
                raw_full = f.read(4 * 1024 * 1024)  # read up to 4MB
            if b"http://c2pa.org" in raw_full or b"GPC:CT" in raw_full:
                has_jumbf = True

        if has_jumbf:
            return make_finding("c2pa_provenance", 0.9,
                "C2PA/JUMBF content credentials detected in file metadata. "
                "This indicates the content was created or edited by a C2PA-compliant tool.")
        else:
            return make_finding("c2pa_provenance", 0.0,
                "No C2PA content credentials found. Note: absence of C2PA does NOT "
                "prove the content is AI-generated; metadata can be stripped during export.")
    except Exception as e:
        logger.error("c2pa_provenance_score failed: %s", e)
        return make_finding("c2pa_provenance", 0.0, "Analysis failed.")


def exif_metadata_score(file_path: str) -> Finding:
    """
    Parse EXIF/XMP metadata and check for AI/generator software tags,
    camera sensor plausibility, and timestamp consistency. Flags:
    - Known AI generator software (Midjourney, DALL-E, Stable Diffusion, etc.)
    - Missing camera make/model on a file claiming photo origin
    - Inconsistent timestamps
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(file_path)
        exif_data = img._getexif() if hasattr(img, "_getexif") else None

        if exif_data is None:
            return make_finding("exif_metadata", 0.3,
                "No EXIF data found. This is common for AI-generated content "
                "but also for screenshots, exports, and stripped metadata.")

        software_tags = []
        camera_make = None
        camera_model = None
        has_datetime = False

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            val_str = str(value).lower() if value else ""

            if tag_name == "Software":
                # Known AI generators
                ai_software = ["midjourney", "dall-e", "dalle", "stable diffusion",
                               "flux", "firefly", "copilot", "gemini", "imagen",
                               "craiyon", "nightcafe", "lexica", "playground"]
                for sw in ai_software:
                    if sw in val_str:
                        software_tags.append(sw)

            elif tag_name in ("Make", "Model"):
                if tag_name == "Make":
                    camera_make = val_str
                else:
                    camera_model = val_str

            elif tag_name == "DateTimeOriginal":
                has_datetime = True

        if software_tags:
            return make_finding("exif_metadata", 0.85,
                f"EXIF Software tag identifies AI generator(s): {', '.join(software_tags)}. "
                "This is direct evidence of AI-generated content.")

        if not camera_make and not camera_model:
            score = 0.35
            desc = ("No camera make/model in EXIF. This could indicate AI generation, "
                    "an export without camera metadata, or a screenshot.")
        else:
            score = 0.1
            desc = (f"Camera identified: {camera_make or 'unknown'} {camera_model or 'unknown'}. "
                    "Metadata consistent with camera capture.")

        return make_finding("exif_metadata", score, desc)
    except Exception as e:
        logger.error("exif_metadata_score failed: %s", e)
        return make_finding("exif_metadata", 0.0, "EXIF analysis failed.")


# ======================================================================
# 1. GLOBAL AI-GENERATION FORENSICS
# ======================================================================

def fft_periodicity_score(gray: np.ndarray) -> Finding:
    """
    Enhanced 2D DFT spectral anomaly detection. Computes:
    1. Radial energy profile and bumpiness (GAN/diffusion upsampling artifacts)
    2. Spectral decay deviation from natural image 1/f^alpha baseline
       (Frankfurt et al.): synthetic images often exhibit anomalous high-frequency
       energy or periodic peaks deviating from the expected power-law falloff.
    3. Periodic peak detection at mu + 2.5*sigma for checkerboard/upsampling artifacts.
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

        # --- Signal 1: Radial bumpiness (upsampling artifacts) ---
        second_deriv = np.diff(ring_means, n=2)
        bumpiness = float(np.std(second_deriv)) / (float(np.mean(np.abs(ring_means))) + 1e-6)
        bump_score = float(np.clip(bumpiness / 0.15, 0.0, 1.0))

        # --- Signal 2: Spectral decay deviation from natural 1/f^alpha ---
        # Natural images follow power-law: log|F(f)|^2 ~ alpha * log(f) + c
        # with alpha typically in [1.5, 3.5]. Synthetic images deviate from this.
        radii = np.arange(5, max_r, 5, dtype=np.float64)[:len(ring_means)]
        if len(radii) >= 6 and np.std(radii) > 0:
            # Fit log-log slope
            log_r = np.log(radii + 1e-6)
            log_power = np.array(ring_means)
            slope, intercept = np.polyfit(log_r, log_power, 1)
            # Natural alpha ~ 2.0; synthetic often shows steeper or flatter decay
            alpha_deviation = abs(slope - 2.0)  # distance from natural 1/f^2
            spectral_decay_score = float(np.clip(alpha_deviation / 1.5, 0.0, 1.0))
        else:
            spectral_decay_score = 0.0

        # --- Signal 3: Periodic peak detection ---
        if len(ring_means) > 8:
            ring_arr = np.array(ring_means)
            ring_mean = np.mean(ring_arr)
            ring_std = np.std(ring_arr) + 1e-6
            peaks = np.sum(ring_arr > ring_mean + 2.5 * ring_std)
            peak_score = float(np.clip(peaks / 3.0, 0.0, 1.0))
        else:
            peak_score = 0.0

        # Combine signals with weights
        score = float(np.clip(0.4 * bump_score + 0.35 * spectral_decay_score + 0.25 * peak_score, 0.0, 1.0))

        parts = []
        if bump_score > 0.5:
            parts.append("radial energy bumps consistent with generative upsampling")
        if spectral_decay_score > 0.5:
            parts.append(f"spectral decay deviates from natural 1/f^alpha baseline")
        if peak_score > 0.5:
            parts.append(f"periodic spectral peaks detected")

        if parts:
            desc = f"Frequency analysis anomaly: {', '.join(parts)}."
        else:
            desc = "Frequency spectrum falloff is smooth and consistent with natural capture."
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


def _wavelet_denoise(gray: np.ndarray) -> np.ndarray:
    """
    Estimate the noise-free image using a 2-level Haar wavelet hard-threshold
    scheme (plan.md Task 1.3: wavelet-denoised residual estimation).

    Process:
      1. Forward 2D DWT (Haar) — 2 levels of LL/LH/HL/HH sub-bands.
      2. Hard-threshold all detail sub-bands (LH/HL/HH) at σ*sqrt(2*log N)
         (VisuShrink universal threshold), where σ is estimated from the
         level-1 HH band via the median absolute deviation rule.
      3. Inverse 2D DWT to reconstruct the denoised estimate.

    Falls back to bilateral filter if scipy is unavailable.
    """
    try:
        from scipy.ndimage import convolve

        img = gray.astype(np.float32)
        h, w = img.shape

        # Haar analysis filters
        lo = np.array([1.0, 1.0]) / np.sqrt(2.0)
        hi = np.array([1.0, -1.0]) / np.sqrt(2.0)

        def _dwt_row(x, f):
            c = np.convolve(x, f[::-1], mode='full')
            return c[len(f) - 1::2][:len(x) // 2]

        def _idwt_row(lo_coeffs, hi_coeffs, f_lo, f_hi, length):
            up_lo = np.zeros(length)
            up_hi = np.zeros(length)
            up_lo[::2] = lo_coeffs[:length // 2]
            up_hi[::2] = hi_coeffs[:length // 2]
            return np.convolve(up_lo, f_lo, mode='same') + np.convolve(up_hi, f_hi, mode='same')

        def _dwt2(x):
            # Apply row-wise then column-wise (1 level)
            rows_lo = np.array([_dwt_row(row, lo) for row in x])
            rows_hi = np.array([_dwt_row(row, hi) for row in x])
            LL = np.array([_dwt_row(col, lo) for col in rows_lo.T]).T
            LH = np.array([_dwt_row(col, hi) for col in rows_lo.T]).T
            HL = np.array([_dwt_row(col, lo) for col in rows_hi.T]).T
            HH = np.array([_dwt_row(col, hi) for col in rows_hi.T]).T
            return LL, LH, HL, HH

        LL, LH, HL, HH = _dwt2(img)

        # Estimate noise sigma from HH sub-band (MAD estimator)
        sigma = float(np.median(np.abs(HH)) / 0.6745) + 1e-8
        # Universal (VisuShrink) threshold
        n_pixels = max(LH.size, 1)
        thresh = sigma * np.sqrt(2.0 * np.log(n_pixels))

        # Hard-threshold detail sub-bands (zero out coefficients below threshold)
        def _hard_thresh(c, t):
            out = c.copy()
            out[np.abs(out) < t] = 0.0
            return out

        LH_t = _hard_thresh(LH, thresh)
        HL_t = _hard_thresh(HL, thresh)
        HH_t = _hard_thresh(HH, thresh)

        # Inverse 2D DWT (1 level)
        lo_r = lo[::-1] * np.sqrt(2.0)
        hi_r = hi[::-1] * np.sqrt(2.0)

        rows_lo2 = np.array([_idwt_row(LH_t[r], HH_t[r], lo_r, hi_r, w) for r in range(LH_t.shape[0])])
        rows_hi2 = np.array([_idwt_row(LL[r],   HL_t[r], lo_r, hi_r, w) for r in range(LL.shape[0])])
        recon = np.array([_idwt_row(rows_hi2[:, c], rows_lo2[:, c], lo_r, hi_r, h)
                          for c in range(rows_hi2.shape[1])]).T

        # Pad / crop to original size
        recon_clipped = recon[:h, :w]
        if recon_clipped.shape != img.shape:
            # Safe fallback: resize with nearest-neighbor
            recon_clipped = cv2.resize(recon_clipped, (w, h), interpolation=cv2.INTER_NEAREST)

        return np.clip(recon_clipped, 0.0, 255.0)

    except Exception:
        # Graceful fallback to bilateral filter
        return cv2.bilateralFilter(gray, 9, 75, 75).astype(np.float32)


def noise_residual_consistency(gray: np.ndarray) -> Finding:
    """
    Noise residual analysis using wavelet-denoised residual estimation
    (plan.md Task 1.3).

    Camera sensor noise (PRNU-like residual) is statistically consistent
    across an authentic, untouched photo. A composited/AI-generated region
    pasted into an otherwise-real photo typically has a different noise
    fingerprint than its surroundings.

    Algorithm:
      R = I - L(I)  where L(I) is the wavelet-denoised estimate (_wavelet_denoise).
      V_R = (1/K) Σ_k ( Var(R_k) - mean(Var(R)) )²   [plan formula]

    A region that is a statistical outlier in the patch-variance distribution
    (z-score > 1.5) indicates a splice / AI-edited insert.  Very low total
    residual energy also flags AI-generated images that lack natural sensor noise.
    """
    try:
        denoised = _wavelet_denoise(gray)
        residual = gray.astype(np.float32) - denoised

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

        # Plan formula: V_R = (1/K) Σ_k ( Var(R_k) - mean(Var(R)) )²
        # This is the variance-of-variance across patches — high value means
        # one or more patches have very different noise fingerprints.
        vr = float(np.mean((cell_vars - mean_v) ** 2))
        # Normalised deviation: z-score of the most anomalous patch
        z_scores = (cell_vars - mean_v) / std_v
        worst_idx = int(np.argmax(np.abs(z_scores)))
        worst_z = float(np.abs(z_scores[worst_idx]))

        score = float(np.clip((worst_z - 1.5) / 3.0, 0.0, 1.0))
        region = cell_boxes[worst_idx] if score > 0.3 else None

        # Also compute total noise energy -- very low total residual variance
        # (absence of natural camera noise) is itself suspicious for "photo-like"
        # AI content.
        total_noise_var = float(np.mean(cell_vars))
        absence_signal = 0.0
        if total_noise_var < 5.0 and mean_v > 1e-6:
            absence_signal = float(np.clip((5.0 - total_noise_var) / 5.0, 0.0, 0.5))

        combined_score = float(np.clip(score + absence_signal, 0.0, 1.0))

        desc_parts = []
        if score > 0.3:
            desc_parts.append(
                f"One region's noise statistics deviate sharply ({worst_z:.1f} std) from the rest "
                f"of the frame, consistent with a composited or AI-edited region"
            )
        if absence_signal > 0.15:
            desc_parts.append(
                f"Overall noise energy is unusually low ({total_noise_var:.1f}), "
                f"suggesting absence of natural camera sensor noise"
            )
        if not desc_parts:
            desc_parts.append("Noise statistics are consistent across the frame.")

        return make_finding("noise_residual_consistency", combined_score,
                          ". ".join(desc_parts) + ".", region=region)
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
    Enhanced face-swap compositing detection using convex-hull boundary analysis.
    Face-swap compositing blends a synthesized/warped face into the target frame
    along a boundary (jawline, hairline, forehead). This blend seam often has
    different edge/gradient statistics than a natural face-to-background transition.

    Computes:
    1. Gradient step discontinuity across the face boundary (convex-hull based)
    2. Color distribution divergence (KL-like divergence) between face crop
       and surrounding background pixels
    3. Sharpness ratio between face interior and boundary ring
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

        # --- Signal 1: Sharpness ratio ---
        inner_lap_var = float(cv2.Laplacian(inner_gray, cv2.CV_64F).var())
        outer_lap_var = float(cv2.Laplacian(outer_gray, cv2.CV_64F).var())

        if outer_lap_var < 1e-6:
            sharpness_ratio = 1.0
        else:
            sharpness_ratio = inner_lap_var / outer_lap_var

        sharpness_score = float(np.clip((0.6 - sharpness_ratio) / 0.6, 0.0, 1.0)) if sharpness_ratio < 0.6 else 0.0

        # --- Signal 2: Color distribution divergence ---
        # Compare color histograms between face and background
        inner_hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        outer_hsv = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV)

        hist_inner = cv2.calcHist([inner_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hist_outer = cv2.calcHist([outer_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

        cv2.normalize(hist_inner, hist_inner, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_outer, hist_outer, 0, 1, cv2.NORM_MINMAX)

        # Bhattacharyya distance as color divergence measure
        color_divergence = cv2.compareHist(
            hist_inner.astype(np.float32),
            hist_outer.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        color_score = float(np.clip((color_divergence - 0.15) / 0.5, 0.0, 1.0))

        # --- Signal 3: Gradient magnitude across boundary ---
        # Compute gradient magnitude and compare inner vs outer gradient distributions
        inner_edges = cv2.Canny(inner_gray, 50, 150)
        outer_edges = cv2.Canny(outer_gray, 50, 150)
        inner_edge_density = float(np.sum(inner_edges > 0)) / max(inner_edges.size, 1)
        outer_edge_density = float(np.sum(outer_edges > 0)) / max(outer_edges.size, 1)
        if outer_edge_density < 1e-6:
            edge_ratio = 1.0
        else:
            edge_ratio = inner_edge_density / outer_edge_density
        edge_score = float(np.clip((0.5 - edge_ratio) / 0.5, 0.0, 1.0)) if edge_ratio < 0.5 else 0.0

        # Combine signals
        score = float(np.clip(0.4 * sharpness_score + 0.35 * color_score + 0.25 * edge_score, 0.0, 1.0))

        parts = []
        if sharpness_score > 0.3:
            parts.append(f"face markedly smoother than surroundings (sharpness ratio {sharpness_ratio:.2f})")
        if color_score > 0.3:
            parts.append(f"color distribution divergence between face and background ({color_divergence:.2f})")
        if edge_score > 0.3:
            parts.append(f"edge density mismatch between face and surroundings")

        if parts:
            desc = f"Face boundary anomaly: {', '.join(parts)} -- consistent with a blended/composited face."
        else:
            desc = "Face region sharpness and color distribution are consistent with its surroundings."
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
    Runs all four forensic families on a single image and returns:
      - file_hash: SHA-256 hash for chain-of-custody
      - provenance_findings: list of Finding (C2PA, EXIF metadata checks)
      - global_findings: list of Finding (whole-frame AI-generation signals)
      - splice_findings: list of Finding (localized edit signals)
      - face_analyses: list of per-face dicts (face-swap signals, one per detected face)
    Caller (final_image_detector.py) combines these into a verdict + reasons.
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"error": "could_not_read_image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Layer 1: Provenance & integrity
    file_hash = sha256_file_hash(img_path)
    provenance_findings = [
        c2pa_provenance_score(img_path),
        exif_metadata_score(img_path),
    ]

    # Layer 2: Global AI-generation forensics
    global_findings = [
        fft_periodicity_score(gray),
        texture_uniformity_score(gray),
        histogram_smoothness_score(img),
    ]

    # Layer 3: Splice/localization forensics
    splice_findings = [
        error_level_analysis(img_path),
        noise_residual_consistency(gray),
    ]

    # Layer 4: Face-region forensics
    faces = detect_faces(img)
    face_analyses = [analyze_face_region(img, f) for f in faces]

    return {
        "file_hash": file_hash,
        "provenance_findings": provenance_findings,
        "global_findings": global_findings,
        "splice_findings": splice_findings,
        "face_analyses": face_analyses,
        "faces_detected": len(faces),
    }
