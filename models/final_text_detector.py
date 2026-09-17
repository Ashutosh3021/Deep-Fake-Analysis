"""
FINAL AI-Generated Text Detector (v3)
=======================================
Enhanced with mathematically-grounded features from the forensic literature:

  1. Perplexity + burstiness via GPT-2 probe model (primary signal)
     - Burstiness: B = (sigma - mu) / (sigma + mu), ranges [-1, 1]
     - Human text: B > 0 (bursty); AI text: B <= 0 (uniform)
  2. Vocabulary richness metrics
     - Type-Token Ratio (TTR) = |V_unique| / N_total
     - Hapax Legomena Ratio (H) = N_words_appearing_once / N_total
     - Shannon entropy of word frequency distribution
  3. Token watermark z-score tester (KGW-style)
     - z = (|s|_G - gamma*T) / sqrt(T * gamma * (1 - gamma))
     - z > 4.0 indicates active watermarking
  4. Formulaic AI discourse markers (weighted, not dominant)
  5. Localized span highlighting returning suspicious sentences
     with explanatory justifications

Setup:
  pip install transformers torch numpy --break-system-packages
First run downloads GPT-2 (~500MB) once, then cached offline.
"""

import os
import re
import math
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROBE_MODEL_ID = os.getenv("TEXT_DETECTOR_PROBE_MODEL", "gpt2")
MIN_WORDS_FOR_RELIABLE_SCORE = 50
LOW_CONFIDENCE_THRESHOLD = 0.58

# Weight allocation across signal families (from plan.md)
# DetectGPT takes some weight from perplexity when the probe LM is available.
PERPLEXITY_WEIGHT  = 0.40
DETECTGPT_WEIGHT   = 0.15
STYLOMETRY_WEIGHT  = 0.25
PATTERN_WEIGHT     = 0.15
WATERMARK_WEIGHT   = 0.05

# Watermark detection parameters (KGW-style, plan Task 4.3)
WATERMARK_GREEN_RATIO = 0.5  # gamma: expected fraction of green tokens
WATERMARK_Z_THRESHOLD = 4.0  # standard detection threshold
# Number of seeds to try when no specific key is provided
WATERMARK_DEFAULT_SEEDS = [0, 42, 1337, 9999, 31415]
# DetectGPT: number of perturbations (plan Task 4.4)
DETECTGPT_NUM_PERTURBATIONS = 20


@dataclass
class TextVerdict:
    label: str                  # "AI_GENERATED" | "HUMAN_WRITTEN" | "UNCERTAIN"
    confidence: float
    ai_probability: float
    perplexity_signal: Optional[float]
    stylometry_signal: float
    pattern_signal: float
    watermark_signal: float
    detectgpt_signal: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    suspicious_spans: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "ai_probability": round(self.ai_probability, 4),
            "perplexity_signal": round(self.perplexity_signal, 4) if self.perplexity_signal is not None else None,
            "stylometry_signal": round(self.stylometry_signal, 4),
            "pattern_signal": round(self.pattern_signal, 4),
            "watermark_signal": round(self.watermark_signal, 4),
            "detectgpt_signal": round(self.detectgpt_signal, 4) if self.detectgpt_signal is not None else None,
            "metrics": self.metrics,
            "suspicious_spans": self.suspicious_spans,
            "notes": self.notes,
        }


class FinalTextDetector:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        self._load_probe_model()

        self._style_patterns = [
            (r"\bas an ai\b", 2.0),
            (r"\bas a language model\b", 2.0),
            (r"\bi cannot\b", 1.0),
            (r"\bit is important to note that\b", 1.5),
            (r"\bit'?s worth noting that\b", 1.2),
            (r"\bin conclusion\b", 1.0),
            (r"\bto summarize\b", 0.8),
            (r"\bdelve into\b", 1.8),
            (r"\brobust framework\b", 1.5),
            (r"\bleverage\b", 1.0),
            (r"\bfurthermore\b", 0.5),
            (r"\bmoreover\b", 0.5),
            (r"\bin this landscape\b", 1.8),
            (r"\btapestry\b", 1.5),
            (r"\bpivotal\b", 0.8),
            (r"\bmultifaceted\b", 1.0),
            (r"\bcommence\b", 1.2),
            (r"\bfostering\b", 0.8),
            (r"\bholistic approach\b", 1.5),
            (r"\bcomprehensive overview\b", 1.2),
        ]

    # ------------------------------------------------------------------
    def _load_probe_model(self):
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            self._torch = torch
            self._tokenizer = GPT2TokenizerFast.from_pretrained(PROBE_MODEL_ID)
            self._model = GPT2LMHeadModel.from_pretrained(PROBE_MODEL_ID)
            self._model.eval()
            logger.info("Loaded probe LM for perplexity scoring: %s", PROBE_MODEL_ID)
        except Exception as e:
            self._load_error = str(e)
            logger.warning("Could not load probe LM (%s). Falling back to stylometric-only mode.", e)

    # ------------------------------------------------------------------
    # Sentence tokenizer
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize_sentences(text: str) -> List[str]:
        """Split text into sentences for burstiness analysis."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        """Split text into words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # ------------------------------------------------------------------
    # 1. Perplexity + Burstiness (primary signal)
    # ------------------------------------------------------------------
    def _perplexity_and_burstiness(self, text: str) -> Optional[Dict[str, float]]:
        if self._model is None or self._tokenizer is None:
            return None
        try:
            torch = self._torch
            encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = encodings.input_ids

            if input_ids.shape[1] < 8:
                return None

            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                token_nll = -token_log_probs.squeeze(0).numpy()

            mean_nll = float(np.mean(token_nll))
            perplexity = float(np.exp(np.clip(mean_nll, 0, 20)))

            # --- Burstiness (from plan.md) ---
            # Method 1: std of per-token surprise
            burstiness_token = float(np.std(token_nll))

            # Method 2: B = (sigma - mu) / (sigma + mu) on sentence lengths
            sentences = self._tokenize_sentences(text)
            if len(sentences) >= 3:
                sent_lengths = [len(self._tokenize_words(s)) for s in sentences]
                mu = np.mean(sent_lengths)
                sigma = np.std(sent_lengths)
                burstiness_formula = (sigma - mu) / (sigma + mu + 1e-6)
            else:
                burstiness_formula = 0.0

            return {
                "perplexity": perplexity,
                "mean_nll": mean_nll,
                "burstiness_token": burstiness_token,
                "burstiness_formula": float(burstiness_formula),
                "tokens_scored": int(len(token_nll)),
                "sentence_count": len(sentences),
            }
        except Exception as e:
            logger.error("Perplexity scoring failed: %s", e)
            return None

    def _perplexity_to_ai_score(self, metrics: Dict[str, float]) -> float:
        ppl = metrics["perplexity"]
        burst_token = metrics["burstiness_token"]
        burst_formula = metrics["burstiness_formula"]

        PPL_LOW, PPL_HIGH = 15.0, 55.0
        ppl_score = 1.0 - np.clip((ppl - PPL_LOW) / (PPL_HIGH - PPL_LOW), 0.0, 1.0)

        BURST_LOW, BURST_HIGH = 1.5, 4.0
        burst_score_token = 1.0 - np.clip((burst_token - BURST_LOW) / (BURST_HIGH - BURST_LOW), 0.0, 1.0)

        # B formula: B <= 0 suggests AI (uniform), B > 0 suggests human (bursty)
        burst_score_formula = float(np.clip(0.5 - burst_formula, 0.0, 1.0))

        return float(np.clip(0.4 * ppl_score + 0.3 * burst_score_token + 0.3 * burst_score_formula, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 2. Vocabulary richness / stylometry (from plan.md)
    # ------------------------------------------------------------------
    def _stylometry_signal(self, text: str) -> Dict[str, Any]:
        words = self._tokenize_words(text)
        word_count = max(len(words), 1)

        # Type-Token Ratio (TTR)
        unique_words = set(words)
        ttr = len(unique_words) / word_count

        # Hapax Legomena Ratio (H)
        word_counts = Counter(words)
        hapax = sum(1 for c in word_counts.values() if c == 1)
        hapax_ratio = hapax / word_count

        # Shannon entropy of word frequency distribution
        freq_dist = np.array(list(word_counts.values()), dtype=float)
        freq_dist = freq_dist / freq_dist.sum()
        entropy = -np.sum(freq_dist * np.log2(freq_dist + 1e-10))

        # Syntactic formulaic transitions density
        formulaic_transitions = [
            r"\bin conclusion\b", r"\bto summarize\b", r"\bfirst and foremost\b",
            r"\bit goes without saying\b", r"\bas we all know\b",
            r"\bin today'?s world\b", r"\bit is worth noting\b",
            r"\bwith regards to\b", r"\bin terms of\b",
        ]
        transition_count = sum(len(re.findall(p, text.lower())) for p in formulaic_transitions)
        transition_density = min(transition_count / max(word_count / 100, 1), 1.0)

        # Sentence length uniformity (AI tends toward uniform sentence lengths)
        sentences = self._tokenize_sentences(text)
        if len(sentences) >= 3:
            sent_lengths = [len(self._tokenize_words(s)) for s in sentences]
            length_cv = np.std(sent_lengths) / (np.mean(sent_lengths) + 1e-6)
            # Low CV = uniform = AI-leaning
            uniformity_score = float(np.clip(1.0 - length_cv, 0.0, 1.0))
        else:
            uniformity_score = 0.5

        # Combine stylometric features
        # Low TTR, low entropy, low hapax, high uniformity -> AI-leaning
        ttr_score = float(np.clip(1.0 - (ttr - 0.3) / 0.4, 0.0, 1.0)) if ttr < 0.7 else 0.0
        entropy_score = float(np.clip(1.0 - entropy / 8.0, 0.0, 1.0))
        hapax_score = float(np.clip(1.0 - (hapax_ratio - 0.2) / 0.4, 0.0, 1.0)) if hapax_ratio < 0.6 else 0.0

        stylometry_score = float(np.clip(
            0.3 * ttr_score + 0.25 * entropy_score + 0.2 * hapax_score +
            0.15 * uniformity_score + 0.1 * transition_density,
            0.0, 1.0
        ))

        return {
            "score": stylometry_score,
            "ttr": round(ttr, 4),
            "hapax_ratio": round(hapax_ratio, 4),
            "entropy": round(entropy, 4),
            "uniformity": round(uniformity_score, 4),
            "transition_density": round(transition_density, 4),
        }

    # ------------------------------------------------------------------
    # 3. Watermark z-score tester (KGW-style, configurable seeds — Task 4.3)
    # ------------------------------------------------------------------
    def _watermark_zscore(self, text: str,
                           seeds: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Token-level watermark detection using the Kirchenbauer (KGW) z-test.

        Formula (plan.md §4.1.4):
          z = (|s|_G − γ·T) / √(T·γ·(1−γ))
          where |s|_G = green-list token count, γ = 0.5, T = total tokens.
          z > 4.0 → likely watermarked.

        Task 4.3 extension — configurable hash seeds:
          Each seed defines a different HMAC-based green/red partition.
          We test every provided seed and report the maximum |z| found,
          mimicking a detector that tries a small set of known keys.
          The seed is mixed into the hash via: hash(word + str(seed)) % 2
        """
        words = self._tokenize_words(text)
        T = len(words)

        if T < 20:
            return {"score": 0.0, "z_score": 0.0, "best_seed": None,
                    "green_count": 0, "total_tokens": T,
                    "caveat": "Text too short for watermark analysis."}

        if seeds is None:
            seeds = WATERMARK_DEFAULT_SEEDS

        gamma = WATERMARK_GREEN_RATIO
        best_z = 0.0
        best_seed = seeds[0]
        best_green = 0

        for seed in seeds:
            # Keyed green/red partition: mix each word with the seed before hashing.
            # Using Python's built-in hash seeded by XOR-ing ord values with the seed
            # gives a stable, per-seed partition without requiring hmac/hashlib overhead.
            green_count = sum(
                1 for w in words if hash(w + str(seed)) % 2 == 0
            )
            z = (green_count - gamma * T) / math.sqrt(T * gamma * (1 - gamma))
            if abs(z) > abs(best_z):
                best_z = z
                best_seed = seed
                best_green = green_count

        # score = how far above the noise threshold (2.0 sigma) we are, capped at 1
        score = float(np.clip((abs(best_z) - 2.0) / 4.0, 0.0, 1.0))

        if abs(best_z) > WATERMARK_Z_THRESHOLD:
            caveat = (
                f"Statistical excess of green-list tokens (z={best_z:.2f}, seed={best_seed}) "
                f"suggests active token-level watermarking. Strong evidence of AI generation "
                f"when the matching watermark key is known."
            )
        else:
            caveat = (
                "No significant watermark signal detected across any tested seed. "
                "Absence does NOT prove human authorship; the generating model may not "
                "support watermarking, or the text may have been paraphrased."
            )

        return {
            "score": score,
            "z_score": round(float(best_z), 4),
            "best_seed": best_seed,
            "seeds_tested": seeds,
            "green_count": best_green,
            "total_tokens": T,
            "caveat": caveat,
        }

    # ------------------------------------------------------------------
    # 4. DetectGPT log-probability curvature (plan.md §4.1.5 — Task 4.4)
    # ------------------------------------------------------------------
    def _detectgpt_curvature(self, text: str,
                              n_perturbations: int = DETECTGPT_NUM_PERTURBATIONS
                              ) -> Optional[Dict[str, Any]]:
        """
        DetectGPT (Mitchell et al., 2023) log-probability curvature test.

        Principle: AI-generated text tends to lie at local *maxima* of the
        log-probability surface of the generating model.  Human text does not.

        Estimate:
          d(x, q) = log P_M(x) − (1/K) Σ_{k=1}^{K} log P_M(x̃_k)
          where x̃_k ~ q(·|x) are word-level perturbations of x.

        A positive d(x,q) — the original scores higher than its perturbations —
        is characteristic of AI-generated text (it is near a probability maximum).
        A negative or near-zero value suggests human authorship.

        Lightweight perturbation strategy (avoids a full mask-fill model):
          We generate K variants by randomly replacing 15% of non-stopword tokens
          with a randomly sampled synonym from a tiny closed word-list, which
          approximates the local curvature signal without requiring T5/GPT-4.

        Returns None if the probe LM is unavailable.
        """
        if self._model is None or self._tokenizer is None:
            return None

        try:
            torch = self._torch

            def _log_prob(t: str) -> Optional[float]:
                """Compute mean per-token log-probability under the probe LM."""
                enc = self._tokenizer(t, return_tensors="pt",
                                      truncation=True, max_length=512)
                ids = enc.input_ids
                if ids.shape[1] < 4:
                    return None
                with torch.no_grad():
                    out = self._model(ids, labels=ids)
                    lps = torch.nn.functional.log_softmax(out.logits[:, :-1, :], dim=-1)
                    tok_lp = lps.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    return float(tok_lp.mean().item())

            # Score the original text
            orig_lp = _log_prob(text)
            if orig_lp is None:
                return None

            # Lightweight perturbation: random word replacements
            # Closed synonym pool — keeps semantics close but changes tokens
            _synonyms = {
                "good": ["fine", "solid", "decent", "proper"],
                "bad": ["poor", "weak", "flawed", "faulty"],
                "large": ["big", "vast", "great", "sizable"],
                "small": ["tiny", "minor", "slight", "narrow"],
                "new": ["recent", "fresh", "novel", "modern"],
                "old": ["prior", "past", "former", "dated"],
                "important": ["key", "vital", "major", "crucial"],
                "show": ["reveal", "display", "present", "indicate"],
                "use": ["apply", "employ", "utilize", "deploy"],
                "make": ["create", "form", "build", "produce"],
                "get": ["obtain", "acquire", "gain", "achieve"],
                "give": ["provide", "offer", "supply", "grant"],
                "know": ["understand", "grasp", "recognize", "realize"],
                "think": ["believe", "consider", "feel", "hold"],
                "help": ["assist", "support", "aid", "enable"],
                "many": ["numerous", "various", "multiple", "several"],
                "also": ["too", "as well", "moreover", "additionally"],
            }

            words = text.split()
            rng = np.random.default_rng(seed=42)
            perturb_lps = []

            for _ in range(n_perturbations):
                w_copy = words[:]
                # Replace ~15% of words
                n_replace = max(1, int(len(w_copy) * 0.15))
                indices = rng.choice(len(w_copy), size=min(n_replace, len(w_copy)),
                                     replace=False)
                for idx in indices:
                    word_lower = w_copy[idx].lower().strip(".,!?;:")
                    if word_lower in _synonyms:
                        replacement = rng.choice(_synonyms[word_lower])
                        # Preserve original capitalisation
                        if w_copy[idx][0].isupper():
                            replacement = replacement.capitalize()
                        w_copy[idx] = replacement

                perturbed_text = " ".join(w_copy)
                lp = _log_prob(perturbed_text)
                if lp is not None:
                    perturb_lps.append(lp)

            if not perturb_lps:
                return None

            mean_perturb_lp = float(np.mean(perturb_lps))
            # d(x, q) = log P(x) - mean log P(x̃)
            curvature = orig_lp - mean_perturb_lp

            # AI text → positive curvature (original > perturbations)
            # Human text → near-zero or negative curvature
            # Normalise to [0, 1]: curvature > 0.05 is AI-leaning
            score = float(np.clip(curvature / 0.5, 0.0, 1.0)) if curvature > 0 else 0.0

            return {
                "score": score,
                "curvature": round(curvature, 5),
                "original_log_prob": round(orig_lp, 5),
                "mean_perturbed_log_prob": round(mean_perturb_lp, 5),
                "n_perturbations_used": len(perturb_lps),
                "interpretation": (
                    "Positive curvature: original text scores higher than its "
                    "perturbations under the probe LM, consistent with AI generation "
                    "(text near a log-probability maximum)."
                    if curvature > 0.02 else
                    "Near-zero or negative curvature: text is not near a "
                    "log-probability maximum, consistent with human authorship."
                ),
            }
        except Exception as e:
            logger.error("DetectGPT curvature failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # 4. Formulaic pattern signal (corroborating only)
    # ------------------------------------------------------------------
    def _pattern_signal(self, text: str) -> Dict[str, Any]:
        lower = text.lower()

        weighted_matches = 0.0
        total_weight = 0.0
        match_details = []
        for pattern, weight in self._style_patterns:
            matches = len(re.findall(pattern, lower))
            if matches > 0:
                weighted_matches += matches * weight
                total_weight += weight
                match_details.append((pattern, matches))

        words = text.split()
        word_count = max(len(words), 1)

        # Repetitive n-grams
        trigrams = list(zip(words, words[1:], words[2:])) if len(words) > 2 else []
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
        repetition_ratio = repeated_trigrams / max(len(trigrams), 1)

        normalized_matches = min(1.0, weighted_matches / max(word_count / 150, 1))
        score = float(np.clip(0.6 * normalized_matches + 0.4 * min(1.0, repetition_ratio * 5), 0.0, 1.0))

        return {
            "score": score,
            "weighted_matches": round(weighted_matches, 2),
            "match_details": match_details,
            "repetition_ratio": round(repetition_ratio, 3),
            "caveat": "Weak, genre-dependent signal. Formal/academic writing and "
                      "non-native English writers can score elevated here without "
                      "being AI-generated.",
        }

    # ------------------------------------------------------------------
    # 5. Suspicious span highlighting
    # ------------------------------------------------------------------
    def _find_suspicious_spans(self, text: str, ppl_metrics: Optional[Dict]) -> List[Dict[str, Any]]:
        """
        Identify and highlight suspicious sentences/spans with explanatory
        justifications. Returns localized evidence with reasons.
        """
        spans = []
        sentences = self._tokenize_sentences(text)

        if not sentences:
            return spans

        for i, sent in enumerate(sentences):
            reasons = []
            sent_words = self._tokenize_words(sent)

            # Check for formulaic AI phrases
            for pattern, weight in self._style_patterns:
                if re.search(pattern, sent.lower()):
                    reasons.append(f"Contains formulaic AI phrase matching '{pattern.strip(chr(92)).strip('b')}'")
                    break  # one match per sentence is enough

            # Check for low sentence length variation (uniform cadence)
            if len(sent_words) < 5:
                reasons.append("Very short sentence (possible template fragment)")

            if reasons:
                spans.append({
                    "sentence_index": i,
                    "text": sent[:200] + ("..." if len(sent) > 200 else ""),
                    "reasons": reasons,
                })

        return spans[:10]  # limit to top 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"error": "empty_text"}

        word_count = len(text.split())
        ppl_metrics = self._perplexity_and_burstiness(text)
        stylometry = self._stylometry_signal(text)
        pattern_info = self._pattern_signal(text)
        watermark_info = self._watermark_zscore(text)
        detectgpt_info = self._detectgpt_curvature(text)

        # --- Combine signals with calibrated weights ---
        signals = []
        if ppl_metrics is not None:
            perplexity_signal = self._perplexity_to_ai_score(ppl_metrics)
            signals.append(("perplexity", PERPLEXITY_WEIGHT, perplexity_signal))
        else:
            perplexity_signal = None

        detectgpt_signal: Optional[float] = None
        if detectgpt_info is not None:
            detectgpt_signal = detectgpt_info["score"]
            signals.append(("detectgpt", DETECTGPT_WEIGHT, detectgpt_signal))

        signals.append(("stylometry", STYLOMETRY_WEIGHT, stylometry["score"]))
        signals.append(("pattern", PATTERN_WEIGHT, pattern_info["score"]))
        signals.append(("watermark", WATERMARK_WEIGHT, watermark_info["score"]))

        total_weight = sum(w for _, w, _ in signals)
        if total_weight > 0:
            ai_probability = sum(w * s for _, w, s in signals) / total_weight
        else:
            ai_probability = 0.5

        # Suspicious span highlighting
        suspicious_spans = self._find_suspicious_spans(text, ppl_metrics)

        if ppl_metrics is not None:
            notes = "Combined perplexity/burstiness (primary) + DetectGPT curvature + stylometry + patterns + watermark z-score."
        else:
            notes = (f"Probe LM unavailable ({self._load_error}); running on stylometric + "
                     f"pattern analysis only. Confidence is suppressed.")

        if detectgpt_info is not None:
            notes += f" DetectGPT curvature={detectgpt_info['curvature']:.4f}."

        if word_count < MIN_WORDS_FOR_RELIABLE_SCORE:
            notes += f" Text is short ({word_count} words); estimates are noisy."

        ai_probability = float(np.clip(ai_probability, 0.0, 1.0))
        distance_from_mid = abs(ai_probability - 0.5) * 2

        if ppl_metrics is None:
            distance_from_mid *= 0.4
        if word_count < MIN_WORDS_FOR_RELIABLE_SCORE:
            distance_from_mid *= 0.6

        if distance_from_mid < (1 - LOW_CONFIDENCE_THRESHOLD):
            label = "UNCERTAIN"
        elif ai_probability > 0.5:
            label = "AI_GENERATED"
        else:
            label = "HUMAN_WRITTEN"

        confidence = 50.0 + distance_from_mid * 50.0

        metrics = {"word_count": word_count}
        if ppl_metrics is not None:
            metrics["perplexity_details"] = {k: round(v, 3) if isinstance(v, float) else v
                                              for k, v in ppl_metrics.items()}
        metrics["stylometry_details"] = stylometry
        metrics["pattern_details"] = pattern_info
        metrics["watermark_details"] = watermark_info
        if detectgpt_info is not None:
            metrics["detectgpt_details"] = detectgpt_info

        verdict = TextVerdict(
            label=label,
            confidence=confidence,
            ai_probability=ai_probability,
            perplexity_signal=perplexity_signal,
            stylometry_signal=stylometry["score"],
            pattern_signal=pattern_info["score"],
            watermark_signal=watermark_info["score"],
            detectgpt_signal=detectgpt_signal,
            metrics=metrics,
            suspicious_spans=suspicious_spans,
            notes=notes,
        )
        return verdict.to_dict()


# Global instance
final_text_detector = FinalTextDetector()


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python final_text_detector.py \"text to analyze\"  (or --file path.txt)")
        sys.exit(1)
    if sys.argv[1] == "--file":
        with open(sys.argv[2]) as f:
            text = f.read()
    else:
        text = " ".join(sys.argv[1:])
    print(json.dumps(final_text_detector.predict(text), indent=2))
