"""
FINAL AI-Generated Text Detector
===================================
Design philosophy: the old code's primary signal was a regex keyword list
("delve", "leverage", "robust", "moreover"...) and an untrained character-
CNN. Two problems with that:
  1. Keyword lists go stale fast -- they target GPT-3.5-era stock phrasing
     and miss modern models, and they false-positive on humans who just
     happen to use those words (which, ironically, became common BECAUSE
     LLMs popularized them -- the signal degrades over time by design).
  2. An untrained CNN on character codes is random noise, same problem as
     the other detectors.

This version's primary signal is **perplexity + burstiness via a small
pretrained causal LM (GPT-2)**. This is the actual standard lightweight
approach (this is what the original DetectGPT / GPTZero-style tools are
built on): AI-generated text tends to sit in the model's high-probability
("low perplexity") region everywhere, fairly uniformly. Human text has
more "burstiness" -- some very predictable spans, some very surprising
word choices, unevenly distributed.

  - GPT-2 (small, ~500MB, CPU-feasible) is used as a *probe* model, not
    because it's a strong LM, but because what we want is a *generic*
    next-token probability landscape to compare against text statistics --
    research on perplexity-based detection has shown even smaller probe
    models transfer reasonably across detecting larger generators.
  - Keyword/pattern matching is kept, but heavily downweighted and labeled
    explicitly as "stylistic indicators, weak signal, prone to false
    positives on non-native English writers" -- this is an important
    fairness note: penalizing "moreover/furthermore" usage disproportionately
    flags ESL writers and academic writing, not just AI text.

Setup:
  pip install transformers torch numpy --break-system-packages
First run downloads GPT-2 (~500MB) once, then cached offline.
"""

import os
import re
import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROBE_MODEL_ID = os.getenv("TEXT_DETECTOR_PROBE_MODEL", "gpt2")
MIN_WORDS_FOR_RELIABLE_SCORE = 50
LOW_CONFIDENCE_THRESHOLD = 0.58

# Perplexity / burstiness weight vs. stylistic-pattern weight. Perplexity
# is the real signal; patterns are corroborating only and explicitly
# capped in influence to avoid penalizing legitimate formal writing.
PERPLEXITY_WEIGHT = 0.75
PATTERN_WEIGHT = 0.25


@dataclass
class TextVerdict:
    label: str                  # "AI_GENERATED" | "HUMAN_WRITTEN" | "UNCERTAIN"
    confidence: float
    ai_probability: float
    perplexity_signal: Optional[float]
    pattern_signal: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 2),
            "ai_probability": round(self.ai_probability, 4),
            "perplexity_signal": round(self.perplexity_signal, 4) if self.perplexity_signal is not None else None,
            "pattern_signal": round(self.pattern_signal, 4),
            "metrics": self.metrics,
            "notes": self.notes,
        }


class FinalTextDetector:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[str] = None
        self._load_probe_model()

        # Kept intentionally short and labeled as weak/stylistic. These are
        # NOT used to make a confident call on their own.
        self._style_patterns = [
            r"\bas an ai\b", r"\bas a language model\b", r"\bi cannot\b",
            r"\bit is important to note that\b", r"\bit'?s worth noting that\b",
            r"\bin conclusion\b", r"\bto summarize\b",
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
            logger.warning("Could not load probe LM (%s). Falling back to pattern-only mode "
                            "with suppressed confidence.", e)

    # ------------------------------------------------------------------
    # Perplexity + burstiness (primary signal)
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
                # outputs.loss is mean per-token cross-entropy (natural log).
                # We also want PER-TOKEN losses (not just the mean) to
                # compute burstiness -- the variance of surprise across
                # the sequence, not just its average.
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                token_nll = -token_log_probs.squeeze(0).numpy()  # per-token negative log-likelihood

            mean_nll = float(np.mean(token_nll))
            perplexity = float(np.exp(np.clip(mean_nll, 0, 20)))
            # Burstiness: std of per-token surprise. AI text tends to stay
            # in a narrower band (lower std); human text swings between
            # very predictable and very surprising tokens more (higher std).
            burstiness = float(np.std(token_nll))

            return {
                "perplexity": perplexity,
                "mean_nll": mean_nll,
                "burstiness": burstiness,
                "tokens_scored": int(len(token_nll)),
            }
        except Exception as e:
            logger.error("Perplexity scoring failed: %s", e)
            return None

    def _perplexity_to_ai_score(self, metrics: Dict[str, float]) -> float:
        """
        Maps (perplexity, burstiness) to an AI-likelihood in [0,1].

        IMPORTANT CALIBRATION NOTE: the thresholds below (perplexity ~20-60
        "typical for human text" / burstiness ~2.5+ "typical human variance")
        are reasonable starting points from published perplexity-detection
        literature using GPT-2 as a probe, but they are NOT fit on your
        specific text domain. Before trusting this in production, run it
        against a labeled set of human vs. AI text samples from YOUR
        use case (emails, essays, social posts -- perplexity baselines
        differ by genre) and adjust PPL_LOW/PPL_HIGH/BURST_LOW below, or
        better: replace this hand-tuned mapping with a logistic regression
        fit on (perplexity, burstiness) -> label from labeled data. That
        is a 10-line sklearn script once you have ~200 labeled examples.
        """
        ppl = metrics["perplexity"]
        burst = metrics["burstiness"]

        # Low perplexity (model finds text very predictable) -> AI-leaning.
        PPL_LOW, PPL_HIGH = 15.0, 55.0
        ppl_score = 1.0 - np.clip((ppl - PPL_LOW) / (PPL_HIGH - PPL_LOW), 0.0, 1.0)

        # Low burstiness (uniform surprise across the text) -> AI-leaning.
        BURST_LOW, BURST_HIGH = 1.5, 4.0
        burst_score = 1.0 - np.clip((burst - BURST_LOW) / (BURST_HIGH - BURST_LOW), 0.0, 1.0)

        return float(np.clip(0.5 * ppl_score + 0.5 * burst_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Stylistic pattern signal (weak, corroborating only)
    # ------------------------------------------------------------------
    def _pattern_signal(self, text: str) -> Dict[str, Any]:
        lower = text.lower()
        matches = sum(len(re.findall(p, lower)) for p in self._style_patterns)

        words = text.split()
        word_count = max(len(words), 1)

        # Repetitive n-grams: a real but weak signal -- some AI text
        # repeats phrase structures more than natural human writing,
        # though this is also genre-dependent (technical docs repeat
        # phrases naturally too).
        trigrams = list(zip(words, words[1:], words[2:])) if len(words) > 2 else []
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
        repetition_ratio = repeated_trigrams / max(len(trigrams), 1)

        normalized_matches = min(1.0, matches / max(word_count / 150, 1))
        score = float(np.clip(0.6 * normalized_matches + 0.4 * min(1.0, repetition_ratio * 5), 0.0, 1.0))

        return {
            "score": score,
            "phrase_matches": matches,
            "repetition_ratio": round(repetition_ratio, 3),
            "caveat": "Weak, genre-dependent signal. Formal/academic writing and "
                      "non-native English writers can score elevated here without "
                      "being AI-generated. Do not treat this alone as evidence.",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"error": "empty_text"}

        word_count = len(text.split())
        ppl_metrics = self._perplexity_and_burstiness(text)
        pattern_info = self._pattern_signal(text)
        pattern_signal = pattern_info["score"]

        if ppl_metrics is not None:
            perplexity_signal = self._perplexity_to_ai_score(ppl_metrics)
            ai_probability = PERPLEXITY_WEIGHT * perplexity_signal + PATTERN_WEIGHT * pattern_signal
            notes = "Combined perplexity/burstiness (primary) + stylistic patterns (weak, corroborating)."
        else:
            perplexity_signal = None
            ai_probability = pattern_signal
            notes = (f"Probe LM unavailable ({self._load_error}); running on weak stylistic "
                     f"patterns only. Confidence is deliberately suppressed -- this mode is "
                     f"not reliable enough to act on alone.")

        if word_count < MIN_WORDS_FOR_RELIABLE_SCORE:
            notes += f" Text is short ({word_count} words); perplexity estimates are noisy below ~{MIN_WORDS_FOR_RELIABLE_SCORE} words."

        ai_probability = float(np.clip(ai_probability, 0.0, 1.0))
        distance_from_mid = abs(ai_probability - 0.5) * 2

        if ppl_metrics is None:
            distance_from_mid *= 0.4  # pattern-only mode: suppress confidence hard
        if word_count < MIN_WORDS_FOR_RELIABLE_SCORE:
            distance_from_mid *= 0.6

        if distance_from_mid < (1 - LOW_CONFIDENCE_THRESHOLD):
            label = "UNCERTAIN"
        elif ai_probability > 0.5:
            label = "AI_GENERATED"
        else:
            label = "HUMAN_WRITTEN"

        confidence = 50.0 + distance_from_mid * 50.0

        metrics = {"word_count": word_count, "pattern_details": pattern_info}
        if ppl_metrics is not None:
            metrics["perplexity_details"] = {k: round(v, 3) if isinstance(v, float) else v
                                              for k, v in ppl_metrics.items()}

        verdict = TextVerdict(
            label=label,
            confidence=confidence,
            ai_probability=ai_probability,
            perplexity_signal=perplexity_signal,
            pattern_signal=pattern_signal,
            metrics=metrics,
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
