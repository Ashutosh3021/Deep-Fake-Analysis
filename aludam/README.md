# aludam

Uniform deepfake / AI-content detection: one entry point, four modalities, one
result type.

**The neural accuracy work lives in `aludam-fullplate`; plain `aludam` is the
documented light tier and gets none of those gains (see "Two tiers" below).**

```python
from aludam import load_detector

det1 = load_detector("text_det");  result1 = det1.predict(text)
det2 = load_detector("img_det");   result2 = det2.predict(image_path)
det3 = load_detector("vdo_det");   result3 = det3.predict(video_path)
det4 = load_detector("aud_det");   result4 = det4.predict(audio_path)

print(result1)
# DetectionResult(score=0.8012, label='ai', confidence=0.5582, explanation='...')
```

Or the one-call form:

```python
from aludam import al
al.detect(type="img_det", src="./photo.png")
```

Every detector returns the same class, so one line of formatting works for all
four.

## Install

```bash
pip install aludam            # light tier
pip install aludam-fullplate  # neural tier (see "Two tiers" below)
```

From a checkout:

```bash
pip install -e "./aludam[neural]"
```

## Two tiers

| install | gets |
|---|---|
| `aludam` | the API, CLI, weight tooling and the light dependency set (pillow, numpy, pypdf, python-docx). No torch, no transformers. |
| `aludam-fullplate` | everything above plus `aludam[neural]`: torch, transformers, huggingface_hub, timm, peft, opencv-python-headless, librosa, soundfile, scipy, onnxruntime - what the four detectors need to actually run. |

`aludam-fullplate` is a thin wrapper whose only dependency is
`aludam[neural]`, so `pip install aludam-fullplate` and
`pip install "aludam[neural]"` install exactly the same stack: identical
import, identical syntax, identical `DetectionResult` either way.

The light tier carries none of the neural accuracy gains (ALUDAM_PLAN.md
sec 5.2): that downgrade is intentional and documented, not an accident. No
accuracy claim is made for either tier until Phase 0 measures one.

## The four detectors

| name | input | reads |
|---|---|---|
| `text_det` | `str` | AI-generated prose vs human writing |
| `img_det` | path | AI-generated or manipulated images |
| `vdo_det` | path | deepfake / manipulated video |
| `aud_det` | path | synthetic speech and AI-generated audio |

## `DetectionResult`

| field | type | meaning |
|---|---|---|
| `score` | `float` 0..1 | continuous evidence, higher = more synthetic |
| `label` | `str` | `"ai"` \| `"real"` \| `"uncertain"` |
| `confidence` | `float` 0..1 | see *Calibration* below |
| `explanation` | `str` | evidence-backed justification |
| `kind` | `str` | `"generated"` \| `"altered"` \| `"both"` \| `"unknown"` |
| `signals` | `tuple` | which signals fired, strongest first |
| `runtime` | `dict` | `calibrated`, `backend`, `degraded` |
| `details` | `dict` | family scores, `upstream_label`, `label_basis`, `placeholder_label`, thresholds |

### How `label` is decided

**Right now `runtime["calibrated"]` is `False` for all four detectors, so
`label` is the upstream detector's own verdict, not a threshold of `score`.**

```python
r.details["label_basis"]       # "upstream"   (today)
r.details["placeholder_label"] # what the unfitted band *would* have said
r.details["label_reason"]      # "upstream_verdict" | "upstream_abstained"
```

Two consequences worth knowing before you read anything else:

* `score` is still reported for every result, and it is the only thing Phase 0
  evaluates (AUROC on raw scores). It just does not set `label` yet.
* The unfitted placeholder band (0.45 / 0.55) is published separately as
  `details["placeholder_label"]`. It can push a result toward `"uncertain"`,
  but it can **never** flip a fake to `"real"` - a contract test asserts that an
  upstream-FAKE input is never returned as `AUTHENTIC`.

This is a deliberate, temporary reintroduction of a `label`/`score` mismatch:
the thresholds are unfitted, and re-thresholding an uncalibrated score would be
inventing numbers. Expect `label` and `score` to disagree near the band until
Phase 0 fits the rule on a held-out calibration split.

### Once calibrated

```python
from aludam.thresholds import publish_blocked
publish_blocked()   # non-empty => do not ship
```

When `runtime["calibrated"]` becomes `True`, the label rule switches to

```
score <= low          -> "real"
low <  score <  high  -> "uncertain"
score >= high         -> "ai"
```

`details["label_basis"]` flips to `"score"`, `label = f(score)` holds again,
and `publish_blocked()` starts returning `[]`. Until it does, **Phase 4
publishing is blocked for every modality** - check `publish_blocked()` before
any release.

`thresholds` live in `aludam.thresholds` and are echoed back in
`details["thresholds"]`.

### `kind` is separate from the verdict

`label` answers *what to do* (accept / reject / abstain). `kind` answers *what
kind* of synthetic content it is, so a face-swap of a real photo reports
`kind="altered"` rather than being forced into the same bucket as content that
was generated from scratch.

## Calibration - read this

**Thresholds and `confidence` are NOT calibrated.**

```python
result.runtime["calibrated"]   # False
```

`confidence` is currently a *margin* - how far `score` sits outside the
abstain band - not a probability that the verdict is correct. It is deliberately
not the upstream detector's `confidence`, because for three of the four
detectors that number is a deterministic function of their own score (video:
`0.5 + 0.5 * score`; text and audio: `0.5 + |score - 0.5|` times a length
penalty), i.e. it carried no independent information.

Real values are fitted on a held-out calibration split (isotonic or Platt,
with a reliability check) and gated by a test split that nothing else touches.
Until then, treat `label` as provisional and read `runtime["calibrated"]`.

## Errors - nothing degrades silently

```python
from aludam import AludamDepsMissing, AludamWeightsMissing

det = load_detector("img_det")   # raises AludamDepsMissing if torch is absent
det.predict(path)                # raises AludamWeightsMissing if weights are absent
```

Missing dependencies and missing weights **raise**. They never fall back to a
heuristic score and never set `runtime["degraded"]`. That flag means something
much narrower: a model loaded, then failed during inference.

`InvalidInputError` covers empty text, missing paths and 0-byte files.

## Command line

```bash
aludam list
aludam predict text_det "some text to classify"
aludam predict img_det ./photo.png
aludam predict vdo_det ./clip.mp4 --json
```

Every subcommand prints the same four fields for every modality.

## Development

```bash
python tests/test_contract.py   # contract tests - no torch, no weights
python ../aludam-fullplate/tests/test_fullplate.py   # same API, both tiers
```

The contract suite asserts the guarantees above. Its core safety matrix is
one-directional: an input the upstream detector calls fake is **never** returned
as `AUTHENTIC`, whatever `score` happens to be, while thresholds are uncalibrated.

## Status

Alpha. The contract and packaging (plan phases 1-2) are built; Phase 0
measurement and the Phase 3 model swaps are still gated on measured
before/after numbers. See `Docs/ALUDAM_PLAN.md`.

`THIRD_PARTY_NOTICES` was fully verified on 2026-10-04: every dependency,
weight and evaluation set resolves to a named license source. Re-check it
before any release that changes a dependency pin or swaps a model.

## License

MIT - see `LICENSE`. Third-party model weights are fetched, not bundled; see
`THIRD_PARTY_NOTICES`.
