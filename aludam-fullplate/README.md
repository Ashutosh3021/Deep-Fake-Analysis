# aludam-fullplate

The neural tier of [aludam](https://pypi.org/project/aludam/): same import,
same syntax, same `DetectionResult` - plus the heavy dependency stack the four
detectors need to actually run.

```bash
pip install aludam-fullplate
```

is exactly

```bash
pip install "aludam[neural]"
```

`aludam-fullplate` is a thin wrapper whose only dependency is
`aludam[neural]`, so there is one dependency list to maintain and two names
to choose from. Both install torch, transformers, huggingface_hub, timm,
peft, opencv-python-headless, librosa, soundfile, scipy and onnxruntime.

## Usage

Identical to `aludam` - either import works:

```python
from aludam import load_detector        # or: from aludam_fullplate import ...

det1 = load_detector("text_det");  result1 = det1.predict(text)
det2 = load_detector("img_det");   result2 = det2.predict(image_path)
det3 = load_detector("vdo_det");   result3 = det3.predict(video_path)
det4 = load_detector("aud_det");   result4 = det4.predict(audio_path)

print(result1)
# DetectionResult(score=0.8012, label='ai', confidence=0.5582, explanation='...')
```

They are the *same* objects: `aludam_fullplate` re-exports `aludam`, it does
not wrap it. The result class, the errors, the CLI (`aludam ...`) and the
weight tooling all come from the one `aludam` package.

## Weights are fetched, never bundled

```bash
aludam fetch          # download model weights into the HF cache, once
aludam status         # what this install is missing
```

Wheels ship code and declarations only. Missing dependencies and missing
weights **raise** (`AludamDepsMissing`, `AludamWeightsMissing`); they never
degrade into a heuristic-looking score.

## Calibration - read this

`runtime["calibrated"]` ships `False`: thresholds and `confidence` are not
fitted yet, so `label` is the upstream detector's own verdict and
`details["placeholder_label"]` shows what the unfitted band would have said.
See the `aludam` README for the full rule.

## Status

Alpha. No accuracy claim is made for either tier until Phase 0 of the plan
measures one; `aludam-fullplate` exists to carry the neural stack, not to
advertise numbers. See `Docs/ALUDAM_PLAN.md`.

## License

MIT - see `LICENSE`. Third-party model weights are fetched, not bundled; see
`THIRD_PARTY_NOTICES` (full inventory in the `aludam` wheel's copy).
