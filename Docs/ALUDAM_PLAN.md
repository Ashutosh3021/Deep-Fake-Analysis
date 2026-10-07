# ALUDAM — Plan v1.1 (revised after review)

**Package names:** `aludam` (light) · `aludam-fullplate` (neural)
**Priorities, in order of what matters:** accuracy · lightweight · simplicity

> **Status: CONDITIONAL APPROVAL.**
> **Phases 0–2 approved for execution. Phases 3–4 are HELD** pending closure of Must-Fix items 1–4 below.
> Accuracy figures throughout are **claimed** (model card / leaderboard), never measured on this repo. Every "after" number is TBD until Phase 0 fills it.

---

## 0. Review response

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | Must fix | Dropping TF makes "RetinaFace if available" a permanent Haar fallback — the face-swap fix never happens | §4.1 — replace with OpenCV **YuNet** (`cv2.FaceDetectorYN`), verified available, zero new pip deps |
| 2 | Must fix | Before/after numbers aren't comparable metrics; RAID may be ModernBERT's tuning set | §5.1 — relabel as *claimed*, separate metric column, add non-RAID held-out set |
| 3 | Must fix | "Net flat" hides runtime cost; 2.3 GB peak is wrong; light tier gets no gains | §5.2 — split disk vs RAM, correct to ~3 GB and mark *measure*, scope "no tradeoff" to fullplate |
| 4 | Must fix | Weight distribution undefined; contradicts "dependency bundle only"; silent degradation on missing weights | §6.3 — explicit fetch step + `AludamWeightsMissing` raises (never degrades) |
| 5 | Should fix | 9% FP needs mitigation; `AI_GENERATED` ≠ `MANIPULATED` | §7.1 — calibrated thresholds, `INCONCLUSIVE` band, <128 px guard; verdict split `SYNTHETIC`/`ALTERED` |
| 6 | Should fix | Video "zero new deps" wrong; research-only license; dead repo | §4.3 — ffmpeg declared (already required), license gate, video is last swap |
| 7 | Minor | Header pre-empted review | Fixed — status line above |
| 8 | Minor | Bombek1 init fix listed in Phase 1, integrated in Phase 3 | Moved to Phase 3 (§8.3) |
| 9 | Minor | Second package only lists deps | §6.2 — `aludam` owns the `[neural]` extra; `fullplate` depends on `aludam[neural]` |
| 10 | Minor | Light tier audio needs scipy/onnxruntime | §6.1 — music moved to fullplate; scipy optional in light (fallback exists) |

---

## 0.1 Progress log

### Phase 2 (contract) — package scaffolded, 4/4 modalities live

New package at `aludam/` (src layout), installed editable into the repo venv.

| File | Purpose |
|---|---|
| `src/aludam/__init__.py` | exports `load_detector`, `DetectionResult`, error types |
| `src/aludam/registry.py` | `load_detector("text_det"\|"img_det"\|"vdo_det"\|"aud_det")`, models-dir resolution, lazy imports |
| `src/aludam/result.py` | frozen `DetectionResult` — the one class all four return |
| `src/aludam/normalize.py` | the four upstream verdict shapes → `DetectionResult` |
| `src/aludam/exceptions.py` | `AludamDepsMissing`, `AludamWeightsMissing`, `UnknownDetectorError`, `InvalidInputError` |
| `src/aludam/cli.py` | `aludam list` / `aludam predict <det> <input> [--json]` |
| `src/aludam/thresholds.py` | `decide(score)` -> label, `PLACEHOLDER` band, `margin()` |
| `src/aludam/al.py` | `al.detect(type=..., src=...)` thin wrapper over `load_detector` |
| `tests/test_contract.py` | 166 assertions, no torch/weights needed |

**Verified:** `python aludam/tests/test_contract.py` -> 166 passed, 0 failed
(+ 41 metrics, + 58 datasets = **265 passing**).
All four detectors ran end-to-end on real fixtures through the exact mandated
snippet (`load_detector(...)` -> `.predict(...)` -> `print(result)`), returning
the same class with the same four printed fields, via both `load_detector()`
and `al.detect()`.

Single-fixture smoke output (**not** an accuracy measurement; post-D4 rule):

| detector | score | label | kind | upstream_label | placeholder_label |
|---|---|---|---|---|---|
| `text_det` | 0.3695 | `uncertain` | `generated` | `UNCERTAIN` | `real` |
| `img_det` | 0.4225 | **`ai`** | `generated` | `FAKE` | `real` |
| `vdo_det` | 0.2650 | **`ai`** | `altered` | `FAKE` | `real` |
| `aud_det` | 0.0000 | `real` | `generated` | `AUTHENTIC` | `real` |

`runtime.calibrated` is `False` for all four. `label` now tracks
`upstream_label` (decision D4); `placeholder_label` shows what the unfitted
band would have said — on the image/video rows the two disagree, which is the
documented interim state, not a result.

Wall-clock (one Windows box, warm cache): load 1–10 s (121 s cold for image),
predict text 6.7 s · image 2.9 s · video 20–37 s · audio 7.6–19.9 s.

### Findings from wiring (feed Phase 0/1)

1. **`confidence` is 0–100 upstream, 0–1 here.** All four detectors compute
   `50.0 + distance * 50.0` (`final_text_detector.py:621`,
   `final_audio_detector.py:524`, `final_video_detector.py:838`; image clamps
   5–95 at `final_image_detector.py:438`). aludam divides by 100; the native
   value is kept in `details.native_confidence`.
2. **`label` and `score` use different rules and can disagree** — e.g. image
   returns `score=0.4225` with `label='ai'`. Cause: image/video set `FAKE` as
   soon as *any* family crosses its own threshold (`final_image_detector.py:410`,
   `final_video_detector.py:836`), while `score` is the *fused* magnitude.
   Recorded as `details.label_basis` (`family_threshold` | `score_threshold`).
   **Decision: preserve upstream behaviour in Phase 0 (measuring, not changing);
   revisit under calibration in Phase 1.**
3. **Signal-key alias was wrong:** video emits `audio_visual_desync`
   (`final_video_detector.py:814`), not `audio_visual_sync`. Mapped to
   `AV_SYNC_MISMATCH`.
4. **Upstream `to_dict()` shapes confirmed** — text: `ai_probability`;
   image: `p_synthetic`; audio: `fake_probability`; video:
   `family_scores.fused_score`. Image has **no** `metadata_analysis` key;
   provenance lives in `family_scores.provenance`.
5. **Counters are not signals.** audio `feature_summary` mixes
   `segment_count: 8` with `v2_amff: 0.91`; ranking them naively put
   "SEGMENT_COUNT 3.00" first. `NON_SIGNAL_KEYS` + a hard `[0,1]` range check
   now filter both.
6. **Audio confidence can reach exactly 1.0** (native 100). image/text/video
   formulas can too. Flagged for calibration — Must-Fix #5.
7. **Upstream bug, not ours:** audio segment scoring leaks a file handle and
   logs `WinError 32 ... tmp*.wav` after the process finishes.
8. **All package strings are ASCII-only** — non-ASCII in `explanation` raised
   `UnicodeEncodeError` on the Windows console (cp1252) and would break JSON
   and log sinks. Enforced across `aludam/src/**.py` (0 non-ASCII lines).
9. **Upstream `confidence` carries no independent information.** For video it is
   `0.5 + 0.5*score`; for text and audio it is `0.5 + |score-0.5|` times a
   length penalty — i.e. a deterministic function of the score the caller
   already has. aludam does not surface it as confidence; the native value is
   retained as `details["upstream_confidence_0_100"]`.
10. **`label` follows `score` only once calibrated.** `thresholds.decide()`
    implements `label = f(score)`, but it is *not* used while
    `runtime.calibrated=False` — see decision D4 below. The upstream verdict is
    preserved as `details["upstream_label"]` (+ `upstream_label_uniform`).
    Which scoring rule feeds `score` (fused vs rescaled max-of-family-margins)
    is decided in Phase 0 by AUROC.
11. **`kind` is derived from the detector's authoritative `fake_type` list,**
    falling back to signal names — not from numeric scores. Deriving it from
    scores mislabelled the image fixture as `both`.
12. **Placeholder thresholds produce false negatives — resolved by D4.**
    `low=0.45, high=0.55` in `thresholds.py` are unfitted; image/video
    fixtures whose upstream verdict is `FAKE` land at score 0.42/0.265 and the
    band alone would label them `real`. Under D4 those inputs now report the
    upstream verdict instead, so the false negative is gone; the unfitted band
    survives only as `details["placeholder_label"]`, which can push toward
    `uncertain` but can never flip a fake to `real`. Thresholds still must be
    fitted on `mage:valid` before `calibrated` can go `True`.
13. **OpenCV conflict in the venv:** `opencv-contrib-python 4.11.0.86` (pulled
    by `mediapipe`) and `opencv-python 4.8.0.76` (pinned by `retina-face`,
    `ultralytics`) are both installed; `cv2` resolves to 4.11.0. Deduping breaks
    the metadata of three packages *and* changes face-detection behaviour,
    which would contaminate the Phase 0 baseline. Deferred as a decision.
14. **Packaging is release-ready but gated.** `python -m build` + `twine check`
    both pass; the wheel ships `LICENSE` and `THIRD_PARTY_NOTICES` under
    `dist-info/licenses/` (PEP 639 `license-files`, because the default
    `LICENSE*` glob does not match `THIRD_PARTY_NOTICES`). Publishing stays
    blocked until `THIRD_PARTY_NOTICES` has no `PENDING` blocking rows.

### Phase 0 progress (datasets + metrics)

`aludam/eval/` now exists: `metrics.py`, `fetch.py`, `datasets.py`, plus
`tests/test_metrics.py` (41) and `tests/test_datasets.py` (58). Network
confirmed; both text golden sets cached and pinned to commit SHAs.

| dataset | split | n | synthetic / authentic | license |
|---|---|---|---|---|
| MAGE `yaful/MAGE` | test | 56,819 | 28,078 / 28,741 | apache-2.0 |
| MAGE | valid (calibration) | 56,792 | 27,993 / 28,799 | apache-2.0 |
| MAGE | ood_gpt (unseen domain + GPT-4) | 1,562 | 800 / 762 | apache-2.0 |
| MAGE | ood_gpt_para (paraphrase attack) | 2,362 | 1,600 / 762 | apache-2.0 |
| HC3 `Hello-SimpleAI/HC3` | all (5 domains) | 85,431 answers | 26,885 / 58,546 | cc-by-sa-4.0 |

223,545,014 bytes cached. MAGE `train.csv` (404 MB) deliberately **not**
fetched - we never score on it, and a later model swap may have trained on it.

15. **MAGE's label convention is inverted relative to ours** - native
    `1 = human-written`, `0 = machine-generated`. `datasets.py` converts and
    then *proves* it from the `src` column (`*_human` must be native 1),
    raising rather than guessing. A silent inversion would invert every
    reported metric while still looking plausible.
16. **52 texts are duplicated between MAGE `valid.csv` and `test.csv`**
    (0.09%; zero duplicates inside either split). Fitting on calibration and
    scoring those same strings on test is leakage. `drop_seen()` removes them;
    the runner must call it. Immaterial to AUROC at this size, but a protocol
    that knowingly leaks is not a protocol.
17. **`hf_hub_download` defaults to `repo_type="model"`** - fetching datasets
    without passing `repo_type="dataset"` returns a confusing 404/401. The
    first network check used `gpt2` (a model) and so missed this entirely;
    verifying "the download path works" against a different repo *type* is
    not verification.
18. **MAGE groups are origin domains** (`cmv_human`, `xsum_machine_...`), 322
    of them across both splits - so calibration and test share group ids by
    design. The leakage unit is the *text*, not the group. HC3 groups are per
    question, keeping a question's human and ChatGPT answers together under
    the bootstrap.


### Decisions taken (user directives, 2026-10-04)

All four open questions below are now closed. Answered in chat, not via the
interactive question tool.

**D1 — OpenCV (finding 13): leave both trees, snapshot as-is. (a)**
Do **not** dedupe before the baseline. `aludam/eval/env_snapshot.py` records
the environment, and it settled the "installed last wins" question
concretely:

```
cv2.__version__ : 4.11.0
active owner    : opencv-contrib-python          (version_match=True)
                  opencv-python 4.8.0.76         (path_match=True, version_match=False)
contrib modules : dnn_superres, face, text, ximgproc
```

`cv2` resolves to **`opencv-contrib-python 4.11.0.86`**; the pinned
`opencv-python 4.8.0.76` is installed but its files were overwritten (both
distributions claim the same `site-packages/cv2`, so path comparison alone
cannot distinguish them — `cv2.__version__` can). Dedupe is scheduled as its
own *measured* change after the baseline, judged by the face-recall
comparison from the YuNet work (§4.1).

**D2 — Scope freeze: strictly additive through all of Phase 0. (a)**
`models/`, `backend/` and `requirements.txt` stay untouched. Findings are
recorded here, not patched in place; fixes land in the ported copies under
`aludam/`. Consequence for reporting: every "after" number includes our ported
fixes *and* any model swap, so the baseline report must carry a
**ported-but-unswapped intermediate arm** to separate the two effects.

**D3 — Legacy `requirements.txt`: fix in a separate commit *after* Phase 0. (a)**
Baseline environment stays untouched. The baseline report must state plainly
that the file's comment is wrong: it claims TensorFlow was removed while
`retina-face==0.0.13` still pulls `tensorflow` (1397 MB, never imported by
default). Finding 19.

**D4 — Thresholds: no stopgap, nothing fitted yet.**
Phase 0 evaluates **raw scores with AUROC**, so unfitted thresholds cannot
move the baseline. Label rule while `runtime.calibrated == False`:

* `label` = the **upstream detector's own verdict**, mapped into
  `ai` / `real` / `uncertain` — legacy-faithful, via
  `thresholds.decide_uncalibrated()`;
* `details["label_basis"]` = `"upstream"` (becomes `"score"` once calibrated);
* `details["placeholder_label"]` = what the unfitted 0.45/0.55 band would
  have said — published, never applied. It may push toward `uncertain` but
  **can never flip a fake to `real`**;
* contract test asserts an upstream-`FAKE` input is never output as
  `AUTHENTIC`;
* `thresholds.publish_blocked()` returns a non-empty list for as long as any
  modality reports `calibrated: false` → **Phase 4 publishing is gated on it.**

This temporarily reintroduces the `label`/`score` mismatch previously argued
against. Accepted knowingly: re-thresholding an uncalibrated score would be
inventing numbers. It ends the moment Phase 0 fits the rule on `mage:valid`.

**D5 — Evaluation data: three sets, fixed by hand. (user directive)**
Per modality, one primary set — chosen so **no set is a model's own training
distribution** — plus Deepfake-Eval-2024 as video primary and a secondary
in-the-wild check on image and audio:

| modality | primary | role of Deepfake-Eval-2024 | excluded |
|---|---|---|---|
| image | `OwensLab/CommunityForensics-Eval` | secondary in-the-wild check | `ComplexDataLab/OpenFake` |
| audio | `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` (ODbL) | secondary in-the-wild check | — |
| video | `nuriachandra/Deepfake-Eval-2024` (gated) | **primary** | — |

Rationale: video is the riskiest swap (dead upstream repo, research-only
card, Sora/Veo2 scope, no temporal reasoning) and so needs the strongest
evidence; the Deepfake-Eval-2024 paper reports large AUC drops for
open-source detectors, so `resnet50-t2v` will likely land far below its
97.3% card number. Deferring video to a later gate would stall the one
subsystem everything else depends on. One set cannot carry three modalities:
the image set has only 1,975 rows (wide CIs), single source, manual labels.

* **OpenFake is excluded because of leakage**: Bombek1's headline 0.9997 AUC
  is measured on OpenFake, its own `~95K train / 5K val` (finding 23). It is
  in-distribution for the model under test.
* **Splits are by source/uploader, never random**, into dev / calibration /
  test; the test split is touched once.
* **Calibration never touches Deepfake-Eval-2024** — see clause 3 below.
* **Sample size 750/class (1,500 rows)** for the open sets; Deepfake-Eval is
  subsampled to 150–200/class, stratified, with clip duration capped (first
  30 s) identically for every arm, per-sample results written to resumable
  JSONL.
* **License (not legal advice).** Evaluating on CC-BY-SA / ODbL data does not
  impose share-alike on our code or wheel: those obligations attach to
  redistributing the data or derivatives of it, and we redistribute neither.
  **No sample or subset is ever committed or shipped** — only ID lists,
  dataset revision SHAs and metrics. The cache is gitignored. All three sets
  are listed in `THIRD_PARTY_NOTICES` under a separate
  **"evaluation data (not distributed)"** section.

**D5.1 — Deepfake-Eval-2024 gated terms, and the one clause that touches us.**
Terms read before acceptance. In full:

> 1. You agree to not use this dataset for the development or improvement of
>    any technologies which have the potential to harm individuals,
>    institutions, or societies.
> 2. CC-BY-SA-4.0 … users are responsible for ensuring its legal use in
>    commercial settings.
> 3. **Users may only use this dataset for evaluation. Use of this dataset for
>    training goes against the terms of use.**

* **Publishing metrics: no conflict.** The card's own Direct Use says the
  dataset "can be used to benchmark detection methods"; nothing anywhere
  restricts publishing results.
* **Clause 3 vs. threshold fitting — resolved conservatively.** Threshold
  fitting is a fit on their rows and reads uncomfortably close to "training".
  Rather than argue the point: **all calibration is fitted on the open sets**
  (ASVspoof2021-DF, CommunityForensics-Eval), which carry no such clause, and
  Deepfake-Eval-2024 is **strictly test-only, touched once**. That also
  honours the "touch test once" rule. Its roles are unchanged: video primary
  + in-the-wild secondary check on image and audio.
* Clause 1 is a behavioral covenant beyond the CC license — our work is
  detection, so it points the way we already go; recorded, not argued.
* Access still requires accepting the form in a browser. Total size 20.3 GB.

**19. **`requirements.txt` comment is false.** Lines 14-16 claim TensorFlow was
    removed; `retina-face==0.0.13` (line 32) still installs it, and
    `opencv-python==4.8.0.76` (line 35) is shadowed by the contrib wheel (D1).
    Both must be called out in the baseline report (D3), fixed after Phase 0.

**20. **A snapshot is the only thing that makes a baseline auditable.**
    `eval/env_snapshot.py` captures git SHA + dirty flag, `pip freeze`, the
    OpenCV listing and owner, `cv2.getBuildInformation()`, every cached HF
    snapshot revision, both pinned dataset SHAs, and the effective value of
    every behaviour-changing env var (`DEEPGUARD_FACE_DETECTOR` unset →
    `haar`, so **the baseline is a Haar baseline**; secrets recorded as
    set/unset only). Written to `eval/data/env_snapshot.json`.

**21. **The upstream modules degrade silently, and by design.** This is the
    concrete evidence behind Must-Fix #4, read out of the shipped code rather
    than inferred:

    * `final_audio_detector.py:107` — `if os.path.exists(AUDIO_CLASSIFIER_PATH)`
      guards the pickled classifier. The file is **absent** from this
      checkout, so `_trained_classifier` stays `None` with no warning.
      `_single_predict` (line 342) then walks trained → HF →
      `_heuristic_predict`, and line 506 tags the result
      `source = "heuristic_fallback"` — still a confident-looking 0–1 score.
    * `final_image_detector.py:146` — `except Exception` around the model load
      logs *"Will fall back to acoustic-heuristic mode"* style text and
      continues; line 459 then appends
      *"Pretrained global classifier unavailable"* to `notes`. The score is
      published regardless.
    * `final_video_detector.py` contains **no** `from_pretrained`, `torch.load`
      or weight file reference at all — `vdo_det` is pure forensics and has
      nothing to fetch.

    `aludam.weights.ensure()` now runs in `registry._build` **before** the
    detector module is imported, which is the only point at which this is
    still catchable — after import the fallback has already happened.
    `models/audio_rf_classifier.pkl` is listed as *optional* (it is trained
    locally by `final_audio_detector.train_classifier`, so it cannot be
    fetched) and is reported by `aludam status` but never raises.

**22. **`text_det` does not separate MAGE's classes — preliminary.** (In-flight
    run, so treat the numbers as indicative until it lands.) Scores were
    verified correct first: `DetectionResult.score` equals the upstream
    `ai_probability` exactly (delta 0.000000000 on three probes), so this is a
    property of the detector, not of our mapping. At n≈175 of 1500:
    `AUROC ≈ 0.48` (synthetic mean 0.319 vs authentic mean 0.326 — the
    authentic class scores *higher*), **zero** of ~80 synthetic rows emitted
    `AI_GENERATED`, upstream abstained on ~75% of rows, and the score never
    exceeded 0.62. Implication for D4: with an upstream that never says `ai`,
    `label` is useless for rejection *regardless of what the calibrated band
    says* — calibration cannot rescue a score that carries no signal. Full run:
    `eval/runs/mage_test.text_det.baseline-legacy.jsonl`.

**23. **Bombek1 trained on OpenFake, and its 9% FP number is measured on
    COCO — both read off its card.** `datasets: nebula-9000/OpenFake` in the
    frontmatter, and the training table says `Dataset | OpenFake (~95K train,
    5K val)`. OpenFake is therefore in-distribution for Bombek1 and is
    excluded (D5). The ~9% FP claim comes from the held-out row
    `COCO 2017 | 300 | 90.67% accuracy | mean P(AI) 0.135`, and COCO appears
    only in the generalization table — never in training — so
    CommunityForensics-Eval's COCO/FFHQ-paired reals test exactly that claim
    on data the model was not fitted to. Two residuals: (a) generator overlap
    — SD / Flux / Midjourney sit in both OpenFake and CommunityForensics, so
    the *images* are disjoint but the *generators* are not; (b) the card's
    own `Performance degrades on images <128×128 (CIFAKE 32×32 → ~50%
    accuracy)` is a second independent source for the `<128 px` guard.

**24. **`img_det`'s current model says it is not a deepfake photo detector.**
    `umm-maybe/AI-image-detector` — the model under test today, not Bombek1 —
    was created **October 2022** and its card states the training data
    "did not include any samples generated by Midjourney 5, SDXL, or
    DALLE-3", that "the intended scope of this tool is **artistic images**;
    that is to say, it is **not a deepfake photo detector**", and it redirects
    the reader to `Organika/sdxl-detector`. Consequences: (a) expect
    CommunityForensics-Eval to score poorly — that is the honest
    out-of-distribution baseline, not a defect, but it must never be printed
    beside Bombek1's 0.9997 without noting that the latter is an in-distribution
    number; (b) the card carries an internal **license contradiction**,
    frontmatter `license: cc-by-4.0` against a body "License Notice" of
    **CC-BY-ND 4.0** (NoDerivatives) — recorded in `THIRD_PARTY_NOTICES`.

**25. **`aud_det`'s training distribution is undisclosed at every level of its
    lineage.** `MelodyMachine/Deepfake-audio-detection-V2` is an
    auto-generated card reading `Training and evaluation data: More
    information needed`, with `datasets: audiofolder` (generic). The lineage
    was followed to the root — `motheecreator/Deepfake-audio-detection` →
    `mo-thecreator/wav2vec2-base-finetuned` → `facebook/wav2vec2-base` — and
    every intermediate card says the same thing. No mention of ASVspoof2021,
    but no exclusion either. Per D5 this is **not** resolved in our favour:
    results on ASVspoof2021-DF are to be reported as **possibly
    in-distribution**, in the report itself, wherever they appear.

### Still to build

- **Finish the in-flight baseline run** (`mage:test`, 750/class, ETA ~3.5 h),
  then `eval/baseline_report.py` for the first real `text_det` number.
  Finding 22 says what to expect.
- **Score `mage:valid`** (its own run) so `eval/fit.py` can fit the band,
  write `src/aludam/data/thresholds.json`, flip `calibrated` to `True` and
  clear `publish_blocked()` (D4). Fitting is the only thing standing between
  us and a calibrated `label` for `text_det`.
- **Image / audio / video golden sets are not yet built** — sets are now
  chosen (D5) but not ingested: `CommunityForensics-Eval` (image),
  `ASVspoof2021_DF` (audio), `Deepfake-Eval-2024` (video, gated — needs a
  browser accept). Still the largest unknown in Phase 0. Both leakage checks
  that gate finalizing them are done (findings 23–25): image is clean,
  audio must ship flagged *possibly in-distribution*.
- **Source/uploader-grouped dev / calibration / test splits** for those three
  sets; test touched once; calibration on open sets only (D5.1).
- `<128 px` guard, calibrated `INCONCLUSIVE` band, `aludam-fullplate`.
- `THIRD_PARTY_NOTICES` `PENDING` rows resolved (publish gate).
- OpenCV dedupe as a separate measured change, judged on face recall (D1).
- `requirements.txt` fix in its own commit after Phase 0 (D3).

### Done this session

| item | state |
|---|---|
| D1-D5 decisions recorded above | done |
| leakage checks: Bombek1 / umm-maybe / MelodyMachine training data | done — findings 23-25 |
| Deepfake-Eval-2024 gated terms read; metric publishing confirmed clean | done (D5.1) |
| `thresholds.decide_uncalibrated` / `publish_blocked` | done |
| label rule in `normalize.py` + `label_basis` / `placeholder_label` | done |
| contract safety matrix (fake never returns `real`) | done — 166 pass |
| `eval/env_snapshot.py` (D1 evidence) | done |
| `eval/run.py` — `plan` / `score` / `report`, resumable JSONL | done (finding 21 fixed: `_rowid` preserved across `drop_seen`, shuffled sampling, `src_domain` stratum) |
| `eval/baseline_report.py` — env + metrics + strata + 2-arm gate | done (generator; the report itself waits on the run) |
| `eval/fit.py` — band fitting + threshold artifact | done (40 tests); execution waits on `mage:valid` |
| score-mapping verified: `score == ai_probability` exactly | done (basis for finding 22) |
| `weights.py` + `aludam fetch` / `status` (Must-Fix #4) | done — `ensure()` runs before import, raises, never degrades |
| `huggingface_hub` declared in the `neural` extra | done |
| `README.md` rewritten for the new label rule | done |
| build + `twine check` | passing |
| **suite total** | **373 passed, 0 failed** |

---

## 1. The problem

### 1.1 The accuracy gap is documented in this repo, not felt

`new_plan.md:475-479` records targets vs. current state:

| metric | current | target | gap |
|---|---|---|---|
| Image AUC (cross-method) | **~88%** | >95% | **−7pt** |
| Text detection accuracy | **~82%** | >90% | −8pt |
| Fusion AUC | **~92%** | >99% | −7pt |

`plan.md:8` states the core problem itself: unimodal models "fail up to 80–88% of the time" out-of-distribution.

**There is no eval harness in the repo.** `tests/test_detectors.py` is a *smoke* test — it asserts each detector imports and returns a dict under `HF_HUB_OFFLINE=1`. Nothing measures AUC. "No accuracy tradeoff" cannot be defended without a number to compare against.

### 1.2 Four root causes of the accuracy drop (found in code)

1. **Face detector defaults to Haar, not RetinaFace** — `forensics_core.py:605`:
   ```python
   backend = os.environ.get("DEEPGUARD_FACE_DETECTOR", "haar").strip().lower()
   ```
   Haar is frontal-only and brittle to pose/occlusion. The `face_swap` family — the most important signal for the "image with face" priority — runs on the weakest available detector by default. Meanwhile **TensorFlow (1397 MB) is installed to support a backend that never loads.**
   *Revised in v1.1: see §4.1 — RetinaFace is not the fix, because it requires the TF we are deleting. YuNet is.*

2. **Failures degrade silently to heuristic-only mode.** `final_image_detector.py:146-149` swallows model-load errors into a warning and still returns HTTP 200. Same pattern in audio (`notes_map`), text (`"Probe LM unavailable"`), video (`"Per-frame image forensics unavailable"`). Accuracy can collapse with zero visible error.

3. **Offline mode is the default.** `tests/test_detectors.py:31-34` sets `HF_HUB_OFFLINE=1` unless `DEEPFAKE_ALLOW_DOWNLOADS=1`. If weights are not cached, every detector falls back to heuristics silently. *This is what makes Must-Fix #4 dangerous: first-run weight fetch must fail loudly, not degrade.*

4. **The current image model is the weakest link.** `umm-maybe/AI-image-detector` (665 MB) is an autotrain classifier with no published metrics — that is the ~88%.

### 1.3 Weight: disk vs RAM (corrected)

These are two different budgets and v1.0 conflated them.

**Disk**

| item | size | note |
|---|---|---|
| tensorflow 2.17.1 (pulled by `retina-face`) | **1397 MB** | **Never imported by default** (`USE_TF=0`, backend=haar) |
| torch 2.5.1+cpu | 1131 MB | required |
| gpt2 weights | 525 MB | droppable |
| `umm-maybe` image model | 665 MB | replaced |
| audio model repo | 1804 MB | kept |
| mediapipe | 97 MB | droppable |
| Render free RAM | **512 MB** | — |
| torch + image model + Flask RSS | ~600 MB → **over** | — |
| torch + gpt2 + Flask RSS | ~820 MB → **over** | — |

Verified: `retina-face==0.0.13` declares `Requires-Dist: tensorflow (>=1.9.0)`, so `requirements.txt`'s claim that TF was removed is false — pip reinstalls it transitively.

**Deleting TF therefore saves ~1397 MB of disk and build time, and saves nothing in RAM**, because it was never loaded. v1.0's "net −54 MB, +9pt AUC" framed a disk win as if it paid for a runtime cost. It does not. The honest statement:

> Removing TF makes the *install* smaller. The image swap makes *runtime* materially heavier: weights go 665 MB → 2008 MB on disk, and fp32 resident weights for 740M params are **~2.96 GB alone**, before torch overhead and activations. **The v1.0 figure of ~2.3 GB peak RAM was wrong.** Estimated peak for the image path is **~3.0–3.5 GB — measure in Phase 3, do not trust this estimate.**

That trade is defensible because accuracy is priority #1. It is a trade, not a free win.

### 1.4 Simplicity debt

- 4 detectors → 4 different result shapes → 4 code paths in every consumer
- `backend/api.py` = **982 lines**, with `mock_*_detection` fallbacks duplicated 4×
- 3 overlapping plan docs (68 KB total) describing a 5-layer architecture the code only partly implements
- **Documents don't work at all:** `get_file_type` accepts `pdf/doc/docx`, but `detect_text` expects a JSON string and no PDF/DOCX extraction library exists in `requirements.txt`.

### 1.5 The OOM mechanism

`backend/api.py:56` — the `models` dict caches every detector forever, plus startup warmup loads them eagerly. All four torch models resident at once. With the image swap this gets *worse* unless eviction lands first.

---

## 2. Reference evaluation

### 2.1 Bombek1/ai-image-detector-siglip-dinov2 (image)

| axis | finding |
|---|---|
| **Accuracy (claimed)** | **0.9997 AUC** (OpenFake 5K val), **97.15% cross-dataset**, AI-or-Not 96.8%/AUC .9986. Real datasets 96.13%, AI datasets 98.17%. Quality-agnostic (clean vs degraded gap = 0.0003). MIT. |
| **vs. current** | Claimed 97.15% vs your measured-in-plan ~88% — **not comparable metrics, see §5.1** |
| **Lightweight** | ❌ **2008 MB** (SigLIP2-SO400M + DINOv2-Large, 740M params) — 3× current 665 MB. Deps add `timm`, `peft` (small). |
| **Simplicity** | ✅ Ships `model.py` → `AIImageDetector(path).predict()` → `{prediction, confidence, P(AI)}` |

Card's own limits: degrades below 128×128; **~9% FP on COCO-style cluttered photos**; ~5% on studio/macro; not for screenshots/graphics.

**Critical scope nuance:** this detects *AI-generated images*. The existing `face_swap`/`splice` families detect *manipulation of real photos*. **Complementary, not interchangeable.** It must be added as a new family, not swapped in — and it must not be allowed to set a `MANIPULATED` verdict on its own (§7.1).

### 2.2 lofcz/ai-music-detector (audio)

| axis | finding |
|---|---|
| **Accuracy (claimed)** | **99.88%** acc, P .9985 / R .9998 / F1 .9991, FPR .31% on 17,866 held-out. MIT. Afchar et al. ISMIR 2025 (deconvolution spectral artifacts). |
| **Lightweight** | ✅✅ **`ai_music_detector.onnx` = 14,795 bytes (14.8 KB)** — confirmed via HF tree API. Feature extraction is FFT + spectral envelope in numpy/scipy. |
| **Simplicity** | ✅ Full Python: `src/python/inference.py`, `extract_fakeprints.py`, `export_onnx.py` |

**But it needs `onnxruntime`, which is not installed** (verified). So despite the 14.8 KB model, this path is **not** light-tier — see §6.1.

**Task mismatch:** it detects **AI-generated music** (Suno ≤5, Udio ≤1.5). The current audio detector targets **spoofed speech** (`MelodyMachine/Deepfake-audio-detection-V2`, ASVspoof-style). Different problem — a *second, complementary* capability, not a replacement.

Architecture fingerprinting → needs retraining when generators change.

### 2.3 gaetanbrison/deepfake-detector-resnet50-t2v-sora-veo2 (video)

| axis | finding |
|---|---|
| **Accuracy (claimed)** | AP .993, **acc 97.3%**, AUROC .988 on FakeParts. sora 98%, veo2 100%, ytb 94%. BSD-3-Clause. |
| **Lightweight** | ✅ **90 MB**, resnet50 backbone |
| **Simplicity** | ✅ resnet50 → 224×224 → softmax; 4–8 frames averaged |

Risks (all confirmed, none speculative):

1. **Code repo is dead.** `pip install git+https://github.com/gaetanbrison/deepfake-detector` → **404**. The user account exists with ~60 repos, none named that. Only weights + `config.yaml` remain on HF. Inference must be reimplemented.
2. **Per-frame only — no temporal reasoning.** Averaging per-frame scores loses `temporal_inconsistency` and `audio_visual_sync` as deep families.
3. Card's own limits: **closed-source generators only** (Sora/Veo2), "cross-generator transfer is poor across the field"; real class = YouTube only → FP risk; **"Not a forensic-grade tool. Research / defensive use only."**
4. **"Zero new dependencies" in v1.0 was wrong.** Audio extraction from video requires a decoder. In practice **ffmpeg is already a hard dependency of the current code** — `final_video_detector.py:540` shells out to `ffmpeg -i <video> -vn -acodec pcm_s16le`, and `tests/test_detectors.py:300-304` probes `shutil.which("ffmpeg")`. So D2 does not introduce a *new* dependency; it leans on an **undeclared** one. It must be declared as a prerequisite (or PyAV/`av` offered as a pure-pip alternative — `av` is not installed).

### 2.4 Cross-cutting issue

Two of three references are **generation** detectors ("was this content *generated*?"). The existing architecture is a **manipulation** detector ("was real content *altered*?" — splice, face swap, AV sync). The fusion layer and signal taxonomy are built around manipulation families. Generation detectors must be added as *additional* families, never substituted — and the verdict vocabulary must stop conflating the two claims (§7.1).

---

## 3. Text reference (sourced independently)

| candidate | verdict |
|---|---|
| **MELD** | Top of RAID leaderboard (AUROC 99.82 attacked / 99.85 clean), **but no released weights found.** Rejected. |
| **Binoculars** | Training-free, strong at low FPR — but requires Falcon-7B ≈ 15 GB. Rejected. |
| **fast-ai-detector** | 40M distilled model, RAID balanced acc 0.9642 — but default `contrast` mode only 0.8078, drops to 0.6731 on Pangram; card warns no GPT-5-era coverage. Rejected as primary. |
| ✅ **ModernBERT-Detect** | `AICodexLab/answerdotai-ModernBERT-base-ai-detector` — **570.7 MB, Apache-2.0, 7054 downloads.** RAID leaderboard: **AUROC 97.65 (with attacks) / 99.12 (clean).** |

**Circularity caveat (Must-Fix #2):** ModernBERT-Detect is a RAID-leaderboard entry. Scoring it with `raid-bench` on RAID would partly re-measure its training distribution. **Phase 0 must include at least one non-RAID held-out set** (HC3, MAGE, or Ghostbuster) or the text gain is unproven.

**Eval harness:** `raid-bench` **v0.2.0 is on PyPI** — `detect_cli.py` / `evaluate_cli.py`.

Research finding that explains the current ~82%: training-free perplexity detectors top out around **0.91 AUROC even when corrected for polarity inversion**, and *"no detector generalizes robustly across llm sources and domains simultaneously."* The existing gpt2-probe approach is not broken — it has hit its method ceiling. This also means the ~82%→97.65% jump is **method change, not tuning**, and must be proven rather than asserted.

---

## 4. Decisions

### 4.1 Image: Bombek1, sole image model, fp32 — **with a TF-free face detector**

Replaces `umm-maybe/AI-image-detector`. Existing provenance/splice/face-swap families are **kept alongside** — Bombek1 answers *"was this generated?"*, the existing families answer *"was real content altered?"*. Both are needed for "image with face".

| disk | before | after |
|---|---|---|
| image weights | `umm-maybe` 665 MB | Bombek1 2008 MB |
| tensorflow (never loaded) | 1397 MB | **0** |
| **net disk** | | **−54 MB** |

**Runtime (corrected, §1.3):** image path resident weights go 665 MB → **~2.96 GB fp32 for 740M params**, est. **~3.0–3.5 GB peak**, from ~1 GB today. **This is a real cost and it is not paid for by the TF removal.** Defensible because accuracy is priority #1 — but it must be stated, and measured.

**Face detection fix (Must-Fix #1).** v1.0 proposed "RetinaFace if available." That is broken: RetinaFace requires tensorflow, and we are deleting tensorflow, so the condition is permanently false and the code falls back to Haar — root cause #1 never gets fixed. Correct fix:

- **Primary: OpenCV YuNet** — `cv2.FaceDetectorYN`, a tiny ONNX model (~230 KB), no TensorFlow, no torch.
  *Verified: `cv2.FaceDetectorYN_create` exists in the installed cv2 4.11.0; YuNet shipped in 4.5.4, well below the `opencv-python==4.8.0.76` pin. **Zero new pip dependencies.***
- **RetinaFace becomes opt-in only** for users who deliberately install tensorflow; it must never be a silent default.
- Haar stays as the last-resort fallback, and **which backend ran is reported in `runtime`** (not logged).

This makes the face-swap accuracy fix genuinely achievable while *also* deleting 1397 MB — the opposite of the v1.0 contradiction.

fp32, not fp16/int8 — quantization would risk the one thing that must not be traded.

### 4.2 Video: resnet50-t2v ⊕ audio detector on the video's audio track

90 MB resnet50 for frame-level T2V (claimed 97.3%), plus **running the existing audio detector on the extracted audio track** — catches speech deepfakes and yields an AV signal using components that already exist. Keep a ~30-line numpy temporal-continuity heuristic. **Delete mediapipe, optical-flow, and per-frame splice.**

- **Dependencies — corrected:** audio extraction already requires **ffmpeg** (undeclared today, see §2.3.4). Must be declared as a prerequisite with a clear error if absent, or PyAV offered as a pip alternative. Claiming "zero new dependencies" was wrong.
- Gains: T2V detection (near-zero today), −97 MB disk, 5 families → 3 signals
- Loses: deep per-frame splice forensics — acceptable, video is not the stated accuracy priority
- Own the ~60-line resnet50 inference (upstream repo is 404; model card gives the exact recipe)
- **Gated:** BSD-3-Clause license review alongside the audio model, and **video is the last swap in Phase 3**, behind image and text.

### 4.3 Text: ModernBERT-Detect + keep stylometry; drop gpt2 probe

- Primary: `AICodexLab/answerdotai-ModernBERT-base-ai-detector` (570.7 MB, Apache-2.0) → **claimed RAID AUROC 97.65 / 99.12** vs current **~82% accuracy**
- Weight: 571 MB vs gpt2's 525 MB → **+46 MB disk**
- **Drop the gpt2 probe + DetectGPT curvature** — a whole code path and 525 MB gone
- Existing stylometry/pattern features become the *light* path: transformer-free, interpretable, and they are what feed `explanation`
- **Gain is unproven until Phase 0 runs a non-RAID held-out set** (§3 circularity caveat)

---

## 5. The ledger (corrected)

### 5.1 Accuracy — claims, not measurements

| modality | current (repo's own `new_plan.md`) | **claimed by source** | metric used by source | **measured on this data** |
|---|---|---|---|---|
| Image | ~88% AUC (unverified) | 97.15% | cross-dataset **accuracy** | **TBD Phase 0** |
| Text | ~82% accuracy (unverified) | 97.65 / 99.12 | **AUROC**, possibly tuned on RAID | **TBD Phase 0 (non-RAID)** |
| Video (T2V) | ~unsupported | 97.3% accuracy | FakeParts held-out | **TBD Phase 3** |
| Audio (speech) | unchanged | — | — | **TBD Phase 0** |
| Audio (music) | **absent** | 99.88% accuracy | 17,866 held-out | **TBD Phase 3** |
| Fusion | ~92% AUC | — | — | **TBD Phase 0** |

**v1.0's "+9pt AUC" and "+15pt" are withdrawn.** Row 1 compares your *accuracy* against a model card's *accuracy* on a different dataset — not comparable, and definitely not "AUC vs AUC." Row 2 compares your *accuracy* against a *AUROC*, two different metrics entirely. **No gain is claimed until Phase 0 produces a number under a single protocol.**

Also carried forward as free accuracy work: face backend → YuNet (§4.1), and every silent heuristic fallback surfaced as an explicit `runtime.degraded` field.

### 5.2 Weight — disk and RAM stated separately

**Disk (real, arithmetic verified):**

```
REMOVED   tensorflow        −1397 MB   (retina-face; never imported by default)
REMOVED   umm-maybe         −665 MB
REMOVED   gpt2 probe        −525 MB
REMOVED   mediapipe         −97 MB
ADDED     Bombek1          +2008 MB
ADDED     ModernBERT       +571 MB
ADDED     resnet50-t2v     +90 MB
ADDED     music fakeprint  +0.015 MB   (14,795 bytes)
─────────────────────────────────────
NET DISK   −15 MB  (essentially flat)
```

**Runtime RAM — not flat, and not saved by the TF removal:**

| path | before (weights resident) | after (weights resident) | Δ |
|---|---|---|---|
| image | ~0.7 GB | **~2.96 GB fp32** (est. peak 3.0–3.5 GB) | **+2.3 GB** |
| text | ~0.5 GB (gpt2) | ~0.57 GB (ModernBERT) | +0.06 GB |
| video | mediapipe + frame buffers | ~0.09 GB (resnet50) | **−large** |
| music | n/a | ~0 (14.8 KB) | — |

TF removal: **1397 MB disk, 0 MB RAM** (it was never loaded).

**Eviction is now mandatory, not optional.** `backend/api.py:56` caches every detector forever; with the image swap, retaining all four would be far worse than today. With lazy load + LRU eviction, peak = `torch + one modality`:

| peak after eviction | estimate | status |
|---|---|---|
| image | ~3.0–3.5 GB | **measure in Phase 3** |
| text | ~0.9 GB | estimate |
| video | ~0.4 GB | estimate |

**Scope limit (Must-Fix #3):** the light tier `aludam` gets **none** of these accuracy gains — no Bombek1, no ModernBERT, no resnet50. It does provenance, documents, and stylometry only. Therefore:

> "No accuracy tradeoff" is a claim about **`aludam-fullplate` only**. The light tier is a deliberate, documented downgrade for users who cannot carry the weights. This must appear in the README's first paragraph, not in a footnote.

### 5.3 Simplicity

One entry point `al.detect()`, one result schema, one result type, 4 shapes → 1. Video 5 families → 3 signals. Documents get built (they don't exist today).

---

## 6. Package architecture (revised)

### 6.1 Tiers — corrected dependency lists

```
aludam/                          → PyPI: aludam
  pyproject.toml                 src layout, py.typed
  src/aludam/
    __init__.py                  al.detect(), __version__, al.fetch()
    schema.py                    ScanResult / Summary / EnsembleResults / MetadataAnalysis
    normalize.py                 raw detector dict → ScanResult   ← the adapter
    explain.py                   explanation built from ranked real signals
    registry.py                  modality → backend, lazy import
    thresholds.py                calibrated bands, INCONCLUSIVE, min-size guard
    document.py                  pdf/docx/txt → str  (pypdf, python-docx)   NEW
    provenance.py                C2PA/EXIF (from forensics_core)           light
    stylometry.py                text signals, pure python                 light
    weights.py                   fetch/cache + AludamWeightsMissing        NEW
    exceptions.py                AludamDepsMissing, AludamWeightsMissing
    detectors/
      image.py  text.py  audio.py  video.py      adapters per modality
      neural/                                  ported from models/, torch guarded
    backends/
      light.py                   provenance + stylometry + frequency
      neural.py                  requires extras, raises AludamDepsMissing otherwise
    eval/                        golden sets + raid-bench wrapper
```

| tier | installs | capabilities |
|---|---|---|
| `aludam` (light) | `pillow`, `numpy`, `pypdf`, `python-docx` · `scipy` **optional** (light frequency path has a bilateral-filter fallback, `forensics_core.py:431-434`) | C2PA/EXIF provenance, **document extraction**, stylometry/word-patterns, frequency heuristics. **No neural accuracy.** |
| `aludam-fullplate` | `aludam` + `torch`, `transformers`, `timm`, `peft`, `opencv-python-headless`, `librosa`, `soundfile`, **`onnxruntime`** | all neural detectors incl. Bombek1, ModernBERT, resnet50, music fakeprint |

**Music fakeprint moved out of light** (Must-Fix #10): the model is 14.8 KB but inference needs `onnxruntime`, which is **not installed** (verified) and is ~50 MB. Shipping it in light would contradict "lightweight." It belongs in fullplate.

**Video prerequisite:** `ffmpeg` on PATH (already required today, now declared) or `av` as a pip alternative.

### 6.2 Two packages, one dependency list (Must-Fix #9)

v1.0 said fullplate is "a dependency bundle only" while the tier table said it "includes all neural weights" — contradictory. And maintaining two parallel dep lists is needless. Resolution that keeps your chosen name:

- **`aludam` owns the single source of truth**: `[project.optional-dependencies] neural = [...]`
- **`aludam-fullplate` is a ~3-line package** whose only dependency is `aludam[neural]`

```toml
# aludam-fullplate/pyproject.toml
[project]
name = "aludam-fullplate"
dependencies = ["aludam[neural]~=1.0.0"]
```

`pip install aludam-fullplate` and `pip install aludam[neural]` behave identically; one dep list, two names, your branding preserved. (If you'd rather drop `fullplate` and ship only extras, say so — it's one fewer package to publish.)

### 6.3 Weight distribution (Must-Fix #4)

**Weights cannot ship inside either wheel.** PyPI file-size limits make multi-GB uploads impractical, and HF already hosts them. So the correct model is:

1. **Wheels ship code + dependency declarations only. Never weights.** This is now consistent everywhere in this document.
2. **Explicit fetch step**, not an implicit surprise:
   ```
   aludam fetch              # all
   aludam fetch image text   # per modality
   ```
   Also callable as `aludam.fetch(...)` / `aludam.detect(..., fetch="auto")`.
3. **Missing weights RAISE — they never degrade.** This is the direct fix for root cause #2/#3 colliding with first-run downloads:
   ```python
   class AludamWeightsMissing(AludamError):
       """Raised when a detector's weights are absent. Run `aludam fetch`."""
   ```
   Mirrors `AludamDepsMissing` exactly. A missing weight is a **programming/ setup error**, not a signal to be swallowed into `degraded: true` and returned as a confident-looking heuristic score.
4. **`runtime.degraded` is reserved for a model that loaded and then failed mid-inference** — never for "weights weren't there." If weights are absent, `al.detect()` raises.
5. Cache location + size printed after fetch; `aludam fetch --dry-run` reports what would be downloaded.

This also de-fangs root cause #3: an offline user gets a clear exception with instructions, instead of silently heuristic scores.

---

## 7. Response schema

### 7.1 Verdict vocabulary split (Must-Fix #5)

`AI_GENERATED` and `face-swapped` are **different claims**. v1.0 mapped both to `MANIPULATED`, which is wrong and would make Bombek1's 9% FP rate masquerade as manipulation accusations.

**Revised verdicts:**

| verdict | claim | set by |
|---|---|---|
| `AUTHENTIC` | no strong evidence against | calibrated score in authentic band |
| `SYNTHETIC` | the content was machine-generated | `AI_GENERATED` / `T2V_GENERATED` / `AI_MUSIC` families |
| `ALTERED` | real content was modified | `FACE_SWAP` / `SPLICE_EDIT` / `AV_SYNC_MISMATCH` families |
| `INCONCLUSIVE` | evidence insufficient or conflicting | score inside the band, or `runtime.degraded`, or quality gates failed |

**Precedence rule** (documented, deterministic): if both generation and manipulation families fire with score ≥ threshold, return `ALTERED` (the more specific claim) and list the generation evidence in `signals`. If only generation fires → `SYNTHETIC`.

**Calibration and guards — added to Phase 0 and Phase 2, not deferred:**

- **`INCONCLUSIVE` band** around the decision boundary (not a hard 0.5 cut). Bombek1's ~9% FP on cluttered real photos means a hard cut will hand out false `SYNTHETIC` verdicts.
- **Minimum-size guard:** images below **128×128 px** skip Bombek1 entirely and return `INCONCLUSIVE` with `reason: "input_below_model_minimum"` — the card is explicit that it degrades to ~50% (chance) there.
- **Thresholds fitted on the Phase 0 golden set**, per family, not hard-coded.
- Thresholds live in `thresholds.py` and are versioned with the package so a verdict is reproducible.

```json
{
  "request_id": "rd_scan_9847120aef",
  "status": "COMPLETED",
  "media_type": "IMAGE",
  "summary": {
    "verdict": "SYNTHETIC",
    "overall_score": 0.94,
    "confidence": "HIGH",
    "confidence_pct": 91.0,
    "claim_basis": "generation"
  },
  "ensemble_results": {
    "visual_manipulation": { "score": 0.96, "signals_detected": ["FACE_SWAP","SPLICE_EDIT"] },
    "visual_generation":   { "score": 0.97, "signals_detected": ["AI_GENERATED"] },
    "audio_manipulation":  { "score": 0.88, "signals_detected": ["SYNTHETIC_SPEECH_RESONANCE"] }
  },
  "metadata_analysis": { "has_c2pa": false, "software_signature": "Unknown / Stripped" },
  "reason": "family scores: fully_ai_generated=0.970, face_swap=0.960; fired families: …; faces_detected=1; tier=4 (Likely Synthetic)",
  "explanation": "…ranked from real signals…",
  "runtime": {
    "backend": "light|neural",
    "face_detector": "yunet|retinaface|haar",
    "explainer": "fallback|llm:<provider>/<model>",
    "aludam_version": "1.0.0",
    "thresholds_version": "1.0.0",
    "degraded": false,
    "elapsed_ms": 842
  }
}
```

Notes on the contract:
- `summary.confidence` is the label (`HIGH`/`MEDIUM`/`LOW`) with exact `confidence_pct` alongside; `overall_score` kept separate (0 = authentic, 1 = synthetic/altered)
- `reason` is the dense technical derivation (family scores, fired families, intervals, segments) — Phase 3 replaces it with aludam's own `reason` when provided; `explanation` is the LLM plain-language rewrite (multi-provider, env-configured, deterministic fallback)
- `media_type` derived from the actual response, never assumed
- `status` retained to match the original spec; `runtime.*` is what tells the truth
- `runtime.face_detector` makes root cause #1 observable instead of buried in a log line

**Signal name mapping** (emitted when score ≥ its calibrated threshold):

`face_swap`→`FACE_SWAP` · `fully_ai_generated`→`FULLY_AI_GENERATED` · `ai_edited_region`/splice→`SPLICE_EDIT` · `temporal_inconsistency`→`TEMPORAL_INCONSISTENCY` · `audio_visual_sync`→`AV_SYNC_MISMATCH` · `provenance_ai_detected`→`AI_PROVENANCE_TAG` · EXIF software→`GENERATOR_SIGNATURE` · `stylometry_signal`→`STYLOMETRY_ANOMALY` · `pattern_signal`→`AI_PHRASE_PATTERN` · `watermark_signal`→`WATERMARK_DETECTED` · Bombek1→`AI_GENERATED` · resnet50-t2v→`T2V_GENERATED` · music fakeprint→`AI_MUSIC`

**Legacy verdict mapping (unified):**

| detector label | aludam verdict |
|---|---|
| `AUTHENTIC` / `LIKELY_AUTHENTIC` | `AUTHENTIC` |
| `UNCERTAIN` / `INDETERMINATE`, or score in band | `INCONCLUSIVE` |
| `FAKE` / `LIKELY_SYNTHETIC` + generation families only | `SYNTHETIC` |
| `FAKE` / `LIKELY_SYNTHETIC` + manipulation families | `ALTERED` |

**`explanation` algorithm:** top 3 signals by score, then concrete evidence — frame count, suspicious intervals, face count, metadata state. If the light backend ran, it says so explicitly rather than pretending. Capped at ~280 chars. If this ends up templated, the feature has failed.

---

## 8. Execution phases

### Phase 0 — Measure (gates everything) · **APPROVED**

Build `aludam/eval/`:

- Golden sets per modality with labels: images (incl. cluttered/real-photo negatives), text, audio, video
- **Text: `raid-bench` for the harness + at least one non-RAID set (HC3 / MAGE / Ghostbuster)** — required by the §3 circularity caveat
- Fit calibration thresholds and the `INCONCLUSIVE` band here (Must-Fix #5)
- Run the *current* pipeline to establish real baselines — the repo has targets nothing verifies
- Record **disk and RSS separately** for each detector so §5.2's estimates become measurements

**Gate: no Phase 3 swap merges without before/after numbers under one protocol.**

### Phase 1 — Free wins (accuracy ↑, disk ↓, no new pip deps) · **APPROVED**

1. **Face backend → YuNet** (`cv2.FaceDetectorYN`, verified available, ~230 KB ONNX); RetinaFace demoted to explicit opt-in; backend reported in `runtime.face_detector` *(replaces the v1.0 "RetinaFace if available", which was self-defeating — Must-Fix #1)*
2. Silent fallbacks → explicit `runtime` fields; distinguish "weights missing" (raises) from "inference failed" (`degraded`) *(Must-Fix #4)*
3. Verify HF weights are actually reachable in the target environment (root cause #3)
4. Drop `retina-face` + tensorflow from `requirements.txt` (−1397 MB **disk**; state plainly that RAM is unchanged — Must-Fix #3)
5. Add `pypdf` + `python-docx` → documents actually work
6. Declare `ffmpeg` as an existing prerequisite with a clear missing-binary error
7. Re-run eval

*Removed from Phase 1: the Bombek1 init optimisation — the model isn't integrated until Phase 3 (Must-Fix #8).*

### Phase 2 — Contract + skeleton · **APPROVED**

- `schema.py`, `normalize.py`, `explain.py`, `thresholds.py` + tests
- Verdict split (`AUTHENTIC`/`SYNTHETIC`/`ALTERED`/`INCONCLUSIVE`) with precedence rules
- `INCONCLUSIVE` band + `<128 px` guard + calibrated thresholds wired in *(Must-Fix #5)*
- `weights.py` with `AludamWeightsMissing`, `aludam fetch`, cache reporting *(Must-Fix #4)*
- `aludam[neural]` extra + `aludam-fullplate` thin wrapper *(Must-Fix #9)*
- Single `al.detect(type=…, src=…)` entry; canonical signal-name mapping table

### Phase 3 — Swaps · **HELD until Must-Fix 1–4 verified**

Order, each gated by Phase 0 numbers:

1. **Image → Bombek1** — including the init fix: build backbones *without* `pretrained=True` / `from_pretrained()` so ~1.5 GB of base weights aren't downloaded and then overwritten by `load_state_dict`. **Measure actual peak RSS and correct §5.2.**
2. **Text → ModernBERT-Detect** (gpt2 probe out), validated on a **non-RAID** set
3. **Audio → music fakeprint** (14.8 KB ONNX + `onnxruntime`) behind the same license review
4. **Video → resnet50-t2v + audio-track reuse** — **last**, BSD-3-Clause license review, ffmpeg prerequisite confirmed, inference reimplemented (upstream repo 404)

### Phase 4 — Publish · **HELD**

License audit (esp. `MelodyMachine/Deepfake-audio-detection-V2`, resnet50 BSD-3) → sdist/wheel → `twine check` → TestPyPI → PyPI.

**Verification at each phase:**
- `python -m unittest tests.test_detectors -v` — must still pass with `HF_HUB_OFFLINE=1`
- schema/normalize/document/threshold tests — no weights needed
- `raid-bench` + non-RAID set for text; golden sets elsewhere
- `python -m build` + `twine check dist/*`
- clean-venv smoke test: `pip install aludam` → works offline, no neural; `pip install aludam-fullplate` → `aludam fetch` → `al.detect()` on a real image and a real PDF; **and** a negative test that missing weights **raises** rather than degrading

---

## 9. Risks

1. **All accuracy figures are claims from model cards and leaderboards, none measured here.** Phase 0 exists precisely to replace them. v1.0's "+9pt AUC" / "+15pt" were withdrawn for non-comparable metrics.
2. **RAID circularity:** ModernBERT-Detect may have been tuned on RAID; a RAID-only eval would overstate the gain.
3. Bombek1's ~9% FP on cluttered real photos — mitigated by the `INCONCLUSIVE` band and thresholds (§7.1), not merely noted.
4. **Image runtime RSS ~3 GB (est. 3.0–3.5 GB)** — materially heavier; must be measured, not assumed. Defensible under accuracy-first, but real.
5. **Light tier delivers none of the accuracy gains** — "no tradeoff" is a fullplate-only claim and must be stated up front.
6. Video: "research/defensive use only," Sora/Veo2-scoped, upstream repo 404, undeclared ffmpeg dependency.
7. `MelodyMachine/Deepfake-audio-detection-V2` license unaudited — publish gate.
8. No GPT-5-era coverage in these benchmarks — real-world numbers will be lower than leaderboard numbers.

## 10. Deferred

- Query assistant (`GEMINI_API_KEY`, ultralytics/YOLO)
- OCR for scanned PDFs
- Hosted API / dashboard (none needed — inference is local)
- Text robustness under adversarial rewrite beyond the ModernBERT swap
- Decision: keep `aludam-fullplate`, or publish extras only (§6.2 — currently planned to keep both)
