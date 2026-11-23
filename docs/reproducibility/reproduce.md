# Reproducing the results

**What has been tested, and how:**
- **Tested from this repository (CPU, no data):** `scripts/privacy_scan.py`, `python -m pytest tests` (registry re-verification + a synthetic training smoke test), the module unit tests listed in `.github/workflows/ci.yml`, and the website build and render check.
- **Run in the original research environment, not re-run from this repository:** every data-dependent command below (feature extraction, fold training, comparisons, final models, submission). The scripts here are copies of those that produced the results, with only the Python interpreter paths replaced by environment variables.

## 1. Environments (tested versions)

| Variable | Purpose | Requirements | Tested |
|---|---|---|---|
| `PY_TRAIN` | training, validation, scoring, eGeMAPS | `requirements.txt` | Python 3.10.20, torch 2.6.0+cu124 |
| `PY_AV` | FaRL, WavLM, ECAPA, x-vector, audeering age/gender, language ID | `requirements-extraction.txt` (+ OpenAI CLIP) | Python 3.10.20, torch 2.7.1+cu118, transformers 5.15.1, speechbrain 1.0.3 |
| `PY_FACE` | InsightFace face age/gender, ArcFace | `requirements-faces.txt` | Python 3.10.22, insightface 0.7.3, onnxruntime-gpu 1.23.2 |

- **Hardware:** one NVIDIA RTX A5000 (24 GB). One fold run takes about 4–5 minutes, and 7 run in parallel. Training fits easily on smaller GPUs; extraction is the slow part.

**Pretrained models** (frozen, not redistributed here):

| Model | Exact version |
|---|---|
| FaRL | `FaRL-Base-Patch16-LAIONFace20M-ep16.pth` (sha256 `67b4d6bc…`, from github.com/FacePerceiver/FaRL) |
| WavLM | `microsoft/wavlm-base-plus-sv` @ `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` |
| ECAPA-TDNN | `yangwang825/ecapa-tdnn-vox2` @ `c605f57148d4e910d4241eec714d7bc948220adb` |
| age/gender | `audeering/wav2vec2-large-robust-24-ft-age-gender` @ `bad2737f529168bc3d3fcd97edfaba9583c7d36d` |
| face age | InsightFace `buffalo_l` |
| language ID | `speechbrain/lang-id-voxlingua107-ecapa` @ `0253049a` |

## 2. Data layout (obtain the data yourself; see `data.md`)

```
train_set/train_set/train_English.txt   # official manifest: pair_id 1 voices/... faces/... identity class
train_set/train_set/{voices,faces}/      # official clips and face crops
meta_file_train_set.csv                  # identity,m/f
evaluation_phase/{gender,no_gender}/     # official evaluation trials
external_data/mavceleb/raw/*.zip, external_data/voxceleb2/raw/*.zip
validation/                              # fold definitions (included in this repository)
```

## 3. Frozen features

```bash
export PY_TRAIN=... PY_AV=... PY_FACE=...
# Official set: FaRL faces and WavLM layers -> all-layer mean (V2 voice input)
$PY_AV FLAG_2027/feature_extraction/extract_face_features_farl.py --manifest train_set/train_set/train_English.txt \
  --image-root train_set/train_set --checkpoint models/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth \
  --output-csv features/face_v2_farl/train_English_faces.csv
HF_HUB_OFFLINE=1 $PY_AV FLAG_2027/feature_extraction/extract_voice_features_wavlm_layers.py \
  --manifest train_set/train_set/train_English.txt --audio-root train_set/train_set \
  --output-npy features/voice_v2_wavlm_layers/train_English_layers.npy --revision feb593a6c23c1cc3d9510425c29b0a14d2b07b1e
$PY_TRAIN FLAG_2027/feature_extraction/build_voice_feature_csv.py --manifest train_set/train_set/train_English.txt \
  --layers-npy features/voice_v2_wavlm_layers/train_English_layers.npy --layers 0-12 --pooling mean \
  --output-csv features/voice_v4_wavlm_allmean/train_English_voices.csv
```

Then follow `scripts/extract_features.sh` (external data preparation and filtering, eGeMAPS, ECAPA, age/gender, traits, evaluation features).

**Validation during extraction:**
- ECAPA and x-vectors for external clips are streamed from the zips by `FLAG_2027/phase2/extract_external_speaker_embeddings.py`. It checks the encoder (official clips must reproduce cached embeddings) and the mapping (zip audio must reproduce the cached WavLM features) before writing.
- `FLAG_2027/phase2/extract_external_arcface.py` checks image SHA-256 hashes the same way.

## 4. Five-fold validation and paired comparisons

```bash
SEEDS="1 2 3" PARALLEL=7 bash scripts/runs/traits_run_all.sh full              # B0
SEEDS="1 2 3" PARALLEL=7 bash scripts/runs/phase2_run_all.sh g3_cat_farl       # C1
$PY_TRAIN FLAG_2027/local_validation/compare_fold_runs.py \
  --run-a 'runs/traits/full/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar' \
  --run-b 'runs/phase2/g3_cat_farl/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar' --seeds 1 2 3 \
  --face-csv features/face_v2_farl/train_English_faces.csv --voice-csv features/voice_v8_wavlm_egemaps/train_English_voices.csv \
  --manifest train_set/train_set/train_English.txt --output-json results/compare_b0_vs_c1.json
```

- **Fixed settings:** 90 epochs (no early stopping), `checkpoint_last` only, and the same trial files for every system (`validation/fold_k/{standard,gender}_pairs.csv`).
- **Comparison tool:** `compare_fold_runs.py` resamples held-out identities within folds (paired bootstrap, 2,000 draws).

## 5. Unseen-language holdouts

```bash
SEEDS="1 2 3" bash scripts/runs/phase3_run_all.sh p2_b0 p2_c1 p3_b0 p3_c1       # Urdu (p2) and German (p3) holdouts
$PY_TRAIN FLAG_2027/phase2/language_transfer_fusion.py --prefix mav3 --unheard-language German \
  --b0 'runs/phase3/p3_b0/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar' \
  --experts 'c1=runs/phase3/p3_c1/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar' --output-json results/german.json
```

## 6. Final system and evaluation zip

```bash
bash scripts/train_final_c1x3.sh        # three full-data models, seeds 1-3
bash scripts/make_submission_c1x3.sh    # ensemble-fitted terms, then mean distance + terms per trial
```

The zip is validated by `generate_cross_modal_submission.py`: one score file per cell, trial IDs in the official order.

## 7. Regenerating the public artefacts

```bash
$PY_TRAIN scripts/build_registry.py --research-root /path/to/research/tree   # public_results/
$PY_TRAIN scripts/make_figures.py                                            # figures/
cd site && npm ci && npm run build && npm run check                          # website
```
