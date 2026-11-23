# Cross-modal identity architecture

This implementation learns directly comparable face and voice embeddings from
the same fixed 4096-dimensional face and 192-dimensional voice features used by
the FOP baseline. It does not load the full-data FOP checkpoint.

## Components

- Face projector: `4096 -> 1024 -> 512 -> 256`, with LayerNorm, GELU,
  dropout, and final L2 normalization.
- Voice projector: `192 -> 512 -> 512 -> 256`, with the same normalization and
  activation design.
- One shared ArcFace-style identity classifier for both modalities.
- Multi-positive symmetric face-to-voice contrastive loss.
- Paired cosine alignment loss.
- Optional momentum voice prototypes for face-to-voice-cluster alignment.
- Optional same-gender hard-negative ranking loss.
- Identity-balanced `P x K` batches; optional equal male/female identity counts.

Only the two projectors are needed for validation or competition inference.
The classifier, gender labels, and voice prototype bank are training aids.

## Experiment sequence

Run commands from the workspace root, not from inside `FLAG_2027`.

### V1: direct shared-space supervision

V1 is the first fair comparison with the FOP baseline. Prototype and hard
negative weights default to zero.

```bash
conda activate flag2027
python FLAG_2027/train_cross_modal.py \
  --fold 0 \
  --validation-root validation \
  --identities-per-batch 16 \
  --samples-per-identity 8 \
  --max-epochs 200 \
  --patience 25 \
  --save-dir runs/cross_modal_v1/fold_0
```

Train all folds only after fold 0 shows a sensible learning curve:

```bash
for fold in 0 1 2 3 4; do
  python FLAG_2027/train_cross_modal.py \
    --fold "$fold" \
    --validation-root validation \
    --identities-per-batch 16 \
    --samples-per-identity 8 \
    --max-epochs 200 \
    --patience 25 \
    --save-dir "runs/cross_modal_v1/fold_${fold}"
done
```

Summarize it with the same tool used for the baseline:

```bash
python -m FLAG_2027.local_validation.summarize_fold_runs \
  --run-root runs/cross_modal_v1
```

### V2: add voice-prototype alignment

Keep every V1 setting fixed and add:

```bash
--prototype-weight 0.5 \
--prototype-start-epoch 10 \
--prototype-momentum 0.9 \
--save-dir runs/cross_modal_v2/fold_0
```

The bank is updated during the first nine warm-up epochs, but its loss begins
at epoch 10. Prototypes belong only to fold-training identities and are never
used for validation scoring.

### V3: add gender-aware hard negatives

Start from the best V2 configuration and add:

```bash
--gender-balanced \
--hard-negative-weight 0.5 \
--hard-negative-margin 0.2 \
--hard-negative-top-k 5 \
--save-dir runs/cross_modal_v3/fold_0
```

Gender controls batch composition and negative selection only. It is not an
input to either projector and is not used during inference.

## Checkpoint selection

`checkpoint_best.pth.tar` minimizes the mean of standard and gender-controlled
EER. `checkpoint_last.pth.tar` records the latest epoch. `training_log.csv`
contains every component loss, positive and negative similarities, both EERs,
both AUCs, and the running best mean EER.

The baseline to beat is:

```text
standard EER: 30.43%
gender EER:   38.69%
mean EER:     34.56% +/- 3.66%
```

Do not compare configurations using only their best fold. Compare five-fold
means and standard deviations after choosing settings on fold 0.

### V4: require alignment with every same-identity sample

V4 replaces positive-set InfoNCE with supervised contrastive averaging. Keep
the V1 settings, disable prototypes and hard negatives, and add:

```bash
--contrastive-mode supcon \
--save-dir runs/cross_modal_v4_supcon/fold_0
```

The original `multi_positive` mode can assign most probability to one easy
positive. `supcon` averages the individual log-probabilities of all eight
same-identity voice candidates for every face, and symmetrically all eight
face candidates for every voice.
