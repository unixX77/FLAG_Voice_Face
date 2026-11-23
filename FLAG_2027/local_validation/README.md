# Local identity-disjoint validation

This package builds five deterministic English validation folds from the 70
FLAG 2027 training identities. Each fold holds out 14 identities (8 male and 6
female) and uses the remaining 56 identities for training.

Each fold contains two balanced verification protocols:

- `standard`: negative face/voice pairs may have any gender combination.
- `gender`: every negative face/voice pair has the same gender.

Pair files do not contain labels. Ground truth is stored separately and all
trials are deterministically shuffled, so trial ordering cannot reveal labels.

## Generate the folds

Run from the workspace root (the directory containing `FLAG_2027`, `train_set`,
and `meta_file_train_set.csv`):

```bash
conda activate flag2027
python FLAG_2027/local_validation/build_protocols.py \
  --output-root validation \
  --seed 2027 \
  --trials-per-identity 50
```

Feature validation checks the two CSV dimensions and verifies that both label
columns align row-for-row with the manifest. Pass `--skip-feature-validation`
only for deterministic rebuild tests after a full validated build succeeds.

## Fold files

Each `validation/fold_N` directory contains:

- `train_indices.npy`: rows used to train fold N.
- `validation_indices.npy`: unseen-identity rows used for verification.
- `standard_pairs.csv` and `gender_pairs.csv`: pair IDs and feature row indices.
- `*_ground_truth.csv`: local labels, deliberately stored separately.
- `metadata.json`: identity lists, the fold-local classifier map, counts, and
  protocol diagnostics.

`validation/folds.json` summarizes all folds. Do not regenerate or modify these
files between model comparisons.

## Metric tests

Run from the repository directory:

```bash
cd FLAG_2027
python -m unittest discover -s local_validation -p 'test_*.py' -v
```

The tests cover perfect, reversed, random, and non-finite scores.

## Evaluate a local score file

A prediction CSV must contain `pair_id,score`, with larger scores indicating a
stronger match. Evaluate it from the repository directory:

```bash
python -m local_validation.evaluate_scores \
  --ground-truth ../validation/fold_0/standard_ground_truth.csv \
  --scores path/to/standard_scores.csv
```

If a model produces Euclidean distances, negate them before writing the score
CSV so that larger values consistently mean a stronger match.

## Train and verify the baseline on a fold

Run this command from the workspace root (the directory containing
`FLAG_2027`, `train_set`, and `validation`):

```bash
conda activate flag2027
python FLAG_2027/train_fold.py \
  --fold 0 \
  --validation-root validation \
  --max-epochs 2 \
  --save-dir runs/fop_fold0_smoke
```

After the smoke run succeeds, use a fresh output directory and train the full
baseline:

```bash
python FLAG_2027/train_fold.py \
  --fold 0 \
  --validation-root validation \
  --max-epochs 200 \
  --patience 20 \
  --save-dir runs/fop_fold0_full
```

The runner trains only on the fold's 56 training identities. Every epoch it
evaluates the 14 unseen validation identities on both protocols and saves:

- `checkpoint_best.pth.tar`: lowest mean of standard and gender EER;
- `checkpoint_last.pth.tar`: most recent epoch, useful for diagnostics;
- `training_log.csv`: training losses and both validation EER/AUC values.

Never initialize a fold run from the 70-identity full-training checkpoint: it
has already seen every validation identity and would make the result invalid.

For the final baseline estimate, train all five folds into consistently named
directories:

```bash
for fold in 0 1 2 3 4; do
  python FLAG_2027/train_fold.py \
    --fold "$fold" \
    --validation-root validation \
    --max-epochs 200 \
    --patience 20 \
    --save-dir "runs/fop_baseline/fold_${fold}"
done
```

This loop is sequential and does not require `tmux`. Closing the SSH session
will still terminate a foreground run, so keep the connection open while it is
training. When all five folds finish, report their aggregate result with:

```bash
python -m FLAG_2027.local_validation.summarize_fold_runs \
  --run-root runs/fop_baseline
```
