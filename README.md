# Matching Faces and Voices Across Languages

**Research project for FLAG 2027** (Face–voice Association across LAnguages and Gender, an ICASSP 2027 Grand Challenge): cross-modal face–voice verification under same-gender and unseen-language conditions.

[Project website](https://unixx77.github.io/FLAG_Voice_Face/) · [Technical report](docs/publication/report.md) · [Research story](docs/publication/research_story.md) · [Experiment registry](public_results/experiments.json)

> **Status:** independent research, not peer reviewed. Local numbers are identity-disjoint cross-validation, **not official scores**. The best verified official CodaBench score is **28.84%** mean EER (evaluation phase, eval #3). The final system below has not been submitted.

## What we found

1. **Better unimodal identification ≠ better face–voice matching.**
   - ArcFace (1.8% same-gender face EER) instead of FaRL (12.3%) costs **+9.96** EER points cross-modally.
   - ECAPA alone (9.8% speaker EER) instead of WavLM + eGeMAPS (30.5%) costs +1.77.
2. **More identities help.** About −1.6 EER points per doubling; 231 external identities gave −1.79 locally and −1.33 on CodaBench.
3. **Pseudo-labelling fails with a weak teacher.** With perfect grouping, cross-modal matching of 77 hidden people reached 24% top-1; the most confident 10% were 57% correct (GAMP-VF, a documented negative result).
4. **Complementary acoustic representations generalise.** Appending frozen ECAPA to WavLM + eGeMAPS is neutral on English folds but improves **unseen-language** same-gender EER:
   - Urdu −2.49 [−3.93, −1.01];
   - German −2.35 [−4.38, −0.30] (German was never used for selection).
   - Random features and speaker-shuffled ECAPA both lose the gain.
5. **Plain concatenation beats fancier fusion:** bottleneck, block normalisation, residual adapter and late fusion all did worse or tied.

## Final system (C1 × 3)

```
face image ─► FaRL ViT-B/16 (frozen, 512) ─► face projector ×3 ──────────────────────────┐
voice clip ─┬► WavLM Base+ all-layer mean (frozen, 768) ┐                                 ├─► mean of 3 × (1 − cos) ─► + age/gender terms ─► distance
            ├► eGeMAPS v02 (88) ─────────────────────────┼─► concat 1,048 ─► voice projector ×3 ┘
            └► ECAPA-TDNN (frozen, 192) ─────────────────┘
projector: z-score → [train: noise 0.1, dropout 0.2] → Linear 256 → LayerNorm → GELU → Dropout 0.3 → Linear 128 → L2
training:  301 identities, SupCon (τ = 0.07) + 0.25 pair + 0.2 trait heads + 0.1 language adversary + 0.1 cross-lingual
```

| System | Local std / gender | Local mean + terms | Urdu std / same-gender | German std / same-gender |
|---|---|---|---|---|
| B0 | 20.34 / 31.97 | 25.94 | 26.06 / 38.86 | 30.37 / 42.14 |
| C1 (+ ECAPA) | 20.48 / 31.94 | 25.73 | 23.94 / 36.32 | 28.99 / 39.73 |
| **C1 × 3** | **19.91 / 31.19** | **25.32** | **23.62 / 35.88** | **28.85 / 39.38** |

EER in %, lower is better. 3 seeds × 5 identity-disjoint folds. Urdu and German are unseen people **and** an unseen language.

## Repository layout

| Path | Content |
|---|---|
| `FLAG_2027/` | model, training, validation, feature extraction, analyses, GAMP-VF, Phase II/III code (our files only) |
| `scripts/` | run scripts for every experiment family, feature extraction, final training and submission, registry and figure builders, privacy scan |
| `validation/` | the exact identity-disjoint folds and trial files |
| `public_results/` | `experiments.json` (63 experiments, every value traceable) + copies of the source result files |
| `figures/` | publication figures (SVG + PNG), generated from the registry |
| `docs/` | report and story (`publication/`), verbatim research reports (`research/`), reproduction, data, licensing, deployment (`reproducibility/`) |
| `site/` | the project website (React + Vite + Recharts, plain CSS in the shared style of the author's other project pages); every number is read from the registry |
| `tests/` | registry re-verification and a CPU smoke test |

## Quick start

```bash
pip install -r requirements-ci.txt
python scripts/privacy_scan.py && python -m pytest -q tests      # no data, CPU only
cd site && npm ci && npm run dev                                  # website at http://localhost:5173/FLAG_Voice_Face/
```

For a full reproduction (data access, three Python environments, feature extraction, fold training, holdouts, final models), see [docs/reproducibility/reproduce.md](docs/reproducibility/reproduce.md). Datasets must be obtained separately ([data.md](docs/reproducibility/data.md)), and no media or checkpoints are redistributed.

## Limitations

- **Small pools:** 70 official people, 14 per held-out fold.
- **Hard error mode:** same-gender, similar-age impostors.
- **Proxies:** the unseen-language tests are not Bangla.
- **Unexplained:** the English-fold result for C1.
- **Selection bias:** model selection over many configurations (the German holdout was kept untouched for this reason).
- **Distance to the field:** a large gap to the leaderboard median (22.47 on 13 Nov).

See §12 of the report.

## Licence and attribution

**No licence has been chosen yet** (see [docs/reproducibility/licensing.md](docs/reproducibility/licensing.md)).
- **Excluded code:** the organisers' starter code has no published licence and is not included.
- **Non-commercial components:** the pipeline uses audeering age/gender (CC BY-NC-SA 4.0), InsightFace models and openSMILE, which carry research / non-commercial terms.
- **Credits:** FaRL, WavLM, ECAPA-TDNN (SpeechBrain / yangwang825), MAV-Celeb, VoxCeleb2, and the FLAG 2027 organisers.

Cite this work with [CITATION.cff](CITATION.cff).
