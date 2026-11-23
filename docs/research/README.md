# Research documentation (verbatim copies)

These are the reports written during the project, unchanged. Later phases supersede earlier recommendations.

| Folder / file | Content |
|---|---|
| `progress_log.md` | chronological lab log (Oct–Nov 2026); cited by line range in `public_results/experiments.json` |
| `gamp_vf/` | GAMP-VF: prototypes, gender-aware hard negatives, pseudo-labelling, matching (negative result) |
| `phase2/` | Phase II: unimodal diagnostics, encoder substitution, fusion, unseen-language proxy |
| `phase3/` | Phase III: ECAPA–WavLM complementarity, controls, German confirmation, final ensemble |
| `codabench/` | official CodaBench result files |

Paths such as `results/phase3/...`, `runs/...` or `checkpoints/...` refer to the private research tree. Every JSON result the
registry uses is mirrored under `public_results/sources/` with the same relative path. Raw media, feature caches and
checkpoints are not published (see `docs/reproducibility/licensing.md`).
