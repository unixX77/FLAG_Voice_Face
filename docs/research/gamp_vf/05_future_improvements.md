# GAMP-VF — 05 Future improvements

Ordered by expected value given today's evidence.

1. **Stronger cross-modal signal before any pseudo-labelling.**
   - Matching only becomes usable if oracle-group top-1 among about 80 people rises well above 24%. That needs better voice representations: fine-tuning the WavLM upper layers or LoRA on many identities, VoxCeleb1+2 dev (about 7k people, `progress_readme` §9.3).
   - Once there, rerun `oracle_group_matching.py` first. It tells within minutes whether pseudo-labelling has become viable.
2. **A genuinely unlabelled pool.** Pseudo-labelling is only worth doing if there is association-free data to exploit, for example face tracks and speech from videos without speaker IDs, excluding Bengali speech (rules §8). There is none today.
3. **A real speaker embedding for clustering.** WavLM all-layer means cluster unseen speakers poorly (ARI ≤ 0.4, about 7 groups per person). An ECAPA/WavLM-SV embedding extracted for the external data would separate grouping errors from matching errors in any future attempt.
4. **Same-gender sampling with language robustness.** `b3_sg` helps heard languages only. Options:
   - combine it with the language-adversarial loss at a higher weight;
   - same-gender, *cross-language* partner pairs;
   - check whether the Urdu gap closes on the P2 proxy.
5. **If `b3_sg` is ever submitted:**
   - refit the attribute terms on its fold models (`finalize_attribute_terms.py`);
   - train the full-data model (`train_cross_modal_full.py` would need the sampler options; not yet ported);
   - require a paired 3-seed gain *with terms*.
6. **Calibration protocol:** for pseudo-label thresholds, calibrate on a pool of the same size as the target pool. Precision falls with pool size: oracle top-1 is 41% at 39 people and 24% at 77.
7. **Not recommended:**
   - prototype InfoNCE in its current form (B1 +0.60);
   - 100% hard-negative batches;
   - more teacher–student iterations at the current signal level.
