# Matching faces and voices across languages: the research story

*Narrative of the FLAG 2027 project (Oct–Nov 2026). Every number is from `public_results/experiments.json`, which cites the original result files. EER is in percent and lower is better. "Local" means identity-disjoint five-fold cross-validation on the 70 official training identities; it is never an official score.*

## 1. The question

**FLAG 2027** (Face–voice Association across LAnguages and Gender, an ICASSP 2027 Grand Challenge on CodaBench) asks: given one face image and one voice clip, are they the same person?

- **Four cells:** a standard track (impostors of any gender) and a same-gender track (impostors share the speaker's gender), each in English (heard) and Bangla (never heard in training; Bengali speech must not be used for training).
- **The difficulty:** a voice does not determine a face. What can transfer are traits both modalities express, such as sex, age, body build, and possibly vocal-tract and facial morphology.
- **Why the same-gender track is harder:** that track removes the easiest of those cues.

## 2. First model and its limits (Stages 1–2)

- **V1:** frozen FaRL face embeddings and WavLM x-vectors with ArcFace-regularised projectors. Local mean EER 31.7.
- **V2:**
  - all 13 WavLM layers averaged instead of the speaker-verification x-vector;
  - symmetric supervised contrastive (SupCon) training;
  - heavily regularised one-layer heads.
  - Result: local 28.0 (best epoch), significant on both tracks.
- **Official scores:** V2 reached **29.11** on the dev leaderboard (organisers' baseline 36.92) and **31.28** in the evaluation phase.
- **The limit:** same-gender matching, at about 34–37% EER against 23–27% on the standard track.

## 3. What did not work at 70 identities (Stage 4)

All of these were paired against V2 over 3 seeds × 5 folds:
- acoustic augmentation (worse);
- learned WavLM layer weights (no change) and per-clip layer attention (significantly worse);
- hyperbolic embeddings (standard −1.0 to −1.3, gender unchanged);
- a learned pairwise verifier, and same-gender or hard-negative losses (no gain);
- ECAPA alone (+3.2) and WavLM + ECAPA (no gain at 70 identities).

The one useful addition was a **fixed age/gender-consistency score term**. Local mean −0.8, and on CodaBench eval #2 = **30.17**.

**What we learned:** with about 35 people per gender, extra inputs and capacity added variance, not signal. A learning curve showed about **−1.6 EER points per doubling of identities**.

## 4. More people (Stage 3)

- **Data added:** MAV-Celeb v1 (English + Urdu), MAV-Celeb v3 (English + German) and VoxCeleb2-test, in 231 identities after:
  - language-ID filtering (no Bengali);
  - outlier-face removal;
  - duplicate checks against the official identities.
- **Training:** batches keep 50% official identities.
- **Result:** local mean −1.79 [−3.45, −0.26] vs V2; **CodaBench eval #3 = 28.84** (local prediction −1.34, official −1.33).
- **Added on top:**
  - eGeMAPS acoustic functionals;
  - shared trait heads (age/build/gender);
  - language-adversarial and cross-lingual losses.
  - Together these gave **B0**: local 26.15, with terms 25.94. That is a small gain, not significant on its own.

## 5. A sophisticated idea that failed: GAMP-VF (Stage 5)

**Hypothesis:** speaker prototypes, gender-aware hard negatives and pseudo-labels from clustering and graph matching would sharpen same-gender discrimination.

- **Prototypes:** prototype losses were neutral or worse.
- **Same-gender partner sampling:** gender −0.41 (n.s.), and the gain did not transfer to a held-out language.
- **Pseudo-labels** (77 hidden identities): even *perfectly* grouped people were matched only 24% top-1 (34% with Hungarian matching), and the most confident 10% of matches were 57% correct, far below a usable 90%. The true labels of those people added nothing measurable either.

**What we learned:** the cross-modal signal itself was too weak for self-training. That pointed to the representations.

## 6. Where is identity lost? (Stage 6, Phase II)

Each modality was measured on its own:
- **B0's voice input** is session-dominated: 30.5% same-gender EER across videos, 10% within one video. A frozen **ECAPA** speaker encoder reaches **9.8% / 11.0%** (same / other language).
- **On faces**, ArcFace reaches 1.8% against FaRL's 12.3%.

**Better unimodal identity did not mean better matching:**
- B0 voice + ArcFace: **+9.96** EER;
- ECAPA + FaRL: +1.77;
- linear, deeper or differently regularised heads, and alternative losses: no help.

**What we learned:** face–voice association runs through attributes that attribute-rich encoders (FaRL, WavLM / eGeMAPS) keep and identity-trained encoders discard.

**One change survived, ECAPA *appended* to B0's voice input (C1):**
- neutral on the official English folds (+0.05; with terms 25.73 vs 25.94);
- on an **unseen-language proxy** (unseen people speaking Urdu, which was never trained on): Urdu standard **−2.06** [−3.15, −0.95], Urdu same-gender **−2.49** [−3.93, −1.01].

## 7. Why does C1 work, and does it generalise? (Stage 7, Phase III)

**Controls:**
- **Fixed random 192-d features** in place of ECAPA match C1 on English but lose the whole Urdu gain: it isn't dimensionality.
- **ECAPA shuffled among training speakers** also loses it: the content must be aligned with the speaker.

**Frozen analysis:** B0's features predict only 18% of ECAPA's variance, and the unpredictable part is a language-stable speaker code (14% EER in both languages) with little face-trait information.

**Other findings:**
- **Fusion alternatives:** bottleneck, per-block normalisation, residual adapter and late fusion never beat plain concatenation on unseen languages.
- **Duration and session:** no dependence on clip duration or recording session.
- **Second held-out language, German:** chosen in advance and never used for selection. C1 vs B0, German same-gender **−2.35** [−4.38, −0.30].
- **Stability:** a 3-seed C1 ensemble is significantly more stable: official −0.66 vs single C1, Urdu −0.33 / −0.38.

## 8. Final system (Stage 8)

**C1 × 3**:
- three seeds of FaRL (512) + [WavLM 768 | eGeMAPS 88 | ECAPA 192] projectors to 128-d;
- the mean of their 1 − cosine distances;
- plus the fixed age/gender terms.

| System | Local official folds + terms | Urdu holdout, same-gender | German holdout, same-gender |
|---|---|---|---|
| B0 | 25.94 | 38.86 | 42.14 |
| C1 | 25.73 | 36.32 | 39.73 |
| **C1 × 3** | **25.32** | **35.88** | **39.38** |

**Submission status:** the C1 × 3 zip exists and is validated. There is **no record of it, or of B0/C1, having been submitted**. Our last verified official score is 28.84 (eval #3), and no final rank is available.

## 9. What remains unknown

- **Which acoustic properties drive the gain.** We showed ECAPA's speaker-aligned, language-stable content is necessary, not which specific properties cause the cross-modal gain.
- **Why the official English folds barely benefit** while the proxies' English cells do.
- **Whether Bangla behaves like Urdu and German.**
- **The gap to the leaderboard median (22.47).** A larger paired-identity pretraining set (VoxCeleb dev) or encoder fine-tuning are the untested levers.
