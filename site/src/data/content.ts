// Narrative content. Every number is pulled from the registry (registry.ts), never typed here.
import { delta, exp, fmt, fmtCI, fmtDelta, metric, transferDelta } from "./registry";

// ---------------------------------------------------------------------------------------------
// Architecture explorer (verified against checkpoints + inference code; docs/publication/01_audit.md)
// ---------------------------------------------------------------------------------------------
export type NodeKind = "input" | "frozen" | "features" | "trainable" | "score" | "training";
export interface PipeNode {
  id: string; label: string; sub?: string; kind: NodeKind; lane: "face" | "voice" | "joint" | "train";
  /** Encoders that run in parallel on the same input are drawn stacked. */
  parallel?: boolean;
  what: string; model?: string; dims?: string; status: string; why?: string; evidence?: string; rejected?: string;
}
export interface Architecture {
  id: string; name: string; tag: string; status: "adopted" | "rejected" | "superseded" | "final";
  summary: string; nodes: PipeNode[]; note?: string;
}

const projector = (inDim: string): Omit<PipeNode, "id" | "lane" | "label"> => ({
  kind: "trainable", sub: `${inDim} → 256 → 128`,
  what: "z-score with stored training statistics → (training only: Gaussian noise σ = 0.1, input dropout 0.2) → Linear → 256 → LayerNorm → GELU → Dropout 0.3 → Linear → 128 → L2 normalisation.",
  dims: `${inDim} in · 128 out (unit sphere)`, status: "Trainable",
  why: "Heavily regularised one-layer heads generalise best to unseen people.",
  evidence: (() => { const d = delta("pj_linear", "mean"); const e = delta("pj_deep", "mean");
    return `Linear heads ${fmtDelta(d.value)} ${fmtCI(d.ci95)} and deeper heads ${fmtDelta(e.value)} mean EER vs B0 (seed 1).`; })(),
  rejected: "Linear heads, two hidden layers, lighter dropout.",
});

const farl: Omit<PipeNode, "id" | "lane"> = {
  label: "FaRL ViT-B/16", sub: "512-d", kind: "frozen",
  what: "Face image embedding with official CLIP preprocessing (224 px, centre crop), no detection or alignment, L2-normalised.",
  model: "FaRL-Base-Patch16-LAIONFace20M-ep16", dims: "512", status: "Frozen",
  why: "A face-attribute encoder keeps the age/appearance cues that a voice can predict.",
  evidence: (() => { const d = delta("g3_b0_arcface", "mean"); return `Replacing FaRL with ArcFace (1.8% unimodal EER) costs ${fmtDelta(d.value)} ${fmtCI(d.ci95)} mean EER.`; })(),
  rejected: "ArcFace (InsightFace), DINOv2, FaRL + DINOv2.",
};
const wavlm = (dims = "768"): Omit<PipeNode, "id" | "lane"> => ({
  label: "WavLM Base+", sub: "all-layer mean · 768-d", kind: "frozen",
  what: "16 kHz audio in 10 s chunks (5 s overlap); frame mean of all 13 hidden states, averaged over layers.",
  model: "microsoft/wavlm-base-plus-sv @ feb593a6", dims, status: "Frozen",
  why: "Intermediate layers keep broad voice traits (pitch, timbre, age) shared with faces.",
  evidence: (() => { const d = delta("v1_to_v2", "gender"); return `V1 → V2 (all layers + SupCon): gender ${fmtDelta(d.value, 1)} ${fmtCI(d.ci95)}.`; })(),
  rejected: "x-vector head (V1), learned layer weights, per-clip layer attention.",
});
const egemaps: Omit<PipeNode, "id" | "lane"> = {
  label: "eGeMAPS v02", sub: "88 functionals", kind: "features",
  what: "openSMILE 2.5.1 acoustic functionals (pitch, loudness, spectral and voice-quality statistics); median-imputed and clipped to training percentiles.",
  dims: "88", status: "Fixed features",
  evidence: (() => { const d = delta("b0_egemaps", "mean"); return `+ eGeMAPS alone ${fmtDelta(d.value)} ${fmtCI(d.ci95)}; part of B0's combined gain.`; })(),
};
const ecapa: Omit<PipeNode, "id" | "lane"> = {
  label: "ECAPA-TDNN", sub: "192-d", kind: "frozen",
  what: "Speaker-verification embedding of the whole clip (VoxCeleb2-trained).", model: "yangwang825/ecapa-tdnn-vox2 @ c605f571",
  dims: "192", status: "Frozen",
  why: "Adds language-stable speaker identity that WavLM + eGeMAPS lacks.",
  evidence: (() => { const u = transferDelta("urdu_holdout_phase2", "C1 vs B0", "gender_Urdu"); const g = transferDelta("german_holdout", "C1 vs B0", "gender_German");
    return `Unseen Urdu same-gender ${fmtDelta(u.value)} ${fmtCI(u.ci95)}; unseen German ${fmtDelta(g.value)} ${fmtCI(g.ci95)} vs B0.`; })(),
  rejected: "ECAPA alone, bottleneck / block-normalised / residual fusion, late fusion with B0.",
};
const terms: Omit<PipeNode, "id" | "lane"> = {
  label: "Age / gender terms", sub: "fixed weights", kind: "score",
  what: "Distance += w_age·|corrected voice age − face age|/10 + w_gender·|p_male(voice) − male(face)|. Voice age/gender: audeering wav2vec2 (linear bias correction per predicted gender); face age/gender: InsightFace buffalo_l. Computed from the trial's own clip and image only.",
  status: "Fitted on training folds, then fixed",
  evidence: (() => { const e = exp("final_c1x3_terms"); const w = e.term_weights!;
    return `Weights: standard track ${w.standard_track.w_age_per_10y} / ${w.standard_track.w_gender}, gender track ${w.gender_track.w_age_per_10y} / ${w.gender_track.w_gender}.`; })(),
};

export const architectures: Architecture[] = [
  {
    id: "v1", name: "V1", tag: "Initial baseline", status: "superseded",
    summary: "FaRL faces and WavLM x-vector voices with ArcFace-regularised projectors.",
    nodes: [
      { id: "f_in", label: "Face image", kind: "input", lane: "face", what: "224 × 224 face crop.", status: "Input" },
      { id: "farl", lane: "face", ...farl, evidence: "Kept through every later version.", rejected: undefined },
      { id: "f_proj", lane: "face", label: "Face projector", sub: "512 → 1024 → 512 → 256", kind: "trainable", what: "Linear → 1024 → LayerNorm → GELU → Dropout 0.2 → Linear → 512 → LayerNorm → GELU → Dropout 0.1 → Linear → 256 → L2 normalisation. No input standardisation, noise or input dropout.", dims: "512 in · 256 out (unit sphere)", status: "Trainable",
        rejected: "Replaced in V2 by one 256-wide hidden layer, a 128-d output and z-scored, noise-regularised inputs." },
      { id: "v_in", label: "Voice clip", kind: "input", lane: "voice", what: "16 kHz mono speech.", status: "Input" },
      { id: "xvec", lane: "voice", label: "WavLM-SV x-vector", sub: "512-d", kind: "frozen", what: "Speaker-verification x-vector head of WavLM Base+ SV.", model: "microsoft/wavlm-base-plus-sv", dims: "512", status: "Frozen",
        rejected: "Replaced by the all-layer mean in V2: speaker-verification heads discard traits shared with faces." },
      { id: "v_proj", lane: "voice", label: "Voice projector", sub: "512 → 512 → 512 → 256", kind: "trainable", what: "Linear → 512 → LayerNorm → GELU → Dropout 0.2 → Linear → 512 → LayerNorm → GELU → Dropout 0.1 → Linear → 256 → L2 normalisation. No input standardisation, noise or input dropout.", dims: "512 in · 256 out (unit sphere)", status: "Trainable" },
      { id: "cos", lane: "joint", label: "1 − cosine", kind: "score", what: "One model: d = 1 − cos(z_v, z_f) = 1 − z_v · z_f on the 256-d unit vectors. No ensemble and no age/gender terms.", dims: "256 · 256 → 1 distance", status: "Scoring" },
      { id: "loss", lane: "train", label: "Multi-positive contrastive + ArcFace + pair", kind: "training", what: "Multi-positive cross-modal contrastive loss (τ = 0.07) + 1.0 × ArcFace identity loss (one classifier over the 56 training identities of the fold, shared by both embeddings; scale 30, margin 0.2) + 0.25 × pair loss. 70 official identities only; early stopping on the validation fold (patience 25, up to 200 epochs).", status: "Training only",
        rejected: "V2 set the ArcFace identity weight to 0 and switched to SupCon, as part of the combined V1 → V2 change (its separate effect was not isolated)." },
    ],
    note: (() => { const v = metric("v1_to_v2", "V1 (FaRL + WavLM x-vector, ArcFace heads), best epoch", "mean"); return `Local mean EER ${fmt(v, 1)} (best epoch per fold).`; })(),
  },
  {
    id: "v2", name: "V2", tag: "Improved representation", status: "superseded",
    summary: "All-layer WavLM, symmetric supervised contrastive loss, heavily regularised heads; 70 official identities.",
    nodes: [
      { id: "f_in", label: "Face image", kind: "input", lane: "face", what: "224 × 224 face crop.", status: "Input" },
      { id: "farl", lane: "face", ...farl },
      { id: "f_proj", lane: "face", label: "Face projector", ...projector("512") },
      { id: "v_in", label: "Voice clip", kind: "input", lane: "voice", what: "16 kHz mono speech.", status: "Input" },
      { id: "wavlm", lane: "voice", parallel: true, ...wavlm() },
      { id: "v_proj", lane: "voice", label: "Voice projector", ...projector("768") },
      { id: "cos", lane: "joint", label: "1 − cosine", kind: "score", what: "One model: d = 1 − cos(z_v, z_f) = 1 − z_v · z_f on the 128-d unit vectors. No ensemble and no age/gender terms.", dims: "128 · 128 → 1 distance", status: "Scoring" },
      { id: "loss", lane: "train", label: "SupCon + pair loss", kind: "training", what: "Symmetric supervised contrastive loss (τ = 0.07, all same-identity samples positive) + 0.25 × pair alignment; the ArcFace identity weight is 0. P16 × K8 identity-balanced batches, AdamW 1e-4, 90 epochs, 70 official identities.", status: "Training only" },
    ],
    note: (() => { const s = metric("v2_reference", "V2", "mean"); return `Local mean EER ${fmt(s)} (3 seeds × 5 folds); CodaBench dev 29.11.`; })(),
  },
  {
    id: "b0", name: "B0", tag: "Expanded data, full model", status: "superseded",
    summary: "301 identities (official + MAV-Celeb v1/v3 + VoxCeleb2-test), eGeMAPS voice features, shared trait heads and language losses.",
    nodes: [
      { id: "f_in", label: "Face image", kind: "input", lane: "face", what: "224 × 224 face crop.", status: "Input" },
      { id: "farl", lane: "face", ...farl },
      { id: "f_proj", lane: "face", label: "Face projector", ...projector("512") },
      { id: "v_in", label: "Voice clip", kind: "input", lane: "voice", what: "16 kHz mono speech.", status: "Input" },
      { id: "wavlm", lane: "voice", parallel: true, ...wavlm() },
      { id: "egemaps", lane: "voice", parallel: true, ...egemaps },
      { id: "concat", lane: "voice", label: "Concatenate", sub: "768 | 88 = 856", kind: "features", what: "Voice input = [WavLM | eGeMAPS].", dims: "856", status: "Fixed" },
      { id: "v_proj", lane: "voice", label: "Voice projector", ...projector("856") },
      { id: "cos", lane: "joint", label: "1 − cosine", kind: "score", what: "One model: d_cos = 1 − cos(z_v, z_f) = 1 − z_v · z_f on the 128-d unit vectors. No seed ensemble; the fixed age/gender terms are added next.", dims: "128 · 128 → 1 distance", status: "Scoring" },
      { id: "terms", lane: "joint", ...terms, evidence: (() => { const e = exp("b0_terms"); return `B0 local mean ${fmt(e.systems!["B0 + terms"].mean)} with terms.`; })() },
      { id: "loss", lane: "train", label: "SupCon + pair + traits + language", kind: "training", what: "SupCon (τ = 0.07) + 0.25 pair + 0.2 shared trait heads (age, build, gender; applied to both the face and the voice embedding) + 0.1 language adversary (gradient reversal, English/Urdu/German) + 0.1 cross-lingual alignment (both on the voice embedding). The heads are discarded after training. 301 identities (36,187 rows), 50% official identities per batch, 50 batches × 90 epochs, AdamW 1e-4.", status: "Training only" },
    ],
  },
  {
    id: "gamp", name: "GAMP-VF", tag: "Investigated, rejected", status: "rejected",
    summary: "Gender-aware multilingual prototype learning: training-side clustering, pseudo-labelling and matching around the unchanged B0 network.",
    nodes: [
      { id: "b0net", lane: "joint", label: "B0 network (unchanged)", kind: "trainable", what: "Verification still scores one face and one voice independently; GAMP-VF only changes training.", status: "Inference unchanged" },
      { id: "proto", lane: "train", label: "Speaker / face prototypes", kind: "training", what: "In-batch identity prototypes (uniform and language-balanced) with a prototype InfoNCE loss.", status: "Training only",
        evidence: (() => { const d = delta("gamp_b1", "mean"); return `Prototype loss ${fmtDelta(d.value)} mean EER vs B0 (seed 1).`; })() },
      { id: "hardneg", lane: "train", label: "Gender-aware hard negatives", kind: "training", what: "Same-gender / most-confusable / similar-age partner sampling in P × K batches.", status: "Training only",
        evidence: (() => { const d = delta("gamp_b3_sg", "gender"); return `Best variant: gender ${fmtDelta(d.value)} ${fmtCI(d.ci95)} (n.s.), English-only.`; })() },
      { id: "cluster", lane: "train", label: "Clustering → pseudo-labels", kind: "training", what: "Voice/face clustering of hidden people, calibrated candidate scoring, Hungarian / Sinkhorn matching, teacher–student retraining.", status: "Training only (simulation)",
        evidence: (() => { const r = exp("gamp_pseudo_labels").recovery!; return `Oracle-grouped matching ${Math.round((r.oracle_groups_top1 ?? 0) * 100)}% top-1; most confident 10% only ${Math.round((r.oracle_groups_precision_top10pct ?? 0) * 100)}% correct (target ${Math.round((r.target_precision ?? 0) * 100)}%).`; })() },
    ],
    note: "Not part of the final system. Clustering, matching and pseudo-labels are never used at inference.",
  },
  {
    id: "c1x3", name: "C1 × 3", tag: "Final system", status: "final",
    summary: "B0 recipe with frozen ECAPA appended to the voice input, trained with three seeds; distances averaged, then fixed age/gender terms.",
    nodes: [
      { id: "f_in", label: "Face image", kind: "input", lane: "face", what: "224 × 224 face crop.", status: "Input" },
      { id: "farl", lane: "face", ...farl },
      { id: "f_proj", lane: "face", label: "Face projector × 3", ...projector("512"), status: "Trainable · 3 seeds" },
      { id: "v_in", label: "Voice clip", kind: "input", lane: "voice", what: "16 kHz mono speech.", status: "Input" },
      { id: "wavlm", lane: "voice", parallel: true, ...wavlm() },
      { id: "egemaps", lane: "voice", parallel: true, ...egemaps },
      { id: "ecapa", lane: "voice", parallel: true, ...ecapa },
      { id: "concat", lane: "voice", label: "Concatenate", sub: "768 | 88 | 192 = 1,048", kind: "features", what: "Voice input = [WavLM | eGeMAPS | ECAPA]. Plain concatenation; each dimension is z-scored inside the projector.", dims: "1,048", status: "Fixed",
        evidence: (() => { const d = transferDelta("urdu_controls_seed1", "Block L2 vs C1", "gender_Urdu"); return `Per-block L2 vs plain concat on unseen Urdu gender: ${fmtDelta(d.value)} ${fmtCI(d.ci95)} (seed 1); no fusion variant beat concatenation.`; })() },
      { id: "v_proj", lane: "voice", label: "Voice projector × 3", ...projector("1,048"), status: "Trainable · 3 seeds" },
      { id: "match", lane: "joint", label: "Cross-modal matching (per seed)", kind: "score", what: "Each seed model k maps the face and the voice to 128-d unit vectors z_f and z_v; its distance is d_k = 1 − cos(z_v, z_f) = 1 − z_v · z_f. Only the trial's own face and clip are used: no cohort, normalisation or cross-trial information.", dims: "128 · 128 → 1 distance per seed", status: "Scoring" },
      { id: "cos", lane: "joint", label: "Mean of 3 × (1 − cosine)", kind: "score", what: "d = (d₁ + d₂ + d₃) / 3, one distance per seed model; equal weights, no score statistics.", status: "Scoring",
        evidence: (() => { const d = delta("seed_ensemble_official", "mean"); return `3-seed ensemble ${fmtDelta(d.value)} ${fmtCI(d.ci95)} mean EER vs single C1.`; })() },
      { id: "terms", lane: "joint", ...terms },
      { id: "loss", lane: "train", label: "SupCon + pair + traits + language", kind: "training", what: "As B0: 301 identities, 36,187 rows, 90 epochs, 50 batches/epoch, AdamW 1e-4.", status: "Training only" },
    ],
  },
];

// ---------------------------------------------------------------------------------------------
// Research journey
// ---------------------------------------------------------------------------------------------
export interface JourneyStep {
  id: string; stage: string; title: string; believed: string; changed: string; experiment: string;
  result: string; decision: string; supported: "yes" | "no" | "partly" | "n/a"; refs: string[];
  chart?: { id: string; systems: string[]; keys: string[]; field?: "systems" | "systems_urdu" | "systems_german" };
}

const d = (id: string, k: string) => { const x = delta(id, k); return `${fmtDelta(x.value)} ${fmtCI(x.ci95)}`; };
const td = (id: string, pair: string, k: string) => { const x = transferDelta(id, pair, k); return `${fmtDelta(x.value)} ${fmtCI(x.ci95)}`; };

export const journey: JourneyStep[] = [
  { id: "problem", stage: "Problem", title: "Is this face the speaker of this voice?", supported: "n/a", refs: ["codabench_dev_v2"],
    believed: "Faces and voices share person-level traits (sex, age, build) that a learned shared space can align.",
    changed: "—", experiment: "FLAG 2027 defines four cells: standard / same-gender × English (heard) / Bangla (never heard).",
    result: `The organisers' FOP baseline scores ${fmt(metric("codabench_dev_v2", "FOP baseline (organisers)", "mean"))} mean EER on the dev leaderboard.`,
    decision: "Build an identity-disjoint local validation first, so decisions never depend on the leaderboard." },
  { id: "baseline", stage: "Initial baseline", title: "V1 → V2: broader voice features, contrastive training", supported: "yes", refs: ["v1_to_v2", "v2_reference", "codabench_dev_v2"],
    believed: "Speaker-verification x-vectors discard traits shared with faces.",
    changed: "All-layer WavLM mean, symmetric SupCon, heavily regularised one-layer heads.",
    experiment: "Five identity-disjoint folds, paired bootstrap V2 vs V1.",
    result: `Standard ${d("v1_to_v2", "standard")}, gender ${d("v1_to_v2", "gender")}. CodaBench dev ${fmt(metric("codabench_dev_v2", "V2", "mean"))}.`,
    decision: "V2 becomes the base; the same-gender track (~34%) is the bottleneck.",
    chart: { id: "v1_to_v2", systems: ["V1 (FaRL + WavLM x-vector, ArcFace heads), best epoch", "V2, best epoch"], keys: ["standard", "gender"] } },
  { id: "explore", stage: "Experimental failures", title: "Ten ideas on 70 identities", supported: "no", refs: ["aug25", "hyperbolic_t010", "layer_attention", "ecapa_only_v2", "wavlm_ecapa_v2", "pairwise_verifier", "age_gender_terms_v2", "learning_curve"],
    believed: "Better geometry, augmentation, layer selection or speaker embeddings would separate same-gender people.",
    changed: "Augmentation, hyperbolic space, layer attention, ECAPA, a learned verifier, alternative losses; fixed age/gender score terms.",
    experiment: "Each paired against V2, 3 seeds × 5 folds, fixed 90 epochs.",
    result: `Augmentation ${d("aug25", "mean")}; layer attention ${d("layer_attention", "mean")}; ECAPA alone ${d("ecapa_only_v2", "mean")}; hyperbolic: standard ${d("hyperbolic_t010", "standard")}, gender ${d("hyperbolic_t010", "gender")}. Only the age/gender terms helped (${d("age_gender_terms_v2", "mean")}).`,
    decision: "With ~35 people per gender, extra capacity adds variance. The learning curve (≈ −1.6 EER per doubling of identities) points to more data." ,
    chart: { id: "learning_curve", systems: [], keys: ["standard", "gender"] } },
  { id: "data", stage: "Data expansion", title: "From 70 to 301 identities", supported: "yes", refs: ["ext_all_share50", "eval3_terms", "codabench_eval_submissions", "b0_full", "b0_terms"],
    believed: "More identities reduce EER; the official domain must not be swamped.",
    changed: "MAV-Celeb v1 (English + Urdu), v3 (English + German), VoxCeleb2-test; language-ID filtering (no Bengali), outlier and duplicate checks; 50% official identities per batch. Then eGeMAPS, shared trait heads and language losses (B0).",
    experiment: "Paired against V2 (data) and against the data-expanded model (B0 components).",
    result: `External data ${d("ext_all_share50", "mean")} mean; CodaBench eval #3 ${fmt(metric("codabench_eval_submissions", "Eval #3: V2 + external data + terms", "mean"))} (local prediction transferred). B0 ${d("b0_full", "mean")} vs that model (n.s.).`,
    decision: "Adopt external data; B0 becomes the reference for the remaining research.",
    chart: { id: "ext_all_share50", systems: ["V2", "+ v1 + v3 + VoxCeleb2 (50% share) = eval #3 model"], keys: ["standard", "gender"] } },
  { id: "gamp", stage: "GAMP-VF", title: "Prototypes, hard negatives and pseudo-labels", supported: "no", refs: ["gamp_b1", "gamp_b3_sg", "gamp_pseudo_labels", "gamp_oracle_headroom"],
    believed: "Gender-aware hard negatives and pseudo-labels from clustering + graph matching would sharpen same-gender discrimination.",
    changed: "A training-side framework around the unchanged B0 network.",
    experiment: "Seed-1 screening, 3-seed promotion, and a hidden-association simulation with 77 identities.",
    result: (() => { const r = exp("gamp_pseudo_labels").recovery!; return `Best sampler: gender ${d("gamp_b3_sg", "gender")} (n.s.). Even oracle-grouped matching reached ${Math.round((r.oracle_groups_top1 ?? 0) * 100)}% top-1; the most confident 10% were ${Math.round((r.oracle_groups_precision_top10pct ?? 0) * 100)}% correct.`; })(),
    decision: "The cross-modal signal is too weak for self-training. Diagnose the representations instead." },
  { id: "diagnose", stage: "Representation diagnostics", title: "Where is identity lost?", supported: "partly", refs: ["p2_audio_unimodal", "p2_face_unimodal", "g3_b0_arcface", "g3_ecapa_farl", "g3_cat_farl", "pj_linear", "urdu_holdout_phase2"],
    believed: "Weak unimodal encoders limit cross-modal matching.",
    changed: "Unimodal verification of every encoder; 2 × 2 encoder substitution; projector and loss ablations.",
    experiment: "Deterministic unimodal trials on unseen people; B0 recipe with only the input swapped.",
    result: `ECAPA ${fmt(metric("p2_audio_unimodal", "ECAPA-TDNN, centred", "same_language_other_video", "unimodal"), 1)}% vs B0's voice input ${fmt(metric("p2_audio_unimodal", "WavLM + eGeMAPS (B0 input), LDA", "same_language_other_video", "unimodal"), 1)}% unimodal EER — yet ArcFace faces cost ${d("g3_b0_arcface", "mean")} and ECAPA alone ${d("g3_ecapa_farl", "mean")} cross-modally. ECAPA appended (C1): unseen Urdu gender ${td("urdu_holdout_phase2", "C1 vs B0", "gender_Urdu")}.`,
    decision: "Better identity ≠ better association. Keep attribute-rich encoders; investigate why ECAPA helps only as a complement.",
    chart: { id: "urdu_holdout_phase2", systems: ["B0", "ECAPA + FaRL", "C1"], keys: ["gender_English", "gender_Urdu"] } },
  { id: "complement", stage: "ECAPA complementarity", title: "Controls, fusion and a second unseen language", supported: "yes", refs: ["urdu_controls_seed1", "p3_bl2", "german_holdout"],
    believed: "ECAPA adds speaker-aligned, language-stable information — not just dimensions.",
    changed: "Random-feature and speaker-shuffled controls; bottleneck, block-L2 and residual fusion; German holdout frozen in advance.",
    experiment: "Seed-1 screens on English folds and Urdu, 3-seed promotion, then the untouched German confirmation.",
    result: `Random features vs C1 on Urdu gender ${td("urdu_controls_seed1", "Control D (random) vs C1", "gender_Urdu")}, shuffled ${td("urdu_controls_seed1", "Control E (shuffled) vs C1", "gender_Urdu")}. German gender, C1 vs B0: ${td("german_holdout", "C1 vs B0", "gender_German")}.`,
    decision: "Keep plain concatenation; the gain generalises to a second unseen language.",
    chart: { id: "german_holdout", systems: ["B0", "C1", "C1 block L2"], keys: ["gender_English", "gender_German"] } },
  { id: "final", stage: "Final ensemble", title: "C1 × 3 with refitted terms", supported: "yes", refs: ["seed_ensemble_official", "seed_ensembles_holdouts", "final_c1x3_terms"],
    believed: "Seed variance is a meaningful share of the remaining noise.",
    changed: "Three seeds of C1, equal-weight mean of distances, terms refitted on the ensemble.",
    experiment: "Matched-size ensembles on official folds, Urdu and German.",
    result: `${d("seed_ensemble_official", "mean")} mean EER vs single C1 on the official folds; final local mean with terms ${fmt(metric("final_c1x3_terms", "C1 x 3 (raw-mean ensemble) + terms", "mean"))}.`,
    decision: "Final candidate. Not submitted; no official score exists for it." },
];

// ---------------------------------------------------------------------------------------------
// Key findings
// ---------------------------------------------------------------------------------------------
export interface Finding { n: number; title: string; body: string; caveat: string; refs: string[] }
export const findings: Finding[] = [
  { n: 1, title: "Better unimodal identification is not better cross-modal alignment", refs: ["p2_face_unimodal", "g3_b0_arcface", "g3_ecapa_farl"],
    body: `ArcFace verifies faces at ${fmt(metric("p2_face_unimodal", "arcface_insightface", "same_gender", "unimodal"), 1)}% same-gender EER vs FaRL's ${fmt(metric("p2_face_unimodal", "farl", "same_gender", "unimodal"), 1)}%, yet replacing FaRL with it costs ${d("g3_b0_arcface", "mean")} EER cross-modally. ECAPA alone, a far better speaker encoder, costs ${d("g3_ecapa_farl", "mean")}.`,
    caveat: "Interpretation (identity training discards voice-predictable attributes) is consistent with probes but not proven causally." },
  { n: 2, title: "More identities help — the domain matters", refs: ["learning_curve", "ext_all_share50", "codabench_eval_submissions"],
    body: `About −1.6 EER points per doubling of in-domain identities; adding 231 external people gave ${d("ext_all_share50", "mean")} locally and ${fmtDelta(metric("codabench_eval_submissions", "Eval #3: V2 + external data + terms", "mean") - metric("codabench_eval_submissions", "Eval #2: V2 + age/gender terms", "mean"))} on CodaBench.`,
    caveat: "The 77 extra hidden identities in the GAMP-VF simulation added no measurable gain on top of 171 others — returns diminish and depend on source." },
  { n: 3, title: "Pseudo-labelling fails when the teacher is weak", refs: ["gamp_pseudo_labels", "gamp_oracle_headroom"],
    body: (() => { const r = exp("gamp_pseudo_labels").recovery!; return `With perfect grouping of 77 hidden people, cross-modal matching reached ${Math.round((r.oracle_groups_top1 ?? 0) * 100)}% top-1 and ${Math.round((r.oracle_groups_hungarian ?? 0) * 100)}% with Hungarian assignment; the most confident 10% were only ${Math.round((r.oracle_groups_precision_top10pct ?? 0) * 100)}% correct.`; })(),
    caveat: "Simulation on training-side data; the project has no genuinely unlabelled pool." },
  { n: 4, title: "Complementary acoustic representations beat either alone", refs: ["g3_cat_farl", "urdu_controls_seed1", "urdu_holdout_phase2"],
    body: `ECAPA appended to WavLM + eGeMAPS improves unseen Urdu same-gender EER by ${td("urdu_holdout_phase2", "C1 vs B0", "gender_Urdu")}; fixed random features (${td("urdu_controls_seed1", "Control D (random) vs C1", "gender_Urdu")} vs C1) and speaker-shuffled ECAPA (${td("urdu_controls_seed1", "Control E (shuffled) vs C1", "gender_Urdu")}) lose the gain.`,
    caveat: "Which acoustic properties cause the gain is not established; ECAPA's added part carries speaker identity but little age/build information." },
  { n: 5, title: "Validate beyond the training language", refs: ["german_holdout", "gamp_b3_sg", "urdu_holdout_phase2"],
    body: `GAMP-VF's same-gender sampling helped English only; C1 holds on two unseen languages — Urdu gender ${td("urdu_holdout_phase2", "C1 vs B0", "gender_Urdu")}, German gender ${td("german_holdout", "C1 vs B0", "gender_German")}.`,
    caveat: "On the official English folds C1 is neutral; Bangla may differ from Urdu and German." },
];

/** Display names and one-line descriptions for the key experiments (registry titles stay unchanged in experiments.json). */
export const KEY_EXPERIMENTS: Array<{ id: string; name: string; desc: string }> = [
  { id: "v1_to_v2", name: "V1 → V2 upgrade", desc: "All-layer WavLM voices and a SupCon loss." },
  { id: "ext_all_share50", name: "External training data", desc: "From 70 to 301 training identities." },
  { id: "b0_full", name: "Auxiliary voice cues (B0)", desc: "eGeMAPS, trait heads and language losses." },
  { id: "g3_b0_arcface", name: "Face-recognition encoder", desc: "ArcFace faces in place of FaRL." },
  { id: "g3_cat_farl", name: "Adding ECAPA (C1)", desc: "Speaker embedding appended to the voice input." },
  { id: "urdu_holdout_phase2", name: "Unseen language: Urdu", desc: "C1 vs B0 on a language never heard." },
  { id: "urdu_controls_seed1", name: "Urdu control tests", desc: "Is the gain from ECAPA's speaker content?" },
  { id: "german_holdout", name: "Unseen language: German", desc: "Checks the Urdu gain on a second language." },
  { id: "seed_ensemble_official", name: "Seed ensemble (C1\u00a0×\u00a03)", desc: "Averaging three C1 models." },
];
/** Clean display name for an experiment, falling back to its registry title. */
export const expName = (e: { id: string; title: string }) => KEY_EXPERIMENTS.find((k) => k.id === e.id)?.name ?? e.title;
