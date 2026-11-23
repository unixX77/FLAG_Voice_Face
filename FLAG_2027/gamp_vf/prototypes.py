"""Speaker (Stage A) and face (Stage B) identity prototypes.

Modes:
* ``uniform`` (A2): p_i = normalize(sum_k w_ik v_ik) with w = 1 (or given weights).
* ``language_balanced`` (A3): average within each language first, then average
  the per-language prototypes, so a speaker with many English clips does not
  dominate. Rows with unknown language (< 0) form their own group.
Individual embeddings are never replaced; prototypes are an extra view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


@dataclass
class PrototypeSet:
    identities: np.ndarray          # (U,) identity ids, sorted
    prototypes: np.ndarray          # (U, d) unit vectors
    counts: np.ndarray              # (U,) rows per identity
    language_counts: list[dict[int, int]]  # per identity: language -> rows
    language_prototypes: list[dict[int, np.ndarray]]  # per identity: language -> unit vector
    within_spread: np.ndarray       # (U,) mean cosine of members to their prototype (quality)


def identity_prototypes(
    embeddings: np.ndarray,
    identities: np.ndarray,
    languages: np.ndarray | None = None,
    mode: str = "uniform",
    weights: np.ndarray | None = None,
) -> PrototypeSet:
    """Prototypes of known identities from (normalised) row embeddings."""
    if mode not in ("uniform", "language_balanced"):
        raise ValueError(f"Unknown prototype mode: {mode}")
    embeddings = _normalize_np(np.asarray(embeddings, dtype=np.float64))
    identities = np.asarray(identities)
    if len(embeddings) != len(identities):
        raise ValueError("Embeddings and identities must align")
    languages = np.full(len(identities), -1) if languages is None else np.asarray(languages)
    weights = np.ones(len(identities)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(languages) != len(identities) or len(weights) != len(identities):
        raise ValueError("Languages/weights must align with identities")
    if (weights < 0).any():
        raise ValueError("Weights must be non-negative")
    ids = np.unique(identities)
    protos, counts, lang_counts, lang_protos, spread = [], [], [], [], []
    for identity in ids:
        rows = np.flatnonzero(identities == identity)
        per_language = {}
        per_language_count = {}
        for language in np.unique(languages[rows]):
            sub = rows[languages[rows] == language]
            w = weights[sub]
            if w.sum() <= 0:
                continue
            per_language[int(language)] = _normalize_np((embeddings[sub] * w[:, None]).sum(0))
            per_language_count[int(language)] = len(sub)
        if not per_language:
            raise ValueError(f"Identity {identity} has zero total weight")
        if mode == "uniform":
            w = weights[rows]
            proto = _normalize_np((embeddings[rows] * w[:, None]).sum(0))
        else:
            proto = _normalize_np(np.mean(list(per_language.values()), axis=0))
        protos.append(proto)
        counts.append(len(rows))
        lang_counts.append(per_language_count)
        lang_protos.append(per_language)
        spread.append(float((embeddings[rows] @ proto).mean()))
    return PrototypeSet(ids, np.stack(protos), np.asarray(counts), lang_counts, lang_protos, np.asarray(spread))


def batch_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    languages: torch.Tensor | None = None,
    language_balanced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable in-batch prototypes: (unique labels (U,), unit prototypes (U, d))."""
    unique, inverse = torch.unique(labels, return_inverse=True)
    one_hot = F.one_hot(inverse, len(unique)).to(embeddings.dtype)  # (N, U)
    if not language_balanced or languages is None:
        summed = one_hot.T @ embeddings
        return unique, F.normalize(summed, dim=1)
    # Per (identity, language) mean, then mean over the languages present.
    lang_unique, lang_inverse = torch.unique(languages, return_inverse=True)
    group = inverse * len(lang_unique) + lang_inverse
    groups = len(unique) * len(lang_unique)
    sums = torch.zeros(groups, embeddings.shape[1], dtype=embeddings.dtype, device=embeddings.device)
    sums.index_add_(0, group, embeddings)
    sizes = torch.zeros(groups, dtype=embeddings.dtype, device=embeddings.device)
    sizes.index_add_(0, group, torch.ones_like(group, dtype=embeddings.dtype))
    present = sizes > 0
    # Each per-language prototype is normalised before averaging (A3), as in identity_prototypes.
    means = torch.where(present[:, None], F.normalize(sums, dim=1), torch.zeros_like(sums))
    means = means.view(len(unique), len(lang_unique), -1)
    present = present.view(len(unique), len(lang_unique)).to(embeddings.dtype)
    balanced = means.sum(1) / present.sum(1, keepdim=True)
    return unique, F.normalize(balanced, dim=1)


def prototype_contrastive_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    languages: torch.Tensor | None = None,
    language_balanced: bool = False,
) -> torch.Tensor:
    """Symmetric instance-to-prototype InfoNCE across modalities.

    Each face is classified among the batch's voice-identity prototypes and each
    voice among the face-identity prototypes. A prototype is one entry per
    identity, so same-identity samples can never act as negatives.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    unique, voice_protos = batch_prototypes(voice_embeddings, labels, languages, language_balanced)
    _, face_protos = batch_prototypes(face_embeddings, labels)
    target = torch.searchsorted(unique, labels)
    face_to_voice = F.cross_entropy(face_embeddings @ voice_protos.T / temperature, target)
    voice_to_face = F.cross_entropy(voice_embeddings @ face_protos.T / temperature, target)
    return 0.5 * (face_to_voice + voice_to_face)


def speaker_similarity_statistics(
    embeddings: np.ndarray, identities: np.ndarray, languages: np.ndarray, max_rows: int = 6000, seed: int = 0
) -> dict[str, float]:
    """Mean cosine for {same, different} speaker x {same, different} language, plus AUCs.

    Only rows with a known language (>= 0) are used. Answers: are recordings of
    one speaker in two languages closer than recordings of two speakers?
    """
    from sklearn.metrics import roc_auc_score

    keep = np.flatnonzero(np.asarray(languages) >= 0)
    rng = np.random.default_rng(seed)
    if len(keep) > max_rows:
        keep = np.sort(rng.choice(keep, max_rows, replace=False))
    e = _normalize_np(np.asarray(embeddings, dtype=np.float64)[keep])
    ids, lang = np.asarray(identities)[keep], np.asarray(languages)[keep]
    sims = e @ e.T
    upper = np.triu(np.ones_like(sims, dtype=bool), 1)
    same_id = (ids[:, None] == ids[None, :]) & upper
    same_lang = lang[:, None] == lang[None, :]
    out = {}
    for name, mask in (("same_speaker_same_language", same_id & same_lang),
                       ("same_speaker_cross_language", same_id & ~same_lang),
                       ("diff_speaker_same_language", upper & ~same_id & same_lang),
                       ("diff_speaker_cross_language", upper & ~same_id & ~same_lang)):
        out[name] = float(sims[mask].mean()) if mask.any() else float("nan")
        out[name + "_pairs"] = int(mask.sum())
    # AUC: same-speaker cross-language vs different-speaker same-language
    # (the hard case: language similarity working against identity).
    pos, neg = sims[same_id & ~same_lang], sims[upper & ~same_id & same_lang]
    if len(pos) and len(neg):
        neg = rng.choice(neg, min(len(neg), 200000), replace=False)
        out["auc_cross_language_same_speaker_vs_same_language_other"] = float(
            roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    pos_all, neg_all = sims[same_id], sims[upper & ~same_id]
    neg_all = rng.choice(neg_all, min(len(neg_all), 200000), replace=False)
    out["auc_same_vs_different_speaker"] = float(
        roc_auc_score(np.r_[np.ones(len(pos_all)), np.zeros(len(neg_all))], np.r_[pos_all, neg_all]))
    return out
