"""Loss functions for direct face--voice shared-space learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import OrthogonalProjectionLoss


def multi_positive_directional_loss(
    similarities: torch.Tensor,
    anchor_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised InfoNCE where every same-identity candidate is positive."""
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    if similarities.shape != (len(anchor_labels), len(candidate_labels)):
        raise ValueError("Similarity matrix and label lengths do not agree")
    positive_mask = anchor_labels[:, None].eq(candidate_labels[None, :])
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every anchor must have at least one positive candidate")
    scaled = similarities / temperature
    positive_logits = scaled.masked_fill(~positive_mask, -torch.inf)
    log_positive = torch.logsumexp(positive_logits, dim=1)
    log_all = torch.logsumexp(scaled, dim=1)
    return (log_all - log_positive).mean()


def supervised_contrastive_directional_loss(
    similarities: torch.Tensor,
    anchor_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Average the log-probability of every same-identity candidate.

    Unlike positive-set InfoNCE, this objective cannot be satisfied by making
    only one of several positive candidates highly similar to the anchor.
    """
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    if similarities.shape != (len(anchor_labels), len(candidate_labels)):
        raise ValueError("Similarity matrix and label lengths do not agree")
    positive_mask = anchor_labels[:, None].eq(candidate_labels[None, :])
    positive_counts = positive_mask.sum(dim=1)
    if (positive_counts == 0).any():
        raise ValueError("Every anchor must have at least one positive candidate")
    scaled = similarities / temperature
    log_probabilities = scaled - torch.logsumexp(scaled, dim=1, keepdim=True)
    mean_positive_log_probability = (
        (log_probabilities * positive_mask).sum(dim=1) / positive_counts
    )
    return -mean_positive_log_probability.mean()


def symmetric_cross_modal_contrastive_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    mode: str = "multi_positive",
    similarities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric multi-positive loss and the full similarity matrix.

    ``similarities`` defaults to the cosine matrix of unit embeddings; pass a
    precomputed matrix (e.g. negative hyperbolic distances) to use another geometry.
    """
    if similarities is None:
        similarities = face_embeddings @ voice_embeddings.T
    if mode == "multi_positive":
        directional_loss = multi_positive_directional_loss
    elif mode == "supcon":
        directional_loss = supervised_contrastive_directional_loss
    else:
        raise ValueError(f"Unknown contrastive mode: {mode}")
    face_to_voice = directional_loss(
        similarities, labels, labels, temperature
    )
    voice_to_face = directional_loss(
        similarities.T, labels, labels, temperature
    )
    return 0.5 * (face_to_voice + voice_to_face), similarities


def _same_gender_supcon_directional_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive_mask = labels[:, None].eq(labels[None, :])
    # Positives are always same-gender; opposite-gender candidates are removed
    # from the softmax so they cannot make the loss small for free.
    candidate_mask = genders[:, None].eq(genders[None, :])
    scaled = (similarities / temperature).masked_fill(~candidate_mask, -torch.inf)
    log_probabilities = scaled - torch.logsumexp(scaled, dim=1, keepdim=True)
    positive_log_probabilities = torch.where(
        positive_mask, log_probabilities, torch.zeros_like(log_probabilities)
    )
    mean_positive_log_probability = (
        positive_log_probabilities.sum(dim=1) / positive_mask.sum(dim=1)
    )
    return -mean_positive_log_probability.mean()


def symmetric_same_gender_supcon_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss whose negatives all share the anchor's gender.

    The ordinary contrastive term is mostly satisfied by separating the two
    genders. This term only rewards ranking same-identity candidates above
    same-gender impostors, matching the gender-controlled trials.
    """
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    if not (len(face_embeddings) == len(labels) == len(genders)):
        raise ValueError("Embeddings, labels, and genders must have equal lengths")
    similarities = face_embeddings @ voice_embeddings.T
    face_to_voice = _same_gender_supcon_directional_loss(
        similarities, labels, genders, temperature
    )
    voice_to_face = _same_gender_supcon_directional_loss(
        similarities.T, labels, genders, temperature
    )
    return 0.5 * (face_to_voice + voice_to_face)


def paired_cosine_alignment_loss(
    face_embeddings: torch.Tensor, voice_embeddings: torch.Tensor
) -> torch.Tensor:
    """Pull each row-aligned face and voice toward the same direction."""
    return (1.0 - (face_embeddings * voice_embeddings).sum(dim=1)).mean()


def fused_orthogonal_projection_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    negative_weight: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the baseline OPL objective to a parameter-free pair embedding.

    The projected face and voice vectors are already in the same normalized
    space. Their normalized mean represents one aligned face--voice training
    pair. OPL then pulls pair representations belonging to the same identity
    together and makes representations from different identities orthogonal.

    This head is used only to calculate a training loss. Inference continues
    to compare the model's face and voice embeddings directly, so enabling the
    objective does not create a two-model ensemble or alter the architecture's
    verification score path.
    """
    if face_embeddings.shape != voice_embeddings.shape:
        raise ValueError("Face and voice embeddings must have the same shape")
    if negative_weight < 0:
        raise ValueError("OPL negative weight must be non-negative")
    fused_pairs = F.normalize(face_embeddings + voice_embeddings, p=2, dim=1)
    return OrthogonalProjectionLoss(negative_weight)(fused_pairs, labels)


def _directional_same_gender_hard_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    positive_scores: torch.Tensor,
    margin: float,
    top_k: int,
) -> torch.Tensor:
    negative_mask = labels[:, None].ne(labels[None, :]) & genders[:, None].eq(
        genders[None, :]
    )
    row_losses: list[torch.Tensor] = []
    for row in range(len(labels)):
        negatives = similarities[row][negative_mask[row]]
        if len(negatives) == 0:
            continue
        hardest = negatives.topk(min(top_k, len(negatives))).values
        row_losses.append(F.relu(margin - positive_scores[row] + hardest).mean())
    if not row_losses:
        return similarities.sum() * 0.0
    return torch.stack(row_losses).mean()


def symmetric_same_gender_hard_negative_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    margin: float = 0.2,
    top_k: int = 5,
) -> torch.Tensor:
    """Rank paired matches above the most similar same-gender impostors."""
    if margin < 0 or top_k < 1:
        raise ValueError("Hard-negative margin must be non-negative and top_k positive")
    similarities = face_embeddings @ voice_embeddings.T
    paired = similarities.diagonal()
    face_to_voice = _directional_same_gender_hard_loss(
        similarities, labels, genders, paired, margin, top_k
    )
    voice_to_face = _directional_same_gender_hard_loss(
        similarities.T, labels, genders, paired, margin, top_k
    )
    return 0.5 * (face_to_voice + voice_to_face)


def _masked_logmeanexp(
    scores: torch.Tensor, mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Smooth maximum per row without bias from different set sizes."""
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every anchor must have at least one selected candidate")
    scaled = scores / temperature
    return temperature * (
        torch.logsumexp(scaled.masked_fill(~mask, -torch.inf), dim=1)
        - counts.float().log()
    )


def _directional_multi_positive_same_gender_loss(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive_mask = labels[:, None].eq(labels[None, :])
    same_gender_negative_mask = (
        labels[:, None].ne(labels[None, :])
        & genders[:, None].eq(genders[None, :])
    )
    positive_scores = _masked_logmeanexp(
        similarities, positive_mask, temperature
    )
    negative_scores = _masked_logmeanexp(
        similarities, same_gender_negative_mask, temperature
    )
    # The softplus ranking loss is expressed in cosine-similarity units.
    losses = temperature * F.softplus(
        (negative_scores - positive_scores + margin) / temperature
    )
    return losses.mean(), positive_scores.mean(), negative_scores.mean()


def symmetric_multi_positive_same_gender_loss(
    face_embeddings: torch.Tensor,
    voice_embeddings: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    margin: float = 0.1,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank all same-ID positives over same-gender different-ID candidates.

    Unlike the earlier diagonal hard-negative loss, this objective represents
    an identity with every positive sample available in the P x K batch. It
    uses only same-gender impostors as negatives and operates symmetrically in
    the face-to-voice and voice-to-face directions.
    """
    if face_embeddings.shape != voice_embeddings.shape:
        raise ValueError("Face and voice embeddings must have the same shape")
    if len(face_embeddings) != len(labels) or len(labels) != len(genders):
        raise ValueError("Embeddings, labels, and genders must have equal lengths")
    if margin < 0 or temperature <= 0:
        raise ValueError("Margin must be non-negative and temperature positive")
    similarities = face_embeddings @ voice_embeddings.T
    forward = _directional_multi_positive_same_gender_loss(
        similarities, labels, genders, margin, temperature
    )
    backward = _directional_multi_positive_same_gender_loss(
        similarities.T, labels, genders, margin, temperature
    )
    return tuple(0.5 * (left + right) for left, right in zip(forward, backward))
