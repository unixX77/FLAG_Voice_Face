"""Auxiliary training-only modules for the voice -> traits -> face architecture.

All of these act on the shared 128-d embeddings during training and are not
used for scoring, so checkpoints of ``CrossModalIdentityNetwork`` and every
evaluation/submission tool stay unchanged (their weights are stored separately
under ``aux_state_dict``).

* ``TraitHeads``: *shared* linear heads that read age, build and gender from an
  embedding. The same head is applied to the face and to the voice embedding,
  so both modalities must place a person's traits along the same directions of
  the shared space (voice -> traits <- face). Targets are person-level labels
  from frozen face models (InsightFace age, FaRL zero-shot build) and metadata
  gender.
* ``LanguageAdversary``: a classifier that predicts the spoken language from
  the voice embedding behind a gradient-reversal layer, so the voice encoder is
  trained to remove language information (English / Urdu / German only; no
  Bengali is ever used).
* ``cross_lingual_alignment``: pulls together voice embeddings of the same
  person spoken in different languages.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def gradient_reversal(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, scale)


class TraitHeads(nn.Module):
    """Shared linear read-outs: age (z-scored), build (z-scored), gender (logit)."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.age = nn.Linear(embedding_dim, 1)
        self.build = nn.Linear(embedding_dim, 1)
        self.gender = nn.Linear(embedding_dim, 1)

    def loss(self, embeddings: torch.Tensor, age: torch.Tensor, build: torch.Tensor,
             gender: torch.Tensor, targets: tuple[str, ...]) -> torch.Tensor:
        parts = []
        if "age" in targets:
            m = torch.isfinite(age)
            if m.any():
                parts.append(F.smooth_l1_loss(self.age(embeddings[m]).squeeze(1), age[m]))
        if "build" in targets:
            m = torch.isfinite(build)
            if m.any():
                parts.append(F.smooth_l1_loss(self.build(embeddings[m]).squeeze(1), build[m]))
        if "gender" in targets:
            m = gender >= 0  # -1 = unknown (pseudo identities); all-known batches are unchanged
            if m.any():
                parts.append(F.binary_cross_entropy_with_logits(self.gender(embeddings[m]).squeeze(1),
                                                                gender[m].float()))
        return torch.stack(parts).mean() if parts else embeddings.sum() * 0.0


class LanguageAdversary(nn.Module):
    def __init__(self, embedding_dim: int, languages: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.GELU(), nn.Linear(hidden, languages))

    def loss(self, voice_embeddings: torch.Tensor, language: torch.Tensor, scale: float,
             class_weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (adversarial loss, classifier accuracy) over rows with a known language."""
        m = language >= 0
        if not m.any():
            zero = voice_embeddings.sum() * 0.0
            return zero, zero.detach()
        logits = self.net(gradient_reversal(voice_embeddings[m], scale))
        target = language[m]
        loss = F.cross_entropy(logits, target, weight=class_weights)
        return loss, (logits.argmax(1) == target).float().mean().detach()


def cross_lingual_alignment(voice_embeddings: torch.Tensor, labels: torch.Tensor,
                            language: torch.Tensor) -> torch.Tensor:
    """Mean (1 - cos) over voice pairs of one identity spoken in two different known languages."""
    same = labels[:, None].eq(labels[None, :])
    known = (language >= 0)[:, None] & (language >= 0)[None, :]
    different = language[:, None].ne(language[None, :])
    mask = same & known & different
    if not mask.any():
        return voice_embeddings.sum() * 0.0
    v = F.normalize(voice_embeddings, dim=1)
    return (1.0 - v @ v.T)[mask].mean()
