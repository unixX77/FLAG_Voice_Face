"""Pairwise verification head built on the original V1 projectors."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .cross_modal_model import CrossModalIdentityNetwork
except ImportError:
    from cross_modal_model import CrossModalIdentityNetwork


class PairwiseVerificationNetwork(nn.Module):
    """Learn a match score from interactions between V1 face/voice embeddings.

    The encoders produce the same normalized 256-D vectors as V1. The verifier
    sees each vector, their absolute difference, and their elementwise product.
    It outputs one logit; larger logits indicate a more likely identity match.
    """

    architecture_name = "cross_modal_v1_pairwise_verifier"

    def __init__(
        self,
        face_dim: int,
        voice_dim: int,
        embedding_dim: int,
        num_classes: int,
        arc_scale: float = 30.0,
        arc_margin: float = 0.2,
        verifier_hidden_dim: int = 256,
        verifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = CrossModalIdentityNetwork(
            face_dim,
            voice_dim,
            embedding_dim,
            num_classes,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
        )
        interaction_dim = 4 * embedding_dim
        self.verifier = nn.Sequential(
            nn.Linear(interaction_dim, verifier_hidden_dim),
            nn.LayerNorm(verifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(verifier_dropout),
            nn.Linear(verifier_hidden_dim, verifier_hidden_dim // 2),
            nn.LayerNorm(verifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(verifier_dropout / 2.0),
            nn.Linear(verifier_hidden_dim // 2, 1),
        )

    @property
    def face_branch(self):
        return self.encoder.face_branch

    @property
    def voice_branch(self):
        return self.encoder.voice_branch

    @property
    def identity_classifier(self):
        return self.encoder.identity_classifier

    def identity_logits(self, embeddings: torch.Tensor, labels: torch.Tensor):
        return self.encoder.identity_logits(embeddings, labels)

    def encode(
        self, faces: torch.Tensor, voices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(faces, voices)

    def pair_features(
        self, face_embeddings: torch.Tensor, voice_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if face_embeddings.shape != voice_embeddings.shape:
            raise ValueError("Paired face and voice embeddings must have equal shapes")
        if face_embeddings.ndim != 2 or face_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected two (batch, {self.embedding_dim}) embedding matrices"
            )
        return torch.cat(
            (
                face_embeddings,
                voice_embeddings,
                torch.abs(face_embeddings - voice_embeddings),
                face_embeddings * voice_embeddings,
            ),
            dim=1,
        )

    def score_embeddings(
        self, face_embeddings: torch.Tensor, voice_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Return one match logit per aligned pair, shape (batch,)."""
        return self.verifier(self.pair_features(face_embeddings, voice_embeddings)).squeeze(1)

    def forward(
        self, faces: torch.Tensor, voices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aligned V1 embeddings for compatibility with metric losses."""
        return self.encode(faces, voices)
