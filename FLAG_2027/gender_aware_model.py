"""Gender-Aware Dual-Head Network (GADH-V1) for face--voice verification.

This module is intentionally separate from ``cross_modal_model.py`` so the
original CrossModalIdentityNetwork V1 remains unchanged and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Package import: python -m FLAG_2027.train_gender_aware
    from .cross_modal_model import ArcMarginClassifier
except ImportError:  # Direct module import used by the local unit tests
    from cross_modal_model import ArcMarginClassifier


@dataclass
class GenderAwareEmbeddings:
    """Global, fine-grained, and inference embeddings for one modality."""

    global_embedding: torch.Tensor
    fine_embedding: torch.Tensor
    inference_embedding: torch.Tensor


class GenderAwareProjector(nn.Module):
    """Adapt one modality and produce global and within-gender embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int],
        global_dim: int = 256,
        fine_dim: int = 128,
        score_global_weight: float = 0.5,
        architecture_mode: str = "dual_head",
    ) -> None:
        super().__init__()
        if architecture_mode not in {"dual_head", "single_256"}:
            raise ValueError(f"Unknown architecture mode: {architecture_mode}")
        if not 0.0 <= score_global_weight <= 1.0:
            raise ValueError("score_global_weight must lie in [0, 1]")
        if input_dim < 1 or global_dim < 1 or fine_dim < 1:
            raise ValueError("All embedding dimensions must be positive")

        self.architecture_mode = architecture_mode
        self.score_global_weight = score_global_weight
        first_hidden, second_hidden = hidden_dims
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(first_hidden, second_hidden),
            nn.LayerNorm(second_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.global_head = nn.Linear(second_hidden, global_dim)
        self.fine_head = (
            nn.Linear(second_hidden, fine_dim)
            if architecture_mode == "dual_head"
            else None
        )

    def components(self, features: torch.Tensor) -> GenderAwareEmbeddings:
        hidden = self.trunk(features)
        global_embedding = F.normalize(self.global_head(hidden), p=2, dim=1)
        if self.fine_head is None:
            # The single-head ablation applies both global and same-gender
            # objectives directly to the original 256-D projected embedding.
            fine_embedding = global_embedding
            inference_embedding = global_embedding
        else:
            fine_embedding = F.normalize(self.fine_head(hidden), p=2, dim=1)
            global_scale = sqrt(self.score_global_weight)
            fine_scale = sqrt(1.0 - self.score_global_weight)
            inference_embedding = torch.cat(
                (global_scale * global_embedding, fine_scale * fine_embedding), dim=1
            )
            # Each component has unit norm and the squared weights sum to one.
            inference_embedding = F.normalize(inference_embedding, p=2, dim=1)
        return GenderAwareEmbeddings(
            global_embedding=global_embedding,
            fine_embedding=fine_embedding,
            inference_embedding=inference_embedding,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the embedding used by the unchanged validation scorer."""
        return self.components(features).inference_embedding


class GenderAwareDualHeadNetwork(nn.Module):
    """GADH-V1 with a global identity head and a same-gender detail head."""

    architecture_name = "gadh_v1"

    def __init__(
        self,
        face_dim: int,
        voice_dim: int,
        num_classes: int,
        global_dim: int = 256,
        fine_dim: int = 128,
        score_global_weight: float = 0.5,
        architecture_mode: str = "dual_head",
        arc_scale: float = 30.0,
        arc_margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.architecture_mode = architecture_mode
        self.face_branch = GenderAwareProjector(
            face_dim,
            (1024, 512),
            global_dim,
            fine_dim,
            score_global_weight,
            architecture_mode,
        )
        self.voice_branch = GenderAwareProjector(
            voice_dim,
            (512, 512),
            global_dim,
            fine_dim,
            score_global_weight,
            architecture_mode,
        )
        # Only the global head is tied to seen training identities. The fine
        # head is trained by identity-relative same-gender metric learning.
        self.identity_classifier = ArcMarginClassifier(
            global_dim, num_classes, arc_scale, arc_margin
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, faces: torch.Tensor, voices: torch.Tensor
    ) -> tuple[GenderAwareEmbeddings, GenderAwareEmbeddings]:
        return self.face_branch.components(faces), self.voice_branch.components(voices)

    def identity_logits(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.identity_classifier(embeddings, labels)
