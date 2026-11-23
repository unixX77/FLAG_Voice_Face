"""Losses shared by fold-based baseline experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalProjectionLoss(nn.Module):
    """Pull same-class fused embeddings together and decorrelate other classes."""

    def __init__(self, negative_weight: float = 0.7):
        super().__init__()
        self.negative_weight = negative_weight

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("Expected features (B,D) and labels (B,)")
        features = F.normalize(features, p=2, dim=1)
        labels = labels.to(features.device)
        same_class = labels[:, None].eq(labels[None, :])
        diagonal = torch.eye(len(labels), dtype=torch.bool, device=features.device)
        positive_mask = same_class & ~diagonal
        negative_mask = ~same_class
        similarities = features @ features.T

        # A shuffled batch can occasionally contain no repeated identity. Keep
        # the loss finite and differentiable in that case instead of producing
        # NaN from the mean of an empty tensor.
        if positive_mask.any():
            positive_mean = similarities[positive_mask].mean()
            positive_loss = 1.0 - positive_mean
        else:
            positive_mean = similarities.new_tensor(0.0)
            positive_loss = similarities.sum() * 0.0
        if negative_mask.any():
            negative_mean = similarities[negative_mask].abs().mean()
        else:
            negative_mean = similarities.sum() * 0.0
        loss = positive_loss + self.negative_weight * negative_mean
        return loss, positive_mean, negative_mean
