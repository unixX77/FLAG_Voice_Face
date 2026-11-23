"""Shared-space face--voice model for unseen-identity verification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Map one modality's fixed backbone feature into a normalized embedding.

    ``hidden_dims=()`` gives a single linear map. Optional input
    standardization statistics are stored as buffers so that validation and
    submission inference apply exactly the training-time normalization.
    Input dropout and Gaussian noise act only in training mode.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        embedding_dim: int,
        dropout: tuple[float, ...],
        standardize_inputs: bool = False,
        input_dropout: float = 0.0,
        input_noise_std: float = 0.0,
        normalize_output: bool = True,
        output_map: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        # Optional parameter-free map applied last (e.g. onto the Poincare ball).
        self.output_map = output_map
        if len(hidden_dims) != len(dropout):
            raise ValueError("Each hidden layer must have one dropout value")
        if not 0 <= input_dropout < 1 or input_noise_std < 0:
            raise ValueError("Input dropout must be in [0, 1) and noise non-negative")
        self.standardize_inputs = standardize_inputs
        if standardize_inputs:
            self.register_buffer("input_mean", torch.zeros(input_dim))
            self.register_buffer("input_std", torch.ones(input_dim))
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_noise_std = input_noise_std
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim, probability in zip(hidden_dims, dropout):
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(probability),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, embedding_dim))
        self.network = nn.Sequential(*layers)

    @torch.no_grad()
    def set_input_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.standardize_inputs:
            raise RuntimeError("Projector was built without input standardization")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.standardize_inputs:
            features = (features - self.input_mean) / self.input_std
        if self.training and self.input_noise_std > 0:
            features = features + self.input_noise_std * torch.randn_like(features)
        features = self.input_dropout(features)
        output = self.network(features)
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=1)
        return output if self.output_map is None else self.output_map(output)


class VoiceBlockFusion(nn.Module):
    """Voice branch for a two-block input [base (split dims) | extra (e.g. ECAPA)] (Phase III E2).

    The full input is standardised (training statistics), then noise/dropout are applied once,
    exactly as in ``Projector``. Modes:
      * ``bottleneck``: extra -> Linear(extra_dim, fusion_dim), concatenated with base, then MLP.
      * ``block_l2``:   each block L2-normalised and rescaled so both blocks carry equal weight
                        and the total norm matches the z-scored input (sqrt(input_dim)).
      * ``residual``:   h = MLP(base) + alpha * adapter(extra); alpha starts at 0, so training
                        starts from a base-only model.
    """

    MODES = ("bottleneck", "block_l2", "residual")

    def __init__(self, input_dim: int, split: int, mode: str, fusion_dim: int, hidden_dims: tuple[int, ...],
                 embedding_dim: int, dropout: tuple[float, ...], standardize_inputs: bool = False,
                 input_dropout: float = 0.0, input_noise_std: float = 0.0) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown voice fusion mode {mode}")
        if not 0 < split < input_dim:
            raise ValueError("voice_fusion_split must lie inside the voice input")
        self.mode, self.split, self.input_dim = mode, split, input_dim
        self.standardize_inputs = standardize_inputs
        if standardize_inputs:
            self.register_buffer("input_mean", torch.zeros(input_dim))
            self.register_buffer("input_std", torch.ones(input_dim))
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_noise_std = input_noise_std
        extra_dim = input_dim - split
        plain = dict(standardize_inputs=False, input_dropout=0.0, input_noise_std=0.0)
        if mode == "bottleneck":
            self.extra_projection = nn.Linear(extra_dim, fusion_dim)
            self.projector = Projector(split + fusion_dim, hidden_dims, embedding_dim, dropout, **plain)
        elif mode == "block_l2":
            self.projector = Projector(input_dim, hidden_dims, embedding_dim, dropout, **plain)
        else:
            self.projector = Projector(split, hidden_dims, embedding_dim, dropout, normalize_output=False, **plain)
            self.adapter = nn.Sequential(nn.Linear(extra_dim, fusion_dim), nn.GELU(),
                                         nn.Linear(fusion_dim, embedding_dim))
            self.alpha = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def set_input_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if not self.standardize_inputs:
            raise RuntimeError("Voice fusion branch was built without input standardization")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.standardize_inputs:
            features = (features - self.input_mean) / self.input_std
        if self.training and self.input_noise_std > 0:
            features = features + self.input_noise_std * torch.randn_like(features)
        features = self.input_dropout(features)
        base, extra = features[:, : self.split], features[:, self.split:]
        if self.mode == "bottleneck":
            return self.projector(torch.cat([base, self.extra_projection(extra)], dim=1))
        if self.mode == "block_l2":
            scale = (self.input_dim / 2) ** 0.5
            return self.projector(torch.cat([F.normalize(base, dim=1) * scale, F.normalize(extra, dim=1) * scale], 1))
        return F.normalize(self.projector(base) + self.alpha * self.adapter(extra), p=2, dim=1)


class PoincareMap(nn.Module):
    """Clip a Euclidean vector's norm, then map it onto the Poincare ball.

    Clipping to ``clip_radius`` before the exponential map at the origin keeps
    points away from the ball boundary, where distances and gradients become
    numerically unstable (feature clipping, Guo et al., CVPR 2022).
    """

    def __init__(self, curvature: float = 1.0, clip_radius: float = 1.0) -> None:
        super().__init__()
        if curvature <= 0 or clip_radius <= 0:
            raise ValueError("Curvature and clip radius must be positive")
        self.curvature = curvature
        self.clip_radius = clip_radius

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        norm = vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
        vectors = vectors * torch.clamp(self.clip_radius / norm, max=1.0)
        norm = vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
        root = self.curvature ** 0.5
        return torch.tanh(root * norm) * vectors / (root * norm)


def poincare_distance_matrix(
    x: torch.Tensor, y: torch.Tensor, curvature: float
) -> torch.Tensor:
    """Pairwise geodesic distances between rows of x and y on the Poincare ball."""
    squared = torch.cdist(x, y).pow(2)
    x_factor = (1 - curvature * x.pow(2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
    y_factor = (1 - curvature * y.pow(2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
    argument = 1 + 2 * curvature * squared / (x_factor * y_factor.T)
    return torch.acosh(argument.clamp_min(1 + 1e-7)) / curvature ** 0.5


def poincare_pair_distance(
    x: torch.Tensor, y: torch.Tensor, curvature: float
) -> torch.Tensor:
    """Geodesic distance between row-aligned points on the Poincare ball."""
    squared = (x - y).pow(2).sum(dim=1)
    x_factor = (1 - curvature * x.pow(2).sum(dim=1)).clamp_min(1e-6)
    y_factor = (1 - curvature * y.pow(2).sum(dim=1)).clamp_min(1e-6)
    argument = 1 + 2 * curvature * squared / (x_factor * y_factor)
    return torch.acosh(argument.clamp_min(1 + 1e-7)) / curvature ** 0.5


class VoiceLayerPool(nn.Module):
    """Pool per-layer WavLM statistics (batch, layers, dim) into one vector.

    Each layer is z-scored with training statistics, then combined either by
    one learned softmax weight per layer shared by all clips (``mix``, which
    starts as a plain average) or by per-clip additive attention over layers
    (``attention``).
    """

    def __init__(self, layers: int, dim: int, mode: str, attention_dim: int = 64) -> None:
        super().__init__()
        if mode not in ("mix", "attention"):
            raise ValueError(f"Unknown layer pooling: {mode}")
        self.mode = mode
        self.register_buffer("layer_mean", torch.zeros(layers, dim))
        self.register_buffer("layer_std", torch.ones(layers, dim))
        if mode == "mix":
            self.layer_logits = nn.Parameter(torch.zeros(layers))
        else:
            self.score = nn.Sequential(nn.Linear(dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    @torch.no_grad()
    def set_input_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.layer_mean.copy_(mean)
        self.layer_std.copy_(std.clamp_min(1e-6))

    def layer_weights(self, layers: torch.Tensor) -> torch.Tensor:
        """Per-clip weights over layers, shape (batch, layers)."""
        if self.mode == "mix":
            return self.layer_logits.softmax(dim=0).expand(len(layers), -1)
        return self.score(layers).squeeze(-1).softmax(dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        layers = (features - self.layer_mean) / self.layer_std
        return (self.layer_weights(layers).unsqueeze(-1) * layers).sum(dim=1)


class VoiceLayerBranch(nn.Module):
    """Layer pooling followed by the usual projector (which then skips z-scoring)."""

    def __init__(self, pool: VoiceLayerPool, projector: Projector) -> None:
        super().__init__()
        self.pool = pool
        self.projector = projector

    def set_input_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.pool.set_input_statistics(mean, std)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(self.pool(features))


class ArcMarginClassifier(nn.Module):
    """A normalized shared classifier with an additive angular target margin."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        if scale <= 0 or margin < 0:
            raise ValueError("ArcFace scale must be positive and margin non-negative")
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        cosine = F.linear(
            F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1)
        )
        if labels is None or self.margin == 0:
            return cosine * self.scale
        target_cosine = cosine.gather(1, labels[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
        target_with_margin = torch.cos(torch.acos(target_cosine) + self.margin)
        logits = cosine.clone()
        logits.scatter_(1, labels[:, None], target_with_margin)
        return logits * self.scale


class VoicePrototypeBank(nn.Module):
    """Momentum-updated voice centers used only as a training objective."""

    def __init__(self, num_classes: int, embedding_dim: int, momentum: float = 0.9):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("Prototype momentum must be in [0, 1)")
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, embedding_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, voice_embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        for identity in labels.unique():
            identity_index = int(identity)
            center = F.normalize(
                voice_embeddings[labels == identity].mean(dim=0), dim=0
            )
            if self.initialized[identity_index]:
                center = F.normalize(
                    self.momentum * self.prototypes[identity_index]
                    + (1.0 - self.momentum) * center,
                    dim=0,
                )
            self.prototypes[identity_index].copy_(center)
            self.initialized[identity_index] = True

    def logits_and_targets(
        self,
        face_embeddings: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError("Prototype temperature must be positive")
        valid_classes = self.initialized.nonzero(as_tuple=False).flatten()
        if len(valid_classes) == 0:
            raise RuntimeError("No voice prototypes have been initialized")
        class_to_column = torch.full(
            (len(self.prototypes),), -1, dtype=torch.long, device=labels.device
        )
        class_to_column[valid_classes] = torch.arange(
            len(valid_classes), device=labels.device
        )
        targets = class_to_column[labels]
        if (targets < 0).any():
            raise RuntimeError("A batch label has no initialized voice prototype")
        logits = face_embeddings @ self.prototypes[valid_classes].T / temperature
        return logits, targets


class CrossModalIdentityNetwork(nn.Module):
    """Independent projectors aligned through direct cross-modal objectives."""

    def __init__(
        self,
        face_dim: int,
        voice_dim: int,
        embedding_dim: int,
        num_classes: int,
        arc_scale: float = 30.0,
        arc_margin: float = 0.2,
        prototype_momentum: float = 0.9,
        face_hidden_dims: tuple[int, ...] = (1024, 512),
        voice_hidden_dims: tuple[int, ...] = (512, 512),
        hidden_dropout: float | None = None,
        standardize_inputs: bool = False,
        input_dropout: float = 0.0,
        input_noise_std: float = 0.0,
        geometry: str = "sphere",
        curvature: float = 1.0,
        clip_radius: float = 1.0,
        voice_layers: int = 0,
        voice_layer_pooling: str = "mix",
        voice_fusion: str = "concat",
        voice_fusion_split: int = 0,
        voice_fusion_dim: int = 64,
    ) -> None:
        super().__init__()
        if geometry not in ("sphere", "hyperbolic"):
            raise ValueError(f"Unknown geometry: {geometry}")
        self.geometry = geometry
        self.curvature = curvature

        def dropout_for(hidden_dims: tuple[int, ...]) -> tuple[float, ...]:
            # None keeps the original V1 schedule of 0.2 then 0.1.
            if hidden_dropout is not None:
                return (hidden_dropout,) * len(hidden_dims)
            return tuple(0.2 if index == 0 else 0.1 for index in range(len(hidden_dims)))

        regularization = dict(
            standardize_inputs=standardize_inputs,
            input_dropout=input_dropout,
            input_noise_std=input_noise_std,
            normalize_output=geometry == "sphere",
        )
        ball = PoincareMap(curvature, clip_radius) if geometry == "hyperbolic" else None
        self.face_branch = Projector(
            face_dim,
            tuple(face_hidden_dims),
            embedding_dim,
            dropout_for(tuple(face_hidden_dims)),
            **regularization,
            output_map=ball,
        )
        voice_regularization = dict(regularization)
        if voice_layers:
            # Per-layer z-scoring happens in the pool instead.
            voice_regularization["standardize_inputs"] = False
        self.voice_branch = Projector(
            voice_dim,
            tuple(voice_hidden_dims),
            embedding_dim,
            dropout_for(tuple(voice_hidden_dims)),
            **voice_regularization,
            output_map=ball,
        )
        if voice_fusion != "concat":
            if voice_layers or geometry != "sphere":
                raise ValueError("Voice block fusion supports plain sphere-geometry voice features only")
            self.voice_branch = VoiceBlockFusion(
                voice_dim, voice_fusion_split, voice_fusion, voice_fusion_dim, tuple(voice_hidden_dims),
                embedding_dim, dropout_for(tuple(voice_hidden_dims)), standardize_inputs, input_dropout,
                input_noise_std)
        if voice_layers:
            # Voice inputs are then (batch, voice_layers, voice_dim).
            self.voice_branch = VoiceLayerBranch(
                VoiceLayerPool(voice_layers, voice_dim, voice_layer_pooling), self.voice_branch
            )
        self.identity_classifier = ArcMarginClassifier(
            embedding_dim, num_classes, arc_scale, arc_margin
        )
        self.voice_prototypes = VoicePrototypeBank(
            num_classes, embedding_dim, prototype_momentum
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.face_branch(faces), self.voice_branch(voices)

    def similarity_matrix(self, faces: torch.Tensor, voices: torch.Tensor) -> torch.Tensor:
        """All face-by-voice similarities; larger means a better match."""
        if self.geometry == "hyperbolic":
            return -poincare_distance_matrix(faces, voices, self.curvature)
        return faces @ voices.T

    def pair_similarity(self, faces: torch.Tensor, voices: torch.Tensor) -> torch.Tensor:
        """Similarity of row-aligned face/voice embeddings; larger is a better match."""
        if self.geometry == "hyperbolic":
            return -poincare_pair_distance(faces, voices, self.curvature)
        return (faces * voices).sum(dim=1)

    def identity_logits(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.identity_classifier(embeddings, labels)
