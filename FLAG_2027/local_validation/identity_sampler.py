"""Identity-balanced batch construction for cross-modal metric learning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class IdentityBalancedBatchSampler(Sampler[list[int]]):
    """Yield P identities and K aligned samples from each identity per batch."""

    def __init__(
        self,
        labels: np.ndarray,
        identities_per_batch: int,
        samples_per_identity: int,
        seed: int,
        batches_per_epoch: int | None = None,
        genders: np.ndarray | None = None,
        gender_balanced: bool = False,
        primary_identities: np.ndarray | None = None,
        primary_share: float | None = None,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("labels must be a non-empty one-dimensional array")
        if identities_per_batch < 2 or samples_per_identity < 2:
            raise ValueError("P and K must both be at least 2")
        self.indices_by_identity: dict[int, np.ndarray] = {}
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[int(label)].append(index)
        for label, indices in grouped.items():
            self.indices_by_identity[label] = np.asarray(indices, dtype=np.int64)
        self.identities = np.asarray(sorted(grouped), dtype=np.int64)
        if identities_per_batch > len(self.identities):
            raise ValueError("P exceeds the number of training identities")
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.seed = seed
        self.epoch = 0
        batch_size = identities_per_batch * samples_per_identity
        self.batches_per_epoch = batches_per_epoch or max(1, len(labels) // batch_size)
        self.gender_balanced = gender_balanced
        self.identities_by_gender: dict[int, np.ndarray] = {}
        if gender_balanced:
            if genders is None or len(genders) != len(labels):
                raise ValueError("Gender-balanced sampling requires one gender per row")
            if identities_per_batch % 2:
                raise ValueError("Gender-balanced P must be even")
            gender_by_identity: dict[int, int] = {}
            for label, gender in zip(labels, np.asarray(genders, dtype=np.int64)):
                previous = gender_by_identity.setdefault(int(label), int(gender))
                if previous != int(gender):
                    raise ValueError("One identity maps to multiple genders")
            for gender in (0, 1):
                available = [
                    identity
                    for identity in self.identities
                    if gender_by_identity[int(identity)] == gender
                ]
                if len(available) < identities_per_batch // 2:
                    raise ValueError("Not enough identities for gender-balanced sampling")
                self.identities_by_gender[gender] = np.asarray(available, dtype=np.int64)

        # Optional domain mixing: a fixed number of each batch's identities comes
        # from the primary (official) pool, the rest from the other identities.
        self.primary_count = None
        if primary_share is not None:
            if gender_balanced:
                raise ValueError("primary_share cannot be combined with gender-balanced sampling")
            if not 0.0 < primary_share < 1.0 or primary_identities is None:
                raise ValueError("primary_share must be in (0, 1) and needs primary identities")
            primary = np.isin(self.identities, np.asarray(primary_identities))
            self.primary_pool, self.other_pool = self.identities[primary], self.identities[~primary]
            self.primary_count = int(round(identities_per_batch * primary_share))
            if not (0 < self.primary_count < identities_per_batch) or self.primary_count > len(self.primary_pool) \
                    or identities_per_batch - self.primary_count > len(self.other_pool):
                raise ValueError("primary_share leaves a pool too small for P")

    @property
    def batch_size(self) -> int:
        return self.identities_per_batch * self.samples_per_identity

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            if self.primary_count is not None:
                selected = np.concatenate([
                    rng.choice(self.primary_pool, self.primary_count, replace=False),
                    rng.choice(self.other_pool, self.identities_per_batch - self.primary_count, replace=False),
                ])
                rng.shuffle(selected)
            elif self.gender_balanced:
                half = self.identities_per_batch // 2
                selected = np.concatenate(
                    [
                        rng.choice(self.identities_by_gender[0], half, replace=False),
                        rng.choice(self.identities_by_gender[1], half, replace=False),
                    ]
                )
                rng.shuffle(selected)
            else:
                selected = rng.choice(
                    self.identities, self.identities_per_batch, replace=False
                )
            batch: list[int] = []
            for identity in selected:
                candidates = self.indices_by_identity[int(identity)]
                chosen = rng.choice(
                    candidates,
                    self.samples_per_identity,
                    replace=len(candidates) < self.samples_per_identity,
                )
                batch.extend(int(index) for index in chosen)
            rng.shuffle(batch)
            yield batch
