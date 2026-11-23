"""Stage C: gender-aware hard-negative identity sampling for P x K batches.

The contrastive loss already treats every other identity in the batch as a
negative, so hard negatives are introduced by *choosing which identities share
a batch*. No separate hard-negative loss term is added, so hard negatives are
not counted twice.

A batch is built from P/2 anchor identities; each anchor gets one partner.
With probability ``hard_ratio`` the partner is chosen by ``strategy``,
otherwise uniformly at random:

* ``random``                - uniform (equivalent to the existing sampler)
* ``same_gender``           - random identity of the anchor's gender
* ``nearest``               - one of the top-k most confusable identities (model similarity)
* ``same_gender_nearest``   - same, restricted to the anchor's gender
* ``same_gender_age``       - one of the k identities of the same gender with the closest age

Safeguards:
* Different labels are different identities by construction. Identities in the
  same ``conflict_group`` (e.g. pseudo identities that may be the same person)
  never share a batch, so they can never act as each other's negatives.
* Unknown gender (-1) never satisfies a same-gender constraint. Such anchors
  fall back to a random partner, and unknown identities are never picked as
  same-gender partners.
* Partners come from the anchor's own pool (official / external), which keeps
  the official share exact.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler

STRATEGIES = ("random", "same_gender", "nearest", "same_gender_nearest", "same_gender_age")


class GenderAwareHardNegativeSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: np.ndarray,
        identities_per_batch: int,
        samples_per_identity: int,
        seed: int,
        batches_per_epoch: int,
        identity_genders: np.ndarray | None = None,
        strategy: str = "random",
        hard_ratio: float = 0.5,
        top_k: int = 5,
        primary_identities: np.ndarray | None = None,
        primary_share: float | None = None,
        row_languages: np.ndarray | None = None,
        language_balanced_k: bool = False,
        identity_weights: np.ndarray | None = None,
        identity_ages: np.ndarray | None = None,
        conflict_groups: np.ndarray | None = None,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy}; choose from {STRATEGIES}")
        if identities_per_batch < 2 or identities_per_batch % 2 or samples_per_identity < 2:
            raise ValueError("P must be even and >= 2, K >= 2")
        if not 0.0 <= hard_ratio <= 1.0 or top_k < 1:
            raise ValueError("hard_ratio must be in [0, 1] and top_k >= 1")
        self.num_identities = int(labels.max()) + 1
        if not np.array_equal(np.unique(labels), np.arange(self.num_identities)):
            raise ValueError("Labels must be contiguous 0..U-1")
        grouped: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            grouped[int(label)].append(i)
        self.rows = [np.asarray(grouped[u]) for u in range(self.num_identities)]
        self.P, self.K, self.seed = identities_per_batch, samples_per_identity, seed
        self.batches_per_epoch = batches_per_epoch
        self.strategy, self.hard_ratio, self.top_k = strategy, hard_ratio, top_k
        self.epoch = 0
        U = self.num_identities
        self.genders = np.full(U, -1) if identity_genders is None else np.asarray(identity_genders, dtype=np.int64)
        self.ages = None if identity_ages is None else np.asarray(identity_ages, dtype=np.float64)
        if strategy == "same_gender_age" and self.ages is None:
            raise ValueError("same_gender_age needs identity ages")
        self.weights = np.ones(U) if identity_weights is None else np.asarray(identity_weights, dtype=np.float64)
        if len(self.genders) != U or len(self.weights) != U or (self.weights <= 0).any():
            raise ValueError("Per-identity arrays must have one positive entry per identity")
        self.conflict = np.arange(U) if conflict_groups is None else np.asarray(conflict_groups, dtype=np.int64)
        # Pools: official (primary) vs other, with a fixed number of anchors from each.
        if primary_share is not None:
            if primary_identities is None or not 0 < primary_share < 1:
                raise ValueError("primary_share needs primary identities and must be in (0, 1)")
            is_primary = np.isin(np.arange(U), primary_identities)
            self.pools = [np.flatnonzero(is_primary), np.flatnonzero(~is_primary)]
            n_primary = int(round(self.P * primary_share))
            if n_primary % 2:
                n_primary += 1 if n_primary < self.P // 2 else -1
            self.anchor_counts = [n_primary // 2, (self.P - n_primary) // 2]
        else:
            self.pools = [np.arange(U)]
            self.anchor_counts = [self.P // 2]
        self.pool_of = np.zeros(U, dtype=np.int64)
        for p, members in enumerate(self.pools):
            self.pool_of[members] = p
            if len(members) < 2 * self.anchor_counts[p]:
                raise ValueError("A pool has too few identities for its share of P")
        # Language-balanced K: per identity, rows grouped by language.
        self.language_balanced_k = language_balanced_k
        if language_balanced_k:
            if row_languages is None or len(row_languages) != len(labels):
                raise ValueError("language_balanced_k needs one language per row")
            row_languages = np.asarray(row_languages)
            self.rows_by_language = [[r[row_languages[r] == l] for l in np.unique(row_languages[r])]
                                     for r in self.rows]
        self.neighbours: np.ndarray | None = None
        if strategy == "same_gender_age":
            self._set_age_neighbours()

    # -- neighbour tables -------------------------------------------------
    def _allowed(self, anchor: int, same_gender: bool) -> np.ndarray:
        ok = (self.pool_of == self.pool_of[anchor]) & (self.conflict != self.conflict[anchor])
        if same_gender:
            ok &= (self.genders == self.genders[anchor]) & (self.genders >= 0)
        return ok

    def _set_age_neighbours(self) -> None:
        table = np.full((self.num_identities, self.top_k), -1)
        for a in range(self.num_identities):
            if self.genders[a] < 0 or not np.isfinite(self.ages[a]):
                continue
            ok = self._allowed(a, True) & np.isfinite(self.ages)
            cand = np.flatnonzero(ok)
            cand = cand[np.argsort(np.abs(self.ages[cand] - self.ages[a]), kind="stable")][: self.top_k]
            table[a, : len(cand)] = cand
        self.neighbours = table

    def set_identity_similarity(self, similarity: np.ndarray) -> None:
        """Refresh hard-negative neighbours from an identity x identity similarity
        (higher = more confusable); the diagonal is ignored."""
        if self.strategy not in ("nearest", "same_gender_nearest"):
            return
        similarity = np.asarray(similarity, dtype=np.float64)
        if similarity.shape != (self.num_identities, self.num_identities):
            raise ValueError("Similarity must be identities x identities")
        same_gender = self.strategy == "same_gender_nearest"
        table = np.full((self.num_identities, self.top_k), -1)
        for a in range(self.num_identities):
            if same_gender and self.genders[a] < 0:
                continue
            ok = self._allowed(a, same_gender)
            ok[a] = False
            cand = np.flatnonzero(ok)
            cand = cand[np.argsort(-similarity[a, cand], kind="stable")][: self.top_k]
            table[a, : len(cand)] = cand
        self.neighbours = table

    # -- sampling -----------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.P * self.K

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _partner(self, anchor: int, taken: np.ndarray, rng: np.random.Generator) -> int:
        blocked = np.isin(self.conflict, self.conflict[taken])
        if rng.random() < self.hard_ratio and self.strategy != "random":
            if self.strategy == "same_gender":
                cand = np.flatnonzero(self._allowed(anchor, True) & ~blocked) if self.genders[anchor] >= 0 else []
            elif self.neighbours is not None:
                cand = [c for c in self.neighbours[anchor] if c >= 0 and not blocked[c]]
            else:
                cand = []  # warm-up before the first similarity refresh: random partner
            if len(cand):
                return int(rng.choice(cand))
        cand = np.flatnonzero((self.pool_of == self.pool_of[anchor]) & ~blocked)
        p = self.weights[cand] / self.weights[cand].sum()
        return int(rng.choice(cand, p=p))

    def _rows_for(self, identity: int, rng: np.random.Generator) -> np.ndarray:
        if not self.language_balanced_k or len(self.rows_by_language[identity]) == 1:
            r = self.rows[identity]
            return rng.choice(r, self.K, replace=len(r) < self.K)
        groups = self.rows_by_language[identity]
        order = rng.permutation(len(groups))
        share = [self.K // len(groups) + (i < self.K % len(groups)) for i in range(len(groups))]
        out = [rng.choice(groups[g], n, replace=len(groups[g]) < n) for g, n in zip(order, share) if n]
        return np.concatenate(out)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            taken: list[int] = []
            for pool, count in zip(self.pools, self.anchor_counts):
                for _ in range(count):
                    blocked = np.isin(self.conflict, self.conflict[taken]) if taken else np.zeros(len(self.conflict), bool)
                    cand = pool[~blocked[pool]]
                    p = self.weights[cand] / self.weights[cand].sum()
                    anchor = int(rng.choice(cand, p=p))
                    taken.append(anchor)
                    taken.append(self._partner(anchor, np.asarray(taken), rng))
            batch = np.concatenate([self._rows_for(u, rng) for u in taken])
            rng.shuffle(batch)
            yield batch.tolist()


def identity_cross_modal_similarity(face_protos: np.ndarray, voice_protos: np.ndarray) -> np.ndarray:
    """sim[i, j] = 0.5 (cos(voice_i, face_j) + cos(face_i, voice_j)): how easily i and j are confused."""
    f = face_protos / np.linalg.norm(face_protos, axis=1, keepdims=True)
    v = voice_protos / np.linalg.norm(voice_protos, axis=1, keepdims=True)
    s = v @ f.T
    return 0.5 * (s + s.T)
