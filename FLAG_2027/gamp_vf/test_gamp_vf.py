"""Unit tests for GAMP-VF components. Run from FLAG_2027/:
    python -m unittest gamp_vf.test_gamp_vf -v
"""

import unittest

import numpy as np
import torch

from gamp_vf.clustering import agglomerative, cluster_metrics, hdbscan
from gamp_vf.matching import hungarian_partial, local_selection, sinkhorn_match, sinkhorn_partial
from gamp_vf.prototypes import (batch_prototypes, identity_prototypes, prototype_contrastive_loss,
                                speaker_similarity_statistics)
from gamp_vf.sampling import GenderAwareHardNegativeSampler, identity_cross_modal_similarity
from gamp_vf.similarity import cosine_matrix, rank_margin, reciprocal_top1, top_k


def _synthetic(n_ids=6, per=10, dim=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_ids, dim))
    ids = np.repeat(np.arange(n_ids), per)
    return centres[ids] + noise * rng.normal(size=(len(ids), dim)), ids, centres


class PrototypeTests(unittest.TestCase):
    def test_prototypes_are_unit_and_uniform_mean(self):
        x, ids, _ = _synthetic()
        p = identity_prototypes(x, ids)
        np.testing.assert_allclose(np.linalg.norm(p.prototypes, axis=1), 1, atol=1e-9)
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        expect = unit[ids == 2].sum(0)
        np.testing.assert_allclose(p.prototypes[2], expect / np.linalg.norm(expect), atol=1e-9)

    def test_language_balanced_ignores_clip_imbalance(self):
        # Identity 0: 9 English clips at e1, 1 Urdu clip at e2. Balanced proto sits midway.
        e1, e2 = np.eye(2)
        x = np.vstack([np.tile(e1, (9, 1)), e2[None]])
        ids, lang = np.zeros(10, int), np.r_[np.zeros(9, int), 1]
        uni = identity_prototypes(x, ids, lang, "uniform").prototypes[0]
        bal = identity_prototypes(x, ids, lang, "language_balanced").prototypes[0]
        np.testing.assert_allclose(bal, np.array([1, 1]) / np.sqrt(2), atol=1e-9)
        self.assertGreater(uni[0], bal[0])

    def test_missing_language_is_its_own_group(self):
        x = np.array([[1.0, 0], [0, 1.0]])
        p = identity_prototypes(x, np.zeros(2, int), np.array([-1, 0]), "language_balanced")
        self.assertEqual(set(p.language_counts[0]), {-1, 0})

    def test_batch_prototypes_match_numpy_and_balanced(self):
        x, ids, _ = _synthetic(4, 6, 8)
        lang = np.tile([0, 0, 0, 0, 1, 2], 4)
        xt = torch.nn.functional.normalize(torch.tensor(x, dtype=torch.float64), dim=1)
        u, p = batch_prototypes(xt, torch.tensor(ids), torch.tensor(lang), language_balanced=True)
        ref = identity_prototypes(x, ids, lang, "language_balanced").prototypes
        np.testing.assert_allclose(p.numpy(), ref, atol=1e-9)
        u, p = batch_prototypes(xt, torch.tensor(ids))
        np.testing.assert_allclose(p.numpy(), identity_prototypes(x, ids).prototypes, atol=1e-9)

    def test_prototype_loss_prefers_aligned_modalities_and_backprops(self):
        x, ids, _ = _synthetic(4, 4, 8, seed=1)
        f = torch.nn.functional.normalize(torch.tensor(x, dtype=torch.float32), dim=1).requires_grad_()
        labels = torch.tensor(ids)
        aligned = prototype_contrastive_loss(f, f, labels)
        shuffled = prototype_contrastive_loss(f, f[torch.randperm(len(f))], labels)
        self.assertLess(aligned.item(), shuffled.item())
        aligned.backward()
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_speaker_statistics_separates_speakers(self):
        x, ids, _ = _synthetic(5, 8, 16, noise=0.1)
        lang = np.tile([0, 1], 20)
        s = speaker_similarity_statistics(x, ids, lang)
        self.assertGreater(s["same_speaker_cross_language"], s["diff_speaker_same_language"])
        self.assertGreater(s["auc_same_vs_different_speaker"], 0.95)


class SimilarityTests(unittest.TestCase):
    def test_cosine_matrix_chunked_equals_dense(self):
        rng = np.random.default_rng(0)
        v, f = rng.normal(size=(37, 5)), rng.normal(size=(11, 5))
        s = cosine_matrix(v, f, chunk=8, device="cpu")
        vn, fn = v / np.linalg.norm(v, axis=1, keepdims=True), f / np.linalg.norm(f, axis=1, keepdims=True)
        np.testing.assert_allclose(s, vn @ fn.T, atol=1e-5)

    def test_topk_reciprocal_margin(self):
        s = np.array([[0.9, 0.1, 0.2], [0.8, 0.7, 0.0]])
        idx, val = top_k(s, 2)
        np.testing.assert_array_equal(idx, [[0, 2], [0, 1]])
        np.testing.assert_array_equal(reciprocal_top1(s), [True, False])
        np.testing.assert_allclose(rank_margin(s), [0.7, 0.1])
        idx_t, _ = top_k(s, 1, axis=0)
        np.testing.assert_array_equal(idx_t[:, 0], [0, 1, 0])


class ClusteringTests(unittest.TestCase):
    def test_agglomerative_recovers_separated_identities(self):
        x, ids, _ = _synthetic(6, 10, 16, noise=0.05)
        m = cluster_metrics(agglomerative(x, 0.3), ids)
        self.assertAlmostEqual(m["ari"], 1.0)
        self.assertAlmostEqual(m["purity"], 1.0)

    def test_noise_does_not_inflate_metrics(self):
        ids = np.repeat(np.arange(3), 4)
        all_noise = np.full(12, -1)
        m = cluster_metrics(all_noise, ids)
        self.assertEqual(m["coverage"], 0.0)
        self.assertLessEqual(m["ari"], 0.0 + 1e-9)

    def test_hdbscan_runs_and_allows_noise(self):
        x, ids, _ = _synthetic(4, 15, 8, noise=0.05)
        labels = hdbscan(x, min_cluster_size=5)
        self.assertGreater(cluster_metrics(labels, ids)["ari"], 0.9)


class MatchingTests(unittest.TestCase):
    def test_hungarian_rectangular_and_no_match(self):
        s = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.25]])  # 3 voices, 2 faces
        np.testing.assert_array_equal(hungarian_partial(s, 0.5), [0, 1, -1])
        np.testing.assert_array_equal(hungarian_partial(s, 0.95), [-1, -1, -1])

    def test_hungarian_prefers_global_optimum(self):
        # Greedy would give row 0 -> col 0 (0.9) and leave row 1 with 0.1.
        s = np.array([[0.9, 0.85], [0.8, 0.1]])
        np.testing.assert_array_equal(hungarian_partial(s, 0.0), [1, 0])

    def test_local_selection_requires_reciprocity_and_margin(self):
        s = np.array([[0.9, 0.1], [0.85, 0.2]])
        np.testing.assert_array_equal(local_selection(s), [0, -1])
        np.testing.assert_array_equal(local_selection(s, min_margin=0.9), [-1, -1])

    def test_sinkhorn_stable_and_marginals(self):
        rng = np.random.default_rng(0)
        extreme = rng.normal(size=(5, 7)) * 50  # logits ~ +-5000: must stay finite (log domain)
        self.assertTrue(np.isfinite(sinkhorn_partial(extreme, temperature=0.01, iterations=500)).all())
        s = np.clip(rng.normal(size=(5, 7)) * 0.3, -1, 1)  # realistic cosines
        plan = sinkhorn_partial(s, temperature=0.05, iterations=500)
        np.testing.assert_allclose(plan[:5].sum(1), 1, atol=1e-3)
        np.testing.assert_allclose(plan[:, :7].sum(0), 1, atol=1e-3)

    def test_sinkhorn_match_one_to_one_with_unmatched(self):
        s = np.array([[0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, -0.5]])
        match, w = sinkhorn_match(s, temperature=0.05, dustbin=0.3)
        np.testing.assert_array_equal(match, [0, 1, -1])
        self.assertEqual(len(set(match[match >= 0])), (match >= 0).sum())


class SamplerTests(unittest.TestCase):
    def _sampler(self, strategy, **kw):
        labels = np.repeat(np.arange(12), 5)
        genders = np.array([0, 1] * 6)
        return GenderAwareHardNegativeSampler(labels, 4, 2, seed=0, batches_per_epoch=50,
                                              identity_genders=genders, strategy=strategy, **kw), labels, genders

    def test_batches_have_distinct_identities_and_k_rows(self):
        for strategy in ("random", "same_gender"):
            s, labels, _ = self._sampler(strategy, hard_ratio=1.0)
            for batch in s:
                ids, counts = np.unique(labels[batch], return_counts=True)
                self.assertEqual(len(ids), 4)
                self.assertTrue((counts == 2).all())

    def test_same_gender_partner(self):
        s, labels, genders = self._sampler("same_gender", hard_ratio=1.0)
        for batch in s:
            ids = labels[batch]
            # 2 anchors x 2: gender counts of identities must be even per gender
            g = genders[np.unique(ids)]
            self.assertEqual((g == 0).sum() % 2, 0)

    def test_nearest_uses_similarity_after_refresh(self):
        s, labels, genders = self._sampler("same_gender_nearest", hard_ratio=1.0, top_k=1)
        sim = np.zeros((12, 12))
        for a in range(12):  # most confusable same-gender identity: a +/- 2
            sim[a, (a + 2) % 12] = sim[(a + 2) % 12, a] = 1
        s.set_identity_similarity(sim)
        for a in range(12):
            self.assertEqual(genders[s.neighbours[a, 0]], genders[a])
            self.assertIn(s.neighbours[a, 0], ((a + 2) % 12, (a - 2) % 12))

    def test_conflict_groups_never_share_a_batch(self):
        groups = np.arange(12)
        groups[1] = 0  # identities 0 and 1 might be the same person
        s, labels, _ = self._sampler("random", conflict_groups=groups)
        for batch in s:
            ids = set(labels[batch])
            self.assertFalse({0, 1} <= ids)

    def test_unknown_gender_never_same_gender_partner(self):
        labels = np.repeat(np.arange(8), 3)
        genders = np.array([0, 0, 0, 0, -1, -1, 1, 1])
        s = GenderAwareHardNegativeSampler(labels, 2, 2, 0, 200, genders, "same_gender", hard_ratio=1.0)
        for batch in s:
            ids = sorted(set(labels[batch]))
            if genders[ids[0]] >= 0 and genders[ids[1]] >= 0 and genders[ids[0]] != genders[ids[1]]:
                self.fail("known-gender anchor paired across genders")

    def test_language_balanced_k(self):
        labels = np.repeat(np.arange(4), 10)
        lang = np.tile(np.r_[np.zeros(9, int), 1], 4)  # 9 English + 1 Urdu per identity
        s = GenderAwareHardNegativeSampler(labels, 2, 4, 0, 20, row_languages=lang, language_balanced_k=True)
        for batch in s:
            for u in np.unique(labels[batch]):
                rows = np.asarray(batch)[labels[batch] == u]
                self.assertEqual((lang[rows] == 1).sum(), 2)

    def test_primary_share_exact(self):
        labels = np.repeat(np.arange(20), 3)
        s = GenderAwareHardNegativeSampler(labels, 8, 2, 0, 30, strategy="random",
                                           primary_identities=np.arange(6), primary_share=0.5)
        for batch in s:
            ids = np.unique(labels[batch])
            self.assertEqual((ids < 6).sum(), 4)

    def test_identity_similarity_symmetric(self):
        rng = np.random.default_rng(0)
        sim = identity_cross_modal_similarity(rng.normal(size=(5, 4)), rng.normal(size=(5, 4)))
        np.testing.assert_allclose(sim, sim.T)


if __name__ == "__main__":
    unittest.main()
