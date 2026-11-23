"""Tests for the proposed shared-space architecture and its training losses."""

import unittest

import numpy as np
import torch

from cross_modal_model import CrossModalIdentityNetwork
from local_validation.cross_modal_losses import (
    fused_orthogonal_projection_loss,
    multi_positive_directional_loss,
    paired_cosine_alignment_loss,
    supervised_contrastive_directional_loss,
    symmetric_cross_modal_contrastive_loss,
    symmetric_same_gender_hard_negative_loss,
    symmetric_same_gender_supcon_loss,
)
from local_validation.identity_sampler import IdentityBalancedBatchSampler


class CrossModalTests(unittest.TestCase):
    def test_projectors_and_shared_classifier_backpropagate(self) -> None:
        model = CrossModalIdentityNetwork(16, 6, 8, 4)
        faces = torch.randn(12, 16)
        voices = torch.randn(12, 6)
        labels = torch.arange(4).repeat_interleave(3)
        face_embeddings, voice_embeddings = model(faces, voices)
        self.assertEqual(face_embeddings.shape, (12, 8))
        self.assertEqual(voice_embeddings.shape, (12, 8))
        torch.testing.assert_close(face_embeddings.norm(dim=1), torch.ones(12))
        torch.testing.assert_close(voice_embeddings.norm(dim=1), torch.ones(12))
        loss = torch.nn.functional.cross_entropy(
            model.identity_logits(face_embeddings, labels), labels
        )
        loss.backward()
        self.assertIsNotNone(model.face_branch.network[0].weight.grad)

    def test_multi_positive_loss_prefers_correct_clusters(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        # Embeddings whose matrix products reproduce the desired comparison
        # are unnecessary here; test the public symmetric function using two
        # identity-aligned bases and a deliberately permuted voice version.
        faces = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        voices_good = faces.clone()
        voices_bad = torch.flip(faces, dims=[0])
        good_loss, _ = symmetric_cross_modal_contrastive_loss(
            faces, voices_good, labels, 0.1
        )
        bad_loss, _ = symmetric_cross_modal_contrastive_loss(
            faces, voices_bad, labels, 0.1
        )
        self.assertLess(good_loss.item(), bad_loss.item())

    def test_supcon_penalizes_one_weak_positive(self) -> None:
        similarities = torch.tensor([[1.0, -1.0, -1.0]])
        anchor_labels = torch.tensor([0])
        candidate_labels = torch.tensor([0, 0, 1])
        set_loss = multi_positive_directional_loss(
            similarities, anchor_labels, candidate_labels, 0.1
        )
        supcon_loss = supervised_contrastive_directional_loss(
            similarities, anchor_labels, candidate_labels, 0.1
        )
        self.assertGreater(supcon_loss.item(), set_loss.item() + 1.0)

    def test_contrastive_mode_validation(self) -> None:
        embeddings = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "Unknown contrastive mode"):
            symmetric_cross_modal_contrastive_loss(
                embeddings, embeddings, labels, mode="invalid"
            )

    def test_pair_and_hard_negative_losses_are_finite(self) -> None:
        face = torch.nn.functional.normalize(torch.randn(8, 6), dim=1)
        voice = torch.nn.functional.normalize(torch.randn(8, 6), dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        genders = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        self.assertTrue(torch.isfinite(paired_cosine_alignment_loss(face, voice)))
        hard = symmetric_same_gender_hard_negative_loss(
            face, voice, labels, genders
        )
        self.assertTrue(torch.isfinite(hard))

    def test_fused_opl_prefers_identity_clusters_and_backpropagates(self) -> None:
        labels = torch.tensor([0, 0, 1, 1])
        good_face = torch.tensor(
            [[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True
        )
        good_voice = good_face.detach().clone().requires_grad_(True)
        bad_voice = torch.flip(good_voice.detach(), dims=[0])
        good_loss, positive, negative = fused_orthogonal_projection_loss(
            good_face, good_voice, labels
        )
        bad_loss, _, _ = fused_orthogonal_projection_loss(
            good_face.detach(), bad_voice, labels
        )
        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertAlmostEqual(positive.item(), 1.0)
        self.assertAlmostEqual(negative.item(), 0.0)
        good_loss.backward()
        self.assertIsNotNone(good_face.grad)
        self.assertIsNotNone(good_voice.grad)

    def test_fused_opl_validates_shapes_and_weight(self) -> None:
        labels = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "same shape"):
            fused_orthogonal_projection_loss(
                torch.randn(2, 3), torch.randn(2, 4), labels
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fused_orthogonal_projection_loss(
                torch.randn(2, 3), torch.randn(2, 3), labels, -0.1
            )

    def test_voice_views_replace_clean_voice_only_when_drawn(self) -> None:
        from types import SimpleNamespace
        from train_cross_modal import CrossModalFeatureDataset

        fold = SimpleNamespace(
            train_faces=np.zeros((4, 3), dtype=np.float32),
            train_voices=np.zeros((4, 2), dtype=np.float32),
            train_labels=np.arange(4, dtype=np.int64),
        )
        views = np.ones((4, 3, 2), dtype=np.float32)
        always_clean = CrossModalFeatureDataset(fold, np.zeros(4), views, 1.0)
        never_clean = CrossModalFeatureDataset(fold, np.zeros(4), views, 0.0)
        self.assertTrue(all(always_clean[i][1].eq(0).all() for i in range(4)))
        self.assertTrue(all(never_clean[i][1].eq(1).all() for i in range(4)))
        with self.assertRaises(ValueError):
            CrossModalFeatureDataset(fold, np.zeros(4), np.ones((3, 3, 2), dtype=np.float32))

    def test_hyperbolic_geometry_stays_in_ball_and_scores_consistently(self) -> None:
        torch.manual_seed(0)
        model = CrossModalIdentityNetwork(
            12, 10, 6, 3, face_hidden_dims=(8,), voice_hidden_dims=(8,),
            hidden_dropout=0.0, geometry="hyperbolic", curvature=1.0, clip_radius=1.0,
        ).eval()
        faces, voices = model(torch.randn(5, 12) * 10, torch.randn(5, 10) * 10)
        self.assertLess(float(faces.norm(dim=1).max()), 1.0)
        matrix = model.similarity_matrix(faces, voices)
        self.assertTrue(torch.allclose(matrix.diagonal(), model.pair_similarity(faces, voices), atol=1e-5))
        self.assertTrue(bool((matrix <= 0).all()))  # negative distances
        sphere = CrossModalIdentityNetwork(12, 10, 6, 3, face_hidden_dims=(8,), voice_hidden_dims=(8,))
        self.assertEqual(set(model.state_dict()), set(sphere.state_dict()))

    def test_identity_sampler_has_exact_p_by_k_composition(self) -> None:
        labels = np.repeat(np.arange(6), [2, 3, 4, 5, 6, 7])
        sampler = IdentityBalancedBatchSampler(labels, 4, 3, seed=7)
        batch = next(iter(sampler))
        sampled_labels = labels[batch]
        identities, counts = np.unique(sampled_labels, return_counts=True)
        self.assertEqual(len(batch), 12)
        self.assertEqual(len(identities), 4)
        np.testing.assert_array_equal(counts, np.full(4, 3))

    def test_gender_balanced_sampler_selects_equal_identity_counts(self) -> None:
        labels = np.repeat(np.arange(8), 3)
        identity_genders = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        genders = identity_genders[labels]
        sampler = IdentityBalancedBatchSampler(
            labels, 4, 2, seed=3, genders=genders, gender_balanced=True
        )
        batch = next(iter(sampler))
        selected_identities = np.unique(labels[batch])
        selected_genders = identity_genders[selected_identities]
        np.testing.assert_array_equal(np.bincount(selected_genders), [2, 2])

    def test_voice_prototypes_update_and_score(self) -> None:
        model = CrossModalIdentityNetwork(8, 4, 6, 3)
        labels = torch.tensor([0, 0, 1, 1])
        voices = torch.nn.functional.normalize(torch.randn(4, 6), dim=1)
        faces = torch.nn.functional.normalize(torch.randn(4, 6), dim=1)
        model.voice_prototypes.update(voices, labels)
        logits, targets = model.voice_prototypes.logits_and_targets(
            faces, labels, temperature=0.1
        )
        self.assertEqual(logits.shape, (4, 2))
        torch.testing.assert_close(targets, labels)

    def test_linear_standardized_projector_matches_in_eval(self) -> None:
        model = CrossModalIdentityNetwork(
            5, 3, 4, 2,
            face_hidden_dims=(),
            voice_hidden_dims=(8,),
            standardize_inputs=True,
            input_dropout=0.5,
            input_noise_std=1.0,
        )
        self.assertEqual(len(model.face_branch.network), 1)
        faces = torch.randn(6, 5) * 10 + 3
        model.face_branch.set_input_statistics(faces.mean(0), faces.std(0))
        self.assertIn("face_branch.input_mean", model.state_dict())
        model.eval()
        # Noise and input dropout are disabled at inference.
        torch.testing.assert_close(model.face_branch(faces), model.face_branch(faces))
        restored = CrossModalIdentityNetwork(
            5, 3, 4, 2, face_hidden_dims=(), voice_hidden_dims=(8,),
            standardize_inputs=True,
        )
        restored.load_state_dict(model.state_dict(), strict=True)
        restored.eval()
        torch.testing.assert_close(restored.face_branch(faces), model.face_branch(faces))

    def test_default_model_state_dict_is_unchanged(self) -> None:
        keys = set(CrossModalIdentityNetwork(16, 6, 8, 4).state_dict())
        self.assertFalse(any("input_" in key for key in keys))

    def test_same_gender_supcon_ignores_opposite_gender_candidates(self) -> None:
        labels = torch.tensor([0, 1, 2])
        genders = torch.tensor([0, 0, 1])
        face = torch.eye(3)
        voice = torch.eye(3)
        base = symmetric_same_gender_supcon_loss(face, voice, labels, genders, 0.1)
        # Making the opposite-gender identity 2 look like identity 0 must not
        # change the loss, while a same-gender impostor must increase it.
        voice_cross = voice.clone()
        voice_cross[2] = torch.nn.functional.normalize(torch.tensor([1., 0., 1.]), dim=0)
        cross = symmetric_same_gender_supcon_loss(face, voice_cross, labels, genders, 0.1)
        voice_same = voice.clone()
        voice_same[1] = torch.nn.functional.normalize(torch.tensor([1., 1., 0.]), dim=0)
        same = symmetric_same_gender_supcon_loss(face, voice_same, labels, genders, 0.1)
        self.assertTrue(torch.isfinite(base))
        self.assertGreater(same.item(), base.item() + 0.1)
        # Only identity 2's own positive changes; its row has no negatives.
        self.assertLess(abs(cross.item() - base.item()) , same.item() - base.item())


if __name__ == "__main__":
    unittest.main()
