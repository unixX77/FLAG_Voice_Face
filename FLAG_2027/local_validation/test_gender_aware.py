"""Tests for the separate Gender-Aware Dual-Head Network (GADH-V1)."""

import unittest

import torch

from gender_aware_model import GenderAwareDualHeadNetwork
from local_validation.cross_modal_losses import (
    symmetric_multi_positive_same_gender_loss,
)


class GenderAwareTests(unittest.TestCase):
    def test_dual_head_shapes_norms_and_weighted_score(self) -> None:
        model = GenderAwareDualHeadNetwork(
            face_dim=16,
            voice_dim=8,
            num_classes=4,
            global_dim=10,
            fine_dim=6,
            score_global_weight=0.25,
        )
        faces = torch.randn(12, 16)
        voices = torch.randn(12, 8)
        face, voice = model(faces, voices)
        self.assertEqual(face.global_embedding.shape, (12, 10))
        self.assertEqual(face.fine_embedding.shape, (12, 6))
        self.assertEqual(face.inference_embedding.shape, (12, 16))
        torch.testing.assert_close(face.inference_embedding.norm(dim=1), torch.ones(12))
        combined_score = face.inference_embedding @ voice.inference_embedding.T
        expected = (
            0.25 * (face.global_embedding @ voice.global_embedding.T)
            + 0.75 * (face.fine_embedding @ voice.fine_embedding.T)
        )
        torch.testing.assert_close(combined_score, expected)

    def test_single_256_reuses_global_embedding(self) -> None:
        model = GenderAwareDualHeadNetwork(
            face_dim=10,
            voice_dim=7,
            num_classes=3,
            architecture_mode="single_256",
        )
        face, voice = model(torch.randn(6, 10), torch.randn(6, 7))
        self.assertEqual(face.inference_embedding.shape, (6, 256))
        self.assertIs(face.global_embedding, face.fine_embedding)
        self.assertIs(face.global_embedding, face.inference_embedding)
        self.assertEqual(voice.inference_embedding.shape, (6, 256))

    def test_same_gender_loss_prefers_identity_aligned_clusters(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        genders = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        bases = torch.eye(4).repeat_interleave(2, dim=0)
        good_voice = bases.clone()
        bad_voice = torch.cat((bases[2:4], bases[0:2], bases[6:8], bases[4:6]))
        good_loss, good_positive, good_negative = (
            symmetric_multi_positive_same_gender_loss(
                bases, good_voice, labels, genders
            )
        )
        bad_loss, _, _ = symmetric_multi_positive_same_gender_loss(
            bases, bad_voice, labels, genders
        )
        self.assertLess(good_loss, bad_loss)
        self.assertGreater(good_positive, good_negative)

    def test_same_gender_loss_backpropagates_to_both_modalities(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        genders = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        face = torch.nn.functional.normalize(torch.randn(8, 5), dim=1).requires_grad_()
        voice = torch.nn.functional.normalize(torch.randn(8, 5), dim=1).requires_grad_()
        loss, _, _ = symmetric_multi_positive_same_gender_loss(
            face, voice, labels, genders
        )
        loss.backward()
        self.assertIsNotNone(face.grad)
        self.assertIsNotNone(voice.grad)
        self.assertTrue(torch.isfinite(face.grad).all())
        self.assertTrue(torch.isfinite(voice.grad).all())


if __name__ == "__main__":
    unittest.main()
