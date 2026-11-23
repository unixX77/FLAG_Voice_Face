"""Small unit tests for fold data, loss, and protocol scoring."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from local_validation.fold_dataset import FaceVoiceFeatureDataset, load_feature_matrices
from local_validation.losses import OrthogonalProjectionLoss
from local_validation.model_evaluation import Protocol, load_protocol, score_protocol


class FoldPipelineTests(unittest.TestCase):
    def test_feature_loader_accepts_dynamic_modality_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            faces = np.column_stack((np.arange(15).reshape(3, 5), [2, 1, 0]))
            voices = np.column_stack((np.arange(12).reshape(3, 4), [2, 1, 0]))
            np.savetxt(root / "faces.csv", faces, delimiter=",")
            np.savetxt(root / "voices.csv", voices, delimiter=",")
            loaded_faces, loaded_voices, labels = load_feature_matrices(
                root / "faces.csv", root / "voices.csv"
            )
            self.assertEqual(loaded_faces.shape, (3, 5))
            self.assertEqual(loaded_voices.shape, (3, 4))
            np.testing.assert_array_equal(labels, [2, 1, 0])

    def test_feature_dataset_keeps_rows_aligned(self) -> None:
        faces = np.arange(12, dtype=np.float32).reshape(3, 4)
        voices = np.arange(6, dtype=np.float32).reshape(3, 2)
        labels = np.asarray([2, 1, 0])
        dataset = FaceVoiceFeatureDataset(faces, voices, labels)
        face, voice, label = dataset[1]
        np.testing.assert_array_equal(face.numpy(), faces[1])
        np.testing.assert_array_equal(voice.numpy(), voices[1])
        self.assertEqual(label.item(), labels[1])

    def test_opl_is_finite_without_positive_pair(self) -> None:
        features = torch.randn(4, 8, requires_grad=True)
        labels = torch.arange(4)
        loss, positive, negative = OrthogonalProjectionLoss()(features, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(positive.item(), 0.0)
        self.assertGreaterEqual(negative.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(features.grad)

    def test_protocol_join_and_higher_is_better_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pairs.csv").write_text(
                "pair_id,face_index,voice_index\np1,10,10\np2,10,20\n"
            )
            # Reverse order verifies that labels are joined by pair_id.
            (root / "truth.csv").write_text("pair_id,label\np2,0\np1,1\n")
            protocol = load_protocol(root / "pairs.csv", root / "truth.csv")
            faces = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            voices = faces.copy()
            scores = score_protocol(protocol, faces, voices, {10: 0, 20: 1})
            self.assertGreater(scores[0], scores[1])
            np.testing.assert_array_equal(protocol.labels, [1, 0])


if __name__ == "__main__":
    unittest.main()
