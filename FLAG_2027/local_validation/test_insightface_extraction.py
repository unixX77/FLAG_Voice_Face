"""Unit tests for the row-safe InsightFace feature extractor."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_extraction.extract_face_features_insightface import (
    normalized_embedding,
    read_manifest,
    select_primary_face,
)


class InsightFaceExtractionTests(unittest.TestCase):
    def test_reads_train_and_evaluation_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.txt"
            train.write_text("pair 1 voice.wav face.jpg id001 7\n")
            rows, is_training = read_manifest(train)
            self.assertTrue(is_training)
            self.assertEqual(rows[0].face_path, "face.jpg")
            self.assertEqual(rows[0].class_index, 7)

            evaluation = root / "eval.txt"
            evaluation.write_text("pair voice.wav face.jpg\n")
            rows, is_training = read_manifest(evaluation)
            self.assertFalse(is_training)
            self.assertEqual(rows[0].voice_path, "voice.wav")

    def test_primary_face_prefers_largest_then_score(self) -> None:
        faces = [
            {"bbox": np.array([20, 20, 80, 80]), "det_score": 0.99},
            {"bbox": np.array([10, 10, 90, 90]), "det_score": 0.80},
        ]
        self.assertIs(select_primary_face(faces, (100, 100, 3)), faces[1])

        tied = [
            {"bbox": np.array([10, 10, 90, 90]), "det_score": 0.80},
            {"bbox": np.array([10, 10, 90, 90]), "det_score": 0.95},
        ]
        self.assertIs(select_primary_face(tied, (100, 100, 3)), tied[1])

    def test_embedding_is_normalized_and_dimension_checked(self) -> None:
        face = {
            "normed_embedding": None,
            "embedding": np.full(512, 2.0, dtype=np.float32),
        }
        embedding = normalized_embedding(face)
        self.assertEqual(embedding.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "512-D"):
            normalized_embedding(
                {"normed_embedding": np.ones(4), "embedding": np.ones(4)}
            )


if __name__ == "__main__":
    unittest.main()
