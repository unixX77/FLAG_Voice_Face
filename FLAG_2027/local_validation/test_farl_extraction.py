"""Unit tests for the row-safe FaRL feature extractor."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_extraction.extract_face_features_farl import read_manifest, write_outputs


class FaRLExtractionTests(unittest.TestCase):
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

    def test_training_output_preserves_rows_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.txt"
            manifest.write_text(
                "p1 1 v1.wav f1.jpg id1 4\n"
                "p2 1 v2.wav f2.jpg id2 9\n"
            )
            rows, is_training = read_manifest(manifest)
            vectors = {
                "a": np.ones(512, dtype=np.float32),
                "b": np.full(512, 2, dtype=np.float32),
            }
            matrix = write_outputs(
                rows,
                ["a", "b"],
                vectors,
                root / "features.csv",
                root / "rowmap.csv",
                is_training,
            )
            written = np.loadtxt(root / "features.csv", delimiter=",")
            self.assertEqual(matrix.shape, (2, 512))
            self.assertEqual(written.shape, (2, 513))
            np.testing.assert_array_equal(written[:, -1], [4, 9])


if __name__ == "__main__":
    unittest.main()
