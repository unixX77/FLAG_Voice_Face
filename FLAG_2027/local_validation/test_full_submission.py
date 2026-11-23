"""Unit tests for full-data V1 submission helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_cross_modal_submission import read_ids, validate_score_file
from train_cross_modal_full import contiguous_labels


class FullSubmissionTests(unittest.TestCase):
    def test_contiguous_labels(self) -> None:
        labels, classes = contiguous_labels(np.asarray([8, 3, 8, 12, 3]))
        np.testing.assert_array_equal(classes, [3, 8, 12])
        np.testing.assert_array_equal(labels, [1, 0, 1, 2, 0])

    def test_score_file_requires_protocol_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "test.txt"
            protocol.write_text("a voice.wav face.jpg\nb voice2.wav face2.jpg\n")
            ids = read_ids(protocol)
            score_file = root / "scores.txt"
            score_file.write_text("a 0.1\nb 0.8\n")
            validate_score_file(score_file, ids)
            score_file.write_text("b 0.8\na 0.1\n")
            with self.assertRaisesRegex(ValueError, "order"):
                validate_score_file(score_file, ids)


if __name__ == "__main__":
    unittest.main()
