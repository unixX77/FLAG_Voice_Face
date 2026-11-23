"""Unit tests for the local verification metric implementation."""

import unittest

import numpy as np

from local_validation.metrics import compute_eer


class MetricTests(unittest.TestCase):
    def test_perfect_scores(self) -> None:
        eer, auc = compute_eer(np.array([0, 0, 1, 1]), np.array([0.0, 0.1, 0.9, 1.0]))
        self.assertAlmostEqual(eer, 0.0)
        self.assertAlmostEqual(auc, 1.0)

    def test_reversed_scores(self) -> None:
        eer, auc = compute_eer(np.array([0, 0, 1, 1]), np.array([1.0, 0.9, 0.1, 0.0]))
        self.assertAlmostEqual(eer, 1.0)
        self.assertAlmostEqual(auc, 0.0)

    def test_random_scores_are_near_chance(self) -> None:
        rng = np.random.default_rng(2027)
        labels = np.tile(np.array([0, 1]), 5000)
        scores = rng.random(len(labels))
        eer, auc = compute_eer(labels, scores)
        self.assertLess(abs(eer - 0.5), 0.03)
        self.assertLess(abs(auc - 0.5), 0.03)

    def test_nonfinite_scores_fail(self) -> None:
        with self.assertRaises(ValueError):
            compute_eer(np.array([0, 1]), np.array([0.0, np.nan]))


if __name__ == "__main__":
    unittest.main()
