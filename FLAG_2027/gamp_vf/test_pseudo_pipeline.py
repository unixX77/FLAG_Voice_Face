"""Tests for Stage E/F pseudo-labelling and the training-side leakage guards. Run from FLAG_2027/:
    python -m unittest gamp_vf.test_pseudo_pipeline -v
Synthetic data checks correctness only; it is not evidence of real performance.
"""

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gamp_vf import pseudo_labels as pl
from gamp_vf.mine_pseudo_labels import Pool, assignments, evaluate, tune_on_calibration
from local_validation.fold_dataset import FoldData
from train_cross_modal import append_external


def synthetic_pool(n_ids, per, seed, dim=32, voice_noise=0.35, face_noise=0.2):
    """Shared-space identities: voices and faces of one person sit near the same centre.
    Voice and face rows are *independent* draws (no row alignment)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_ids, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    truth = np.repeat(np.arange(n_ids), per)
    ve = centres[truth] + voice_noise / np.sqrt(dim) * rng.normal(size=(len(truth), dim))
    fe = centres[truth] + face_noise / np.sqrt(dim) * rng.normal(size=(len(truth), dim))
    gender = (np.arange(n_ids) % 2)[truth]
    gp = np.where(gender == 1, 0.9, 0.1).astype(float)
    langs = np.tile([0, 1], len(truth) // 2)
    return Pool(np.arange(len(truth)), fe, ve, gp, gp, fe, langs, 0.5, 0.5, "projected"), truth, gender


class PseudoLabelTests(unittest.TestCase):
    def test_end_to_end_recovers_constructed_assignment(self):
        calibration, truth_c, _ = synthetic_pool(20, 12, seed=1)
        hidden, truth_h, gender_h = synthetic_pool(30, 12, seed=2)
        params = tune_on_calibration(calibration, truth_c)
        decisions, cand, conf, decision = assignments(hidden, params)
        for mode in ("local", "hungarian", "local_plus_hungarian"):
            r = evaluate(hidden, decisions[mode], truth_h, gender_h, cap=20)
            self.assertGreaterEqual(r["pair_precision"], 0.9, mode)
            self.assertGreaterEqual(r["identity_recall"], 0.7, mode)
            self.assertGreaterEqual(r["row_precision"], 0.9, mode)
        # One-to-one on faces in every mode.
        for mode, m in decisions.items():
            if mode != "pairwise_top1":
                self.assertEqual(len(np.unique(m[m >= 0])), (m >= 0).sum(), mode)

    def test_unreliable_pool_is_not_forced(self):
        # Voices carry no identity information: nothing should pass the precision threshold.
        calibration, truth_c, _ = synthetic_pool(20, 12, seed=3, voice_noise=30.0)
        params = tune_on_calibration(calibration, truth_c)
        hidden, truth_h, gender_h = synthetic_pool(30, 12, seed=4, voice_noise=30.0)
        decisions = assignments(hidden, params)[0]
        r = evaluate(hidden, decisions["local"], truth_h, gender_h, cap=20)
        self.assertLessEqual(r["accepted_pairs"], 3)

    def test_precision_threshold_and_decide(self):
        conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([1, 1, 1, 0, 0], bool)
        self.assertEqual(pl.precision_threshold(conf, correct, 0.9), 0.7)
        self.assertEqual(pl.precision_threshold(conf, np.zeros(5, bool), 0.9), float("inf"))
        np.testing.assert_array_equal(pl.decide(conf, 0.75, 0.55), ["high", "high", "ambiguous", "ambiguous", "unreliable"])

    def test_gender_compatibility_keeps_unknowns(self):
        v = pl.Groups(np.arange(3), np.eye(3), [], [], np.array([0.95, 0.05, np.nan]))
        f = pl.Groups(np.arange(2), np.eye(3)[:2], [], [], np.array([0.9, 0.55]))
        np.testing.assert_array_equal(pl.gender_compatible(v, f), [[True, True], [False, True], [True, True]])

    def test_conflict_groups_union_close_prototypes(self):
        e = np.eye(4)
        voice = np.vstack([e[0], e[0] * 0.99 + e[1] * 0.14, e[2]])
        voice /= np.linalg.norm(voice, axis=1, keepdims=True)
        face = np.vstack([e[0], e[1], e[2]])
        g = pl.conflict_groups(voice, face, voice_sim=0.95, face_sim=0.95)
        self.assertEqual(g[0], g[1])
        self.assertNotEqual(g[0], g[2])


class LeakageGuardTests(unittest.TestCase):
    """append_external: hidden identities are removed; pseudo rows must come from the hidden pool."""

    def _fixture(self, tmp: Path):
        rows, faces, voices = [], [], []
        names = ["ext_a", "ext_b", "ext_c", "ext_d"]
        for k, n in enumerate(names):
            for i in range(3):
                rows.append(f"key{k}{i} 1 voices/English/{n}/{i}.wav faces/English/{n}/{i}.jpg {n} {100 + k}")
                faces.append([k, i, 100 + k])
                voices.append([k, -i, 100 + k])
        (tmp / "manifest.txt").write_text("\n".join(rows) + "\n")
        np.savetxt(tmp / "faces.csv", np.array(faces, float), delimiter=",")
        np.savetxt(tmp / "voices.csv", np.array(voices, float), delimiter=",")
        (tmp / "gender.csv").write_text("".join(f"{n},{'m' if i % 2 else 'f'}\n" for i, n in enumerate(names)))
        (tmp / "hidden.txt").write_text("ext_c\next_d\n")
        fold = FoldData(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32), np.array([0, 1]),
                        np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32), np.array([5]),
                        ["off_1", "off_2"], ["off_3"])
        args = argparse.Namespace(extra_face_csv=tmp / "faces.csv", extra_voice_csv=tmp / "voices.csv",
                                  extra_manifest=tmp / "manifest.txt", extra_gender_metadata=tmp / "gender.csv",
                                  holdout_external_prefix=None, fold=0, exclude_train_languages=[],
                                  exclude_external_identities=tmp / "hidden.txt", pseudo_pairs_csv=None)
        return fold, args

    def _write_pseudo(self, path, pairs):
        with path.open("w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["face_row", "voice_row", "pseudo_id", "weight", "gender", "conflict_group"])
            for f, v in pairs:
                w.writerow([f, v, "p0", 0.8, -1, "g0"])

    def test_hidden_identities_removed_and_pseudo_rows_added(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fold, args = self._fixture(tmp)
            self._write_pseudo(tmp / "pseudo.csv", [(6, 9), (7, 10)])  # rows of ext_c / ext_d
            args.pseudo_pairs_csv = tmp / "pseudo.csv"
            genders, primary, info = append_external(fold, np.zeros(2, np.int64), args)
            self.assertNotIn("ext_c", fold.train_identities)
            self.assertNotIn("ext_d", fold.train_identities)
            self.assertIn("pseudo:p0", fold.train_identities)
            self.assertEqual(info["pseudo_rows"], 2)
            self.assertEqual(genders[-1], -1)
            # pseudo face/voice features come from the referenced rows, not row-aligned pairs
            np.testing.assert_array_equal(fold.train_faces[-2:], [[2, 0], [2, 1]])
            np.testing.assert_array_equal(fold.train_voices[-2:], [[3, 0], [3, -1]])

    def test_empty_pseudo_file_is_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fold, args = self._fixture(tmp)
            self._write_pseudo(tmp / "pseudo.csv", [])
            args.pseudo_pairs_csv = tmp / "pseudo.csv"
            genders, _, info = append_external(fold, np.zeros(2, np.int64), args)
            self.assertEqual(info["pseudo_rows"], 0)
            self.assertEqual(len(fold.train_labels), len(fold.train_faces))

    def test_pseudo_rows_outside_hidden_pool_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fold, args = self._fixture(tmp)
            self._write_pseudo(tmp / "pseudo.csv", [(0, 9)])  # face row of a labelled identity
            args.pseudo_pairs_csv = tmp / "pseudo.csv"
            with self.assertRaises(ValueError):
                append_external(fold, np.zeros(2, np.int64), args)

    def test_external_overlapping_official_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fold, args = self._fixture(tmp)
            fold.validation_identities = ["ext_a"]
            with self.assertRaises(ValueError):
                append_external(fold, np.zeros(2, np.int64), args)


if __name__ == "__main__":
    unittest.main()
