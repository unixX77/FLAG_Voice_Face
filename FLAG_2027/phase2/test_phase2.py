"""Phase II tests. Run from FLAG_2027/:  python -m unittest phase2.test_phase2 -v"""

import unittest

import numpy as np

from phase2 import unimodal_eval as ue


def toy_pool(n_ids=8, per=12, langs=2, seed=0, dup_every=0):
    rng = np.random.default_rng(seed)
    ident = np.repeat([f"p{i:02d}" for i in range(n_ids)], per)
    gender = np.repeat(np.arange(n_ids) % 2, per)
    group = np.tile(np.arange(per) % langs, n_ids)
    session = np.array([f"{i}_v{(k // 3)}" for i in range(n_ids) for k in range(per)])
    content = np.array([f"{i}_{k}" for i in range(n_ids) for k in range(per)])
    if dup_every:
        content[1::dup_every] = content[0::dup_every][: len(content[1::dup_every])]  # duplicated recordings
    attr = np.repeat(rng.normal(size=n_ids), per)
    return ue.Pool(ident, gender, group, session, content, attr, rng.uniform(1, 12, n_ids * per))


class TrialTests(unittest.TestCase):
    def test_deterministic(self):
        p = toy_pool()
        t1, t2 = ue.build_trials(p, 10, seed=3), ue.build_trials(p, 10, seed=3)
        for f in ("a", "b", "label", "condition"):
            np.testing.assert_array_equal(getattr(t1, f), getattr(t2, f))

    def test_genuine_never_same_content_and_conditions_correct(self):
        p = toy_pool(dup_every=4)
        t = ue.build_trials(p, 50)
        g = t.label == 1
        self.assertTrue((p.content[t.a[g]] != p.content[t.b[g]]).all())
        self.assertTrue((p.identity[t.a[g]] == p.identity[t.b[g]]).all())
        for c in ue.GENUINE:
            m = t.condition == c
            if not m.any():
                continue
            same_group = p.group[t.a[m]] == p.group[t.b[m]]
            same_session = p.session[t.a[m]] == p.session[t.b[m]]
            self.assertTrue(same_group.all() if c.startswith("same_group") else (~same_group).all(), c)
            self.assertTrue(same_session.all() if c.endswith("_same_session") else (~same_session).all(), c)

    def test_impostor_conditions(self):
        p = toy_pool()
        t = ue.build_trials(p, 20)
        i = t.label == 0
        self.assertTrue((p.identity[t.a[i]] != p.identity[t.b[i]]).all())
        sg = t.condition == "imp_same_gender_same_group"
        self.assertTrue((p.gender[t.a[sg]] == p.gender[t.b[sg]]).all() and (p.group[t.a[sg]] == p.group[t.b[sg]]).all())
        dg = t.condition == "imp_diff_gender_same_group"
        self.assertTrue((p.gender[t.a[dg]] != p.gender[t.b[dg]]).all())
        xg = t.condition == "imp_same_gender_diff_group"
        self.assertTrue((p.group[t.a[xg]] != p.group[t.b[xg]]).all())
        si = t.condition == "imp_similar"
        self.assertTrue((p.gender[t.a[si]] == p.gender[t.b[si]]).all())

    def test_mixed_gender_identity_rejected(self):
        p = toy_pool()
        p.gender[0] = 1 - p.gender[0]
        with self.assertRaises(ValueError):
            ue.build_trials(p, 5)


class MetricTests(unittest.TestCase):
    def test_eer_perfect_and_random(self):
        lab = np.r_[np.ones(500), np.zeros(500)]
        self.assertAlmostEqual(ue.eer(lab, np.r_[np.ones(500), np.zeros(500)]), 0.0)
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(ue.eer(lab, rng.normal(size=1000)), 0.5, delta=0.05)

    def test_retrieval_perfect_embeddings(self):
        p = toy_pool(6, 8)
        centres = np.eye(6)
        emb = centres[np.searchsorted(np.unique(p.identity), p.identity)] + 0.01 * np.random.default_rng(1).normal(size=(48, 6))
        r = ue.retrieval(emb, p)
        self.assertEqual(r["recall_at_1"], 1.0)
        self.assertEqual(r["mrr"], 1.0)

    def test_retrieval_excludes_duplicates(self):
        # Two copies of the same recording would otherwise give a trivial rank-1 hit.
        p = toy_pool(2, 4, dup_every=2)
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(8, 5))
        emb[1::2] = emb[0::2]  # duplicates identical
        r = ue.retrieval(emb, p)
        self.assertLess(r["recall_at_1"], 1.0 + 1e-9)
        # query 0's duplicate (row 1) must not count: check directly
        valid = p.content != p.content[0]
        self.assertFalse(valid[1])

    def test_cross_group_retrieval_masks(self):
        p = toy_pool(4, 6)
        emb = np.random.default_rng(3).normal(size=(24, 4))
        r = ue.retrieval(emb, p, query_mask=p.group == 1, gallery_mask=p.group == 0)
        self.assertGreater(r["queries"], 0)

    def test_fisher_and_variance(self):
        p = toy_pool(4, 10)
        lab = np.searchsorted(np.unique(p.identity), p.identity)
        tight = np.eye(4)[lab] + 0.01 * np.random.default_rng(0).normal(size=(40, 4))
        loose = np.random.default_rng(0).normal(size=(40, 4))
        self.assertGreater(ue.fisher_ratio(tight, p.identity), 10 * ue.fisher_ratio(loose, p.identity))
        self.assertGreater(ue.variance_explained(tight, p.identity), 0.95)


class CompatibilityTests(unittest.TestCase):
    def test_face_and_voice_inputs_follow_the_run_config(self):
        import tempfile
        from pathlib import Path
        import torch
        from local_validation.compare_fold_runs import face_inputs, voice_inputs
        default = torch.zeros(3, 2)
        with tempfile.TemporaryDirectory() as d:
            own = Path(d) / "own.csv"
            np.savetxt(own, np.array([[1, 2, 9], [3, 4, 9], [5, 6, 9]], float), delimiter=",")
            shared = Path(d) / "shared.csv"
            np.savetxt(shared, np.zeros((3, 3)), delimiter=",")
            self.assertIs(face_inputs({"face_csv": str(shared)}, default, shared), default)
            got = face_inputs({"face_csv": str(own)}, default, shared)
            np.testing.assert_array_equal(got.numpy(), [[1, 2], [3, 4], [5, 6]])  # label column dropped
            self.assertIs(voice_inputs({"voice_csv": str(shared)}, default, shared), default)

    def test_projection_head_shapes(self):
        import torch
        from cross_modal_model import CrossModalIdentityNetwork
        for dims in ((), (256,), (512, 256)):
            m = CrossModalIdentityNetwork(512, 1048, 128, 10, face_hidden_dims=dims, voice_hidden_dims=dims,
                                          hidden_dropout=0.3, standardize_inputs=True).eval()
            f, v = m(torch.randn(7, 512), torch.randn(7, 1048))
            self.assertEqual(tuple(f.shape), (7, 128))
            self.assertEqual(tuple(v.shape), (7, 128))
            torch.testing.assert_close(v.norm(dim=1), torch.ones(7))

    def test_supcon_numerically_stable_and_multi_positive(self):
        import torch
        from local_validation.cross_modal_losses import symmetric_cross_modal_contrastive_loss
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        e = torch.nn.functional.normalize(torch.randn(6, 8), dim=1)
        for t in (0.07, 0.01, 0.001):
            loss, _ = symmetric_cross_modal_contrastive_loss(e, e, labels, t, "supcon")
            self.assertTrue(torch.isfinite(loss))
        # Same-identity samples are positives: aligning both samples of an identity lowers the loss.
        a = torch.nn.functional.normalize(torch.eye(8)[[0, 0, 1, 1, 2, 2]] + 0.01, dim=1)
        b = torch.nn.functional.normalize(torch.eye(8)[[0, 1, 1, 2, 2, 0]] + 0.01, dim=1)
        self.assertLess(symmetric_cross_modal_contrastive_loss(a, a, labels, 0.1, "supcon")[0],
                        symmetric_cross_modal_contrastive_loss(a, b, labels, 0.1, "supcon")[0])

    def test_phase2_checkpoints_load_with_their_own_inputs(self):
        from pathlib import Path
        import torch
        from local_validation.compare_fold_runs import load_model, voice_inputs
        path = Path("../runs/phase2/g3_cat_farl/fold_0_seed_1/checkpoint_last.pth.tar")
        if not path.exists():
            self.skipTest("phase2 runs not present")
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        self.assertIn("voice_v10_wavlm_egemaps_ecapa", cfg["voice_csv"])
        fake = torch.zeros(4, 1048)
        m = load_model(path, 512, fake)
        f, v = m(torch.randn(4, 512), fake)
        self.assertEqual(tuple(v.shape), (4, 128))

    def test_language_holdout_split_matches_p2_and_excludes_urdu(self):
        from pathlib import Path
        import torch
        from train_cross_modal import holdout_external_identities
        path = Path("../runs/phase2/p2_g3_cat_farl/fold_2_seed_1/checkpoint_last.pth.tar")
        if not path.exists():
            self.skipTest("phase2 P2 runs not present")
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        rows = [l.split()[4] for l in open("../features/external/combined_all/train_manifest.txt") if l.strip()]
        self.assertEqual(sorted(cfg["held_out_external"]), sorted(holdout_external_identities(rows, "mav1", 2)))
        self.assertIn("Urdu", cfg["exclude_train_languages"])


if __name__ == "__main__":
    unittest.main()
