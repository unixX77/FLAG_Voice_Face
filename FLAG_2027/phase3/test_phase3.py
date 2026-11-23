"""Phase III tests. Run from FLAG_2027/:  python -m unittest phase3.test_phase3 -v"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from cross_modal_model import CrossModalIdentityNetwork, VoiceBlockFusion
from local_validation.fold_dataset import FoldData
from train_cross_modal import shuffle_voice_block


def net(mode, **kw):
    return CrossModalIdentityNetwork(512, 1048, 128, 10, face_hidden_dims=(256,), voice_hidden_dims=(256,),
                                     hidden_dropout=0.3, standardize_inputs=True, input_dropout=0.2,
                                     input_noise_std=0.1, voice_fusion=mode, voice_fusion_split=856, **kw)


class FusionTests(unittest.TestCase):
    def test_shapes_and_unit_norm_all_modes(self):
        for mode in ("concat", "bottleneck", "block_l2", "residual"):
            m = net(mode).eval()
            f, v = m(torch.randn(5, 512), torch.randn(5, 1048))
            self.assertEqual(tuple(v.shape), (5, 128), mode)
            torch.testing.assert_close(v.norm(dim=1), torch.ones(5))

    def test_concat_default_is_plain_projector(self):
        from cross_modal_model import Projector
        self.assertIsInstance(net("concat").voice_branch, Projector)

    def test_residual_starts_at_base_only(self):
        m = net("residual").eval()
        b = m.voice_branch
        x = torch.randn(6, 1048)
        z = (x - b.input_mean) / b.input_std
        expected = F.normalize(b.projector(z[:, :856]), dim=1)
        torch.testing.assert_close(b(x), expected)
        # ECAPA block has no effect at initialisation (alpha = 0) ...
        x2 = x.clone(); x2[:, 856:] = torch.randn(6, 192)
        torch.testing.assert_close(b(x), b(x2))
        # ... but alpha receives gradient, so the adapter can switch on.
        b.train(); b(x).sum().backward()
        self.assertIsNotNone(b.alpha.grad)

    def test_block_l2_equal_block_weight(self):
        b = net("block_l2").voice_branch.eval()
        captured = {}
        def hook(mod, inp, out):  # must return None, or the hook would replace the module output
            captured["x"] = inp[0]
        b.projector.network[0].register_forward_hook(hook)
        b(torch.randn(4, 1048) * 50)
        x = captured["x"]
        torch.testing.assert_close(x[:, :856].norm(dim=1), x[:, 856:].norm(dim=1))
        torch.testing.assert_close(x.norm(dim=1), torch.full((4,), 1048 ** 0.5), rtol=1e-4, atol=1e-3)

    def test_bottleneck_dimension(self):
        b = net("bottleneck", voice_fusion_dim=32).voice_branch
        self.assertEqual(b.extra_projection.out_features, 32)
        self.assertEqual(b.projector.network[0].in_features, 856 + 32)

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            VoiceBlockFusion(100, 0, "residual", 8, (16,), 8, (0.1,))


class ShuffleControlTests(unittest.TestCase):
    def test_shuffle_breaks_speaker_alignment_on_training_rows_only(self):
        rng = np.random.default_rng(0)
        names = ["id01", "id02", "id03", "id04", "mav1_a", "mav1_b", "mav1_c"]
        labels = np.repeat(np.arange(7), 4)
        genders = np.repeat([0, 0, 1, 1, 0, 0, 0], 4)
        voices = np.zeros((28, 6), np.float32)
        voices[:, :3] = rng.normal(size=(28, 3))
        voices[:, 3:] = labels[:, None]  # extra block = speaker id, to track donors
        val = rng.normal(size=(5, 6)).astype(np.float32)
        fd = FoldData(np.zeros((28, 2), np.float32), voices.copy(), labels, np.zeros((5, 2), np.float32), val.copy(),
                      np.arange(5), names, ["v1"])
        perm = shuffle_voice_block(fd, genders, 3, seed=1)
        np.testing.assert_array_equal(fd.validation_voices, val)          # validation untouched
        np.testing.assert_array_equal(fd.train_voices[:, :3], voices[:, :3])  # base block untouched
        donors = fd.train_voices[:, 3].astype(int)
        for u in range(7):
            d = set(donors[labels == u])
            self.assertEqual(d, {perm[u]})
            self.assertNotEqual(perm[u], u)
            self.assertEqual(genders[labels == perm[u]][0], genders[labels == u][0])  # same gender
            src = lambda n: n.split("_")[0] if "_" in n else "official"
            self.assertEqual(src(names[perm[u]]), src(names[u]))              # same source


class PipelineTests(unittest.TestCase):
    def test_concatenated_voice_cache_ordering(self):
        """voice_v10 row = [voice_v8 (WavLM 768 | eGeMAPS 88) | ECAPA 192 | label]; voice_v11 = [v8 | random | label]."""
        import itertools
        from pathlib import Path
        root = Path("..")
        paths = {n: root / p for n, p in (("v8", "features/voice_v8_wavlm_egemaps/train_English_voices.csv"),
                                         ("ecapa", "features/voice_v6_ecapa/train_English_voices.csv"),
                                         ("v10", "features/voice_v10_wavlm_egemaps_ecapa/train_English_voices.csv"),
                                         ("v11", "features/voice_v11_wavlm_egemaps_random192/train_English_voices.csv"))}
        if not all(p.exists() for p in paths.values()):
            self.skipTest("feature caches not present")
        rows = {n: [np.array(l.split(","), float) for l in itertools.islice(open(p), 5)] for n, p in paths.items()}
        for i in range(5):
            v8, e, v10, v11 = rows["v8"][i], rows["ecapa"][i], rows["v10"][i], rows["v11"][i]
            np.testing.assert_allclose(v10[:856], v8[:856], rtol=1e-6)
            np.testing.assert_allclose(v10[856:1048], e[:192], rtol=1e-6)
            self.assertEqual(v10[-1], v8[-1])
            self.assertEqual(len(v10), 1049)
            np.testing.assert_allclose(v11[:856], v8[:856], rtol=1e-6)
            self.assertEqual(v11[-1], v8[-1])

    def test_seed_ensemble_of_identical_seeds_equals_single(self):
        import json, subprocess, sys, tempfile
        from pathlib import Path
        rng = np.random.default_rng(0)
        arrays = {}
        for f in range(5):
            for t in ("standard", "gender"):
                n = 120
                lab = np.tile([1, 0, 0], n // 3)
                ident = np.repeat(np.arange(6), n // 6)
                lang = np.where(np.arange(n) % 2, "English", "Urdu")
                s = lab * 0.5 + rng.normal(size=n)
                for seed in (1, 2, 3):
                    arrays[f"m__f{f}_s{seed}_{t}"] = s  # identical seeds
                arrays[f"meta__f{f}_{t}_identity"], arrays[f"meta__f{f}_{t}_language"] = ident, lang
                arrays[f"meta__f{f}_{t}_label"] = lab
        with tempfile.TemporaryDirectory() as d:
            npz, out = Path(d) / "s.npz", Path(d) / "o.json"
            np.savez(npz, **arrays)
            subprocess.run([sys.executable, "-m", "phase3.ensemble_transfer", "--scores", str(npz), "--systems", "m",
                            "--bootstrap", "20", "--output-json", str(out)], check=True, capture_output=True)
            r = json.loads(out.read_text())["point"]
        for k in r["m|single"]:  # z-scoring is monotone within a fold, so identical seeds give identical EERs
            self.assertAlmostEqual(r["m|single"][k], r["m|x3"][k], places=9)
            self.assertAlmostEqual(r["m|single"][k], r["m|x2"][k], places=9)

    def test_generator_ensemble_is_mean_of_member_distances(self):
        from generate_cross_modal_submission import score
        torch.manual_seed(0)
        members = [CrossModalIdentityNetwork(8, 6, 4, 3, face_hidden_dims=(), voice_hidden_dims=()).eval() for _ in range(3)]
        faces, voices = np.random.default_rng(1).normal(size=(10, 8)).astype(np.float32), \
            np.random.default_rng(2).normal(size=(10, 6)).astype(np.float32)
        singles = [score(m, faces, voices, torch.device("cpu"), 4) for m in members]
        for s in singles:
            self.assertTrue(((s >= 0) & (s <= 2)).all())  # 1 - cosine
        # each trial depends only on its own face/voice row: scoring a subset gives the same values
        np.testing.assert_allclose(score(members[0], faces[:3], voices[:3], torch.device("cpu"), 2), singles[0][:3])
        ensemble = np.mean(singles, axis=0)
        np.testing.assert_allclose(ensemble, np.mean([score(m, faces, voices, torch.device("cpu"), 7) for m in members], 0))

    def test_reserved_expert_names_rejected(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, "phase2/language_transfer_fusion.py", "--b0", "x{fold}{seed}",
                            "--experts", "b0=y", "--output-json", "/dev/null"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("reserved", r.stderr)


if __name__ == "__main__":
    unittest.main()
