"""CPU smoke test without any dataset: the real model, loss and scoring on synthetic features."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FLAG_2027"))
from cross_modal_model import CrossModalIdentityNetwork  # noqa: E402
from local_validation.cross_modal_losses import paired_cosine_alignment_loss, symmetric_cross_modal_contrastive_loss  # noqa: E402
from phase2.unimodal_eval import eer  # noqa: E402


class SmokeTests(unittest.TestCase):
    def test_tiny_training_run_learns_a_synthetic_association(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        ids, per = 24, 6
        latent = rng.normal(size=(ids, 8))
        lab = np.repeat(np.arange(ids), per)
        # faces 512-d and voices 1,048-d share an identity latent through fixed random maps (+ noise)
        faces = latent[lab] @ rng.normal(size=(8, 512)) + 0.5 * rng.normal(size=(len(lab), 512))
        voices = latent[lab] @ rng.normal(size=(8, 1048)) + 0.5 * rng.normal(size=(len(lab), 1048))
        F, V, L = (torch.tensor(x, dtype=torch.float32) for x in (faces, voices, lab))
        L = L.long()
        tr, te = lab < 16, lab >= 16  # identity-disjoint split
        m = CrossModalIdentityNetwork(512, 1048, 128, 16, face_hidden_dims=(256,), voice_hidden_dims=(256,),
                                      hidden_dropout=0.3, standardize_inputs=True, input_dropout=0.2, input_noise_std=0.1)
        m.face_branch.set_input_statistics(F[tr].mean(0), F[tr].std(0))
        m.voice_branch.set_input_statistics(V[tr].mean(0), V[tr].std(0))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for _ in range(60):
            m.train()
            fe, ve = m(F[tr], V[tr])
            loss = symmetric_cross_modal_contrastive_loss(fe, ve, L[tr], 0.07, "supcon")[0] + 0.25 * paired_cosine_alignment_loss(fe, ve)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])
        m.eval()
        with torch.no_grad():
            fe, ve = m(F[te], V[te])
        s = (fe @ ve.T).numpy()  # all face x voice pairs of unseen identities
        y = (lab[te][:, None] == lab[te][None, :]).astype(int)
        self.assertLess(eer(y.ravel(), s.ravel()), 0.35)  # far better than chance (0.5)

    def test_distance_is_one_minus_cosine_and_unit_norm(self):
        m = CrossModalIdentityNetwork(512, 1048, 128, 3, face_hidden_dims=(256,), voice_hidden_dims=(256,)).eval()
        with torch.no_grad():
            f, v = m(torch.randn(4, 512), torch.randn(4, 1048))
        torch.testing.assert_close(f.norm(dim=1), torch.ones(4))
        d = 1 - m.pair_similarity(f, v)
        self.assertTrue(bool(((d >= 0) & (d <= 2)).all()))


if __name__ == "__main__":
    unittest.main()
