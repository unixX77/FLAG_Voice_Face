"""Phase 3 diagnostic: how much face-related trait information each representation carries.

Person-level ridge probes (identity-disjoint GroupKFold, 5 folds over all 301 people):
predict the *face* traits (median InsightFace age, median FaRL zero-shot build) from each voice
representation, and from each face representation. Reported: Pearson r between prediction and
target over held-out people, overall and within gender (gender-only signals can't explain it).
Clip-level inputs are averaged per person after the backend (z-score with training-fold stats).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2.audio_diagnostics import EXT, read_csv_matrix  # noqa: E402


def main() -> None:
    off = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    ext = [l.split() for l in open(EXT / "train_manifest.txt") if l.strip()]
    ident = np.array([r[4] for r in off] + [r[4] for r in ext])
    g = dict(l.strip().split(",") for l in open("meta_file_train_set.csv"))
    g.update(dict(l.strip().split(",") for l in open(EXT / "gender.csv")))
    tr = pd.concat([pd.read_csv("features/traits/train_English_traits.csv"),
                    pd.read_csv("features/traits/combined_all_traits.csv")])
    tr = tr.set_index("key").loc[[r[0] for r in off] + [r[0] for r in ext]]
    reps = {
        "voice: WavLM+eGeMAPS (B0)": ("features/voice_v8_wavlm_egemaps/train_English_voices.csv", EXT / "train_voices_egemaps.csv", None),
        "voice: WavLM all-layer mean": ("features/voice_v4_wavlm_allmean/train_English_voices.csv", EXT / "train_voices.csv", None),
        "voice: eGeMAPS": ("features/voice_v8_wavlm_egemaps/train_English_voices.csv", EXT / "train_voices_egemaps.csv", slice(768, None)),
        "voice: ECAPA": ("features/voice_v6_ecapa/train_English_voices.csv", EXT / "train_voices_ecapa.csv", None),
        "voice: x-vector (per clip)": ("features/voice_v9_xvector_perclip/train_English_voices.csv", EXT / "train_voices_xvector.csv", None),
        "face: FaRL (B0)": ("features/face_v2_farl/train_English_faces.csv", EXT / "train_faces.csv", None),
    }
    if (EXT / "train_faces_arcface.csv").exists():
        reps["face: ArcFace"] = ("features/face_v1_insightface/train_English_faces.csv", EXT / "train_faces_arcface.csv", None)
    people = np.unique(ident)
    gender = np.array([g[p] for p in people])
    targets = {t: np.array([np.nanmedian(tr[t].to_numpy()[ident == p]) for p in people]) for t in ("age", "build")}
    groups = np.arange(len(people))
    report = {"people": int(len(people)), "probes": {}}
    for name, (a, b, cols) in reps.items():
        X = np.vstack([read_csv_matrix(a), read_csv_matrix(b)])
        if cols is not None:
            X = X[:, cols]
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12) if "ECAPA" in name or "face" in name else X
        P = np.stack([X[ident == p].mean(0) for p in people])
        out = {}
        for t, y in targets.items():
            ok = np.isfinite(y)
            pred = np.full(len(y), np.nan)
            for trn, tst in GroupKFold(5).split(P[ok], y[ok], groups[ok]):
                idx_tr, idx_te = np.flatnonzero(ok)[trn], np.flatnonzero(ok)[tst]
                mu, sd = P[idx_tr].mean(0), P[idx_tr].std(0) + 1e-6
                m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit((P[idx_tr] - mu) / sd, y[idx_tr])
                pred[idx_te] = m.predict((P[idx_te] - mu) / sd)
            out[f"{t}_r_all"] = float(np.corrcoef(pred[ok], y[ok])[0, 1])
            for gg in ("m", "f"):
                mm = ok & (gender == gg)
                out[f"{t}_r_within_{gg}"] = float(np.corrcoef(pred[mm], y[mm])[0, 1])
        report["probes"][name] = out
        print(f"{name:30s}", {k: round(v, 3) for k, v in out.items()}, flush=True)
    Path("results/phase2/phase3_alignment/trait_probe.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
