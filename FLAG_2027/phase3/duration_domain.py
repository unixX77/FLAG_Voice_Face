"""E4: does C1's change vs B0 depend on voice-clip duration or recording session?

Per-trial scores of two systems (per seed, 3 seeds) on identical trials:
  official : fixed fold protocols (English), voice durations from the wav headers
  p2       : Urdu-holdout proxy trials (evaluate_language_transfer builder), durations from the
             external duration cache; strata also by voice language and by same/other source video
             of the genuine face (impostors are always other people/videos)
For each stratum: EER of A and B (mean over folds x seeds), B - A with a paired identity bootstrap
(people resampled within fold). Interaction: bootstrap distribution of
(delta in the shortest-duration bin - delta in the longest bin).
Output: results/phase3/e4_duration/*.json and per-trial score arrays (.npz).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import face_inputs, load_model, voice_inputs, weighted_eer  # noqa: E402
from local_validation.evaluate_language_transfer import build_trials, embed  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices, load_manifest_identities  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402
from train_cross_modal import holdout_external_identities  # noqa: E402

BINS = ((0, 3), (3, 6), (6, 10), (10, 1e9))
OUT = Path("results/phase3/e4_duration")


def bin_name(b):
    return f"{b[0]}-{'inf' if b[1] >= 1e9 else int(b[1])}s"


def strata_report(cells, systems, seeds, n_boot=1000):
    """cells: {stratum: [ (fold, labels, voice_person_idx, n_people, {sys: {seed: scores}}) ]}"""
    rng = np.random.default_rng(0)
    out, boots = {}, {}
    for name, parts in cells.items():
        def eer_of(sysname, weights=None):
            vals = []
            for k, (fold, lab, pid, n_people, sc) in enumerate(parts):
                w = np.ones(len(lab)) if weights is None else weights[k][pid]
                m = w > 0
                if lab[m].sum() == 0 or (1 - lab[m]).sum() == 0:
                    continue
                vals += [weighted_eer(lab[m], sc[sysname][s][m], w[m]) for s in seeds]
            return float(np.mean(vals))
        a, b = eer_of(systems[0]), eer_of(systems[1])
        d = []
        for _ in range(n_boot):
            ws = [np.bincount(rng.integers(0, p[3], p[3]), minlength=p[3]).astype(float) for p in parts]
            d.append(eer_of(systems[1], ws) - eer_of(systems[0], ws))
        d = np.asarray(d)
        boots[name] = d
        out[name] = {"n_trials": int(sum(len(p[1]) for p in parts)), "n_genuine": int(sum(p[1].sum() for p in parts)),
                     systems[0]: a, systems[1]: b, "b_minus_a": b - a,
                     "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                     "p_b_better": float((d < 0).mean())}
    return out, boots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default="b0=runs/traits/full/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar")
    p.add_argument("--b", default="c1=runs/phase2/g3_cat_farl/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar")
    p.add_argument("--a-p2", default="b0=runs/traits/p2_full/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar")
    p.add_argument("--b-p2", default="c1=runs/phase2/p2_g3_cat_farl/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds
    # ---------------- official folds ----------------
    (na, pa), (nb, pb) = (args.a.split("=", 1), args.b.split("=", 1))
    systems = (na, nb)
    F, V, _ = load_feature_matrices(Path("features/face_v2_farl/train_English_faces.csv"),
                                    Path("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)
    rows = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    dur = np.array([sf.info(f"train_set/train_set/{r[2]}").duration for r in rows])
    ident = load_manifest_identities(Path("train_set/train_set/train_English.txt"))
    cells = {}
    raw = {}
    for fold in range(5):
        fd = Path(f"validation/fold_{fold}")
        for t in ("standard", "gender"):
            pr = load_protocol(fd / f"{t}_pairs.csv", fd / f"{t}_ground_truth.csv")
            fi, vi, lab = np.asarray(pr.face_indices), np.asarray(pr.voice_indices), np.asarray(pr.labels)
            people = np.unique(ident[vi])
            pid = np.searchsorted(people, ident[vi])
            sc = {}
            for name, pat in ((na, pa), (nb, pb)):
                sc[name] = {}
                for s in seeds:
                    path = Path(pat.format(fold=fold, seed=s))
                    cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
                    v = voice_inputs(cfg, Vt, Path("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
                    f = face_inputs(cfg, Ft, Path("features/face_v2_farl/train_English_faces.csv"))
                    m = load_model(path, f.shape[1], v)
                    with torch.no_grad():
                        sc[name][s] = m.pair_similarity(m.face_branch(f[fi]), m.voice_branch(v[vi])).numpy()
                    raw[f"official__{name}__f{fold}_s{s}_{t}"] = sc[name][s]
            for b in BINS:
                k = (dur[vi] >= b[0]) & (dur[vi] < b[1])
                cells.setdefault(f"{t}|{bin_name(b)}", []).append(
                    (fold, lab[k], pid[k], len(people), {n: {s: x[k] for s, x in d.items()} for n, d in sc.items()}))
            cells.setdefault(f"{t}|all", []).append((fold, lab, pid, len(people), sc))
        print("official fold", fold, flush=True)
    off_report, off_boot = strata_report(cells, systems, seeds)
    inter = {}
    for t in ("standard", "gender"):
        d = off_boot[f"{t}|{bin_name(BINS[0])}"] - off_boot[f"{t}|{bin_name(BINS[-1])}"]
        inter[t] = {"short_minus_long_delta": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    (OUT / "official_duration.json").write_text(json.dumps({"systems": systems, "strata": off_report,
                                                            "interaction_short_vs_long": inter,
                                                            "clips_per_bin": {bin_name(b): int(((dur >= b[0]) & (dur < b[1])).sum()) for b in BINS}}, indent=1))
    # ---------------- P2 Urdu-holdout proxy ----------------
    (na2, pa2), (nb2, pb2) = (args.a_p2.split("=", 1), args.b_p2.split("=", 1))
    first = torch.load(pa2.format(fold=0, seed=seeds[0]), map_location="cpu", weights_only=False)["config"]
    xrows = [l.split() for l in Path(first["extra_manifest"]).open() if l.strip()]
    gender_of = dict(l.strip().split(",") for l in Path(first["extra_gender_metadata"]).open() if l.strip())
    xdur = np.load("features/external/combined_all/train_voice_durations.npy")
    vvid = np.array([r[2].split("/")[-1].rsplit("_", 1)[0] for r in xrows])
    fvid = np.array([r[3].split("/")[-1].rsplit("_", 1)[0] for r in xrows])
    names = [r[4] for r in xrows]
    cache, cells2 = {}, {}

    def feats(path):
        if path not in cache:
            cache[path] = np.loadtxt(path, delimiter=",", dtype=np.float32)[:, :-1]
        return cache[path]
    for fold in range(5):
        held = holdout_external_identities(names, "mav1", fold)
        trials = build_trials(xrows, gender_of, held, np.random.default_rng(100 + fold), 3, 3)
        for t in ("standard", "gender"):
            tr = trials[t]
            v = np.array([x[0] for x in tr]); f = np.array([x[1] for x in tr]); lab = np.array([x[2] for x in tr])
            lg = np.array([x[4] for x in tr])
            people = sorted({x[3] for x in tr}); pos = {q: i for i, q in enumerate(people)}
            pid = np.array([pos[x[3]] for x in tr])
            sc = {}
            for name, pat in ((na2, pa2), (nb2, pb2)):
                sc[name] = {}
                for s in seeds:
                    path = pat.format(fold=fold, seed=s)
                    cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
                    fe, ve = embed(path, feats(cfg["extra_face_csv"]), feats(cfg["extra_voice_csv"]))
                    sc[name][s] = (fe[f] * ve[v]).sum(1)
                    raw[f"p2__{name}__f{fold}_s{s}_{t}"] = sc[name][s]
            same_vid = (vvid[v] == fvid[f]) & (lab == 1)
            for lang in ("English", "Urdu"):
                for b in BINS:
                    k = (lg == lang) & (xdur[v] >= b[0]) & (xdur[v] < b[1])
                    cells2.setdefault(f"{t}|{lang}|{bin_name(b)}", []).append(
                        (fold, lab[k], pid[k], len(people), {n: {s: x[k] for s, x in d.items()} for n, d in sc.items()}))
                for sess, m_gen in (("genuine_same_video", same_vid), ("genuine_other_video", (lab == 1) & ~same_vid)):
                    k = (lg == lang) & (m_gen | (lab == 0))
                    cells2.setdefault(f"{t}|{lang}|{sess}", []).append(
                        (fold, lab[k], pid[k], len(people), {n: {s: x[k] for s, x in d.items()} for n, d in sc.items()}))
        print("p2 fold", fold, flush=True)
    p2_report, p2_boot = strata_report(cells2, (na2, nb2), seeds)
    inter2 = {}
    for t in ("standard", "gender"):
        for lang in ("English", "Urdu"):
            d = p2_boot[f"{t}|{lang}|{bin_name(BINS[0])}"] - p2_boot[f"{t}|{lang}|{bin_name(BINS[-1])}"]
            inter2[f"{t}|{lang}"] = {"short_minus_long_delta": float(np.mean(d)),
                                     "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    (OUT / "p2_duration_session.json").write_text(json.dumps({"systems": (na2, nb2), "strata": p2_report,
                                                              "interaction_short_vs_long": inter2}, indent=1))
    np.savez_compressed(OUT / "per_trial_scores.npz", **raw)
    for rep in (off_report, p2_report):
        for k, v in rep.items():
            print(f"{k:38s} n={v['n_trials']:6d} {systems[0]} {v[systems[0]]*100:.2f} {systems[1]} {v[systems[1]]*100:.2f} "
                  f"d {v['b_minus_a']*100:+.2f} [{v['ci95'][0]*100:+.2f},{v['ci95'][1]*100:+.2f}]")
    print("interaction official", inter, "\ninteraction p2", inter2)


if __name__ == "__main__":
    main()
