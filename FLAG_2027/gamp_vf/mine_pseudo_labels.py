"""Stages E + F: mine pseudo face-voice associations in the hidden pool (simulation).

Hidden-association protocol (docs/gamp_vf/02_architecture.md):
  H = hidden external identities: their voice clips and face images are pooled
      *separately*, without identity labels and without row alignment.
  C = calibration identities: labelled, never trained on by any simulation model;
      used to tune clustering thresholds, fit the confidence calibrator and the
      matching parameters. Thresholds are frozen before they are applied to H.
Ground-truth identities of H are read only after all decisions, for diagnostics.

Per fold, writes results/gamp_vf/stage_E/<iteration>/fold_<k>/:
  candidates.csv    every voice group's candidate with features, confidence, decision, reason, truth
  pseudo_pairs.csv  training rows for the chosen mode (face_row, voice_row, pseudo_id, weight, gender, conflict_group)
  summary.json      thresholds, clustering quality, recovery metrics for every mode on C and H
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamp_vf import pseudo_labels as pl  # noqa: E402
from gamp_vf.clustering import agglomerative, cluster_metrics  # noqa: E402
from gamp_vf.matching import hungarian_partial, local_selection, sinkhorn_match  # noqa: E402
from local_validation.compare_fold_runs import load_model  # noqa: E402

LANG = {"English": 0, "Urdu": 1, "German": 2}
TARGET_PRECISION = 0.9


def read_list(path: Path) -> list[str]:
    return [l.strip() for l in path.open() if l.strip()]


def embed(model, faces, voices, gender_weight, gender_bias):
    with torch.no_grad():
        fe = model.face_branch(torch.from_numpy(faces)).numpy()
        ve = model.voice_branch(torch.from_numpy(voices)).numpy()
    sig = lambda e: 1 / (1 + np.exp(-(e @ gender_weight + gender_bias)))  # noqa: E731
    return fe, ve, sig(fe), sig(ve)


class Pool:
    """One identity pool (H or C) seen through the teacher, grouped without labels."""

    def __init__(self, rows, fe, ve, fg, vg, raw_faces, langs, face_t, voice_t, face_space):
        self.rows = rows
        self.voice_labels = agglomerative(ve, voice_t, min_size=3)
        face_input = raw_faces if face_space == "raw" else fe
        self.face_labels = agglomerative(face_input, face_t, min_size=3)
        self.voice = pl.make_groups(ve, self.voice_labels, np.arange(len(rows)), langs, True, vg)
        self.face = pl.make_groups(fe, self.face_labels, np.arange(len(rows)), None, False, fg)
        s = self.voice.prototypes @ self.face.prototypes.T
        self.scores = np.where(pl.gender_compatible(self.voice, self.face), s, pl.MASKED)
        self.langs = langs

    def pair_features(self, g: int, j: int) -> np.ndarray:
        s = self.scores
        others = np.delete(s[g], j)
        margin = s[g, j] - (others.max() if len(others) else -1.0)
        reciprocal = float(s[:, j].argmax() == g and s[g].argmax() == j)
        known = [v for lang, v in self.voice.language_prototypes[g].items() if lang >= 0]
        if len(known) >= 2:
            allowed = s[g] > pl.MASKED + 0.5
            consistency = np.mean([np.argmax(np.where(allowed, self.face.prototypes @ v, pl.MASKED)) == j for v in known])
        else:
            consistency = 1.0
        return np.array([s[g, j], margin, reciprocal, consistency,
                         np.log(len(self.voice.members[g])), np.log(len(self.face.members[j]))])


def evaluate(pool: Pool, match: np.ndarray, truth: np.ndarray, genders: np.ndarray, cap: int) -> dict:
    """Recovery diagnostics for a voice-group -> face-group assignment (truth used here only)."""
    v_major, v_pur = pl.group_majority(pool.voice, truth)
    f_major, f_pur = pl.group_majority(pool.face, truth)
    accepted = np.flatnonzero(match >= 0)
    correct = np.array([v_major[g] == f_major[match[g]] for g in accepted], dtype=bool)
    # Row-level purity of the pseudo rows: P(face row and voice row truly share identity).
    rng = np.random.default_rng(0)
    same, total = 0, 0
    for g in accepted:
        f, v = pl.pair_rows(pool.voice.members[g], pool.face.members[match[g]], cap, rng)
        same += int((truth[f] == truth[v]).sum())
        total += len(f)
    ids_true = np.unique(truth)
    recovered = np.unique(v_major[accepted][correct]) if len(accepted) else np.array([])
    out = {"accepted_pairs": int(len(accepted)), "correct_pairs": int(correct.sum()),
           "pair_precision": float(correct.mean()) if len(accepted) else float("nan"),
           "row_precision": float(same / total) if total else float("nan"), "pseudo_rows": int(total),
           "identity_recall": float(len(recovered) / len(ids_true)),
           "voice_row_coverage": float(sum(len(pool.voice.members[g]) for g in accepted) / len(truth)),
           "unmatched_voice_groups": int((match < 0).sum()), "voice_groups": int(len(match)),
           "face_groups": int(len(pool.face.ids))}
    # Errors by gender and language of the voice group (majority).
    by = {}
    for g, ok in zip(accepted, correct):
        gender = "m" if genders[pool.voice.members[g]].mean() > 0.5 else "f"
        lang_values = pool.langs[pool.voice.members[g]]
        lang = {0: "English", 1: "Urdu", 2: "German", -1: "Mixed"}[int(np.bincount(lang_values + 1).argmax()) - 1]
        for key in (f"gender_{gender}", f"language_{lang}"):
            d = by.setdefault(key, {"accepted": 0, "errors": 0})
            d["accepted"] += 1
            d["errors"] += int(not ok)
    out["errors_by_group"] = by
    return out


def tune_on_calibration(pool: Pool, truth: np.ndarray):
    """Fit calibrator + thresholds + matching parameters on C (labelled, unseen by the teacher)."""
    v_major, _ = pl.group_majority(pool.voice, truth)
    f_major, _ = pl.group_majority(pool.face, truth)
    cand = pl.candidate_features(pool.voice, pool.face, pool.scores)
    X = pl.feature_matrix(cand)
    correct = v_major == f_major[cand["best_face"]]
    calibrator = pl.ConfidenceCalibrator().fit(X, correct)
    conf = calibrator.predict(X)
    high = pl.precision_threshold(conf, correct, TARGET_PRECISION)
    low = min(0.5, high)

    def best_by_count(options):
        best, best_key = None, None
        for key, match in options:
            acc = np.flatnonzero(match >= 0)
            ok = (v_major[acc] == f_major[match[acc]]).sum() if len(acc) else 0
            prec = ok / len(acc) if len(acc) else 0
            if prec >= TARGET_PRECISION and (best is None or ok > best[0]):
                best, best_key = (ok, prec), key
        return best_key
    hung = best_by_count([(t, hungarian_partial(pool.scores, t)) for t in np.arange(0.0, 0.8, 0.025)])
    sink = best_by_count([((temp, d), sinkhorn_match(pool.scores, temp, d)[0])
                          for temp in (0.02, 0.05, 0.1) for d in np.arange(0.0, 0.8, 0.05)])
    return {"calibrator": calibrator, "high": high, "low": low,
            "hungarian_min_score": 0.8 if hung is None else float(hung),
            "sinkhorn": (0.05, 0.8) if sink is None else (float(sink[0]), float(sink[1])),
            "calibration_candidates": int(len(correct)), "calibration_top1_accuracy": float(correct.mean())}


def assignments(pool: Pool, params) -> dict[str, np.ndarray]:
    """Decisions on a pool with *frozen* parameters (no labels involved)."""
    cand = pl.candidate_features(pool.voice, pool.face, pool.scores)
    conf = params["calibrator"].predict(pl.feature_matrix(cand))
    decision = pl.decide(conf, params["high"], params["low"])
    n = len(conf)
    local = np.where(decision == "high", cand["best_face"], -1)
    # Keep local selection one-to-one on faces (highest confidence wins).
    for j in np.unique(local[local >= 0]):
        rows = np.flatnonzero(local == j)
        local[rows[rows != rows[np.argmax(conf[rows])]]] = -1
    hung = hungarian_partial(pool.scores, params["hungarian_min_score"])
    sink, _ = sinkhorn_match(pool.scores, *params["sinkhorn"])
    # B6: high-confidence local decisions first, then global matching among the ambiguous rest.
    hybrid = local.copy()
    rest_v = np.flatnonzero((local < 0) & (decision == "ambiguous"))
    rest_f = np.setdiff1d(np.arange(pool.scores.shape[1]), local[local >= 0])
    if len(rest_v) and len(rest_f):
        sub = hungarian_partial(pool.scores[np.ix_(rest_v, rest_f)], params["hungarian_min_score"])
        hybrid[rest_v[sub >= 0]] = rest_f[sub[sub >= 0]]
    return {"local": local, "hungarian": hung, "sinkhorn": sink, "local_plus_hungarian": hybrid,
            "pairwise_top1": cand["best_face"]}, cand, conf, decision


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, help="checkpoint pattern with {fold}")
    p.add_argument("--iteration", required=True, help="output name, e.g. it1")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ext", type=Path, default=Path("features/external/combined_all"))
    p.add_argument("--lists", type=Path, default=Path("results/gamp_vf/stage_E"))
    p.add_argument("--mode", default="auto", help="mode for pseudo_pairs.csv; auto = best on C")
    p.add_argument("--rows-per-pair", type=int, default=100)
    args = p.parse_args()

    rows = [l.split() for l in (args.ext / "train_manifest.txt").open() if l.strip()]
    names = np.array([r[4] for r in rows])
    langs = np.array([LANG.get(r[2].split("/")[1], -1) for r in rows])
    gender_of = dict(l.strip().split(",") for l in (args.ext / "gender.csv").open())
    genders = np.array([1 if gender_of[n] == "m" else 0 for n in names])
    faces = np.loadtxt(args.ext / "train_faces.csv", delimiter=",", dtype=np.float32)[:, :-1]
    voices = np.loadtxt(args.ext / "train_voices_egemaps.csv", delimiter=",", dtype=np.float32)[:, :-1]
    H, C = read_list(args.lists / "hidden_identities.txt"), read_list(args.lists / "calibration_identities.txt")
    rows_H, rows_C = np.flatnonzero(np.isin(names, H)), np.flatnonzero(np.isin(names, C))
    assert not set(H) & set(C)

    for fold in args.folds:
        out = args.lists / args.iteration / f"fold_{fold}"
        out.mkdir(parents=True, exist_ok=True)
        teacher_path = Path(args.teacher.format(fold=fold))
        ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
        excluded = set(read_list(Path(ckpt["config"]["exclude_external_identities"])))
        if not (set(H) | set(C)) <= excluded:
            raise ValueError("Teacher was trained on hidden or calibration identities")
        model = load_model(teacher_path, faces.shape[1], torch.from_numpy(voices))
        gw = ckpt["aux_state_dict"]["traits.gender.weight"].numpy()[0]
        gb = float(ckpt["aux_state_dict"]["traits.gender.bias"].numpy()[0])
        emb = {}
        for key, r in (("H", rows_H), ("C", rows_C)):
            emb[key] = embed(model, faces[r], voices[r], gw, gb)

        # Clustering thresholds: best ARI on C.
        fe_C, ve_C = emb["C"][0], emb["C"][1]
        truth_C = names[rows_C]
        grid_v = [(t, cluster_metrics(agglomerative(ve_C, t, 3), truth_C)) for t in np.arange(0.15, 0.65, 0.05)]
        grid_f = [((space, t), cluster_metrics(agglomerative(faces[rows_C] if space == "raw" else fe_C, t, 3), truth_C))
                  for space in ("raw", "projected") for t in np.arange(0.15, 0.65, 0.05)]
        voice_t = max(grid_v, key=lambda x: x[1]["ari"])[0]
        face_space, face_t = max(grid_f, key=lambda x: x[1]["ari"])[0]

        def pool(key, r):
            fe, ve, fg, vg = emb[key]
            return Pool(r, fe, ve, fg, vg, faces[r], langs[r], face_t, voice_t, face_space)
        pC, pH = pool("C", rows_C), pool("H", rows_H)
        params = tune_on_calibration(pC, truth_C)
        decisions_C = assignments(pC, params)[0]
        # ---- H: decisions with frozen parameters; truth read afterwards ----
        decisions_H, cand_H, conf_H, decision_H = assignments(pH, params)
        truth_H = names[rows_H]
        summary = {"fold": fold, "teacher": str(teacher_path), "iteration": args.iteration,
                   "target_precision": TARGET_PRECISION,
                   "thresholds": {"voice_cluster_distance": float(voice_t), "face_cluster_space": face_space,
                                  "face_cluster_distance": float(face_t), "confidence_high": params["high"],
                                  "confidence_low": params["low"],
                                  "hungarian_min_score": params["hungarian_min_score"],
                                  "sinkhorn_temperature_dustbin": params["sinkhorn"]},
                   "calibration_top1_accuracy": params["calibration_top1_accuracy"],
                   "clustering": {"C_voice": cluster_metrics(pC.voice_labels, truth_C),
                                  "C_face": cluster_metrics(pC.face_labels, truth_C),
                                  "H_voice": cluster_metrics(pH.voice_labels, truth_H),
                                  "H_face": cluster_metrics(pH.face_labels, truth_H)},
                   "recovery": {"C": {m: evaluate(pC, a, truth_C, genders[rows_C], args.rows_per_pair)
                                      for m, a in decisions_C.items()},
                                "H": {m: evaluate(pH, a, truth_H, genders[rows_H], args.rows_per_pair)
                                      for m, a in decisions_H.items()}}}
        # Sensitivity to incomplete candidate pools: remove 30% of H face groups.
        rng = np.random.default_rng(fold)
        keep_f = np.sort(rng.choice(pH.scores.shape[1], int(0.7 * pH.scores.shape[1]), replace=False))
        reduced = {}
        full_scores = pH.scores
        pH.scores = full_scores[:, keep_f]
        face_backup = pH.face
        pH.face = pl.Groups(face_backup.ids[keep_f], face_backup.prototypes[keep_f],
                            [face_backup.members[j] for j in keep_f],
                            [face_backup.language_prototypes[j] for j in keep_f], face_backup.gender_prob[keep_f])
        for m, a in assignments(pH, params)[0].items():
            reduced[m] = evaluate(pH, a, truth_H, genders[rows_H], args.rows_per_pair)
        pH.scores, pH.face = full_scores, face_backup
        summary["recovery"]["H_with_30pct_faces_missing"] = reduced

        # Mode for training: most correct pairs on C at the target precision (training-side choice).
        if args.mode == "auto":
            eligible = {m: r for m, r in summary["recovery"]["C"].items()
                        if m != "pairwise_top1" and r["accepted_pairs"] and r["pair_precision"] >= TARGET_PRECISION}
            mode = max(eligible, key=lambda m: eligible[m]["correct_pairs"]) if eligible else "local"
        else:
            mode = args.mode
        summary["chosen_mode"] = mode
        match = decisions_H[mode]

        # Candidate provenance table (all voice groups).
        v_major, v_pur = pl.group_majority(pH.voice, truth_H)
        f_major, f_pur = pl.group_majority(pH.face, truth_H)
        with (out / "candidates.csv").open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["voice_group", "voice_rows", "best_face_group", "score_raw_cosine", "margin", "reciprocal",
                        "language_consistency", "confidence_calibrated", "decision", "reason",
                        *[f"match_{m}" for m in decisions_H], "teacher", "iteration",
                        "diag_voice_identity", "diag_voice_purity", "diag_best_face_identity", "diag_correct_top1"])
            for g in range(len(conf_H)):
                d = decision_H[g]
                reason = {"high": f"confidence>={params['high']:.3f}",
                          "ambiguous": f"{params['low']:.3f}<=confidence<{params['high']:.3f}",
                          "unreliable": f"confidence<{params['low']:.3f}"}[d]
                if d == "high" and decisions_H["local"][g] < 0:
                    reason += "; lost face-group conflict"
                b = cand_H["best_face"][g]
                w.writerow([g, len(pH.voice.members[g]), b, f"{cand_H['score'][g]:.4f}", f"{cand_H['margin'][g]:.4f}",
                            int(cand_H["reciprocal"][g]), f"{cand_H['language_consistency'][g]:.2f}",
                            f"{conf_H[g]:.4f}", d, reason, *[int(a[g]) for a in decisions_H.values()],
                            str(teacher_path), args.iteration, v_major[g], f"{v_pur[g]:.3f}", f_major[b],
                            int(v_major[g] == f_major[b])])

        # Pseudo pairs for the student.
        acc = np.flatnonzero(match >= 0)
        conf_pairs = params["calibrator"].predict(np.stack([pH.pair_features(g, match[g]) for g in acc])) \
            if len(acc) else np.array([])
        conflicts = pl.conflict_groups(pH.voice.prototypes[acc], pH.face.prototypes[match[acc]],
                                       1 - voice_t, 1 - face_t) if len(acc) else np.array([])
        rng = np.random.default_rng(1000 + fold)
        n_rows = 0
        with (out / "pseudo_pairs.csv").open("w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["face_row", "voice_row", "pseudo_id", "weight", "gender", "conflict_group"])
            for k, g in enumerate(acc):
                j = match[g]
                gp = np.nanmean([pH.voice.gender_prob[g], pH.face.gender_prob[j]])
                gender = 1 if gp >= 0.75 else 0 if gp <= 0.25 else -1
                f, v = pl.pair_rows(pH.voice.members[g], pH.face.members[j], args.rows_per_pair, rng)
                for fi, vi in zip(f, v):
                    face_row, voice_row = int(rows_H[fi]), int(rows_H[vi])  # external CSV row indices
                    assert names[face_row] in H and names[voice_row] in H  # never C, never official
                    w.writerow([face_row, voice_row, f"{args.iteration}_f{fold}_v{g}",
                                f"{max(conf_pairs[k], 0.05):.4f}", gender, f"c{conflicts[k]}"])
                    n_rows += 1
        summary["pseudo_pairs_written"] = {"pairs": int(len(acc)), "rows": n_rows,
                                           "conflict_groups": int(len(np.unique(conflicts))) if len(acc) else 0}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        r = summary["recovery"]["H"][mode]
        print(f"fold {fold}: clusters voice t={voice_t:.2f} face {face_space} t={face_t:.2f} | mode={mode} "
              f"pairs={r['accepted_pairs']} precision={r['pair_precision']:.3f} row_precision={r['row_precision']:.3f} "
              f"identity_recall={r['identity_recall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
