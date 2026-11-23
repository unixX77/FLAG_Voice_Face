"""Phase 1 supplement: dedicated speaker encoders on the same trials as audio_diagnostics.py.

ECAPA-TDNN (VoxCeleb2) and the per-clip WavLM-SV x-vector, now available for official and
external rows with identical preprocessing (extract_external_speaker_embeddings.py).
Same pools, groups, deterministic trials and backends (raw / centred / LDA fitted on
identity-disjoint people). Also: duration bins for the bilingual pool, and fusion with the
WavLM features as a check of complementarity.
Output: results/phase2/phase1_audio/metrics_speaker_encoders.json (+ scores).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2 import unimodal_eval as ue  # noqa: E402
from phase2.audio_diagnostics import (CONDITIONS_BILINGUAL, CONDITIONS_OFFICIAL, EXT, OUT, Backend,  # noqa: E402
                                      clustering_check, duration_bins, evaluate_groups, load_external_all,
                                      load_official, probes, read_csv_matrix, subset)


def summarise(r):
    return {k: round(v["eer"] * 100, 2) for k, v in r["metrics"].items() if isinstance(v, dict) and "eer" in v} | {
        "R@1": round(r["retrieval"]["recall_at_1"], 3)}


def main() -> None:
    opool, oreps, _, ofolds, _ = load_official()
    xpool, xreps, _, _ = load_external_all()
    oreps = {"ecapa_vox2": oreps["ecapa_vox2"],
             "wavlm_sv_xvector_perclip": read_csv_matrix("features/voice_v9_xvector_perclip/train_English_voices.csv"),
             "wavlm_allmean": oreps["wavlm_allmean"]}
    xr = {"ecapa_vox2": read_csv_matrix(EXT / "train_voices_ecapa.csv"),
          "wavlm_sv_xvector_perclip": read_csv_matrix(EXT / "train_voices_xvector.csv"),
          "wavlm_allmean": xreps["wavlm_allmean"]}
    # Fusion check: z-scored concatenation of ECAPA and WavLM (does WavLM add anything?).
    oreps["ecapa+wavlm"] = np.hstack([oreps["ecapa_vox2"], oreps["wavlm_allmean"]])
    xr["ecapa+wavlm"] = np.hstack([xr["ecapa_vox2"], xr["wavlm_allmean"]])
    dur_x = np.load(EXT / "train_voice_durations.npy")
    report = {"official": {}, "bilingual": {}}
    scores = {}
    for name in ("ecapa_vox2", "wavlm_sv_xvector_perclip", "ecapa+wavlm"):
        for kind in ("raw", "centred", "lda") if name != "ecapa+wavlm" else ("lda",):
            b = Backend(xr[name], xpool.identity, kind)
            r = evaluate_groups(opool, ofolds, lambda gid, m, x=oreps[name], b=b: b(x[m]), CONDITIONS_OFFICIAL, 40,
                                dur=opool.quality)
            report["official"][f"{name}|{kind}"] = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
            scores[f"official__{name}__{kind}"] = r["scores"]
            print("official", name, kind, summarise(r), flush=True)
    names = xpool.identity
    groups = np.full(len(names), -1)
    for src in ("mav1", "mav3"):
        ids = sorted({n for n in names if n.startswith(src + "_")})
        for i, n in enumerate(ids):
            groups[names == n] = i % 5 + (0 if src == "mav1" else 5)
    bil = groups >= 0
    bpool, bgroups = subset(xpool, bil), groups[bil]
    bpool.quality = dur_x[bil]
    fit = ~bil
    for name in ("ecapa_vox2", "wavlm_sv_xvector_perclip", "ecapa+wavlm"):
        fx = np.concatenate([oreps[name], xr[name][fit]])
        fid = np.concatenate([opool.identity, xpool.identity[fit]])
        for kind in ("raw", "centred", "lda") if name != "ecapa+wavlm" else ("lda",):
            b = Backend(fx, fid, kind)
            x = xr[name][bil]
            r = evaluate_groups(bpool, bgroups, lambda gid, m, x=x, b=b: b(x[m]), CONDITIONS_BILINGUAL, 40,
                                retrieval_session=True)
            # duration bins (same-language, different-session genuine vs same-gender impostors)
            t, s = r["trials"], r["scores"]
            keep = np.isin(t.condition, ("same_group_diff_session", "imp_same_gender_same_group"))
            sub = ue.Trials(t.a[keep], t.b[keep], t.label[keep], t.condition[keep])
            r["metrics"]["duration_same_gender"] = duration_bins(sub, s[keep], dur_x[bil])
            entry = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
            if kind != "raw":
                entry["probes"] = probes(b(x), bpool, bgroups)
                entry["clustering"] = clustering_check(b(x), bpool, bgroups)
            report["bilingual"][f"{name}|{kind}"] = entry
            scores[f"bilingual__{name}__{kind}"] = s
            print("bilingual", name, kind, summarise(r), flush=True)
    report["external_duration_seconds"] = {"median": float(np.median(dur_x)), "p10": float(np.percentile(dur_x, 10)),
                                           "p90": float(np.percentile(dur_x, 90))}
    report["official_duration_seconds"] = {"median": float(np.median(opool.quality)),
                                           "p10": float(np.percentile(opool.quality, 10)),
                                           "p90": float(np.percentile(opool.quality, 90))}
    (OUT / "metrics_speaker_encoders.json").write_text(json.dumps(report, indent=1, default=float))
    np.savez_compressed(OUT / "scores_speaker_encoders.npz", **scores)
    print("saved")


if __name__ == "__main__":
    main()
