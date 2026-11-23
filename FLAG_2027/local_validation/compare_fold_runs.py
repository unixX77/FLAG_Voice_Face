"""Paired comparison of two five-fold configurations on the local protocols.

Each configuration is a directory pattern holding one checkpoint per fold and
seed (``{fold}`` and ``{seed}`` placeholders). Both are scored on the same
validation trials with the same clean features. EERs are averaged over seeds
and folds. A paired cluster bootstrap resamples held-out identities within each
fold (trials weighted by the multiplicity of their voice identity) and reports
the B - A difference with a 95 % interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_modal_model import CrossModalIdentityNetwork  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices, load_manifest_identities  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402

TRACKS = ("standard", "gender")


def weighted_eer(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


_LAYER_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def voice_inputs(config: dict, default: torch.Tensor, default_csv: Path | None = None) -> torch.Tensor:
    """The voice input a run was trained on: the shared CSV, its own CSV, or its WavLM layer array."""
    path = config.get("voice_layers_npy")
    if path is None:
        own_csv = config.get("voice_csv")
        if own_csv is None or default_csv is None or Path(own_csv).resolve() == Path(default_csv).resolve():
            return default
        key = (str(own_csv), "csv")
        if key not in _LAYER_CACHE:
            values = np.loadtxt(own_csv, delimiter=",", dtype=np.float32)[:, :-1]  # drop label column
            _LAYER_CACHE[key] = torch.from_numpy(np.ascontiguousarray(values))
        return _LAYER_CACHE[key]
    key = (str(path), config.get("voice_layer_features", "mean"))
    if key not in _LAYER_CACHE:
        layers = np.load(path).astype(np.float32)
        if key[1] == "mean":
            layers = layers[:, :, : layers.shape[2] // 2]
        _LAYER_CACHE[key] = torch.from_numpy(np.ascontiguousarray(layers))
    return _LAYER_CACHE[key]


def face_inputs(config: dict, default: torch.Tensor, default_csv: Path | None = None) -> torch.Tensor:
    """The face input a run was trained on: the shared CSV or its own (e.g. ArcFace instead of FaRL)."""
    own_csv = config.get("face_csv")
    if own_csv is None or default_csv is None or Path(own_csv).resolve() == Path(default_csv).resolve():
        return default
    key = (str(own_csv), "face_csv")
    if key not in _LAYER_CACHE:
        values = np.loadtxt(own_csv, delimiter=",", dtype=np.float32)[:, :-1]
        _LAYER_CACHE[key] = torch.from_numpy(np.ascontiguousarray(values))
    return _LAYER_CACHE[key]


def load_model(path: Path, face_dim: int, voice: torch.Tensor) -> CrossModalIdentityNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    c = checkpoint["config"]
    voice_dim = voice.shape[-1]
    model = CrossModalIdentityNetwork(
        face_dim=face_dim, voice_dim=voice_dim, embedding_dim=c["embedding_dim"],
        num_classes=checkpoint["state_dict"]["identity_classifier.weight"].shape[0],
        arc_scale=c["arc_scale"], arc_margin=c["arc_margin"], prototype_momentum=c["prototype_momentum"],
        face_hidden_dims=tuple(c["face_hidden_dims"]), voice_hidden_dims=tuple(c["voice_hidden_dims"]),
        hidden_dropout=c["hidden_dropout"], standardize_inputs=c["standardize_inputs"],
        input_dropout=c["input_dropout"], input_noise_std=c["input_noise_std"],
        geometry=c.get("geometry", "sphere"), curvature=c.get("curvature", 1.0),
        clip_radius=c.get("clip_radius", 1.0),
        voice_layers=voice.shape[1] if voice.ndim == 3 else 0,
        voice_layer_pooling=c.get("voice_layer_pooling", "mix"),
        voice_fusion=c.get("voice_fusion", "concat"), voice_fusion_split=c.get("voice_fusion_split", 0),
        voice_fusion_dim=c.get("voice_fusion_dim", 64))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="checkpoint pattern with {fold} and {seed}")
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--face-csv", type=Path, required=True)
    parser.add_argument("--voice-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, default=Path("validation"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    identities = load_manifest_identities(args.manifest)
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)

    # trials[fold][track] = (labels, voice identity, {run: [scores per seed]})
    trials = []
    for fold in range(5):
        fold_dir = args.validation_root / f"fold_{fold}"
        per_track = {}
        for track in TRACKS:
            protocol = load_protocol(fold_dir / f"{track}_pairs.csv", fold_dir / f"{track}_ground_truth.csv")
            face_idx = np.asarray(protocol.face_indices, dtype=np.int64)
            voice_idx = np.asarray(protocol.voice_indices, dtype=np.int64)
            per_track[track] = {"labels": np.asarray(protocol.labels), "voice_id": identities[voice_idx],
                                "face_idx": face_idx, "voice_idx": voice_idx, "scores": {"a": [], "b": []}}
        for run, pattern in (("a", args.run_a), ("b", args.run_b)):
            for seed in args.seeds:
                path = Path(pattern.format(fold=fold, seed=seed))
                config = torch.load(path, map_location="cpu", weights_only=False)["config"]
                voice = voice_inputs(config, Vt, args.voice_csv)
                face = face_inputs(config, Ft, args.face_csv)
                model = load_model(path, face.shape[1], voice)
                with torch.no_grad():
                    fe, ve = model.face_branch(face), model.voice_branch(voice)
                    for t in per_track.values():
                        t["scores"][run].append(model.pair_similarity(
                            fe[t["face_idx"]], ve[t["voice_idx"]]).numpy())
        trials.append(per_track)

    def summary(weights_for):
        out = {}
        for run in ("a", "b"):
            for track in TRACKS:
                out[(run, track)] = float(np.mean([
                    weighted_eer(f[track]["labels"], s, weights_for(k, track))
                    for k, f in enumerate(trials) for s in f[track]["scores"][run]]))
        return out

    point = summary(lambda k, track: np.ones(len(trials[k][track]["labels"])))
    rng = np.random.default_rng(0)
    diffs = {track: [] for track in (*TRACKS, "mean")}
    for _ in range(args.bootstrap):
        counts = []
        for f in trials:
            ids = np.unique(f["standard"]["voice_id"])
            drawn, n = np.unique(rng.choice(ids, len(ids)), return_counts=True)
            counts.append(dict(zip(drawn, n)))
        boot = summary(lambda k, track: np.array([counts[k].get(i, 0) for i in trials[k][track]["voice_id"]], float))
        for track in TRACKS:
            diffs[track].append(boot[("b", track)] - boot[("a", track)])
        diffs["mean"].append(np.mean([boot[("b", t)] - boot[("a", t)] for t in TRACKS]))

    per_fold = {run: {track: [round(float(np.mean([weighted_eer(f[track]["labels"], s, np.ones(len(s)))
                                                       for s in f[track]["scores"][run]])), 4) for f in trials]
                      for track in TRACKS} for run in ("a", "b")}
    report = {"run_a": args.run_a, "run_b": args.run_b, "seeds": args.seeds}
    for track in (*TRACKS, "mean"):
        a = np.mean([point[("a", t)] for t in TRACKS]) if track == "mean" else point[("a", track)]
        b = np.mean([point[("b", t)] for t in TRACKS]) if track == "mean" else point[("b", track)]
        lo, hi = np.percentile(diffs[track], [2.5, 97.5])
        report[track] = {"a": round(float(a), 4), "b": round(float(b), 4), "b_minus_a": round(float(b - a), 4),
                         "ci95": [round(float(lo), 4), round(float(hi), 4)],
                         "p_b_better": round(float(np.mean(np.array(diffs[track]) < 0)), 3)}
    report["per_fold_eer"] = per_fold
    text = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
