#!/usr/bin/env python3
"""Extract frozen speaker-verification and age/gender voice features for a manifest.

Two public pretrained models, used fully frozen (allowed by the FLAG 2027 rules):

* ``--model ecapa``: SpeechBrain ECAPA-TDNN trained on VoxCeleb2
  (``yangwang825/ecapa-tdnn-vox2``, Apache-2.0). Writes one 192-d speaker
  embedding per clip to ``<output-prefix>_voices.csv``.
* ``--model agegender``: audeering ``wav2vec2-large-robust-24-ft-age-gender``
  (CC BY-NC-SA 4.0; non-commercial). Writes the 1024-d mean-pooled hidden
  state to ``<output-prefix>_voices.csv``, and the predicted age (years) with
  female/male/child probabilities to ``<output-prefix>_attributes.csv``.

Every clip is loaded with the same 16 kHz mono PCM loader as the WavLM features
and encoded independently as a whole clip (the models' standard inference).
Training manifests get the class label appended as the last CSV column, as in
the other voice feature files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from extract_voice_features_wavlm import load_pcm16_wav, read_manifest, resolve_audio_paths

MODELS = {
    "ecapa": ("yangwang825/ecapa-tdnn-vox2", "c605f57148d4e910d4241eec714d7bc948220adb"),
    "agegender": ("audeering/wav2vec2-large-robust-24-ft-age-gender", "bad2737f529168bc3d3fcd97edfaba9583c7d36d"),
}


def load_ecapa(device: torch.device, cache_dir: Path):
    from speechbrain.inference.interfaces import Pretrained

    class Encoder(Pretrained):
        """From the model card (same wrapper as extract_voice_features_ecapa.py)."""

        MODULES_NEEDED = ["compute_features", "mean_var_norm", "embedding_model"]

        def encode(self, wav: torch.Tensor) -> torch.Tensor:
            wav = wav.unsqueeze(0).to(self.device).float()
            lengths = torch.ones(1, device=self.device)
            feats = self.mods.compute_features(wav)
            feats = self.mods.mean_var_norm(feats, lengths)
            return self.mods.embedding_model(feats, lengths).reshape(-1)

    source, revision = MODELS["ecapa"]
    from huggingface_hub import snapshot_download

    local = snapshot_download(source, revision=revision, cache_dir=str(cache_dir))
    # The model's hyperparams.yaml points its checkpoints at the hub; point them at
    # the pinned local snapshot instead.
    model = Encoder.from_hparams(source=local, savedir=str(cache_dir / "ecapa_savedir"),
                                 overrides={"pretrained_path": local},
                                 run_opts={"device": str(device)})
    model.eval()

    def encode(wave: np.ndarray):
        with torch.inference_mode():
            return model.encode(torch.from_numpy(wave)).cpu().numpy(), None

    return encode


def load_agegender(device: torch.device, cache_dir: Path):
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class ModelHead(nn.Module):
        """Classification head, from the model card."""

        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            return self.out_proj(torch.tanh(self.dense(self.dropout(x))))

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        """Age/gender model, from the model card."""

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

    source, revision = MODELS["agegender"]
    processor = Wav2Vec2Processor.from_pretrained(source, revision=revision, cache_dir=str(cache_dir))
    model = AgeGenderModel.from_pretrained(source, revision=revision, cache_dir=str(cache_dir),
                                           use_safetensors=True).to(device).eval()

    def encode(wave: np.ndarray):
        values = processor(wave, sampling_rate=16000)["input_values"][0].reshape(1, -1)
        with torch.inference_mode():
            hidden, age, gender = model(torch.from_numpy(values).to(device))
        # The age head outputs approximately 0..1 for 0..100 years.
        attributes = np.concatenate([[float(age.reshape(-1)[0]) * 100.0], gender.reshape(-1).cpu().numpy()])
        return hidden.reshape(-1).cpu().numpy(), attributes

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True,
                        help="e.g. features/voice_v6_ecapa/train_English")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/hf_cache"))
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    paths = resolve_audio_paths(rows, args.audio_root)
    encode = (load_ecapa if args.model == "ecapa" else load_agegender)(device, args.cache_dir)

    features, attributes = [], []
    for path in tqdm(paths, desc=f"Encoding ({args.model})", ncols=80):
        feature, attribute = encode(load_pcm16_wav(path))
        features.append(feature)
        if attribute is not None:
            attributes.append(attribute)
    matrix = np.stack(features).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite voice feature")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([row.class_index for row in rows], dtype=np.int64) if is_training else None
    voices_csv = Path(f"{args.output_prefix}_voices.csv")
    np.savetxt(voices_csv, matrix if labels is None else np.column_stack((matrix, labels)),
               delimiter=",", fmt="%.8g")
    if attributes:
        with open(f"{args.output_prefix}_attributes.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "age_years", "p_female", "p_male", "p_child"])
            for index, values in enumerate(attributes):
                writer.writerow([index, *(f"{value:.6f}" for value in values)])
    source, revision = MODELS[args.model]
    report = {"model": source, "revision": revision, "manifest": str(args.manifest),
              "audio_root": str(args.audio_root), "rows": len(rows), "feature_dim": matrix.shape[1],
              "torch": torch.__version__, "device": str(device)}
    Path(f"{args.output_prefix}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
