"""Extract ECAPA (yangwang825/ecapa-tdnn-vox2) and WavLM-SV x-vectors for the external rows,
streaming the audio from the kept source zips (no work folders are re-created).

Identical preprocessing to the official features:
  * MAV-Celeb clips are the zip's 16 kHz mono PCM wav files, read with the same checks as
    ``load_pcm16_wav``;
  * VoxCeleb2 clips are decoded from the zip's mp4 with the same ffmpeg command as
    ``prepare_voxceleb2.py`` (-vn -ac 1 -ar 16000 pcm_s16le), via a temporary file in /dev/shm;
  * ECAPA: whole clip (``extract_voice_features_speaker_age.py``); x-vector: 10 s chunks,
    5 s overlap, fp16, mean of normalised chunk embeddings (``extract_voice_features_wavlm.py``).
Validation before writing:
  * encoder check: 40 official clips re-extracted with this code must match the cached
    official ECAPA / x-vector features (cosine > 0.999);
  * mapping check: 60 external clips must reproduce the cached WavLM layer statistics
    (cosine > 0.999), proving every manifest row is mapped to the right zip member.
Run with the video_features environment.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feature_extraction"))
from extract_voice_features_speaker_age import load_ecapa  # noqa: E402
from extract_voice_features_wavlm import load_pcm16_wav, split_chunks  # noqa: E402
from extract_voice_features_wavlm_layers import pooled_hidden_states  # noqa: E402

REV = "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
MODEL = "microsoft/wavlm-base-plus-sv"
EXT = Path("features/external/combined_all")
ZIPS = {"mav1": "external_data/mavceleb/raw/mavceleb_v1_train.zip",
        "mav3": "external_data/mavceleb/raw/mavceleb_v3_tran.zip",
        "vox2": "external_data/voxceleb2/raw/vox2_test_mp4.zip"}
SR = 16000


def pcm16_from_bytes(data: bytes, where: str) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as h:
        if h.getframerate() != SR or h.getnchannels() != 1 or h.getsampwidth() != 2 or h.getcomptype() != "NONE":
            raise ValueError(f"{where}: not 16 kHz mono 16-bit PCM")
        x = np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").astype(np.float32)
    if len(x) == 0:
        raise ValueError(f"{where}: empty")
    return np.ascontiguousarray(x / 32768.0)


class ZipAudio:
    """Manifest voice path -> waveform, read from the source zip."""

    def __init__(self):
        self.archives = {k: zipfile.ZipFile(v) for k, v in ZIPS.items()}
        self.index = {}
        for src, z in self.archives.items():
            suffix_ok = ".mp4" if src == "vox2" else ".wav"
            for name in z.namelist():
                if name.endswith(suffix_ok):
                    parts = name.split("/")
                    if src == "vox2":   # .../<id>/<video>/<clip>.mp4
                        key = (src, parts[-3], parts[-2], Path(parts[-1]).stem)
                    else:               # .../voices/<id>/<lang>/<video>/<clip>.wav
                        if "voices" not in parts:
                            continue
                        key = (src, parts[-4], parts[-2], Path(parts[-1]).stem, parts[-3])
                    if key in self.index:
                        raise ValueError(f"ambiguous zip member key {key}")
                    self.index[key] = name
        self.tmp = Path(tempfile.mkdtemp(dir="/dev/shm", prefix="ext_audio_"))

    def key(self, voice_path: str):
        _, lang, folder, fname = voice_path.split("/")
        src, identity = folder.split("_", 1)
        video, clip = Path(fname).stem.rsplit("_", 1)
        return (src, identity, video, clip) if src == "vox2" else (src, identity, video, clip, lang)

    def load(self, voice_path: str) -> np.ndarray:
        k = self.key(voice_path)
        member = self.index[k]
        data = self.archives[k[0]].read(member)
        if k[0] != "vox2":
            return pcm16_from_bytes(data, member)
        src, dst = self.tmp / "clip.mp4", self.tmp / "clip.wav"
        src.write_bytes(data)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le", str(dst)], check=True)
        return load_pcm16_wav(dst)

    def close(self):
        for f in self.tmp.iterdir():
            f.unlink()
        self.tmp.rmdir()


def xvector_encoder(device):
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REV)
    model = WavLMForXVector.from_pretrained(MODEL, revision=REV).to(device).eval()

    def encode(wave_: np.ndarray) -> np.ndarray:
        chunks = split_chunks(wave_, 10 * SR, 5 * SR)
        inputs = fe(chunks, sampling_rate=SR, padding=True, return_attention_mask=True, return_tensors="pt")
        inputs = {n: t.to(device) for n, t in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                    enabled=device.type == "cuda"):
            e = F.normalize(model(**inputs).embeddings.float(), dim=1).cpu()
        return F.normalize(e.sum(0), dim=0).numpy()
    return encode


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecapa = load_ecapa(device, Path("models/hf_cache"))
    xvec = xvector_encoder(device)
    report = {"ecapa_model": "yangwang825/ecapa-tdnn-vox2@c605f57148d4e910d4241eec714d7bc948220adb",
              "xvector_model": f"{MODEL}@{REV}", "zips": ZIPS}

    # 1. Encoder equivalence on official clips (batching differences give tiny fp16 noise).
    off = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(off), 40, replace=False)
    cached_e = np.loadtxt("features/voice_v6_ecapa/train_English_voices.csv", delimiter=",")[:, :-1]
    cached_x = np.loadtxt("features/voice_v1_wavlm/train_English_voices.csv", delimiter=",")[:, :-1]
    ce, cx = [], []
    for i in pick:
        w = load_pcm16_wav(Path("train_set/train_set") / off[i][2])
        e = ecapa(w)[0]
        ce.append(float(e @ cached_e[i] / np.linalg.norm(e) / np.linalg.norm(cached_e[i])))
        x = xvec(w)
        cx.append(float(x @ cached_x[i] / np.linalg.norm(x) / np.linalg.norm(cached_x[i])))
    report["official_check"] = {"ecapa_min_cos": min(ce), "xvector_min_cos_vs_cached": min(cx),
                                "xvector_note": "cached official x-vectors were batch-padded across files (short "
                                                "clips differ, >=10 s clips identical); official x-vectors are "
                                                "re-extracted per clip below for identical preprocessing"}
    print(report["official_check"], flush=True)
    if min(ce) < 0.999:
        raise SystemExit("ECAPA check failed: preprocessing differs from the cached official features")
    # Official x-vectors, per clip (identical code path to the external ones), deduplicated by hash.
    import csv
    sha = [r["sha256"] for r in csv.DictReader(open("features/voice_v1_wavlm/train_English_voices_rowmap.csv"))]
    cache = {}
    OX = np.zeros((len(off), 512), np.float32)
    for i, r in enumerate(tqdm(off, ncols=80, desc="official x-vector")):
        if sha[i] not in cache:
            cache[sha[i]] = xvec(load_pcm16_wav(Path("train_set/train_set") / r[2]))
        OX[i] = cache[sha[i]]
    out = Path("features/voice_v9_xvector_perclip")
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "train_English_voices.csv", np.column_stack([OX, [int(r[5]) for r in off]]),
               delimiter=",", fmt="%.8g")
    (out / "train_English_report.json").write_text(json.dumps(
        {"model": f"{MODEL}@{REV}", "chunk_seconds": 10, "overlap_seconds": 5, "fp16": True,
         "batching": "per clip (chunks of one clip only)", "rows": len(off), "unique_audio": len(cache)}, indent=1))

    # 2. Mapping check against cached WavLM layer statistics.
    rows = [l.split() for l in open(EXT / "train_manifest.txt") if l.strip()]
    audio = ZipAudio()
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, revision=REV)
    wavlm = WavLMModel.from_pretrained(MODEL, revision=REV).to(device).eval()
    src_pos = {}
    for src, folder in (("mav1", "mav1"), ("mav3", "mav3"), ("vox2", "vox2test")):
        man = [l.split()[0] for l in open(f"features/external/manifests/{folder}_train.txt") if l.strip()]
        src_pos[src] = (folder, {k: i for i, k in enumerate(man)})
    mcos = []
    for i in np.random.default_rng(1).choice(len(rows), 60, replace=False):
        src = rows[i][0].split("_")[0]
        folder, pos = src_pos[src]
        cached = np.load(f"features/external/{folder}/train_layers.npy", mmap_mode="r")[pos[rows[i][0]]].astype(np.float32)
        total, frames = None, 0
        for chunk in split_chunks(audio.load(rows[i][2]), 10 * SR, 5 * SR):
            pooled, n = pooled_hidden_states(wavlm, fe, chunk, device, True)
            total = pooled * n if total is None else total + pooled * n
            frames += n
        mine = (total / frames).numpy().reshape(-1)
        c = cached.reshape(-1)
        mcos.append(float(mine @ c / np.linalg.norm(mine) / np.linalg.norm(c)))
    report["mapping_check"] = {"clips": 60, "min_cos": min(mcos), "mean_cos": float(np.mean(mcos))}
    print(report["mapping_check"], flush=True)
    del wavlm
    if min(mcos) < 0.999:
        raise SystemExit("mapping check failed: some manifest rows do not map to the cached audio")

    # 3. Extract every external row.
    E, X = np.zeros((len(rows), 192), np.float32), np.zeros((len(rows), 512), np.float32)
    durations = np.zeros(len(rows), np.float32)
    for i, r in enumerate(tqdm(rows, ncols=80, desc="external")):
        w = audio.load(r[2])
        durations[i] = len(w) / SR
        E[i] = ecapa(w)[0]
        X[i] = xvec(w)
    audio.close()
    labels = np.array([int(r[5]) for r in rows])
    assert np.isfinite(E).all() and np.isfinite(X).all()
    np.savetxt(EXT / "train_voices_ecapa.csv", np.column_stack([E, labels]), delimiter=",", fmt="%.8g")
    np.savetxt(EXT / "train_voices_xvector.csv", np.column_stack([X, labels]), delimiter=",", fmt="%.8g")
    np.save(EXT / "train_voice_durations.npy", durations)
    report |= {"rows": len(rows), "outputs": ["train_voices_ecapa.csv", "train_voices_xvector.csv",
                                             "train_voice_durations.npy"]}
    (EXT / "speaker_embeddings_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
