#!/usr/bin/env python3
"""Turn a VoxCeleb2 mp4 archive into an official-format training manifest.

VoxCeleb2 clips are 224x224 face-track videos (25 fps) with 16 kHz mono audio.
Per identity, up to ``--clips-per-identity`` clips are chosen round-robin over
its videos (so many videos are covered). For each chosen clip:

* the voice is the clip's own audio, written as 16 kHz mono PCM wav;
* the face is the middle frame of a clip from a *different* video of the same
  identity (when the identity has more than one video), mirroring MAV-Celeb's
  pairing so the model cannot match faces and voices through a shared
  recording.

Clips are read straight from the zip; the zip stays as the raw copy.
Manifest rows (6 columns, like ``train_English.txt``)::

    <pair_id> 1 voices/Mixed/<identity>/<video>_<clip>.wav faces/Mixed/<identity>/<video>_<clip>.jpg <identity> <class>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True, help="vox2_meta.csv (identity, gender)")
    parser.add_argument("--prefix", default="vox2")
    parser.add_argument("--class-offset", type=int, default=3000)
    parser.add_argument("--clips-per-identity", type=int, default=80)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gender-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    archive = zipfile.ZipFile(args.zip)
    clips = defaultdict(lambda: defaultdict(list))
    for name in archive.namelist():
        if name.endswith(".mp4"):
            identity, video, clip = name.split("/")[-3:]
            clips[identity][video].append(name)
    gender = {}
    for line in args.meta.open(encoding="utf-8-sig"):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 3 and fields[2] in ("m", "f"):
            gender[fields[0]] = fields[2]

    rng = np.random.default_rng(args.seed)
    jobs, rows = [], []
    for k, identity in enumerate(sorted(clips)):
        videos = {v: sorted(c) for v, c in clips[identity].items()}
        order = list(rng.permutation(sorted(videos)))
        for v in order:
            rng.shuffle(videos[v])
        chosen, depth = [], 0
        while len(chosen) < args.clips_per_identity and any(len(videos[v]) > depth for v in order):
            for v in order:
                if depth < len(videos[v]) and len(chosen) < args.clips_per_identity:
                    chosen.append((v, videos[v][depth]))
            depth += 1
        name = f"{args.prefix}_{identity}"
        for i, (video, member) in enumerate(chosen):
            others = [v for v in order if v != video] or [video]
            face_video = others[i % len(others)]
            face_member = videos[face_video][i % len(videos[face_video])]
            stem = Path(member).stem
            voice_rel = f"voices/Mixed/{name}/{video}_{stem}.wav"
            face_rel = f"faces/Mixed/{name}/{face_video}_{Path(face_member).stem}.jpg"
            jobs.append((member, face_member, voice_rel, face_rel))
            rows.append(f"{args.prefix}_{len(rows):06d} 1 {voice_rel} {face_rel} {name} {args.class_offset + k}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    # zipfile reads are not thread-safe on one handle; give each worker its own.
    local = {}

    def safe(job):
        tid = threading.get_ident()
        if tid not in local:
            local[tid] = zipfile.ZipFile(args.zip)
        return work_with(local[tid], job)

    def work_with(handle, job):
        member, face_member, voice_rel, face_rel = job
        voice_out, face_out = args.work_dir / voice_rel, args.work_dir / face_rel
        with tempfile.TemporaryDirectory(dir=args.work_dir) as tmp:
            if not voice_out.exists():
                src = Path(tmp) / "voice.mp4"
                src.write_bytes(handle.read(member))
                voice_out.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(voice_out)])
            if not face_out.exists():
                src = Path(tmp) / "face.mp4"
                src.write_bytes(handle.read(face_member))
                duration = float(subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                    capture_output=True, text=True, check=True).stdout.strip() or 0)
                face_out.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg(["-ss", f"{duration / 2:.2f}", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(face_out)])

    with ThreadPoolExecutor(args.workers) as pool:
        for done, _ in enumerate(pool.map(safe, jobs), start=1):
            if done % 1000 == 0:
                print(f"{done}/{len(jobs)}", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(rows) + "\n")
    names = sorted({r.split()[4] for r in rows})
    missing_gender = [n for n in names if n[len(args.prefix) + 1:] not in gender]
    args.gender_csv.write_text("".join(f"{n},{gender[n[len(args.prefix) + 1:]]}\n" for n in names
                                       if n not in missing_gender))
    report = {"zip": str(args.zip), "identities": len(names), "rows": len(rows),
              "clips_per_identity": args.clips_per_identity, "missing_gender": missing_gender,
              "male": sum(gender.get(n[len(args.prefix) + 1:]) == "m" for n in names),
              "female": sum(gender.get(n[len(args.prefix) + 1:]) == "f" for n in names)}
    Path(f"{args.manifest}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
