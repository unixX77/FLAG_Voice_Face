"""InsightFace ArcFace (buffalo_l) embeddings for the external face rows, streamed from the zips.

Identical pipeline to the official ``features/face_v1_insightface`` cache: the same
``encode_image`` (detection, padded retry, low-threshold retry, centre-crop fallback),
det_size 640, det_threshold 0.5, retry 0.3, failure policy centre_crop, insightface 0.7.3,
onnxruntime 1.23.2 (env ``face_features``).
Images: MAV-Celeb jpgs are read from the zip; VoxCeleb2 frames are re-extracted from the mp4
at duration/2 with the same ffprobe/ffmpeg commands as prepare_voxceleb2.py.
Checks before writing:
  * mapping: every image's SHA-256 must equal the one recorded when the cached FaRL features
    were extracted (per-source train_faces_rowmap.csv) - byte-identical inputs;
  * encoder: 40 official images must reproduce the cached official ArcFace (cos > 0.999).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feature_extraction"))
from extract_face_features_insightface import encode_image  # noqa: E402

EXT = Path("features/external/combined_all")
ZIPS = {"mav1": "external_data/mavceleb/raw/mavceleb_v1_train.zip",
        "mav3": "external_data/mavceleb/raw/mavceleb_v3_tran.zip",
        "vox2": "external_data/voxceleb2/raw/vox2_test_mp4.zip"}


class ZipImages:
    def __init__(self):
        self.z = {k: zipfile.ZipFile(v) for k, v in ZIPS.items()}
        self.index = {}
        for src, z in self.z.items():
            for name in z.namelist():
                parts = name.split("/")
                if src == "vox2" and name.endswith(".mp4"):
                    self.index[(src, parts[-3], parts[-2], Path(parts[-1]).stem)] = name
                elif src != "vox2" and name.endswith(".jpg") and "faces" in parts:
                    # .../faces/<id>/<lang>/<video>/<frame>.jpg; the same video can sit under two
                    # language folders with different frames of the same name, so the key needs <lang>.
                    key = (src, parts[-4], parts[-2], Path(parts[-1]).stem, parts[-3])
                    if key in self.index:
                        raise ValueError(f"ambiguous zip member key {key}")
                    self.index[key] = name
        self.tmp = Path(tempfile.mkdtemp(dir="/dev/shm", prefix="ext_img_"))

    def load_bytes(self, face_path: str) -> bytes:
        _, lang, folder, fname = face_path.split("/")
        src, identity = folder.split("_", 1)
        video, frame = Path(fname).stem.rsplit("_", 1)
        member = self.index[(src, identity, video, frame) if src == "vox2" else (src, identity, video, frame, lang)]
        data = self.z[src].read(member)
        if src != "vox2":
            return data
        mp4, jpg = self.tmp / "face.mp4", self.tmp / "face.jpg"
        mp4.write_bytes(data)
        duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                         "csv=p=0", str(mp4)], capture_output=True, text=True, check=True
                                        ).stdout.strip() or 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{duration / 2:.2f}", "-i", str(mp4), "-frames:v",
                        "1", "-q:v", "2", str(jpg)], check=True)
        return jpg.read_bytes()


def main() -> None:
    import cv2
    import onnxruntime
    from insightface.app import FaceAnalysis
    if hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls(directory="")
    app = FaceAnalysis(name="buffalo_l", root=str(Path.home() / ".insightface"),
                       allowed_modules=["detection", "recognition"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

    def embed(data: bytes):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return encode_image(app, image, cv2, "center_crop", 0.3)

    report = {"model": "insightface buffalo_l (w600k_r50)", "det_size": 640, "det_threshold": 0.5,
              "retry_det_threshold": 0.3, "failure_policy": "center_crop"}
    # Encoder check on official images.
    off = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    cached = np.loadtxt("features/face_v1_insightface/train_English_faces.csv", delimiter=",")[:, :-1]
    cos = []
    for i in np.random.default_rng(0).choice(len(off), 40, replace=False):
        e = embed(Path("train_set/train_set", off[i][3]).read_bytes()).embedding
        cos.append(float(e @ cached[i] / np.linalg.norm(cached[i])))
    report["official_check_min_cos"] = min(cos)
    print("official check", min(cos), flush=True)
    if min(cos) < 0.999:
        raise SystemExit("encoder check failed")
    # Expected image hashes from the FaRL extraction (per-source row maps), keyed by pair id.
    expected = {}
    for folder in ("mav1", "mav3", "vox2test"):
        for r in csv.DictReader(open(f"features/external/{folder}/train_faces_rowmap.csv")):
            expected[r["pair_id"]] = r["sha256"]
    rows = [l.split() for l in open(EXT / "train_manifest.txt") if l.strip()]
    imgs = ZipImages()
    E = np.zeros((len(rows), 512), np.float32)
    status, det = [], np.full(len(rows), np.nan, np.float32)
    mismatched, cache = 0, {}
    for i, r in enumerate(tqdm(rows, ncols=80, desc="external faces")):
        data = imgs.load_bytes(r[3])
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected[r[0]]:
            mismatched += 1
            if mismatched <= 5:
                print("hash mismatch", r[0], r[3], flush=True)
        if digest not in cache:
            rec = embed(data)
            cache[digest] = (rec.embedding, rec.status, rec.detection_score)
        E[i], st, ds = cache[digest]
        status.append(st)
        det[i] = np.nan if ds is None else ds
    for f in imgs.tmp.iterdir():
        f.unlink()
    imgs.tmp.rmdir()
    report |= {"rows": len(rows), "unique_images": len(cache), "hash_mismatches_vs_farl_inputs": mismatched,
               "status_counts": {s: status.count(s) for s in sorted(set(status))}}
    print(json.dumps(report), flush=True)
    if mismatched:
        raise SystemExit("mapping check failed: some images differ from the FaRL extraction inputs")
    labels = np.array([int(r[5]) for r in rows])
    np.savetxt(EXT / "train_faces_arcface.csv", np.column_stack([E, labels]), delimiter=",", fmt="%.8g")
    np.save(EXT / "train_faces_arcface_detscore.npy", det)
    (EXT / "arcface_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
