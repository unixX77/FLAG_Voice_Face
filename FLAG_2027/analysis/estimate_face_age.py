"""Estimate apparent age and gender for every face in a manifest with InsightFace.

Uses the buffalo_l detector and genderage model. The largest detected face is
used; images without a detection are padded and retried once, and rows that
still fail are written with empty values.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    rows = [line.split() for line in args.manifest.read_text().splitlines() if line.strip()]
    failures = 0
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "key", "identity", "age", "gender_pred", "det_score"])
        for index, row in enumerate(tqdm(rows, ncols=80)):
            # Training manifests: key label voice face identity class; trial lists: key voice face.
            face_path, identity = (row[3], row[4]) if len(row) == 6 else (row[2], "")
            image = cv2.imread(str(args.image_root / face_path))
            faces = app.get(image)
            if not faces:
                pad = max(image.shape[:2]) // 2
                faces = app.get(cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT))
            if not faces:
                failures += 1
                writer.writerow([index, row[0], identity, "", "", ""])
                continue
            face = max(faces, key=lambda f: float(np.prod(f.bbox[2:] - f.bbox[:2])))
            writer.writerow([index, row[0], identity, int(face.age),
                             "m" if int(face.gender) == 1 else "f", round(float(face.det_score), 3)])
    print(f"rows={len(rows)} failures={failures} -> {args.output_csv}")


if __name__ == "__main__":
    main()
