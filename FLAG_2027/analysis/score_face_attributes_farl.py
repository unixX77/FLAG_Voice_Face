"""Zero-shot attribute scores for stored FaRL face embeddings via the FaRL text encoder.

Each attribute is the difference between the mean cosine similarity to a set of
"high" prompts and to a set of "low" prompts. Age and gender are scored the same
way, so they can validate the method against InsightFace ages and metadata.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch

ATTRIBUTES = {
    "build": (
        ["a photo of a fat person.", "a photo of an overweight person.", "a photo of a person with a chubby face.",
         "a photo of a heavy-set person.", "a photo of a person with a round, full face."],
        ["a photo of a thin person.", "a photo of a slim person.", "a photo of a person with a skinny face.",
         "a photo of a lean person.", "a photo of a person with a narrow, slender face."],
    ),
    "age": (
        ["a photo of an old person.", "a photo of an elderly person.", "a photo of a middle-aged person."],
        ["a photo of a young person.", "a photo of a person in their twenties.", "a photo of a youthful person."],
    ),
    "male": (
        ["a photo of a man.", "a photo of a male face."],
        ["a photo of a woman.", "a photo of a female face."],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-csv", type=Path, required=True, help="FaRL features with label column last")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    model, _ = clip.load("ViT-B/16", device="cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    faces = pd.read_csv(args.face_csv, header=None, dtype=np.float32).to_numpy()[:, :-1]
    faces /= np.linalg.norm(faces, axis=1, keepdims=True)

    columns = {}
    with torch.no_grad():
        for name, (high, low) in ATTRIBUTES.items():
            means = []
            for prompts in (high, low):
                text = model.encode_text(clip.tokenize(prompts)).float()
                text = text / text.norm(dim=1, keepdim=True)
                means.append(faces @ text.numpy().T)
            columns[name] = means[0].mean(1) - means[1].mean(1)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *columns])
        for i in range(len(faces)):
            writer.writerow([i, *(f"{columns[k][i]:.6f}" for k in columns)])
    print(f"rows={len(faces)} -> {args.output_csv}")


if __name__ == "__main__":
    main()
