"""Plot pooled genuine/impostor score histograms (raw cosine) from evaluate_gamp.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    d = json.loads(args.diagnostics.read_text())["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = ["#2a6fdb", "#e0732c", "#2b9e6b", "#a04fc2"]
    for ax, track in zip(axes, ("standard", "gender")):
        for c, color in zip(args.configs, colors):
            t = d[c][track]
            bins = np.asarray(t["hist_bins"])
            mid = 0.5 * (bins[1:] + bins[:-1])
            for kind, style in (("genuine", "-"), ("impostor", "--")):
                h = np.asarray(t[f"hist_{kind}"], float)
                ax.plot(mid, h / h.sum(), style, color=color, label=f"{c} {kind} (EER {t['eer']*100:.1f})"
                        if kind == "genuine" else f"{c} {kind}")
        ax.set_title(f"{track} trials (pooled folds)")
        ax.set_xlabel("raw cosine score")
        ax.set_xlim(-0.2, 1.0)
    axes[0].set_ylabel("share of trials")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
