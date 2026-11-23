"""Summarise the learning curve: final-epoch (90) EER versus training identities.

Reads runs/learning_curve/ids_{N}/fold_{f}_seed_{s}/training_log.csv and the
56-identity reference runs/voice_aug/base. Only (fold, seed) pairs finished at
every size are used, so the sizes are compared on the same runs. Fits
EER = a + b * log(N) per track and extrapolates.
"""
import csv, sys
from pathlib import Path
import numpy as np

SIZES = {20: "runs/learning_curve/ids_20", 28: "runs/learning_curve/ids_28",
         42: "runs/learning_curve/ids_42", 56: "runs/voice_aug/base"}

def final(path):
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 90:
        return None
    r = rows[89]
    return float(r["standard_eer"]), float(r["gender_eer"])

res = {n: {} for n in SIZES}
for n, root in SIZES.items():
    for f in range(5):
        for s in (1, 2, 3):
            p = Path(root) / f"fold_{f}_seed_{s}" / "training_log.csv"
            if p.exists() and (v := final(p)):
                res[n][(f, s)] = v
common = set.intersection(*(set(r) for r in res.values()))
print(f"(fold, seed) pairs finished at every size: {len(common)}")
if not common:
    sys.exit()
xs, table = [], {}
for n in SIZES:
    a = np.array([res[n][k] for k in sorted(common)])
    table[n] = a.mean(0)
    print(f"N={n:3d}  standard {100*table[n][0]:5.2f}  gender {100*table[n][1]:5.2f}  mean {100*table[n].mean():5.2f}")
logn = np.log(list(SIZES))
for i, name in enumerate(("standard", "gender")):
    y = np.array([table[n][i] for n in SIZES])
    b, a = np.polyfit(logn, y, 1)
    print(f"{name}: EER change per doubling of identities = {100*b*np.log(2):+.2f} points")
    for target_n in (70, 184, 500, 1000, 6000):
        print(f"   extrapolated N={target_n:5d}: {100*(a + b*np.log(target_n)):5.2f}")
