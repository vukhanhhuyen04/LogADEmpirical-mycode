"""Bước 10: gộp nhiều seed -> trung bình ± độ lệch chuẩn (results_seeds.md).

  python 10_aggregate.py --runs runs/regex_seed42 runs/regex_seed1 runs/regex_seed2
"""
import argparse
import glob
import os

import numpy as np

from common import load_json

ORDER = ["C0", "C1", "C2", "C3", "C4", "C4A", "C5", "C5A", "C5_C1", "C5M", "C5M_C1", "C6", "C6_C0", "RF_mac_dinh", "RF_A", "IF_A", "IF_C"]
COLS = [("Precision", 100), ("Recall", 100), ("F1", 100), ("FP", 1), ("FN", 1)]


def ms(v, scale):
    v = np.asarray(v, dtype=float) * scale
    return (f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}").replace(".", ",") if scale != 1 \
        else (f"{v.mean():,.0f} ± {v.std(ddof=1):,.0f}" if len(v) > 1 else f"{v.mean():,.0f}").replace(",", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results_seeds.md")
    a = ap.parse_args()
    data = {}
    for r in a.runs:
        for f in glob.glob(os.path.join(r, "metrics_*.json")):
            data.setdefault(os.path.basename(f)[8:-5], []).append(load_json(f)["test"]["all"])
    names = [k for k in ORDER if k in data] + sorted(k for k in data if k not in ORDER)
    L = [f"# Kết quả trên {len(a.runs)} seed ({', '.join(os.path.basename(r) for r in a.runs)})\n",
         "| Cấu hình | Số seed | P (%) | R (%) | F1 (%) | FP | FN |", "|---|---|---|---|---|---|---|"]
    for k in names:
        rows = data[k]
        L.append(f"| {k} | {len(rows)} | " + " | ".join(ms([m[c] for m in rows], s) for c, s in COLS) + " |")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
