"""Bước 13 (v4): C5M - ngưỡng trượt bền: tau_t = trung vị + k * MAD của N cửa sổ trước đó.
MAD = trung vị của |s - trung vị|. Thống kê bền ít bị kéo theo đợt bất thường dồn dập hay
các giá trị trần hơn phân vị (sửa hai hiện tượng của C5). (N, k) chọn trên validation.

  python 13_robust_threshold.py --run runs/regex_seed42 --src C4 --name C5M
  python 13_robust_threshold.py --run runs/regex_seed42 --src C1 --name C5M_C1 --fixed C3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import fmt_row, load_json, metrics, metrics_by_oov, save_json

N_GRID = [500, 1000, 2000, 5000]
K_GRID = [1, 2, 3, 5, 8, 12]
MIN_HIST = 200


def robust_tau(history, scores, n, k, fallback):
    s = pd.Series(np.concatenate([history[-n:], scores])).shift(1)
    med = s.rolling(n, min_periods=MIN_HIST).median()
    mad = (s - med).abs().rolling(n, min_periods=MIN_HIST).median()   # xấp xỉ MAD trượt
    tau = (med + k * mad).to_numpy()[-len(scores):]
    return np.where(np.isnan(tau), fallback, tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src", default="C4")
    ap.add_argument("--name", default="C5M")
    ap.add_argument("--fixed", default=None)
    ap.add_argument("--delta", type=float, default=0.02)
    a = ap.parse_args()
    v = pd.read_csv(os.path.join(a.run, f"windows_val_{a.src}.csv"))
    t = pd.read_csv(os.path.join(a.run, f"windows_test_{a.src}.csv"))
    fixed_name = a.fixed or a.src
    fallback = load_json(os.path.join(a.run, f"metrics_{fixed_name}.json")).get("tau", float(np.quantile(v.score, .99)))
    r0 = load_json(os.path.join(a.run, "metrics_C0.json"))["val"]["Recall"]
    grid = []
    for n in N_GRID:
        for k in K_GRID:
            tau = robust_tau(np.array([]), v.score.to_numpy(), n, k, fallback)
            grid.append(dict(N_hist=n, k=k, **metrics(v.label, v.score.to_numpy() >= tau)))
    g = pd.DataFrame(grid); g.to_csv(os.path.join(a.run, f"grid_{a.name}_val.csv"), index=False)
    ok = g[g.Recall >= r0 - a.delta]
    best = ok.sort_values(["FP", "F1"], ascending=[True, False]).iloc[0] if len(ok) else g.sort_values("Recall").iloc[-1]
    n, k = int(best.N_hist), float(best.k)
    tau_t = robust_tau(v.score.to_numpy(), t.score.to_numpy(), n, k, fallback)
    pred = t.score.to_numpy() >= tau_t
    out = {"config": a.name, "nguon_diem": a.src, "kieu_nguong": "truot_mad", "N_hist": n, "k": k,
           "tieu_chi": "B", "chon_tren": "validation", "tau_min": float(tau_t.min()),
           "tau_median": float(np.median(tau_t)), "tau_max": float(tau_t.max()),
           "val": {c: best[c] for c in ("TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Specificity")},
           "test": metrics_by_oov(t.label, pred, t.has_oov)}
    save_json(out, os.path.join(a.run, f"metrics_{a.name}.json"))
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=160)
    ax.plot(x, tau_t, color="#2a6fb0", lw=1, label=f"Trung vị + {k:g}·MAD (N = {n})")
    ax.axhline(fallback, color="#d1495b", ls="--", lw=1, label=f"Ngưỡng cố định ({fixed_name}) = {fallback:.2f}")
    ax.set_xlabel("Thứ tự cửa sổ trong tập test"); ax.set_ylabel("Ngưỡng τ"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.run, f"{a.name}_theo_thoi_gian.png")); plt.close()
    print(f"{a.name}: N={n}, k={k:g}; ngưỡng test từ {tau_t.min():.2f} đến {tau_t.max():.2f}")
    for c, m in out["test"].items():
        print(fmt_row(f"{a.name} test " + c, m))


if __name__ == "__main__":
    main()
