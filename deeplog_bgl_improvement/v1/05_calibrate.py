"""Bước 5: Hiệu chỉnh ngưỡng thích nghi trên phân bố điểm của VALIDATION, khóa ngưỡng rồi chấm test.
  C2 = điểm của C0 + ngưỡng hiệu chỉnh;  C3 = điểm của C1 + ngưỡng hiệu chỉnh.

  python 05_calibrate.py --run runs/regex_seed42 --src C0 --name C2 --criterion B
  python 05_calibrate.py --run runs/regex_seed42 --src C1 --name C3 --criterion B
Tiêu chí: A = F1 lớn nhất; B = FP nhỏ nhất, Recall_val >= Recall_val(C0) - delta; C = ngân sách FPR.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import choose_threshold, fmt_row, load_json, metrics, metrics_by_oov, save_json


def plots(yv, sv, tau, title, path_prefix):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    bins = np.linspace(sv.min(), sv.max(), 60)
    ax.hist(sv[yv == 0], bins=bins, alpha=.6, label="Bình thường", color="#2a6fb0")
    ax.hist(sv[yv == 1], bins=bins, alpha=.6, label="Bất thường", color="#d1495b")
    ax.axvline(tau, color="k", ls="--", label=f"Ngưỡng τ = {tau:.3f}")
    ax.set_yscale("log"); ax.set_xlabel("Điểm bất thường của cửa sổ"); ax.set_ylabel("Số cửa sổ (log)")
    ax.set_title(title); ax.legend(frameon=False); plt.tight_layout(); plt.savefig(path_prefix + "_hist.png"); plt.close()
    taus = np.unique(np.quantile(sv, np.linspace(0, 1, 300)))
    P, R = [], []
    for t in taus:
        m = metrics(yv, sv >= t); P.append(m["Precision"]); R.append(m["Recall"])
    m = metrics(yv, sv >= tau)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(R, P, color="#2a6fb0"); ax.scatter([m["Recall"]], [m["Precision"]], color="#d1495b", zorder=3, label="Ngưỡng đã chọn")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(path_prefix + "_pr.png"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src", required=True, help="C0, C1, RF, IF ...")
    ap.add_argument("--name", required=True, help="tên cấu hình đầu ra, ví dụ C2")
    ap.add_argument("--criterion", default="B", choices=["A", "B", "C"])
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--fpr_budget", type=float, default=0.01)
    ap.add_argument("--ref", default="C0", help="cấu hình tham chiếu để lấy Recall_val cho tiêu chí B")
    a = ap.parse_args()
    v = pd.read_csv(os.path.join(a.run, f"windows_val_{a.src}.csv"))
    t = pd.read_csv(os.path.join(a.run, f"windows_test_{a.src}.csv"))
    rb = None
    if a.criterion == "B":
        rb = load_json(os.path.join(a.run, f"metrics_{a.ref}.json"))["val"]["Recall"]
    tau, mv = choose_threshold(v.label, v.score, a.criterion, rb, a.delta, a.fpr_budget)
    pred = t.score.to_numpy() >= tau
    out = {"config": a.name, "nguon_diem": a.src, "tieu_chi": a.criterion, "tau": tau,
           "recall_val_tham_chieu": rb, "delta": a.delta, "chon_tren": "validation", "val": mv,
           "test": metrics_by_oov(t.label, pred, t.has_oov)}
    save_json(out, os.path.join(a.run, f"metrics_{a.name}.json"))
    plots(v.label.to_numpy(), v.score.to_numpy(), tau, f"{a.name} – validation", os.path.join(a.run, a.name))
    print(f"{a.name}: tau = {tau:.4f} (tiêu chí {a.criterion}, chọn trên validation)")
    for k, m in out["test"].items():
        print(fmt_row(f"{a.name} test " + k, m))


if __name__ == "__main__":
    main()
