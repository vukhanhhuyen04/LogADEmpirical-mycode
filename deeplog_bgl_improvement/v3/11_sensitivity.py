"""Bước 11: phân tích độ nhạy -> 2 hình cho báo cáo.
  (a) F1 validation và số vị trí được ánh xạ theo tau_sim (từ grid_C1_val.csv)
  (b) FP và FN trên test theo delta của tiêu chí B, với điểm mềm C4

  python 11_sensitivity.py --run runs/regex_seed42
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import choose_threshold, load_json, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    g = pd.read_csv(os.path.join(a.run, "grid_C1_val.csv"))
    gp = g.g_prime.max(); g = g[g.g_prime == gp].sort_values("tau_sim")
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=160)
    ax.plot(g.tau_sim, g.F1 * 100, "o-", color="#2a6fb0", label="F1 validation (%)")
    ax.set_xlabel("Ngưỡng tương đồng τ_sim"); ax.set_ylabel("F1 validation (%)", color="#2a6fb0")
    ax2 = ax.twinx(); ax2.plot(g.tau_sim, g.chap_nhan, "s--", color="#e08a2b", label="Số vị trí được ánh xạ")
    ax2.set_ylabel("Số vị trí được ánh xạ", color="#e08a2b"); ax2.set_yscale("log")
    ax.set_title(f"Độ nhạy theo τ_sim (g′ = {gp})"); plt.tight_layout()
    plt.savefig(os.path.join(a.run, "do_nhay_tau_sim.png")); plt.close()

    v = pd.read_csv(os.path.join(a.run, "windows_val_C4.csv")); t = pd.read_csv(os.path.join(a.run, "windows_test_C4.csv"))
    r0 = load_json(os.path.join(a.run, "metrics_C0.json"))["val"]["Recall"]
    rows = []
    for d in [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        tau, _ = choose_threshold(v.label, v.score, "B", r0, d)
        m = metrics(t.label, t.score >= tau); rows.append(dict(delta=d, tau=tau, **m))
    r = pd.DataFrame(rows); r.to_csv(os.path.join(a.run, "do_nhay_delta_C4.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=160)
    ax.plot(r.delta * 100, r.FP, "o-", color="#d1495b", label="FP (test)")
    ax.set_xlabel("δ – mức giảm Recall cho phép trên validation (điểm %)"); ax.set_ylabel("FP trên test", color="#d1495b")
    ax2 = ax.twinx(); ax2.plot(r.delta * 100, r.Recall * 100, "s--", color="#2a6fb0"); ax2.set_ylabel("Recall test (%)", color="#2a6fb0")
    ax.set_title("Đánh đổi FP – Recall theo δ (C4, điểm mềm)"); plt.tight_layout()
    plt.savefig(os.path.join(a.run, "do_nhay_delta_C4.png")); plt.close()
    print(r[["delta", "tau", "FP", "FN", "Recall", "F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
