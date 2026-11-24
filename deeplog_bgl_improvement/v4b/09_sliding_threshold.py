"""Bước 9 (v3): C5 - ngưỡng thích nghi theo cửa sổ trượt.

Thay vì một ngưỡng cố định chọn một lần trên validation, ngưỡng tại cửa sổ thứ t là
phân vị q của điểm N cửa sổ NGAY TRƯỚC đó:
    tau_t = Quantile_q( s(w_{t-N}), ..., s(w_{t-1}) )
Không dùng nhãn, chỉ dùng điểm của dữ liệu quá khứ nên không rò rỉ thông tin.
(N, q) chọn trên validation; khi chấm test, lịch sử khởi đầu là N cửa sổ cuối của validation
(đứng ngay trước test theo thời gian).

  python 09_sliding_threshold.py --run runs/regex_seed42 --src C4 --name C5
  python 09_sliding_threshold.py --run runs/regex_seed42 --src C1 --name C5_C1
"""
import argparse
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import fmt_row, load_json, metrics, metrics_by_oov, save_json

N_GRID = [500, 1000, 2000, 5000]
Q_GRID = [0.90, 0.95, 0.97, 0.99, 0.995]
MIN_HIST = 200


def sliding_tau(history, scores, n, q, fallback, mode="all"):
    """Ngưỡng tại mỗi cửa sổ = phân vị q của n điểm quá khứ.
    mode="all": lịch sử gồm mọi cửa sổ trước đó.
    mode="normal": lịch sử chỉ gồm các cửa sổ đã được dự đoán bình thường, để một đợt bất thường
    dồn dập không kéo ngưỡng lên. Trả về (tau, lịch sử cuối cùng)."""
    if mode == "all":
        s = pd.Series(np.concatenate([history[-n:], scores]))
        tau = s.shift(1).rolling(n, min_periods=MIN_HIST).quantile(q).to_numpy()[-len(scores):]
        tau = np.where(np.isnan(tau), fallback, tau)
        return tau, np.concatenate([history, scores])[-n:]
    hist = deque(list(history)[-n:], maxlen=n); tau = np.empty(len(scores))
    for i, x in enumerate(scores):
        tau[i] = np.quantile(hist, q) if len(hist) >= MIN_HIST else fallback
        if x < tau[i]:
            hist.append(x)
    return tau, np.array(hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src", default="C4", help="nguồn điểm: C4 (điểm mềm) hoặc C1")
    ap.add_argument("--name", default="C5")
    ap.add_argument("--fixed", default=None, help="cấu hình ngưỡng cố định để so sánh/fallback, mặc định = src")
    ap.add_argument("--criterion", default="B", choices=["A", "B"])
    ap.add_argument("--delta", type=float, default=0.02)
    a = ap.parse_args()
    v = pd.read_csv(os.path.join(a.run, f"windows_val_{a.src}.csv"))
    t = pd.read_csv(os.path.join(a.run, f"windows_test_{a.src}.csv"))
    fixed_name = a.fixed or a.src
    fixed = load_json(os.path.join(a.run, f"metrics_{fixed_name}.json"))
    fallback = fixed.get("tau", float(np.quantile(v.score, 0.99)))
    r0 = load_json(os.path.join(a.run, "metrics_C0.json"))["val"]["Recall"]

    grid = []
    for mode in ("all", "normal"):
        for n in N_GRID:
            for q in Q_GRID:
                tau, _ = sliding_tau(np.array([]), v.score.to_numpy(), n, q, fallback, mode)
                m = metrics(v.label, v.score.to_numpy() >= tau)
                grid.append(dict(mode=mode, N_hist=n, q=q, **m))
    g = pd.DataFrame(grid); g.to_csv(os.path.join(a.run, f"grid_{a.name}_val.csv"), index=False)
    if a.criterion == "B":
        ok = g[g.Recall >= r0 - a.delta]
        best = (ok.sort_values(["FP", "F1"], ascending=[True, False]).iloc[0] if len(ok)
                else g.sort_values("Recall", ascending=False).iloc[0])
    else:
        best = g.sort_values(["F1", "q"], ascending=[False, False]).iloc[0]
    n, q, mode = int(best.N_hist), float(best.q), best["mode"]

    _, hist_val = sliding_tau(np.array([]), v.score.to_numpy(), n, q, fallback, mode)   # lịch sử ngay trước test
    tau_t, _ = sliding_tau(hist_val, t.score.to_numpy(), n, q, fallback, mode)
    pred = t.score.to_numpy() >= tau_t
    out = {"config": a.name, "nguon_diem": a.src, "kieu_nguong": "truot", "che_do_lich_su": mode, "N_hist": n, "q": q,
           "tieu_chi": a.criterion, "chon_tren": "validation",
           "tau_min": float(tau_t.min()), "tau_median": float(np.median(tau_t)), "tau_max": float(tau_t.max()),
           "val": {k: best[k] for k in ("TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Specificity")},
           "test": metrics_by_oov(t.label, pred, t.has_oov)}
    save_json(out, os.path.join(a.run, f"metrics_{a.name}.json"))
    pd.DataFrame({"first_line": t.first_line, "label": t.label, "score": t.score - tau_t,   # điểm đã trừ ngưỡng
                  "has_oov": t.has_oov}).to_csv(os.path.join(a.run, f"windows_test_{a.name}.csv"), index=False)

    # Hình: ngưỡng trượt theo thời gian trên test, so với ngưỡng cố định, và tỷ lệ FP theo đoạn
    x = np.arange(len(t)); seg = max(1, len(t) // 50)
    fixed_pred = t.score.to_numpy() >= fallback
    fp_fixed = pd.Series(((t.label == 0) & fixed_pred).astype(int)).groupby(x // seg).mean() * 100
    fp_slide = pd.Series(((t.label == 0) & pred).astype(int)).groupby(x // seg).mean() * 100
    oov_rate = pd.Series(t.has_oov).groupby(x // seg).mean() * 100
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), dpi=160, sharex=True)
    ax[0].plot(x, tau_t, color="#2a6fb0", lw=1, label=f"Ngưỡng trượt (N={n}, q={q}, lịch sử={mode})")
    ax[0].axhline(fallback, color="#d1495b", ls="--", lw=1, label=f"Ngưỡng cố định ({fixed_name}) = {fallback:.2f}")
    ax[0].set_ylabel("Ngưỡng τ"); ax[0].legend(frameon=False, fontsize=8)
    xs = fp_fixed.index * seg
    ax[1].plot(xs, fp_fixed.values, color="#d1495b", label="FP (%) – ngưỡng cố định")
    ax[1].plot(xs, fp_slide.values, color="#2a6fb0", label="FP (%) – ngưỡng trượt")
    ax[1].plot(xs, oov_rate.values, color="gray", ls=":", label="Cửa sổ có template mới (%)")
    ax[1].set_xlabel("Thứ tự cửa sổ trong tập test (theo thời gian)"); ax[1].set_ylabel("%")
    ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.run, f"{a.name}_theo_thoi_gian.png")); plt.close()

    print(f"{a.name}: chế độ {mode}, N={n}, q={q} (chọn trên validation, tiêu chí {a.criterion}); "
          f"ngưỡng test từ {tau_t.min():.2f} đến {tau_t.max():.2f}")
    for k, m in out["test"].items():
        print(fmt_row(f"{a.name} test " + k, m))


if __name__ == "__main__":
    main()
