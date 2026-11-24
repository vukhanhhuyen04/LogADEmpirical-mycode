"""Bước 12 (v4): C6 - thích nghi nhờ phản hồi của người vận hành (human-in-the-loop).

Mô phỏng: hệ thống chấm các cửa sổ test theo đúng thứ tự thời gian. Khi một cửa sổ bị cảnh báo,
người vận hành xem xét (tối đa B lần). Nếu đó là báo nhầm, các template mới chưa ánh xạ được
trong cửa sổ được ghi vào danh sách "đã xác nhận bình thường"; từ đó trở đi, vị trí chứa các
template này không còn bị coi là bằng chứng bất thường. Mô hình KHÔNG được huấn luyện lại.
Nhãn chỉ được dùng cho đúng các cửa sổ người vận hành xem (ngân sách B), như ngoài thực tế.

  python 12_feedback.py --parsed out/regex --run runs/regex_seed42
Kết quả: metrics_C6.json (B=100, nền C1), metrics_C6_C0.json (B=100, nền C0),
         feedback_curve.csv, C6_theo_ngan_sach.png, C6_theo_thoi_gian.png
"""
import argparse
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import (PAD, UNK, build_model, fmt_row, get_device, load_json, load_pkl,
                    metrics, metrics_by_oov, save_json)

spec = importlib.util.spec_from_file_location("m04", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_oov_mapper.py"))
m04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m04)
BUDGETS = [0, 10, 25, 50, 100, 200, 500, 1000]


def window_evidence(model, dev, raw, seqs, h, batch=8192):
    """Cho mỗi cửa sổ: thứ hạng lớn nhất trên các vị trí ĐÃ BIẾT, và tập template mới còn lại (UNK)."""
    V = model.fc.out_features
    owner, xs, ys = [], [], []
    unk_sets = [set() for _ in seqs]
    for i, s in enumerate(seqs):
        for t in range(len(s)):
            if s[t] == UNK:
                unk_sets[i].add(raw[i][t])          # kể cả vị trí đầu cửa sổ (sửa lỗi v2)
                continue
            if t == 0:
                continue
            hist = s[max(0, t - h):t]
            xs.append([PAD] * (h - len(hist)) + list(hist)); ys.append(s[t]); owner.append(i)
    known_max = np.zeros(len(seqs), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for a in range(0, len(ys), batch):
            xb = torch.tensor(xs[a:a + batch], dtype=torch.long, device=dev)
            yb = torch.tensor(ys[a:a + batch], dtype=torch.long, device=dev)
            lg = model(xb); lg[:, PAD] = -1e9; lg[:, UNK] = -1e9
            r = (lg > lg.gather(1, yb.unsqueeze(1))).sum(1) + 1
            np.maximum.at(known_max, np.asarray(owner[a:a + batch]), r.cpu().numpy())
    return known_max, unk_sets


def simulate(known_max, unk_sets, labels, g, budget, strategy="tat_ca"):
    """strategy="tat_ca": xem lần lượt mọi cảnh báo cho đến khi hết ngân sách.
    strategy="moi": chỉ xem cảnh báo có template mới CHƯA TỪNG được xem (ưu tiên thông tin mới)."""
    confirmed, reviewed, used = set(), set(), 0
    pred = np.zeros(len(labels), dtype=bool); fb_at = []
    for i in range(len(labels)):
        rest = unk_sets[i] - confirmed
        pred[i] = known_max[i] > g or len(rest) > 0
        if not pred[i] or used >= budget:
            continue
        if strategy == "moi" and not (rest - reviewed):
            continue
        used += 1; reviewed |= rest
        if labels[i] == 0 and rest:            # người vận hành xác nhận: báo nhầm
            confirmed |= rest; fb_at.append(i)
    return pred, confirmed, fb_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--budget", type=int, default=100, help="ngân sách dùng cho metrics_C6.json")
    ap.add_argument("--strategy", default="tat_ca", choices=["tat_ca", "moi"])
    ap.add_argument("--suffix", default="", help="hậu tố tên kết quả, ví dụ _moi")
    a = ap.parse_args()
    dev = get_device()
    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    c0 = load_json(os.path.join(a.run, "metrics_C0.json")); c1 = load_json(os.path.join(a.run, "metrics_C1.json"))
    g, h = c0["g"], c0["history"]
    model = build_model(len(D["vocab"]) + 2).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "deeplog.pt"), map_location=dev)); model.eval()
    tpl = pd.read_csv(os.path.join(a.parsed, "templates.csv"))
    T = dict(zip(tpl.EventId, tpl.EventTemplate.astype(str)))
    cands = m04.build_candidates(D["vocab"], T, 5)
    raw, idx, y = D["test"]["raw"], D["test"]["idx"], D["test"]["y"]
    oov = D["test"]["has_oov"]

    bases = {"C1": m04.map_split(model, dev, raw, idx, cands, c1["tau_sim"], c1["g_prime"], h)[0],
             "C0": [list(s) for s in idx]}
    rows, store = [], {}
    for base, seqs in bases.items():
        km, us = window_evidence(model, dev, raw, seqs, h)
        for B in BUDGETS:
            pred, conf, fb = simulate(km, us, y, g, B, a.strategy)
            m = metrics(y, pred)
            rows.append(dict(nen=base, ngan_sach=B, so_template_xac_nhan=len(conf), **m))
            if B == a.budget:
                store[base] = (pred, conf, fb)
            print(f"nền {base} B={B:5d}: FP={m['FP']:6d} FN={m['FN']:4d} F1={m['F1']*100:6.2f} "
                  f"template xác nhận={len(conf)}")
    cur = pd.DataFrame(rows); cur.to_csv(os.path.join(a.run, f"feedback_curve{a.suffix}.csv"), index=False)

    for base, name in (("C1", "C6" + a.suffix), ("C0", "C6" + a.suffix + "_C0")):
        pred, conf, fb = store[base]
        out = {"config": name, "chien_luoc": a.strategy, "nen": base, "ngan_sach_phan_hoi": a.budget, "g": g,
               "so_template_xac_nhan": len(conf), "so_phan_hoi_bao_nham": len(fb),
               "kieu_nguong": "phan_hoi", "test": metrics_by_oov(y, pred, oov)}
        save_json(out, os.path.join(a.run, f"metrics_{name}.json"))
        for k, m in out["test"].items():
            print(fmt_row(f"{name} test " + k, m))
    ex = sorted(store["C1"][1])[:15]
    pd.DataFrame({"template_xac_nhan_binh_thuong": [T.get(e, e) for e in ex]}).to_csv(
        os.path.join(a.run, f"feedback_templates_vi_du{a.suffix}.csv"), index=False)

    # Hình 1: FP và FN theo ngân sách phản hồi
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=160)
    for base, col in (("C1", "#2a6fb0"), ("C0", "#999999")):
        s = cur[cur.nen == base]; xs = s.ngan_sach.replace(0, 1)
        ax.plot(xs, s.FP, "o-", color=col, label=f"FP – nền {base}")
    ax.set_xscale("log"); ax.set_xlabel("Ngân sách phản hồi B (số cảnh báo được xem, B=0 vẽ ở 1)")
    ax.set_ylabel("FP trên test"); ax2 = ax.twinx()
    s = cur[cur.nen == "C1"]; ax2.plot(s.ngan_sach.replace(0, 1), s.FN, "s--", color="#d1495b", label="FN – nền C1")
    ax2.set_ylabel("FN", color="#d1495b")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8); ax.set_title("Cảnh báo sai theo số lần phản hồi")
    plt.tight_layout(); plt.savefig(os.path.join(a.run, f"C6{a.suffix}_theo_ngan_sach.png")); plt.close()

    # Hình 2: tỷ lệ FP theo thời gian, C1 so với C6
    x = np.arange(len(y)); seg = max(1, len(y) // 50)
    base_pred, _, _ = simulate(*window_evidence(model, dev, raw, bases["C1"], h), y, g, 0)
    fp1 = pd.Series(((y == 0) & base_pred).astype(int)).groupby(x // seg).mean() * 100
    fp6 = pd.Series(((y == 0) & store["C1"][0]).astype(int)).groupby(x // seg).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    ax.plot(fp1.index * seg, fp1.values, color="#d1495b", label="C1 (không phản hồi)")
    ax.plot(fp6.index * seg, fp6.values, color="#2a6fb0", label=f"C6 (B = {a.budget} phản hồi)")
    for i in store["C1"][2]:
        ax.axvline(i, color="#2a6fb0", alpha=.15, lw=.6)
    ax.set_xlabel("Thứ tự cửa sổ trong tập test (vạch mờ: thời điểm có phản hồi báo nhầm)")
    ax.set_ylabel("Tỷ lệ cảnh báo sai (%)"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.run, f"C6{a.suffix}_theo_thoi_gian.png")); plt.close()


if __name__ == "__main__":
    main()
