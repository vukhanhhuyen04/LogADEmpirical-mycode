"""Bước 2: Gom cửa sổ 20 dòng (bước 20), chia theo thời gian train/val/test = 72/8/20,
xây từ điển từ các cửa sổ BÌNH THƯỜNG của train và thống kê template mới (OOV).

  python 02_make_windows.py --parsed out/regex
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import PAD, UNK, save_json, save_pkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True, help="thư mục chứa structured.csv")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--train", type=float, default=0.72)
    ap.add_argument("--val", type=float, default=0.08)
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.parsed, "structured.csv"))
    ev = df["EventId"].to_numpy(); lab = df["Label"].to_numpy(); lid = df["LineId"].to_numpy()
    starts = range(0, len(df) - a.window + 1, a.step)
    W = [(ev[s:s + a.window].tolist(), int(lab[s:s + a.window].max()), int(lid[s])) for s in starts]
    n = len(W); n_tr = int(n * a.train); n_va = int(n * a.val)
    parts = {"train": W[:n_tr], "val": W[n_tr:n_tr + n_va], "test": W[n_tr + n_va:]}

    known = sorted({e for seq, y, _ in parts["train"] if y == 0 for e in seq})
    vocab = {e: i + 2 for i, e in enumerate(known)}          # 0 = PAD, 1 = UNK
    data = {"vocab": vocab, "window": a.window}
    stats = {"so_cua_so": n, "so_template_tu_dien": len(vocab)}
    for name, ws in parts.items():
        raw = [seq for seq, _, _ in ws]
        y = np.array([yy for _, yy, _ in ws])
        idx = [[vocab.get(e, UNK) for e in seq] for seq in raw]
        has_oov = np.array([any(i == UNK for i in s) for s in idx])
        data[name] = {"raw": raw, "idx": idx, "y": y, "has_oov": has_oov,
                      "first_line": np.array([f for _, _, f in ws])}
        distinct = {e for seq in raw for e in seq}
        stats[name] = {"so_cua_so": len(ws), "bat_thuong": int(y.sum()),
                       "ty_le_bat_thuong_%": round(float(100 * y.mean()), 3) if len(y) else 0,
                       "cua_so_co_template_moi_%": round(float(100 * has_oov.mean()), 2) if len(y) else 0,
                       "template_phan_biet": len(distinct),
                       "template_moi_phan_biet": len(distinct - set(vocab)),
                       "ty_le_template_moi_%": round(100 * len(distinct - set(vocab)) / max(1, len(distinct)), 2)}
    save_pkl(data, os.path.join(a.parsed, "windows.pkl"))
    save_json(stats, os.path.join(a.parsed, "window_stats.json"))
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
