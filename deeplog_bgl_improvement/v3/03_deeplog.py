"""Bước 3: Huấn luyện DeepLog trên cửa sổ bình thường của train, chấm điểm val/test,
chọn top-g trên validation (cấu hình C0) và xuất điểm từng cửa sổ cho các bước sau.

  python 03_deeplog.py --parsed out/regex --out runs/regex_seed42 --seed 42
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (UNK, build_model, fmt_row, get_device, histories, load_pkl,
                    metrics, metrics_by_oov, save_json, score_sequences)

G_GRID = list(range(1, 31)) + [40, 50, 75, 100, 150]


def pairs(seqs, h):
    xs, ys = [], []
    for s in seqs:
        x, y = histories(s, h); xs += x; ys += y
    arr = np.unique(np.column_stack([np.asarray(xs), np.asarray(ys)]), axis=0)  # khử trùng lặp
    return arr[:, :-1], arr[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = get_device(); print("Thiết bị:", dev)

    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    V = len(D["vocab"]) + 2
    tr = [s for s, y in zip(D["train"]["idx"], D["train"]["y"]) if y == 0]
    va_norm = [s for s, y in zip(D["val"]["idx"], D["val"]["y"]) if y == 0]
    Xtr, Ytr = pairs(tr, a.history)
    Xva, Yva = pairs(va_norm, a.history)
    keep = Yva != UNK; Xva, Yva = Xva[keep], Yva[keep]       # loss validation chỉ trên sự kiện đã biết
    print(f"Từ điển {V-2} template | cặp train {len(Ytr):,} | cặp val {len(Yva):,}")

    model = build_model(V).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva), torch.tensor(Yva)
    best, bad, t0 = 1e9, 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(Ytr_t)); tot = 0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b].to(dev)), Ytr_t[b].to(dev))
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        model.eval(); vl = 0
        with torch.no_grad():
            for i in range(0, len(Yva_t), 4096):
                vl += lossf(model(Xva_t[i:i+4096].to(dev)), Yva_t[i:i+4096].to(dev)).item() * len(Yva_t[i:i+4096])
        vl /= max(1, len(Yva_t))
        print(f"epoch {ep:3d} train_loss {tot/len(Ytr_t):.4f} val_loss {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(model.state_dict(), os.path.join(a.out, "deeplog.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("Dừng sớm"); break
    train_time = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(a.out, "deeplog.pt"), map_location=dev))

    res = {}
    for split in ("val", "test"):
        t1 = time.time()
        nll, rk = score_sequences(model, D[split]["idx"], a.history, dev)
        res[split] = (nll, rk, time.time() - t1)
        pd.DataFrame({"first_line": D[split]["first_line"], "label": D[split]["y"],
                      "score": nll, "max_rank": rk, "has_oov": D[split]["has_oov"].astype(int)}
                     ).to_csv(os.path.join(a.out, f"windows_{split}_C0.csv"), index=False)

    yv = D["val"]["y"]; best_g = None
    for g in G_GRID:                                  # chọn g trên VALIDATION
        m = metrics(yv, res["val"][1] > g)
        if best_g is None or (m["F1"], g) > (best_g[1]["F1"], best_g[0]):
            best_g = (g, m)
    g = best_g[0]
    yt = D["test"]["y"]; pred = res["test"][1] > g
    out = {"config": "C0", "g": g, "chon_tren": "validation", "val": best_g[1],
           "test": metrics_by_oov(yt, pred, D["test"]["has_oov"]),
           "train_time_s": train_time, "test_time_s": res["test"][2], "seed": a.seed,
           "history": a.history, "vocab": V - 2}
    save_json(out, os.path.join(a.out, "metrics_C0.json"))
    print(f"\ng = {g} (chọn trên validation)")
    for k, m in out["test"].items():
        print(fmt_row("C0 test " + k, m))


if __name__ == "__main__":
    main()
