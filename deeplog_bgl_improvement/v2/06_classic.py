"""Bước 6: Mô hình học máy truyền thống trên cùng cửa sổ/cùng split:
  RF  - Random Forest có giám sát (vector đếm template + cột đếm OOV), score = predict_proba.
  IF  - Isolation Forest chỉ học cửa sổ bình thường, score = -score_samples.
Xuất windows_{val,test}_{RF,IF}.csv để hiệu chỉnh ngưỡng bằng 05_calibrate.py.

  python 06_classic.py --parsed out/regex --run runs/regex_seed42
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from common import UNK, fmt_row, load_pkl, metrics, metrics_by_oov, save_json


def featurize(idx, V):
    rows, cols = [], []
    for i, s in enumerate(idx):
        for e in s:
            rows.append(i); cols.append(e if e != UNK else V)   # cột cuối = số sự kiện chưa biết
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(idx), V + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    D = load_pkl(os.path.join(a.parsed, "windows.pkl")); V = len(D["vocab"]) + 2
    X = {s: featurize(D[s]["idx"], V) for s in ("train", "val", "test")}
    y = {s: D[s]["y"] for s in ("train", "val", "test")}

    models = {
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=a.seed, n_jobs=-1),
        "IF": IsolationForest(n_estimators=200, contamination="auto", random_state=a.seed, n_jobs=-1),
    }
    for name, m in models.items():
        t0 = time.time()
        if name == "RF":
            m.fit(X["train"], y["train"])
            score = lambda Z: m.predict_proba(Z)[:, 1]
        else:
            m.fit(X["train"][y["train"] == 0])
            score = lambda Z: -m.score_samples(Z)
        tt = time.time() - t0
        for s in ("val", "test"):
            pd.DataFrame({"first_line": D[s]["first_line"], "label": y[s], "score": score(X[s]),
                          "has_oov": D[s]["has_oov"].astype(int)}).to_csv(
                os.path.join(a.run, f"windows_{s}_{name}.csv"), index=False)
        if name == "RF":   # ngưỡng mặc định 0,5 làm mốc
            sv = score(X["val"]); st = score(X["test"])
            out = {"config": "RF_mac_dinh", "tau": 0.5, "val": metrics(y["val"], sv >= .5),
                   "test": metrics_by_oov(y["test"], st >= .5, D["test"]["has_oov"]), "train_time_s": tt}
            save_json(out, os.path.join(a.run, "metrics_RF_mac_dinh.json"))
            print(fmt_row("RF ngưỡng 0,5 test all", out["test"]["all"]))
        print(f"{name}: huấn luyện {tt:.1f}s, đã xuất điểm val/test")


if __name__ == "__main__":
    main()
