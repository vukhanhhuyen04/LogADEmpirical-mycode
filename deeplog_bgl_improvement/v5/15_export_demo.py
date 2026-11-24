"""Bước 15: Đóng gói dữ liệu cho ứng dụng demo (chạy trên Kaggle sau run_v4.sh).
Lấy N cửa sổ liên tiếp của tập test kèm dòng log gốc, mô hình DeepLog đã huấn luyện và tham số.

  python 15_export_demo.py --parsed out/regex --run runs/regex_seed42 --log /duong/dan/BGL.log --start 18000 --n 6000
Kết quả: demo_bundle.zip (khoảng 10–20 MB) -> tải về máy, giải nén vào thư mục demo/
"""
import argparse
import json
import os
import shutil

import pandas as pd

from common import load_json, load_pkl, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--start", type=int, default=18000, help="cửa sổ test bắt đầu")
    ap.add_argument("--n", type=int, default=6000, help="số cửa sổ đưa vào demo")
    ap.add_argument("--out", default="demo_bundle")
    a = ap.parse_args()
    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    W = D["window"]; off = len(D["train"]["y"]) + len(D["val"]["y"])
    te = D["test"]; end = min(len(te["y"]), a.start + a.n)
    df = pd.read_csv(os.path.join(a.parsed, "structured.csv"), usecols=["LineId"])
    need = {}
    for k in range(a.start, end):
        g = off + k
        need[k] = df.LineId.iloc[g * W:(g + 1) * W].tolist()
    want = {lid for v in need.values() for lid in v}
    text = {}
    with open(a.log, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i + 1 in want:
                tok = line.rstrip("\n").split(maxsplit=9)
                text[i + 1] = (tok[4] if len(tok) > 4 else "", tok[3] if len(tok) > 3 else "", tok[9] if len(tok) > 9 else line.strip())
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for k in range(a.start, end):
        rows.append({"id": k, "label": int(te["y"][k]), "events": te["raw"][k],
                     "lines": [list(text.get(l, ("", "", ""))) for l in need[k]]})
    with open(os.path.join(a.out, "windows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    shutil.copy(os.path.join(a.run, "deeplog.pt"), os.path.join(a.out, "deeplog.pt"))
    shutil.copy(os.path.join(a.parsed, "templates.csv"), os.path.join(a.out, "templates.csv"))
    c0 = load_json(os.path.join(a.run, "metrics_C0.json")); c1 = load_json(os.path.join(a.run, "metrics_C1.json"))
    save_json({"vocab": D["vocab"], "g": c0["g"], "history": c0["history"], "tau_sim": c1["tau_sim"],
               "g_prime": c1["g_prime"], "window": W, "nguon": f"BGL test, cửa sổ {a.start}–{end - 1}"},
              os.path.join(a.out, "params.json"))
    shutil.make_archive(a.out, "zip", a.out)
    print(f"Đã đóng gói {len(rows)} cửa sổ -> {a.out}.zip")


if __name__ == "__main__":
    main()
