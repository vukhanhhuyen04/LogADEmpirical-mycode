"""Bước 4: Xử lý template mới (C1) - ánh xạ template chưa biết sang template đã biết khi
(1) độ tương đồng cosine TF-IDF >= tau_sim, (2) ứng viên nằm trong top-g' dự đoán của DeepLog
với lịch sử hiện tại (ngữ cảnh), (3) không lệch các từ mang nghĩa lỗi/phủ định.
Tham số (tau_sim, g') chọn trên VALIDATION; tập test chỉ chấm một lần.

  python 04_oov_mapper.py --parsed out/regex --run runs/regex_seed42
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from common import (PAD, UNK, build_model, fmt_row, get_device, load_json, load_pkl,
                    metrics, metrics_by_oov, save_json, score_sequences)

NEG = {"fail", "failed", "failure", "failures", "error", "errors", "not", "unable", "denied",
       "panic", "fatal", "invalid", "cannot", "timeout", "timed", "lost", "severed", "refused",
       "reset", "abort", "aborted", "bad", "corrupt", "corrupted", "uncorrectable"}
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]


def tokens(text):
    return [w for w in re.split(r"[^A-Za-z0-9]+", text.lower()) if w and not w.isdigit()]


def build_candidates(vocab, templates, k):
    known = sorted(vocab, key=vocab.get)
    vec = TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None)
    Xk = vec.fit_transform([templates.get(e, "") for e in known])        # fit CHỈ trên template đã biết
    oov = sorted(set(templates) - set(vocab))
    cands = {}
    if oov:
        S = (vec.transform([templates[e] for e in oov]) @ Xk.T).toarray()
        for i, e in enumerate(oov):
            top = np.argsort(-S[i])[:k]
            neg_e = NEG & set(tokens(templates[e]))
            cands[e] = [(vocab[known[j]], float(S[i, j]), known[j],
                         (NEG & set(tokens(templates[known[j]]))) == neg_e) for j in top]
    return cands


def map_split(model, dev, raw, idx, cands, tau, gp, h, log=None):
    seqs = [list(s) for s in idx]; n_map = np.zeros(len(seqs), dtype=int)
    st = dict(oov_vi_tri=0, chap_nhan=0, tu_choi_tuong_dong=0, tu_choi_tu_khoa=0, tu_choi_ngu_canh=0)
    L = max(len(s) for s in seqs) if seqs else 0
    for t in range(L):
        ws = [i for i, s in enumerate(seqs) if t < len(s) and s[t] == UNK]
        if not ws:
            continue
        st["oov_vi_tri"] += len(ws)
        logp = None
        if t > 0:
            X = [[PAD] * max(0, h - t) + seqs[i][max(0, t - h):t] for i in ws]
            with torch.no_grad():
                lg = model(torch.tensor(X, dtype=torch.long, device=dev))
                lg[:, PAD] = -1e9; lg[:, UNK] = -1e9
                logp = torch.log_softmax(lg, 1).cpu().numpy()
        for r, i in enumerate(ws):
            e = raw[i][t]; chosen = None; reason = "tuong_dong"
            for vidx, sim, vname, ok_kw in cands.get(e, []):
                if sim < tau:
                    break
                if not ok_kw:
                    reason = "tu_khoa"; continue
                if logp is not None:
                    rank = int((logp[r] > logp[r, vidx]).sum()) + 1
                    if rank > gp:
                        reason = "ngu_canh"; continue
                chosen = (vidx, sim, vname); break
            if chosen:
                seqs[i][t] = chosen[0]; n_map[i] += 1; st["chap_nhan"] += 1
            else:
                st["tu_choi_" + {"tuong_dong": "tuong_dong", "tu_khoa": "tu_khoa", "ngu_canh": "ngu_canh"}[reason]] += 1
            if log is not None:
                top = cands.get(e, [(None, 0.0, "", True)])[0]
                log.append(dict(window=i, pos=t, oov_event=e, chon=chosen[2] if chosen else "",
                                sim=round(chosen[1] if chosen else top[1], 4),
                                ung_vien_tot_nhat=top[2], ket_qua="chap_nhan" if chosen else "tu_choi_" + reason))
    return seqs, n_map, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True, help="thư mục kết quả của 03_deeplog.py")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    dev = get_device()
    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    c0 = load_json(os.path.join(a.run, "metrics_C0.json"))
    g, h = c0["g"], c0["history"]
    model = build_model(len(D["vocab"]) + 2).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "deeplog.pt"), map_location=dev)); model.eval()
    tpl = pd.read_csv(os.path.join(a.parsed, "templates.csv"))
    templates = dict(zip(tpl.EventId, tpl.EventTemplate.astype(str)))
    cands = build_candidates(D["vocab"], templates, a.k)
    print(f"{len(cands)} template mới có ứng viên | g = {g}")

    grid = []
    for gp in sorted({g, 2 * g, 3 * g}):
        for tau in TAU_GRID:
            seqs, _, st = map_split(model, dev, D["val"]["raw"], D["val"]["idx"], cands, tau, gp, h)
            _, rk = score_sequences(model, seqs, h, dev)
            m = metrics(D["val"]["y"], rk > g)
            grid.append(dict(tau_sim=tau, g_prime=gp, **m, **st))
            print(f"val tau={tau:.1f} g'={gp:3d} F1={m['F1']*100:6.2f} FP={m['FP']} FN={m['FN']} chấp nhận={st['chap_nhan']}")
    pd.DataFrame(grid).to_csv(os.path.join(a.run, "grid_C1_val.csv"), index=False)
    best = max(grid, key=lambda r: (r["F1"], r["tau_sim"]))
    tau, gp = best["tau_sim"], best["g_prime"]

    for split in ("val", "test"):
        log = [] if split == "test" else None
        seqs, n_map, st = map_split(model, dev, D[split]["raw"], D[split]["idx"], cands, tau, gp, h, log)
        nll, rk = score_sequences(model, seqs, h, dev)
        pd.DataFrame({"first_line": D[split]["first_line"], "label": D[split]["y"], "score": nll,
                      "max_rank": rk, "has_oov": D[split]["has_oov"].astype(int), "n_mapped": n_map}
                     ).to_csv(os.path.join(a.run, f"windows_{split}_C1.csv"), index=False)
        if split == "test":
            pd.DataFrame(log).to_csv(os.path.join(a.run, "oov_decisions_test.csv"), index=False)
            out = {"config": "C1", "g": g, "tau_sim": tau, "g_prime": gp, "chon_tren": "validation",
                   "val": {k: best[k] for k in ("TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Specificity")},
                   "test": metrics_by_oov(D["test"]["y"], rk > g, D["test"]["has_oov"]), "thong_ke_anh_xa_test": st}
            save_json(out, os.path.join(a.run, "metrics_C1.json"))
            print(f"\nChọn tau_sim={tau}, g'={gp}. Thống kê test: {st}")
            for k, m in out["test"].items():
                print(fmt_row("C1 test " + k, m))


if __name__ == "__main__":
    main()
