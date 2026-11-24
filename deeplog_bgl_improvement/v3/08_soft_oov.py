"""Bước 8 (v2): C4 - điểm mềm cho template mới, kết hợp với hiệu chỉnh ngưỡng.

Ở C1/C3, template mới không ánh xạ được nhận điểm cố định UNK_NLL (~13,8), nên mọi cửa sổ
chứa nó đều vượt ngưỡng và hiệu chỉnh ngưỡng không có tác dụng với nhóm này.
C4 thay điểm cố định bằng điểm mềm, không thêm tham số:
    p_mem(e | h) = max_v  p(v | h) * sim(e, v)     (v thỏa điều kiện từ khóa)
    l(e) = min(-log p_mem(e | h), UNK_NLL)
Vị trí đầu cửa sổ (không có lịch sử) dùng tần suất của v trong train thay cho p(v | h).
Sau đó điểm cửa sổ s(w) = max_t l_t được đưa vào 05_calibrate.py.

  python 08_soft_oov.py --parsed out/regex --run runs/regex_seed42
  python 05_calibrate.py --run runs/regex_seed42 --src C4 --name C4 --criterion B
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
import torch

from common import PAD, UNK, UNK_NLL, build_model, get_device, load_json, load_pkl

spec = importlib.util.spec_from_file_location("m04", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_oov_mapper.py"))
m04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m04)


def soft_scores(model, dev, raw, seqs, cands, prior_logp, h, batch=8192):
    V = model.fc.out_features
    owner, xs, ys, oov = [], [], [], []
    for i, s in enumerate(seqs):
        for t in range(len(s)):
            if t == 0 and s[0] != UNK:
                continue                       # sự kiện đầu đã biết: không có lịch sử, bỏ qua như DeepLog
            hist = s[max(0, t - h):t]
            xs.append([PAD] * (h - len(hist)) + list(hist)); ys.append(s[t]); owner.append(i)
            oov.append((t, raw[i][t]) if s[t] == UNK else None)
    nll = np.zeros(len(ys))
    model.eval()
    with torch.no_grad():
        for a in range(0, len(ys), batch):
            xb = torch.tensor(xs[a:a + batch], dtype=torch.long, device=dev)
            lg = model(xb); lg[:, PAD] = -1e9; lg[:, UNK] = -1e9
            logp = torch.log_softmax(lg, 1).cpu().numpy()
            for j in range(len(xb)):
                k = a + j; y = ys[k]
                if y != UNK:
                    nll[k] = -logp[j, y]; continue
                t, e = oov[k]
                best = -np.inf
                for vidx, sim, _, ok_kw in cands.get(e, []):
                    if sim <= 0 or not ok_kw:
                        continue
                    lp = prior_logp[vidx] if t == 0 else logp[j, vidx]
                    best = max(best, lp + np.log(sim))
                nll[k] = min(-best, UNK_NLL) if np.isfinite(best) else UNK_NLL
    out = np.zeros(len(seqs)); np.maximum.at(out, np.asarray(owner, dtype=np.int64), nll)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    dev = get_device()
    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    c0 = load_json(os.path.join(a.run, "metrics_C0.json")); c1 = load_json(os.path.join(a.run, "metrics_C1.json"))
    h = c0["history"]
    V = len(D["vocab"]) + 2
    model = build_model(V).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "deeplog.pt"), map_location=dev)); model.eval()
    tpl = pd.read_csv(os.path.join(a.parsed, "templates.csv"))
    cands = m04.build_candidates(D["vocab"], dict(zip(tpl.EventId, tpl.EventTemplate.astype(str))), a.k)
    cnt = np.ones(V)
    for s, y in zip(D["train"]["idx"], D["train"]["y"]):
        if y == 0:
            np.add.at(cnt, s, 1)
    cnt[PAD] = cnt[UNK] = 0
    prior_logp = np.log(cnt / cnt.sum() + 1e-12)
    for split in ("val", "test"):
        seqs, n_map, _ = m04.map_split(model, dev, D[split]["raw"], D[split]["idx"], cands,
                                       c1["tau_sim"], c1["g_prime"], h)          # dùng lại tham số C1 đã chọn trên val
        score = soft_scores(model, dev, D[split]["raw"], seqs, cands, prior_logp, h)
        pd.DataFrame({"first_line": D[split]["first_line"], "label": D[split]["y"], "score": score,
                      "has_oov": D[split]["has_oov"].astype(int), "n_mapped": n_map}
                     ).to_csv(os.path.join(a.run, f"windows_{split}_C4.csv"), index=False)
        print(f"{split}: đã ghi điểm mềm C4 cho {len(score):,} cửa sổ")


if __name__ == "__main__":
    main()
