"""Bước 14 (v4): mở rộng lưới tau_sim xuống 0,3 và 0,4 (phân tích độ nhạy bổ sung).
Không thay đổi C1 chính thức; chỉ báo cáo: với mỗi tau_sim, F1/FP trên validation và test,
cùng ví dụ các ánh xạ chỉ xuất hiện khi hạ tau_sim (để đánh giá rủi ro ánh xạ sai).

  python 14_tau_extended.py --parsed out/regex --run runs/regex_seed42
"""
import argparse
import importlib.util
import os

import pandas as pd
import torch

from common import build_model, get_device, load_json, load_pkl, metrics, score_sequences

spec = importlib.util.spec_from_file_location("m04", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_oov_mapper.py"))
m04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m04)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    dev = get_device()
    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    c0 = load_json(os.path.join(a.run, "metrics_C0.json")); c1 = load_json(os.path.join(a.run, "metrics_C1.json"))
    g, h, gp = c0["g"], c0["history"], c1["g_prime"]
    model = build_model(len(D["vocab"]) + 2).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "deeplog.pt"), map_location=dev)); model.eval()
    tpl = pd.read_csv(os.path.join(a.parsed, "templates.csv"))
    T = dict(zip(tpl.EventId, tpl.EventTemplate.astype(str)))
    cands = m04.build_candidates(D["vocab"], T, 5)
    rows, logs = [], {}
    for tau in [0.3, 0.4, 0.5]:
        r = {"tau_sim": tau, "g_prime": gp}
        for split in ("val", "test"):
            log = [] if split == "test" else None
            seqs, _, st = m04.map_split(model, dev, D[split]["raw"], D[split]["idx"], cands, tau, gp, h, log)
            _, rk = score_sequences(model, seqs, h, dev)
            m = metrics(D[split]["y"], rk > g)
            r.update({f"{split}_F1": m["F1"], f"{split}_FP": m["FP"], f"{split}_FN": m["FN"],
                      f"{split}_Recall": m["Recall"], f"{split}_chap_nhan": st["chap_nhan"]})
            if log is not None:
                logs[tau] = pd.DataFrame(log)
        rows.append(r); print(r)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(a.run, "tau_sim_mo_rong.csv"), index=False)
    # ví dụ ánh xạ chỉ có khi hạ tau_sim xuống 0,3 (sim trong [0,3; 0,5))
    d = logs[0.3]; d = d[(d.ket_qua == "chap_nhan") & (d.sim < 0.5)].drop_duplicates("oov_event").head(10)
    d.assign(template_moi=d.oov_event.map(T), anh_xa_sang=d.chon.map(T))[["template_moi", "anh_xa_sang", "sim"]].to_csv(
        os.path.join(a.run, "tau_sim_vi_du_anh_xa_thap.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
