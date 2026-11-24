"""Bước 7: Gom kết quả thành results.md để chép vào báo cáo (bảng chính, bảng theo nhóm
template mới, thống kê ánh xạ và ví dụ log thật cho phân tích lỗi).

  python 07_summary.py --parsed out/regex --run runs/regex_seed42 --best C3
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from common import load_json, load_pkl

ORDER = ["C0", "C1", "C2", "C3", "C4", "C4A", "C5", "C5A", "C5_C1", "C5M", "C5M_C1", "C6", "C6_C0", "RF_mac_dinh", "RF_A", "IF_A", "IF_C"]


def pct(x):
    return f"{x*100:.2f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--best", default="C3", help="cấu hình lấy ví dụ FP/FN")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    M = {}
    for f in glob.glob(os.path.join(a.run, "metrics_*.json")):
        M[os.path.basename(f)[8:-5]] = load_json(f)
    names = [k for k in ORDER if k in M] + sorted(k for k in M if k not in ORDER)
    base = M.get("C0", {}).get("test", {}).get("all")
    L = ["# Kết quả thực nghiệm\n", "## Bảng chính (tập test, ngưỡng chọn trên validation)\n",
         "| Cấu hình | Ngưỡng | P (%) | R (%) | F1 (%) | Spec (%) | FP | FN | ΔFP | ΔFN |", "|" + "---|" * 10]
    for k in names:
        m = M[k]["test"]["all"]; thr = M[k].get("tau", M[k].get("g"))
        if M[k].get("kieu_nguong") == "phan_hoi":
            thr = f"g={M[k]['g']}, phản hồi B={M[k]['ngan_sach_phan_hoi']}"
        elif M[k].get("kieu_nguong") == "truot_mad":
            thr = f"trung vị+{M[k]['k']:g}·MAD, N={M[k]['N_hist']}"
        elif M[k].get("kieu_nguong") == "truot":
            thr = f"trượt {M[k].get('che_do_lich_su','all')}, N={M[k]['N_hist']}, q={M[k]['q']}"
        else:
            thr = f"g={M[k]['g']}" if "g" in M[k] and "tau" not in M[k] else f"τ={thr:.4f}" if isinstance(thr, float) else str(thr)
        d = (f"{m['FP']-base['FP']:+d}", f"{m['FN']-base['FN']:+d}") if base and k[0] == "C" else ("", "")
        L.append(f"| {k} | {thr} | {pct(m['Precision'])} | {pct(m['Recall'])} | {pct(m['F1'])} | "
                 f"{m['Specificity']*100:.4f} | {m['FP']} | {m['FN']} | {d[0]} | {d[1]} |")
    L += ["\n## Tách theo cửa sổ có / không có template mới (test)\n",
          "| Cấu hình | Nhóm | Số cửa sổ | Bất thường | FP | FN | FPR (%) | Recall (%) |", "|" + "---|" * 8]
    for k in names:
        for grp, lab in (("co_template_moi", "Có template mới"), ("khong_template_moi", "Không có")):
            m = M[k]["test"].get(grp)
            if m:
                L.append(f"| {k} | {lab} | {m['N']} | {m['TP']+m['FN']} | {m['FP']} | {m['FN']} | {pct(m['FPR'])} | {pct(m['Recall'])} |")
    if "C1" in M:
        L += ["\n## Thống kê ánh xạ template mới (test)\n", f"tau_sim = {M['C1']['tau_sim']}, g' = {M['C1']['g_prime']}\n"]
        for k, v in M["C1"]["thong_ke_anh_xa_test"].items():
            L.append(f"- {k}: {v}")

    D = load_pkl(os.path.join(a.parsed, "windows.pkl"))
    tpl = pd.read_csv(os.path.join(a.parsed, "templates.csv"))
    T = dict(zip(tpl.EventId, tpl.EventTemplate.astype(str)))
    dec = os.path.join(a.run, "oov_decisions_test.csv")
    if os.path.exists(dec):
        d = pd.read_csv(dec)
        L.append("\n## Ví dụ ánh xạ (test)\n")
        for kq in ("chap_nhan", "tu_choi_ngu_canh", "tu_choi_tu_khoa", "tu_choi_tuong_dong"):
            sub = d[d.ket_qua == kq].drop_duplicates("oov_event").head(a.n)
            if len(sub):
                L.append(f"\n**{kq}**\n")
                for _, r in sub.iterrows():
                    tgt = r.chon if isinstance(r.chon, str) and r.chon else r.ung_vien_tot_nhat
                    L.append(f"- `{T.get(r.oov_event, '')}` → `{T.get(tgt, '')}` (sim={r.sim})")
    best = M.get(a.best)
    if best:
        src = best.get("nguon_diem", a.best)
        w = pd.read_csv(os.path.join(a.run, f"windows_test_{src}.csv"))
        thr = best.get("tau"); pred = w.score >= thr if thr is not None else w.max_rank > best["g"]
        raw = D["test"]["raw"]
        for name, mask in (("Cảnh báo sai (FP)", (w.label == 0) & pred), ("Bỏ sót (FN)", (w.label == 1) & ~pred)):
            L.append(f"\n## {name} của {a.best} – {a.n} ví dụ (dòng bắt đầu, các template khác nhau)\n")
            for i in np.where(mask)[0][:a.n]:
                ev = list(dict.fromkeys(raw[i]))
                L.append(f"- Cửa sổ bắt đầu dòng {w.first_line[i]} (template mới: {bool(w.has_oov[i])}, điểm {w.score[i]:.3f}):")
                for e in ev[:6]:
                    L.append(f"    - {'[MỚI] ' if e not in D['vocab'] else ''}`{T.get(e, e)}`")
    open(os.path.join(a.run, "results.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:30])); print(f"... -> {os.path.join(a.run, 'results.md')}")


if __name__ == "__main__":
    main()
