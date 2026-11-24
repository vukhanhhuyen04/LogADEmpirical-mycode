"""Bước 1: Phân tích cú pháp BGL bằng Drain (drain3), có/không chuẩn hóa tham số bằng regex.

Ví dụ:
  python 01_parse_bgl.py --log data/BGL.log --out out/regex
  python 01_parse_bgl.py --log data/BGL.log --out out/noregex --no_regex
"""
import argparse
import csv
import os
import time

from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

# Thứ tự quan trọng: mẫu dài/đặc thù trước, số đơn lẻ sau cùng.
BGL_REGEX = [
    r"core\.\d+",                                   # core.1234
    r"(\d{1,3}\.){3}\d{1,3}(:\d+)?",                # địa chỉ IP[:cổng]
    r"0x[0-9a-fA-F]+",                              # số hex có tiền tố 0x
    r"\bR\d{2}-M\d[-\w:]*",                         # mã nút phần cứng R02-M1-N0-C:J12-U11
    r"(/[\w.\-]+)+",                                # đường dẫn tệp
    r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b",       # chuỗi hex dài có chứa chữ số
    r"\b\d+\b",                                     # số nguyên
]


def make_miner(use_regex):
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = 0.5
    cfg.drain_depth = 4
    cfg.mask_prefix, cfg.mask_suffix = "<", ">"
    cfg.parametrize_numeric_tokens = use_regex
    cfg.masking_instructions = [MaskingInstruction(p, "*") for p in BGL_REGEX] if use_regex else []
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_regex", action="store_true")
    ap.add_argument("--max_lines", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    miner = make_miner(not a.no_regex)

    t0 = time.time(); rows = []; skipped = 0
    with open(a.log, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if a.max_lines and i >= a.max_lines:
                break
            tok = line.rstrip("\n").split(maxsplit=9)
            if len(tok) < 10:          # thiếu phần nội dung -> bỏ qua như LogADEmpirical
                skipped += 1
                continue
            label = 0 if tok[0] == "-" else 1
            r = miner.add_log_message(tok[9])
            rows.append((i + 1, label, tok[1], r["cluster_id"]))
            if (i + 1) % 500000 == 0:
                print(f"  {i+1:,} dòng, {len(miner.drain.clusters):,} template, {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(a.out, "structured.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["LineId", "Label", "Timestamp", "EventId"])
        for lid, lab, ts, cid in rows:
            w.writerow([lid, lab, ts, f"E{cid}"])
    with open(os.path.join(a.out, "templates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["EventId", "EventTemplate", "Occurrences"])
        for c in miner.drain.clusters:
            w.writerow([f"E{c.cluster_id}", c.get_template(), c.size])
    print(f"Xong: {len(rows):,} dòng hợp lệ, bỏ {skipped:,} dòng, "
          f"{len(miner.drain.clusters):,} template, {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
