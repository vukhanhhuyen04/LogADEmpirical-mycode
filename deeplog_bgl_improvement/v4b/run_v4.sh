#!/usr/bin/env bash
# Bản v4: chạy lại pipeline seed 42 (bỏ bản không regex cho nhanh) rồi thêm C6, C5M, tau_sim mở rộng.
# Dùng:  bash run_v4.sh /duong/dan/BGL.log
set -e
LOG=${1:?"Thiếu đường dẫn BGL.log"}; RUN=runs/regex_seed42
SKIP_NOREGEX=1 bash run_all.sh "$LOG" 42
echo "== 12. C6: phản hồi của người vận hành =="
python 12_feedback.py --parsed out/regex --run $RUN --budget 100
echo "== 13. C5M: ngưỡng trượt bền (trung vị + k*MAD) =="
python 13_robust_threshold.py --run $RUN --src C4 --name C5M
python 13_robust_threshold.py --run $RUN --src C1 --name C5M_C1 --fixed C3
echo "== 14. Mở rộng lưới tau_sim =="
python 14_tau_extended.py --parsed out/regex --run $RUN
echo "== 7. Tổng hợp lại =="
python 07_summary.py --parsed out/regex --run $RUN --best C4
