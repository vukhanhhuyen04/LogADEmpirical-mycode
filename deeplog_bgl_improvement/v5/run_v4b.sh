#!/usr/bin/env bash
# Chỉ chạy C6 với chiến lược "moi" (xem cảnh báo có template chưa từng xem), dùng lại kết quả v4 đã có.
set -e
RUN=runs/regex_seed42
python 12_feedback.py --parsed out/regex --run $RUN --budget 100 --strategy moi --suffix _moi
python 07_summary.py --parsed out/regex --run $RUN --best C4
