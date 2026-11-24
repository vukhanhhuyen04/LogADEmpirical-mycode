#!/usr/bin/env bash
# Chạy đủ 3 seed rồi gộp kết quả. Dùng:  bash run_seeds.sh /duong/dan/BGL.log
# Seed 42 chạy đầy đủ (kể cả bản không regex); seed 1, 2 dùng lại phần parse.
set -e
LOG=${1:?"Thiếu đường dẫn BGL.log"}
bash run_all.sh "$LOG" 42
for S in 1 2; do SKIP_NOREGEX=1 bash run_all.sh "$LOG" $S; done
python 10_aggregate.py --runs runs/regex_seed42 runs/regex_seed1 runs/regex_seed2 --out results_seeds.md
