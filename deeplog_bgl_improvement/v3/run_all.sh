#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm cải tiến DeepLog/BGL.  Dùng:  bash run_all.sh /duong/dan/BGL.log 42
set -e
LOG=${1:?"Thiếu đường dẫn BGL.log"}; SEED=${2:-42}
OUT=out; RUN=runs/regex_seed$SEED

echo "== 1. Parse (có regex và không regex để so sánh) =="
[ -f $OUT/regex/structured.csv ]   || python 01_parse_bgl.py --log "$LOG" --out $OUT/regex
[ -n "$SKIP_NOREGEX" ] || [ -f $OUT/noregex/structured.csv ] || python 01_parse_bgl.py --log "$LOG" --out $OUT/noregex --no_regex

echo "== 2. Cửa sổ 20 dòng, chia 72/8/20 theo thời gian =="
[ -f $OUT/regex/windows.pkl ]   || python 02_make_windows.py --parsed $OUT/regex
[ -n "$SKIP_NOREGEX" ] || [ -f $OUT/noregex/windows.pkl ] || python 02_make_windows.py --parsed $OUT/noregex

echo "== 3. C0: DeepLog gốc, g chọn trên validation =="
python 03_deeplog.py --parsed $OUT/regex --out $RUN --seed $SEED

echo "== 4. C1: xử lý template mới =="
python 04_oov_mapper.py --parsed $OUT/regex --run $RUN

echo "== 5. C2, C3: hiệu chỉnh ngưỡng (tiêu chí B) =="
python 05_calibrate.py --run $RUN --src C0 --name C2 --criterion B
python 05_calibrate.py --run $RUN --src C1 --name C3 --criterion B

echo "== 5b. C4: điểm mềm cho template mới + hiệu chỉnh ngưỡng (v2) =="
python 08_soft_oov.py --parsed $OUT/regex --run $RUN
python 05_calibrate.py --run $RUN --src C4 --name C4 --criterion B
python 05_calibrate.py --run $RUN --src C4 --name C4A --criterion A

echo "== 5c. C5: ngưỡng thích nghi theo cửa sổ trượt (v3) =="
python 09_sliding_threshold.py --run $RUN --src C4 --name C5 --criterion B
python 09_sliding_threshold.py --run $RUN --src C4 --name C5A --criterion A --fixed C4A
python 09_sliding_threshold.py --run $RUN --src C1 --name C5_C1 --criterion B --fixed C3

echo "== 6. Mô hình truyền thống =="
python 06_classic.py --parsed $OUT/regex --run $RUN --seed $SEED
python 05_calibrate.py --run $RUN --src RF --name RF_A --criterion A
python 05_calibrate.py --run $RUN --src IF --name IF_A --criterion A
python 05_calibrate.py --run $RUN --src IF --name IF_C --criterion C --fpr_budget 0.01

echo "== 6b. Phân tích độ nhạy =="
python 11_sensitivity.py --run $RUN

echo "== 7. Tổng hợp =="
python 07_summary.py --parsed $OUT/regex --run $RUN --best C4
