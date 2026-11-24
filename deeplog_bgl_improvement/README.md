# Cải tiến DeepLog trên BGL

Bộ mã độc lập (không phụ thuộc phần còn lại của LogADEmpirical).

- `v1/`: bản đầu (C0 DeepLog gốc, C1 ánh xạ template mới, C2/C3 hiệu chỉnh ngưỡng, RF/IF).
- `v2/`: v1 + `08_soft_oov.py` (C4: điểm mềm cho template mới); cập nhật `common.py`, `07_summary.py`, `run_all.sh`.
- `v3/`: v2 + `09_sliding_threshold.py`, `10_aggregate.py`, `11_sensitivity.py`, `run_seeds.sh`; cập nhật `07_summary.py`, `run_all.sh`.

Cách chạy: xem `v3/LENH_CHAY.md` (v3 là bản mới nhất). Ví dụ: `cd deeplog_bgl_improvement/v3 && bash run_all.sh /duong/dan/BGL.log 42`.
Kết quả sinh ra trong `out/` và `runs/` (đã được .gitignore).
