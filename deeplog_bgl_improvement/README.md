# Cải tiến DeepLog trên BGL

Bộ mã độc lập (không phụ thuộc phần còn lại của LogADEmpirical).

- `v1/`: bản đầu (C0 DeepLog gốc, C1 ánh xạ template mới, C2/C3 hiệu chỉnh ngưỡng, RF/IF).
- `v2/`: v1 + `08_soft_oov.py` (C4: điểm mềm cho template mới); cập nhật `common.py`, `07_summary.py`, `run_all.sh`.

Cách chạy: xem `v2/LENH_CHAY.md` (v2 là bản mới nhất). Ví dụ: `cd deeplog_bgl_improvement/v2 && bash run_all.sh /duong/dan/BGL.log 42`.
Kết quả sinh ra trong `out/` và `runs/` (đã được .gitignore).
