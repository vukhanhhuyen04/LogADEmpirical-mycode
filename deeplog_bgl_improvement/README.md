# Cải tiến DeepLog trên BGL

Bộ mã độc lập (không phụ thuộc phần còn lại của LogADEmpirical).

- `v1/`: bản đầu (C0 DeepLog gốc, C1 ánh xạ template mới, C2/C3 hiệu chỉnh ngưỡng, RF/IF).
- `v2/`: v1 + `08_soft_oov.py` (C4: điểm mềm cho template mới); cập nhật `common.py`, `07_summary.py`, `run_all.sh`.
- `v3/`: v2 + `09_sliding_threshold.py`, `10_aggregate.py`, `11_sensitivity.py`, `run_seeds.sh`; cập nhật `07_summary.py`, `run_all.sh`.
- `v4/`: v3 + `12_feedback.py`, `13_robust_threshold.py`, `14_tau_extended.py`, `run_v4.sh`; cập nhật `07_summary.py`, `10_aggregate.py`.
- `v4b/`: v4 + `run_v4b.sh`; cập nhật `07_summary.py`, `12_feedback.py`.
- `notebooks/`: notebook Kaggle `deeplog-bgl.ipynb`, `-v2`, `-v3`, `-v4` (kèm output của các lần chạy).

Cách chạy: xem `v4b/LENH_CHAY.md` (v4b là bản mới nhất). Ví dụ: `cd deeplog_bgl_improvement/v3 && bash run_all.sh /duong/dan/BGL.log 42`.
Kết quả sinh ra trong `out/` và `runs/` (đã được .gitignore).
