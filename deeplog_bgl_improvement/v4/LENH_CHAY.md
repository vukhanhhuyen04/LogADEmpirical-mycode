# Hướng dẫn chạy thí nghiệm cải tiến DeepLog/BGL

Bộ mã độc lập, không phụ thuộc LogADEmpirical. Đã chạy thử trọn quy trình trên mẫu BGL_2k của Loghub.

| Tệp | Việc | Kết quả chính → dùng ở báo cáo |
|---|---|---|
| `01_parse_bgl.py` | Drain (drain3), có / không regex | `structured.csv`, `templates.csv` → Bảng 4.14 |
| `02_make_windows.py` | Cửa sổ 20 dòng, chia 72/8/20 theo thời gian, từ điển, thống kê OOV | `window_stats.json` → Bảng 4.14 |
| `03_deeplog.py` | C0: DeepLog gốc, g chọn trên validation | `metrics_C0.json`, `windows_*_C0.csv` |
| `04_oov_mapper.py` | C1: ánh xạ template mới (tương đồng + ngữ cảnh + từ khóa) | `metrics_C1.json`, `grid_C1_val.csv`, `oov_decisions_test.csv` |
| `05_calibrate.py` | C2/C3 và RF/IF: ngưỡng hiệu chỉnh trên validation | `metrics_*.json`, `*_hist.png`, `*_pr.png` → Hình 4.5, 4.6 |
| `06_classic.py` | Random Forest, Isolation Forest | `windows_*_RF.csv`, `windows_*_IF.csv` |
| `07_summary.py` | Gom bảng + ví dụ log | `results.md` → Bảng 4.15–4.18 |

## 1. Cài đặt (một lần)

Trên máy (Mac của Huyền):

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Trên Kaggle (Settings → Accelerator: GPU T4; tải thư mục `code` và `BGL.log` lên làm Dataset):

```python
!cp -r /kaggle/input/<ten-dataset-code>/code /kaggle/working/ && cd /kaggle/working/code
!pip install -q drain3
```

## 2. Chạy toàn bộ (một lệnh)

```bash
bash run_all.sh /duong/dan/BGL.log 42
```

Trên Kaggle: `!cd /kaggle/working/code && bash run_all.sh /kaggle/input/<ten-dataset-bgl>/BGL.log 42`

Thời gian ước tính trên BGL đầy đủ (4,7 triệu dòng):

| Bước | Mac 8 GB (MPS) | Kaggle T4 |
|---|---|---|
| Parse ×2 | 10–20 phút | 10–20 phút |
| DeepLog C0 | 30–90 phút | 15–30 phút |
| Ánh xạ C1 (quét 15 cấu hình trên val) | 15–40 phút | 10–20 phút |
| Còn lại | < 10 phút | < 10 phút |

## 3. Chạy từng bước (nên làm lần đầu để kiểm tra)

```bash
python 01_parse_bgl.py --log BGL.log --out out/regex
python 01_parse_bgl.py --log BGL.log --out out/noregex --no_regex
python 02_make_windows.py --parsed out/regex
python 02_make_windows.py --parsed out/noregex
python 03_deeplog.py --parsed out/regex --out runs/regex_seed42 --seed 42
python 04_oov_mapper.py --parsed out/regex --run runs/regex_seed42
python 05_calibrate.py --run runs/regex_seed42 --src C0 --name C2 --criterion B
python 05_calibrate.py --run runs/regex_seed42 --src C1 --name C3 --criterion B
python 06_classic.py --parsed out/regex --run runs/regex_seed42
python 05_calibrate.py --run runs/regex_seed42 --src RF --name RF_A --criterion A
python 05_calibrate.py --run runs/regex_seed42 --src IF --name IF_A --criterion A
python 05_calibrate.py --run runs/regex_seed42 --src IF --name IF_C --criterion C --fpr_budget 0.01
python 07_summary.py --parsed out/regex --run runs/regex_seed42 --best C3
```

Muốn có 3 seed: chạy lại bước 3–7 với `--seed 1`, `--seed 2` (thư mục `runs/regex_seed1`, ...). Parse và cửa sổ không cần chạy lại.

## 4. Kiểm tra sau mỗi bước

- Bước 1: số template bản `regex` phải **ít hơn rõ rệt** bản `noregex`.
- Bước 2: mở `out/regex/window_stats.json`, chép `ty_le_template_moi_%` và `cua_so_co_template_moi_%` của test vào Bảng 4.14. Tập `val` phải có `bat_thuong` > 0.
- Bước 3: `val_loss` giảm dần; dòng `C0 test co_template_moi` cho biết bao nhiêu FP rơi vào cửa sổ có template mới.
- Bước 4: `grid_C1_val.csv` có 15 dòng; nếu `chap_nhan` = 0 ở mọi dòng thì cơ chế không ánh xạ được gì → xem mục 6.
- Bước 7: mở `runs/regex_seed42/results.md`, chép bảng vào báo cáo.

## 5. Gửi kết quả lại cho Claude

Nén và gửi các tệp: `out/*/window_stats.json`, `runs/regex_seed42/metrics_*.json`, `grid_C1_val.csv`, `results.md`, `*.png`. **Không** cần gửi `windows.pkl`, `deeplog.pt`, các `.csv` lớn.

## 6. Lệnh ngắn cho Claude Code (VS Code) khi cần chỉnh

- RAM không đủ ở bước 3:
  > Trong 03_deeplog.py, hàm pairs dùng nhiều RAM. Sửa để tạo cặp theo từng khối 20.000 cửa sổ, ép kiểu int32 và khử trùng lặp dần. Không đổi kết quả.
- Không có ánh xạ nào được chấp nhận ở bước 4:
  > Trong 04_oov_mapper.py, thêm TAU_GRID 0.3 và 0.4, thêm g' = 5g, và in phân bố độ tương đồng cao nhất của các template mới trên val (min/median/max).
- FN tăng nhiều ở C1:
  > Trong 04_oov_mapper.py, chỉ ánh xạ khi template mới không chứa từ trong NEG (bỏ qua mọi template mới có từ lỗi), chạy lại và so sánh FN.
- Muốn thêm tiêu chí ngưỡng:
  > Chạy 05_calibrate.py cho C0 và C1 với --criterion A và --criterion C --fpr_budget 0.01, đặt tên C2A, C3A, C2C, C3C, rồi chạy lại 07_summary.py.
- Muốn so sánh bản không regex:
  > Chạy 03 đến 05 với --parsed out/noregex, thư mục runs/noregex_seed42, rồi lập bảng so sánh C0 và C3 giữa hai bản.

## 7. Quy tắc bắt buộc (ghi vào báo cáo)

1. g, tau_sim, g', tau đều chọn trên **validation**; test chỉ chấm một lần.
2. Từ điển và TF-IDF chỉ học từ **train**.
3. C0–C3 dùng **cùng một bản parse và cùng split**.
4. Báo cáo cả ΔFP và ΔFN so với C0, không chỉ FP giảm.

## 8. Bản v3: ngưỡng trượt (C5), 3 seed, phân tích độ nhạy

| Tệp | Việc | Kết quả |
|---|---|---|
| `08_soft_oov.py` | C4: điểm mềm cho template mới | `windows_*_C4.csv` |
| `09_sliding_threshold.py` | C5: ngưỡng = phân vị q của N cửa sổ gần nhất; chọn (chế độ lịch sử, N, q) trên validation | `metrics_C5*.json`, `C5_theo_thoi_gian.png` |
| `10_aggregate.py` | Gộp nhiều seed: trung bình ± độ lệch chuẩn | `results_seeds.md` |
| `11_sensitivity.py` | Độ nhạy theo τ_sim và δ | `do_nhay_tau_sim.png`, `do_nhay_delta_C4.png` |
| `run_seeds.sh` | Chạy seed 42 đầy đủ + seed 1, 2 (dùng lại phần parse) rồi gộp | toàn bộ |

Chạy:  `bash run_seeds.sh /duong/dan/BGL.log`  (Kaggle T4: khoảng 2–2,5 giờ)

## 9. Bản v4: phản hồi người vận hành (C6), ngưỡng bền MAD (C5M), mở rộng τ_sim

| Tệp | Việc | Kết quả |
|---|---|---|
| `12_feedback.py` | C6: mô phỏng người vận hành xem tối đa B cảnh báo; báo nhầm thì các template mới trong cửa sổ được xác nhận bình thường | `metrics_C6.json`, `feedback_curve.csv`, `C6_theo_ngan_sach.png`, `C6_theo_thoi_gian.png` |
| `13_robust_threshold.py` | C5M: ngưỡng trượt = trung vị + k·MAD | `metrics_C5M*.json`, `C5M_theo_thoi_gian.png` |
| `14_tau_extended.py` | Độ nhạy khi hạ τ_sim xuống 0,3 và 0,4 | `tau_sim_mo_rong.csv`, `tau_sim_vi_du_anh_xa_thap.csv` |
| `run_v4.sh` | Chạy lại seed 42 rồi chạy 3 bước trên | toàn bộ |

Chạy:  `bash run_v4.sh /duong/dan/BGL.log`  (Kaggle T4: khoảng 1–1,5 giờ)
