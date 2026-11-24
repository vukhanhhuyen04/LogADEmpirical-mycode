"""Hàm dùng chung: đọc/ghi, chỉ số đánh giá, mô hình DeepLog, chọn ngưỡng."""
import json
import math
import os
import pickle

import numpy as np

PAD, UNK = 0, 1
UNK_NLL = -math.log(1e-6)  # điểm gán cho sự kiện chưa biết (~13.8)


# ---------------------------------------------------------------- I/O
def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- metrics
def metrics(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=p, Recall=r, F1=f1,
                Specificity=spec, FPR=1 - spec if tn + fp else 0.0, N=len(y))


def metrics_by_oov(y, pred, has_oov):
    has_oov = np.asarray(has_oov).astype(bool)
    return {"all": metrics(y, pred),
            "co_template_moi": metrics(np.asarray(y)[has_oov], np.asarray(pred)[has_oov]),
            "khong_template_moi": metrics(np.asarray(y)[~has_oov], np.asarray(pred)[~has_oov])}


def fmt_row(name, m):
    return (f"{name:<28} P={m['Precision']*100:6.2f} R={m['Recall']*100:6.2f} "
            f"F1={m['F1']*100:6.2f} Spec={m['Specificity']*100:8.4f} "
            f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}")


# ---------------------------------------------------------------- threshold
def candidate_thresholds(scores, max_n=3000):
    u = np.unique(np.asarray(scores, dtype=float))
    if len(u) > max_n:
        u = np.unique(np.quantile(u, np.linspace(0, 1, max_n)))
    return np.concatenate([u, [u[-1] + 1e-9]])


def choose_threshold(y, scores, criterion="B", recall_base=None, delta=0.02, fpr_budget=0.01):
    """Chọn ngưỡng tau trên validation; dự đoán bất thường khi score >= tau.
    A: F1 lớn nhất (hòa thì lấy tau lớn hơn).
    B: FP nhỏ nhất với ràng buộc Recall >= recall_base - delta.
    C: tau = phân vị (1 - fpr_budget) của điểm các cửa sổ bình thường."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    if criterion == "C":
        tau = float(np.quantile(s[y == 0], 1 - fpr_budget))
        return tau, metrics(y, s >= tau)
    best = None
    for tau in candidate_thresholds(s):
        m = metrics(y, s >= tau)
        if criterion == "A":
            key = (m["F1"], tau)
        else:
            if recall_base is None:
                raise ValueError("Tiêu chí B cần recall_base")
            if m["Recall"] < recall_base - delta - 1e-12:
                continue
            key = (-m["FP"], tau)
        if best is None or key > best[0]:
            best = (key, float(tau), m)
    if best is None:  # không ngưỡng nào thỏa ràng buộc -> lấy ngưỡng nhỏ nhất
        tau = float(s.min())
        return tau, metrics(y, s >= tau)
    return best[1], best[2]


# ---------------------------------------------------------------- DeepLog
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(vocab_size, emb=64, hidden=128, layers=2):
    import torch.nn as nn

    class DeepLog(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
            self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True)
            self.fc = nn.Linear(hidden, vocab_size)

        def forward(self, x):
            out, _ = self.lstm(self.emb(x))
            return self.fc(out[:, -1, :])

    return DeepLog()


def histories(seq, h):
    """Cặp (lịch sử h sự kiện, sự kiện kế tiếp) cho mọi vị trí t >= 1."""
    xs, ys = [], []
    for t in range(1, len(seq)):
        hist = seq[max(0, t - h):t]
        xs.append([PAD] * (h - len(hist)) + list(hist))
        ys.append(seq[t])
    return xs, ys


def score_sequences(model, seqs, h, device, batch=8192):
    """Trả về (max_nll, max_rank) cho từng cửa sổ.
    Sự kiện kế tiếp là UNK -> nll = UNK_NLL, rank = vocab_size (luôn vượt top-g)."""
    import torch
    model.eval()
    V = model.fc.out_features
    owner, xs, ys = [], [], []
    for i, s in enumerate(seqs):
        x, y = histories(s, h)
        xs += x; ys += y; owner += [i] * len(y)
    owner = np.asarray(owner, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    nll = np.zeros(len(ys)); rank = np.zeros(len(ys), dtype=np.int64)
    with torch.no_grad():
        for a in range(0, len(ys), batch):
            xb = torch.tensor(xs[a:a + batch], dtype=torch.long, device=device)
            yb = torch.tensor(ys[a:a + batch], dtype=torch.long, device=device)
            logits = model(xb)
            logits[:, PAD] = -1e9; logits[:, UNK] = -1e9
            logp = torch.log_softmax(logits, dim=1)
            tgt = logp.gather(1, yb.clamp(min=0).unsqueeze(1)).squeeze(1)
            r = (logp > tgt.unsqueeze(1)).sum(1) + 1
            nll[a:a + len(yb)] = (-tgt).cpu().numpy()
            rank[a:a + len(yb)] = r.cpu().numpy()
    unk = ys == UNK
    nll[unk] = UNK_NLL; rank[unk] = V
    n = len(seqs)
    max_nll = np.zeros(n); max_rank = np.zeros(n, dtype=np.int64)
    np.maximum.at(max_nll, owner, nll)
    np.maximum.at(max_rank, owner, rank)
    # v2: sự kiện đầu cửa sổ không có lịch sử để dự đoán; nếu là template chưa biết
    # thì vẫn tính là bằng chứng bất thường (bản v1 bỏ qua vị trí này).
    first_unk = np.array([len(s) > 0 and s[0] == UNK for s in seqs], dtype=bool)
    max_nll[first_unk] = UNK_NLL
    max_rank[first_unk] = V
    return max_nll, max_rank
