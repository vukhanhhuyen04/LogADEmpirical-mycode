"""Random Forest baseline on template-count vectors.

Uses exactly the same windows and the same chronological 80/20 split as the deep models
(experimental_results/full/random/<dataset>/20/{train,test}.pkl: window 20, step 20).
A window's label is max(Label) over its lines. 10% of train windows are held out for
validation (random, seed 42), like the 10% validation of the deep models.

Usage: python rf_baseline.py --dataset bgl|tbird [--data_dir DIR] [--out_dir results_week/rf/]
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier


def load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    events = [list(s["EventId"]) for s in data]
    labels = np.array([int(np.max(s["Label"])) for s in data], dtype=np.int8)
    return events, labels


def vectorize(events, vocab):
    """CSR matrix of template counts; the last column counts templates not seen in train."""
    oov = len(vocab)
    indptr = np.zeros(len(events) + 1, dtype=np.int64)
    cols = []
    for i, ev in enumerate(events):
        cols.extend(vocab.get(e, oov) for e in ev)
        indptr[i + 1] = len(cols)
    cols = np.asarray(cols, dtype=np.int32)
    m = sp.csr_matrix((np.ones(len(cols), dtype=np.float32), cols, indptr),
                      shape=(len(events), oov + 1))
    m.sum_duplicates()  # repeated templates in a window become counts
    return m


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = 100 * tp / (tp + fp) if tp + fp else 0.0
    r = 100 * tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    sp_ = tn / (tn + fp) if tn + fp else 0.0
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, precision=p, recall=r, f1=f1, specificity=sp_)


def show(name, m):
    print("{}: TP={TP} TN={TN} FP={FP} FN={FN} | Precision={precision:.3f}% Recall={recall:.3f}% "
          "F1={f1:.3f}% Specificity={specificity:.4f}".format(name, **m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bgl", "tbird"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="results_week/rf/")
    args = ap.parse_args()
    data_dir = args.data_dir or "experimental_results/full/random/{}/20/".format(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    tr_events, tr_y = load(os.path.join(data_dir, "train.pkl"))
    te_events, te_y = load(os.path.join(data_dir, "test.pkl"))
    print("loaded train={} (abnormal {}) test={} (abnormal {}) in {:.1f}s".format(
        len(tr_y), int(tr_y.sum()), len(te_y), int(te_y.sum()), time.time() - t0))

    # vocabulary from all train windows (the model is supervised)
    vocab = {}
    for ev in tr_events:
        for e in ev:
            if e not in vocab:
                vocab[e] = len(vocab)
    t0 = time.time()
    X_all = vectorize(tr_events, vocab)
    X_te = vectorize(te_events, vocab)
    feat_time = time.time() - t0
    oov = int(X_te[:, len(vocab)].sum())
    print("templates in train: {} | windows 20 lines | feature time {:.1f}s | unseen-template "
          "occurrences in test: {}".format(len(vocab), feat_time, oov))

    perm = np.random.RandomState(42).permutation(len(tr_y))
    n_val = int(len(perm) * 0.1)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    X_fit, y_fit = X_all[fit_idx], tr_y[fit_idx]
    X_val, y_val = X_all[val_idx], tr_y[val_idx]
    print("fit={} (abnormal {}) val={} (abnormal {})".format(
        len(y_fit), int(y_fit.sum()), len(y_val), int(y_val.sum())))

    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_fit, y_fit)
    train_time = time.time() - t0
    print("train time {:.1f}s".format(train_time))

    t0 = time.time()
    pred_te = rf.predict(X_te)
    test_time = time.time() - t0
    res_test = metrics(te_y, pred_te)
    res_val = metrics(y_val, rf.predict(X_val))
    show("VALIDATION", res_val)
    show("TEST", res_test)
    print("test predict time {:.2f}s (+ feature time {:.1f}s for train+test)".format(test_time, feat_time))

    out = dict(dataset=args.dataset, data_dir=data_dir, n_templates=len(vocab),
               n_train_fit=int(len(y_fit)), n_val=int(len(y_val)), n_test=int(len(te_y)),
               test_abnormal=int(te_y.sum()), unseen_template_occurrences_in_test=oov,
               train_time_s=train_time, test_predict_time_s=test_time, feature_time_s=feat_time,
               validation=res_val, test=res_test)
    with open(os.path.join(args.out_dir, "rf_{}.json".format(args.dataset)), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
