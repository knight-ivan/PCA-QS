#!/usr/bin/env python3
"""Confirmation rerun of the REAL-DATA downstream-classification study.

Matches the original scale and seed (n_repeats = 1000, per-replicate seed 42+i,
deduction 0.1, PC configs fixed_5 / fixed_10 / dynamic_0.7, logistic regression),
but applies the review's protocol fixes:
  * split train/test on RAW data BEFORE preprocessing (no leakage);
  * fit StandardScaler + PCA on the TRAINING frame only;
  * PCA-QS and SRS subsets of the SAME exact size from the training frame;
  * occupancy-limited bins (m^k <= r);
  * threshold-free AUC (primary) + F1/recall/precision at a common 0.5 threshold
    (not the retained-sample positive rate, which confounds sampling with
    thresholding).

Datasets: Credit Card is bundled/available. APS Failure raw data
(aps_failure_training_set.csv) is NOT in the archive; provide it via --aps-path to
include it.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
PC_CONFIGS = [("fixed_5", 5), ("fixed_10", 10), ("dynamic", None)]


def load_xy(path, label_col):
    df = pd.read_csv(path)
    if label_col is None:
        label_col = df.columns[-1]
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok].astype(int)


def one_rep(X, y, seed, retain, var_target):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    if not np.isfinite(Xtr).all():                     # APS has missing values
        imp = SimpleImputer(strategy="median").fit(Xtr)   # fit on train only
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
    scaler = StandardScaler().fit(Xtr)                 # fit on train only
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    pca = PCA().fit(Xtr_s)
    r = int(round(retain * len(Xtr)))
    gen = np.random.default_rng(seed)
    rows = []
    for cfg, pc in PC_CONFIGS:
        k = pc if pc else int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_target) + 1)
        k = min(k, Xtr.shape[1], max(2, int(np.floor(np.log2(r)))))   # occupancy cap: 2^k <= r
        m = max(2, int(np.floor(r ** (1.0 / k))))
        qs = PCAQS(n_components=k, n_bins=m, retention=retain,
                   random_state=int(gen.integers(1 << 31)))
        qs.fit(Xtr)
        qi = qs.sample_indices(Xtr, exact_size=r)
        si = srs_indices(len(Xtr), r, random_state=int(gen.integers(1 << 31)))
        for meth, idx in (("PCA-QS", qi), ("SRS", si)):
            clf = LogisticRegression(max_iter=1000).fit(Xtr_s[idx], ytr[idx])
            p = clf.predict_proba(Xte_s)[:, 1]
            yhat = (p >= 0.5).astype(int)
            rows.append(dict(pc_config=cfg, k=k, m=m, method=meth,
                             AUC=roc_auc_score(yte, p),
                             F1=f1_score(yte, yhat, zero_division=0),
                             recall=recall_score(yte, yhat, zero_division=0),
                             precision=precision_score(yte, yhat, zero_division=0)))
    return rows


def run(X, y, name, reps, retain, var_target, base_seed, jobs):
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(one_rep)(X, y, base_seed + i, retain, var_target) for i in range(reps))
    df = pd.DataFrame([dict(dataset=name, rep=i, seed=base_seed + i, **row)
                       for i, rows in enumerate(out) for row in rows])
    return df


def summarize(df):
    out = []
    for name, gd in df.groupby("dataset"):
        for cfg, g in gd.groupby("pc_config"):
            for m in ["AUC", "F1", "recall", "precision"]:
                q = g[g.method == "PCA-QS"].sort_values("rep")[m].values
                s = g[g.method == "SRS"].sort_values("rep")[m].values
                d = q - s
                se = d.std(ddof=1) / np.sqrt(len(d))
                out.append(dict(dataset=name, pc_config=cfg, metric=m,
                                QS_mean=q.mean(), SRS_mean=s.mean(), diff=d.mean(),
                                ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se,
                                frac_QS_better=float(np.mean(q > s))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    _proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--credit-path", default=os.path.join(
        _proj, "LaTeX", "Final Version", "Real Data Logistic Regression",
        "crediccard", "UCI_Credit_Card.csv"))
    ap.add_argument("--aps-path", default=None, help="raw aps_failure_training_set.csv (else fetched from OpenML)")
    ap.add_argument("--no-aps", action="store_true", help="skip the APS Failure dataset")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--retain", type=float, default=0.1)
    ap.add_argument("--var", type=float, default=0.7)
    ap.add_argument("--base-seed", type=int, default=42, help="original seed = 42 + i")
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    frames = []
    if os.path.exists(a.credit_path):
        X, y = load_xy(a.credit_path, None)
        print(f"CreditCard: N={len(X)} d={X.shape[1]} pos_rate={y.mean():.3f} reps={a.reps} seed=42+i", flush=True)
        frames.append(run(X, y, "CreditCard", a.reps, a.retain, a.var, a.base_seed, a.jobs))
    else:
        print(f"[skip] CreditCard: {a.credit_path} not found", flush=True)
    if not a.no_aps:
        if a.aps_path and os.path.exists(a.aps_path):
            X, y = load_xy(a.aps_path, "class")
        else:                                          # fetch from OpenML (authorised)
            from sklearn.datasets import fetch_openml
            d = fetch_openml("APSFailure", version=1, as_frame=True)
            X = d.data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            y = (d.target.to_numpy() == "pos").astype(int)
        print(f"APS: N={len(X)} d={X.shape[1]} pos_rate={y.mean():.3f} reps={a.reps} seed=42+i", flush=True)
        frames.append(run(X, y, "APS", a.reps, a.retain, a.var, a.base_seed, a.jobs))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(OUT, "confirm_classification_real.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(OUT, "confirm_classification_real_summary.csv"), index=False)
    pd.set_option("display.width", 170, "display.max_columns", 20)
    print("\n=== paired PCA-QS - SRS on held-out test (positive diff => PCA-QS better) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
