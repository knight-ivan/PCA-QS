#!/usr/bin/env python3
"""Confirmation rerun of the downstream-classification study, with the review's
protocol fixes applied:

  * split train/test on RAW data BEFORE any preprocessing (no leakage);
  * fit StandardScaler + PCA on the TRAINING frame only;
  * draw PCA-QS and SRS subsets of the SAME exact size from the training frame;
  * evaluate on the untouched test set with a threshold-free metric (AUC) as
    primary, plus F1/recall/precision at a COMMON fixed threshold 0.5
    (not the retained-sample positive rate, which confounds sampling with
    thresholding).

Goal: verify PCA-QS is at least as good as SRS on AUC (statistically), not to
reproduce exact numbers. Reports paired PCA-QS - SRS differences with 95% CIs.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices, choose_design
from pcaqs.data import classification_gmm

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def one_run(gen, N, n_test, retain, k, var_target=0.7):
    X, y = classification_gmm(N + n_test, random_state=gen.integers(1 << 31))
    # raw split BEFORE preprocessing
    perm = gen.permutation(len(X))
    te, tr = perm[:n_test], perm[n_test:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    scaler = StandardScaler().fit(Xtr)                 # fit on train only
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    pca = PCA().fit(Xtr_s)                             # fit on train only
    r = int(round(retain * len(Xtr)))
    k_eff, m, _ = choose_design(r, pca.explained_variance_ratio_, k=k, var_target=var_target, stratification="profile")

    # PCA-QS strata on the training frame; equal exact retained size for both
    qs = PCAQS(n_components=k_eff, n_bins=m, retention=retain, stratification="profile",
               random_state=int(gen.integers(1 << 31)))
    qs.fit(Xtr)
    qi = qs.sample_indices(Xtr, allocation="floor")
    si = srs_indices(len(Xtr), len(qi), random_state=int(gen.integers(1 << 31)))

    res = {}
    for meth, idx in (("PCA-QS", qi), ("SRS", si)):
        for name, clf in (("logistic", LogisticRegression(max_iter=200, class_weight="balanced")),
                          ("rf", RandomForestClassifier(n_estimators=100, n_jobs=1,
                                                        class_weight="balanced", random_state=0))):
            clf.fit(Xtr_s[idx], ytr[idx])
            p = clf.predict_proba(Xte_s)[:, 1]
            yhat = (p >= 0.5).astype(int)              # common fixed threshold
            res[(meth, name)] = dict(
                AUC=roc_auc_score(yte, p),
                F1=f1_score(yte, yhat, zero_division=0),
                recall=recall_score(yte, yhat, zero_division=0),
                precision=precision_score(yte, yhat, zero_division=0))
    return res, k_eff


def _one(b, N, n_test, retain, k, seed):
    res, k_eff = one_run(np.random.default_rng([seed, b]), N, n_test, retain, k)
    return [dict(run=b, method=meth, clf=clf, k=k_eff, **m)
            for (meth, clf), m in res.items()]


def run(runs=60, N=100_000, n_test=20_000, retain=0.05, k=None, seed=7, jobs=-1):
    out = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_one)(b, N, n_test, retain, k, seed) for b in range(runs))
    return pd.DataFrame([r for pair in out for r in pair])


def summarize(df):
    out = []
    for clf in sorted(df.clf.unique()):
        for m in ["AUC", "F1", "recall", "precision"]:
            q = df[(df.clf == clf) & (df.method == "PCA-QS")].sort_values("run")[m].values
            s = df[(df.clf == clf) & (df.method == "SRS")].sort_values("run")[m].values
            d = q - s
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append(dict(clf=clf, metric=m, QS_mean=q.mean(), SRS_mean=s.mean(),
                            diff=d.mean(), ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se,
                            frac_QS_better=float(np.mean(q > s))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--N", type=int, default=100_000)
    ap.add_argument("--retain", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=None, help="fixed k; default = dynamic 70% variance")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    df = run(runs=a.runs, N=a.N, retain=a.retain, k=a.k, jobs=a.jobs)
    df.to_csv(os.path.join(OUT, "confirm_classification.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(OUT, "confirm_classification_summary.csv"), index=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n=== paired PCA-QS - SRS on held-out test (positive diff => PCA-QS better) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
