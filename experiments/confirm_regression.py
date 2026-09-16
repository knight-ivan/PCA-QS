#!/usr/bin/env python3
"""Confirmation rerun of the appendix regression comparison: PCA-QS vs SRS vs
leverage-score sampling vs a leverage-based coreset, on CASP, Bike Sharing, and
Appliances Energy. Datasets are fetched from OpenML / UCI (no archived code/data
existed for this study). Metrics: test MSE and R^2 of a ridge regression trained
on each equal-size retained subset, over many seeded replicates.

Success is read the same way as elsewhere: PCA-QS should be competitive --- at or
near the best --- not necessarily strictly dominant.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def load(name):
    if name == "CASP":
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv")
        y = df["RMSD"].to_numpy(float); X = df.drop(columns=["RMSD"]).to_numpy(float)
    elif name == "BikeSharing":
        d = fetch_openml(data_id=42712, as_frame=True)
        y = pd.to_numeric(d.target, errors="coerce").to_numpy()
        X = d.data.select_dtypes(include=[np.number]).to_numpy(float)
    elif name == "Appliances":
        d = fetch_openml("appliances_energy_prediction", version=1, as_frame=True)
        y = pd.to_numeric(d.target, errors="coerce").to_numpy()
        X = d.data.select_dtypes(include=[np.number]).to_numpy(float)
    ok = np.isfinite(y)
    return X[ok], y[ok]


def leverage(Xtr):
    """Statistical leverage of each training row (design with intercept), via QR."""
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Q, _ = np.linalg.qr(A)
    return np.clip((Q ** 2).sum(1), 1e-12, None)


def one_rep(X, y, seed, retain, var_target):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    if not np.isfinite(Xtr).all():
        imp = SimpleImputer(strategy="median").fit(Xtr); Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    n = len(Xtr); r = int(round(retain * n)); gen = np.random.default_rng(seed)

    lev = leverage(Xtr_s)
    # PCA-QS config (occupancy-capped)
    from sklearn.decomposition import PCA
    pca = PCA().fit(Xtr_s)
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_target) + 1)
    k = min(k, Xtr.shape[1], max(2, int(np.floor(np.log2(r)))))
    m = max(2, int(np.floor(r ** (1.0 / k))))
    qs = PCAQS(n_components=k, n_bins=m, retention=retain, random_state=int(gen.integers(1 << 31)))
    qs.fit(Xtr)

    methods = {}
    methods["SRS"] = (srs_indices(n, r, random_state=int(gen.integers(1 << 31))), None)
    methods["PCA-QS"] = (qs.sample_indices(Xtr, exact_size=r), None)
    # leverage-score sampling: prob ∝ leverage, unweighted subset (matches others)
    pl = lev / lev.sum()
    methods["Leverage"] = (gen.choice(n, size=r, replace=False, p=pl), None)
    # coreset: leverage importance sample WITH reweighting (weighted ridge)
    idx_c = gen.choice(n, size=r, replace=True, p=pl)
    w_c = 1.0 / (r * pl[idx_c])
    methods["Coreset"] = (idx_c, w_c)

    rows = []
    for meth, (idx, w) in methods.items():
        reg = Ridge(alpha=1.0).fit(Xtr_s[idx], ytr[idx], sample_weight=w)
        p = reg.predict(Xte_s)
        rows.append(dict(method=meth, k=k, m=m,
                         MSE=mean_squared_error(yte, p), R2=r2_score(yte, p)))
    return rows


def run(name, reps, retain, var_target, base_seed, jobs):
    X, y = load(name)
    print(f"{name}: N={len(X)} d={X.shape[1]} reps={reps} seed={base_seed}+i", flush=True)
    out = Parallel(n_jobs=jobs)(delayed(one_rep)(X, y, base_seed + i, retain, var_target) for i in range(reps))
    return pd.DataFrame([dict(dataset=name, rep=i, **row) for i, rows in enumerate(out) for row in rows])


def summarize(df):
    rows = []
    for name, g in df.groupby("dataset"):
        base = g[g.method == "SRS"].sort_values("rep")
        for meth in ["PCA-QS", "SRS", "Leverage", "Coreset"]:
            m = g[g.method == meth].sort_values("rep")
            d = (m["MSE"].values - base["MSE"].values)   # vs SRS (negative = better)
            rows.append(dict(dataset=name, method=meth, MSE=m["MSE"].mean(), R2=m["R2"].mean(),
                             MSE_vs_SRS=d.mean(),
                             win_vs_SRS=float(np.mean(m["MSE"].values < base["MSE"].values))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["CASP", "BikeSharing", "Appliances"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--retain", type=float, default=0.05)
    ap.add_argument("--var", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    df = pd.concat([run(n, a.reps, a.retain, a.var, a.seed, a.jobs) for n in a.datasets], ignore_index=True)
    df.to_csv(os.path.join(OUT, "confirm_regression.csv"), index=False)
    summ = summarize(df); summ.to_csv(os.path.join(OUT, "confirm_regression_summary.csv"), index=False)
    pd.set_option("display.width", 170, "display.max_columns", 20)
    print("\n=== regression: mean test MSE / R^2 per method (lower MSE / higher R2 better) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
