#!/usr/bin/env python3
"""Confirmation rerun of the real-data structure-preservation study.

For each dataset: standardize the numeric features, fit PCA, and compare PCA-QS
(occupancy-limited bins) against SRS at an identical exact retained size, over
several random subsampling replicates. Metrics (corrected definitions) are
computed between the retained subset and the full data in the top-k PCA-score
space (``--space pca``, the default) or in the original standardized feature
space (``--space original``); sampling strata always come from the PCA-score
space, only the scoring space changes. Reports, per dataset and metric, the mean
PCA-QS and SRS values and the fraction of replicates on which PCA-QS is closer to
the full data.

Datasets are read from --data-root; only lightweight ones are enabled by default.
"""
import os, sys, argparse, glob
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices
from pcaqs.metrics import quantile_error, energy_distance, mmd_rbf, mahalanobis_mean

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]

# dataset -> (glob, drop-by-name-substring, header, drop-by-column-index)
DATASETS = {
    "CreditCard":  ("**/UCI_Credit_Card.csv", ["ID", "default"], "infer", []),
    "MAGIC":       ("**/magic04.csv", [], "infer", []),
    "EEG":         ("**/EEG_Eye_State.csv", [], "infer", []),
    "Epileptic":   ("**/Epileptic_Seizure_Recognition.csv", ["Unnamed", "y"], "infer", []),
    "OnlineNews":  ("**/OnlineNewsPopularity.csv", ["url", "shares"], "infer", []),
    # heavy datasets: HIGGS-2 is a 197k-row representative subset (the 2.6G HIGGS.csv
    # is gzip); YearPrediction col 0 is the year target (dropped).
    "HIGGS":       ("**/HIGGS-2.csv", [], None, []),
    "YearPrediction": ("**/YearPredictionMSD.csv", [], None, [0]),
}


def load_numeric(path, drop_subs, header="infer", drop_idx=None):
    df = pd.read_csv(path, header=header)
    df = df.select_dtypes(include=[np.number])
    if drop_idx:
        df = df.drop(columns=[df.columns[i] for i in drop_idx])
    for s in drop_subs:
        df = df[[c for c in df.columns if s.lower() not in str(c).lower()]]
    X = df.to_numpy(dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    return X


def marginal_hist_kl_js(A, B, bins=30):
    kl, js = [], []
    for j in range(A.shape[1]):
        lo = min(A[:, j].min(), B[:, j].min()); hi = max(A[:, j].max(), B[:, j].max())
        if hi <= lo:
            continue
        grid = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(A[:, j], grid); q, _ = np.histogram(B[:, j], grid)
        p = (p + 1.0) / (p.sum() + bins); q = (q + 1.0) / (q.sum() + bins)
        kl.append(np.sum(p * np.log(p / q)))
        m = 0.5 * (p + q)
        js.append(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return float(np.mean(kl)), float(np.mean(js))


def pairwise_w1(A, B, n_anchor, gen):
    def pd_(M):
        S = M[gen.choice(len(M), size=min(n_anchor, len(M)), replace=False)]
        D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
        return D[np.triu_indices(len(S), 1)]
    return float(wasserstein_distance(pd_(A), pd_(B)))


def one_rep(sc, r, k, m, seed, score_on=None):
    # Strata are always built on the PCA-score space `sc`; metrics are evaluated
    # on `score_on` (defaults to `sc` = PCA subspace; pass Xs for original space).
    if score_on is None:
        score_on = sc
    gen = np.random.default_rng(seed)
    qs = PCAQS(n_components=k, n_bins=m, retention=r / len(sc),
               random_state=int(gen.integers(1 << 31)))
    # scores already computed; emulate strata on the score space directly:
    qs.mean_ = np.zeros(sc.shape[1]); qs.scale_ = np.ones(sc.shape[1])
    qs.components_ = np.eye(sc.shape[1])
    qi = qs.sample_indices(sc, exact_size=r)
    si = srs_indices(len(sc), r, random_state=int(gen.integers(1 << 31)))
    rows = []
    for meth, idx in (("PCA-QS", qi), ("SRS", si)):
        A, B = score_on, score_on[idx]
        kl, js = marginal_hist_kl_js(A, B)
        rows.append(dict(method=meth,
                         quantile_error=quantile_error(A, B), KL=kl, JS=js,
                         energy=energy_distance(A, B, random_state=0),
                         MMD=mmd_rbf(A, B, random_state=0),
                         Mahalanobis=mahalanobis_mean(A, B),
                         pairwise_W1=pairwise_w1(A, B, 400, gen)))
    return rows


def run_dataset(name, path, drop_subs, reps, retain, var_target, jobs, fixed_k=None,
                base_seed=20260806, header="infer", drop_idx=None, space="pca"):
    X = load_numeric(path, drop_subs, header, drop_idx)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    k = fixed_k if fixed_k else int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_target) + 1)
    r = int(round(retain * len(Xs)))
    m = max(2, int(np.floor(r ** (1.0 / k))))
    sc = Xs @ pca.components_[:k].T
    score_on = sc if space == "pca" else Xs   # PCA subspace (default) or original feature space
    # Original matrix-comparison code used an UNSEEDED rng; we set an explicit,
    # reproducible per-replicate seed (base_seed + rep) and record it.
    out = Parallel(n_jobs=jobs)(delayed(one_rep)(sc, r, k, m, base_seed + s, score_on) for s in range(reps))
    df = pd.DataFrame([dict(dataset=name, space=space, N=len(Xs), d=X.shape[1], k=k, m=m, r=r,
                            rep=i, seed=base_seed + i, **row)
                       for i, pair in enumerate(out) for row in pair])
    print(f"{name}: N={len(Xs)} d={X.shape[1]} k={k} m={m} r={r} reps={reps} base_seed={base_seed}", flush=True)
    return df


def summarize(df):
    out = []
    for name, g in df.groupby("dataset"):
        k = int(g["k"].iloc[0]); m = int(g["m"].iloc[0])
        for metric in METRICS:
            q = g[g.method == "PCA-QS"].sort_values("rep")[metric].values
            s = g[g.method == "SRS"].sort_values("rep")[metric].values
            d = q - s                                    # paired difference
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append(dict(dataset=name, k=k, m=m, metric=metric,
                            QS_mean=q.mean(), SRS_mean=s.mean(), diff=d.mean(),
                            ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se,
                            frac_QS_better=float(np.mean(q < s))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    _proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--data-root", default=os.path.join(
        _proj, "LaTeX", "Final Version", "Real Data Marix Comparions csv"))
    ap.add_argument("--datasets", nargs="+", default=["CreditCard", "MAGIC", "EEG", "Epileptic"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--retain", type=float, default=0.05)
    ap.add_argument("--var", type=float, default=0.70)
    ap.add_argument("--k", type=int, default=None, help="fixed k (default: dynamic var threshold)")
    ap.add_argument("--seed", type=int, default=20260806, help="base seed; per-rep seed = seed + rep")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--space", choices=["pca", "original"], default="pca",
                    help="score metrics in the top-k PCA subspace (default) or the original feature space")
    a = ap.parse_args()
    frames = []
    for name in a.datasets:
        pat, drop, header, drop_idx = DATASETS[name]
        hits = glob.glob(os.path.join(a.data_root, pat), recursive=True)
        if not hits:
            print(f"[skip] {name}: no file matching {pat}", flush=True); continue
        frames.append(run_dataset(name, sorted(hits)[0], drop, a.reps, a.retain, a.var,
                                   a.jobs, a.k, a.seed, header, drop_idx, a.space))
    df = pd.concat(frames, ignore_index=True)
    tag = "" if a.space == "pca" else "_origspace"
    df.to_csv(os.path.join(OUT, f"confirm_real_data{tag}.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(OUT, f"confirm_real_data{tag}_summary.csv"), index=False)
    pd.set_option("display.width", 170, "display.max_columns", 20)
    print("\n=== per dataset: fraction of reps PCA-QS closer to full data ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
