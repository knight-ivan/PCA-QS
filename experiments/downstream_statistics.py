#!/usr/bin/env python3
"""Information preservation for standard downstream statistical analyses.

For each real dataset (standardized features), a PCA-QS partition is built once on the full
data and then, over 1000 replicates with fixed seeds, a PCA-QS subsample (companion
allocation m_g = max(1, floor(delta N_g))) and a simple random subsample of the SAME
realized size are drawn. On each subsample we run the analyses a practitioner would run on a
reduced dataset and compare them with the same analyses on the full data:

  mean        feature means                      ||mean_sub - mean_full||^2
  sd          feature standard deviations        ||sd_sub - sd_full||^2
  quantiles   feature deciles                     mean over features of sum_q (q_sub - q_full)^2
  corr        correlation matrix                  ||R_sub - R_full||_F^2
  eigvals     top-5 covariance eigenvalues       sum_j (lam_sub - lam_full)^2
  subspace    top-3 principal subspace           ||sin Theta(V_sub, V_full)||_F^2

Designs (--design):
  profile     cutoff-count profiles with B = 5 and k = min(k*, 10), k* the smallest k
              explaining 70% of the variance -- the companion paper's configuration;
  profile1    profile design with k = 1 (the theory's optimum for linear summaries);
  grid        full cross-classification, k = 3, m = floor((r/10)^(1/3)).

Estimates use design weights (Proposition 1). The efficiency of PCA-QS for each analysis is
MSE_QS / MSE_SRS (< 1 favours PCA-QS) with a bootstrap 95% interval; for the means the exact
design-variance ratio is also reported (no Monte Carlo error), together with the Gaussian
average-limit predictions 1 - rho_k / k (profile) and 1 - rho_k (grid).
"""
import os, sys, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory_validation import (load_real, REAL, pc_directions, percentile_cuts, strata_keys,
                               floor_alloc, grid_bins, FIG)

QS_LEVELS = np.linspace(0.1, 0.9, 9)


def analyses(X, w):
    """Weighted analyses of a subsample X with weights w (sum to 1)."""
    mu = w @ X
    Z = X - mu
    C = (Z * w[:, None]).T @ Z
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    ev, V = np.linalg.eigh(C)
    ev, V = ev[::-1], V[:, ::-1]
    order = np.argsort(X, axis=0)
    cw = np.cumsum(w[order], axis=0)
    idx = np.array([[np.searchsorted(cw[:, j], q) for q in QS_LEVELS] for j in range(X.shape[1])])
    idx = np.minimum(idx, len(X) - 1)
    quant = np.array([X[order[idx[j], j], j] for j in range(X.shape[1])])
    return dict(mean=mu, mean_unw=X.mean(0), sd=sd, R=R, ev=ev[:5], V3=V[:, :3], quant=quant)


def losses(a, f):
    P = a["V3"] @ a["V3"].T; Q = f["V3"] @ f["V3"].T
    return dict(mean=np.sum((a["mean"] - f["mean"]) ** 2),
                mean_unweighted=np.sum((a["mean_unw"] - f["mean"]) ** 2),
                sd=np.sum((a["sd"] - f["sd"]) ** 2),
                quantiles=np.mean(np.sum((a["quant"] - f["quant"]) ** 2, axis=1)),
                corr=np.sum((a["R"] - f["R"]) ** 2),
                eigvals=np.sum((a["ev"] - f["ev"]) ** 2),
                subspace=0.5 * np.sum((P - Q) ** 2))


def one_rep(X, key_inv, Nh, rh, starts, full, seed):
    rng = np.random.default_rng(seed)
    N, r = len(X), int(rh.sum())
    # within-cell SRSWOR for all cells at once: random order inside each cell, keep first r_h
    order = np.argsort(key_inv + rng.random(N), kind="stable")
    within = np.arange(N) - starts[key_inv[order]]
    pick = order[within < rh[key_inv[order]]]
    assert len(pick) == r
    w = (Nh[key_inv[pick]] / N) / rh[key_inv[pick]]
    lq = losses(analyses(X[pick], w / w.sum()), full)
    srs = rng.choice(N, size=r, replace=False)
    ls = losses(analyses(X[srs], np.full(r, 1.0 / r)), full)
    return lq, ls


def configure(design, X, delta):
    N = len(X)
    V_all, lam = pc_directions(X, min(X.shape[1], 50))
    cum = np.cumsum(lam) / lam.sum()
    if design == "profile":
        k = int(min(np.searchsorted(cum, 0.70) + 1, 10)); B = 5
    elif design == "profile1":
        k, B = 1, 5
    else:
        k = 3; B = grid_bins(delta * N, k)
    return k, B, lam


def run_dataset(name, design, delta, reps, jobs, seed=20260915):
    X = load_real(name); N, d = X.shape
    k, B, lam = configure(design, X, delta)
    V, _ = pc_directions(X, k)
    rho = lam[:k].sum() / lam.sum()
    sc = X @ V
    key = strata_keys(sc, percentile_cuts(sc, B), "grid" if design == "grid" else "profile")
    uniq, inv, Nh = np.unique(key, return_inverse=True, return_counts=True)
    rh = floor_alloc(Nh, delta)
    r = int(rh.sum())
    starts = np.r_[0, np.cumsum(Nh)[:-1]]
    full = analyses(X, np.full(N, 1.0 / N))
    Sh2 = np.zeros((len(Nh), d))
    for j in range(d):
        s1 = np.bincount(inv, X[:, j]); s2 = np.bincount(inv, X[:, j] ** 2)
        Sh2[:, j] = np.where(Nh > 1, (s2 - s1 ** 2 / Nh) / np.maximum(Nh - 1, 1), 0.0)
    exact_qs = float((((Nh / N) ** 2 * (1 - rh / Nh))[:, None] * Sh2 / rh[:, None]).sum())
    exact_srs = float((1 - r / N) * X.var(0, ddof=1).sum() / r)
    predicted = 1 - rho / k if design != "grid" else 1 - rho
    res = Parallel(n_jobs=jobs)(delayed(one_rep)(X, inv, Nh, rh, starts, full, seed + i)
                                for i in range(reps))
    LQ = pd.DataFrame([a for a, _ in res]); LS = pd.DataFrame([b for _, b in res])
    rng = np.random.default_rng(1)
    rows = []
    for c in LQ.columns:
        q, s = LQ[c].to_numpy(), LS[c].to_numpy()
        boot = [q[i].mean() / s[i].mean() for i in (rng.integers(0, reps, reps) for _ in range(500))]
        rows.append(dict(dataset=name, design=design, N=N, d=d, k=k, B=B, delta=delta, r=r,
                         realized_retention=r / N, H_N=len(Nh), singletons=int((Nh == 1).sum()),
                         rho_k=rho, exact_mean_ratio=exact_qs / exact_srs, gaussian_limit=predicted,
                         reps=reps, analysis=c, mse_QS=q.mean(), mse_SRS=s.mean(),
                         efficiency=q.mean() / s.mean(),
                         ci_lo=np.quantile(boot, 0.025), ci_hi=np.quantile(boot, 0.975),
                         frac_QS_closer=float(np.mean(q < s))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(REAL))
    ap.add_argument("--design", choices=["profile", "profile1", "grid"], default="profile")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    rows = []
    for n in a.datasets:
        rows += run_dataset(n, a.design, a.delta, a.reps, a.jobs)
        print(pd.DataFrame(rows)[lambda d: d.dataset == n][["analysis", "efficiency", "ci_lo", "ci_hi", "exact_mean_ratio", "gaussian_limit", "k", "B", "H_N", "realized_retention"]].round(3).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG, f"downstream_statistics_{a.design}_d{int(round(100 * a.delta))}.csv"), index=False)
    print(df.pivot(index="dataset", columns="analysis", values="efficiency").round(3).to_string())
