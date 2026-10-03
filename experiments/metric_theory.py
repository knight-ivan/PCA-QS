#!/usr/bin/env python3
"""Expected discrepancy metrics between a subset and the full data: theory (exact) vs simulation.

For a kernel k with feature map phi, MMD^2(P_hat, P_N) = || mean embedding of the (weighted) subset
- mean embedding of the full data ||^2. Three metrics of this form are studied, in the standardized
original feature space:
  rbf     Gaussian kernel, bandwidth b^2 = median squared distance (median heuristic);
  energy  energy distance = 2 * MMD^2 for the distance-induced kernel
          k(x,y) = (||x|| + ||y|| - ||x-y||)/2;
  mahal   squared Mahalanobis distance between the two means (linear kernel x' A y,
          A = (Sigma_N + 1e-6 I)^{-1}).

Designs at delta = 0.05 (same r for every method within a data set, r = realized PCA-QS size):
  PCA-QS grid     k = 3, m = floor((delta N / 10)^(1/3)), floor allocation, design weights;
  PCA-QS profile  m = 5, k = min(k*, 10), floor allocation, design weights;
  SRS             simple random sampling without replacement of the same size;
  Leverage        rows drawn with replacement with probability proportional to the leverage of the
                  standardized design with intercept, used as an unweighted subset;
  Coreset         the same draws with inverse-probability (Hansen-Hurwitz) weights, as in the
                  companion paper's coreset construction.

Exact expectations (no approximation):
  stratified WOR  E D^2 = sum_h (N_h/N)^2 (1 - r_h/N_h) S_h^2 / r_h,
                  S_h^2 = N_h/(N_h-1) [mean_i k_ii - mean_{i,j in h} k_ij];
  SRS WOR         the case of one stratum;
  weighted WR     E D^2 = (1/r) [ (1/N^2) sum_i k_ii / p_i - ||mu_N||^2 ];
  unweighted WR   E D^2 = ||mu_p - mu_N||^2 + (1/r) [ sum_i p_i k_ii - ||mu_p||^2 ].
Observed values use the exact identity D^2 = w'K_ss w - 2 w'g_s + c with g_i = mean_j k(X_i, X_j).
1000 replicates, seeds 20261003 + i. Output: ../figures/metric_theory_<dataset>.csv
"""
import os, sys, argparse, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_validation import (load_real, pc_directions, percentile_cuts, strata_keys, floor_alloc,
                               grid_bins, FIG)

KERNELS = ("rbf", "energy", "mahal")


def leverage(X):
    A = np.hstack([np.ones((len(X), 1)), X])
    Q, _ = np.linalg.qr(A)
    return np.clip((Q ** 2).sum(1), 1e-12, None)


class Kern:
    """Kernel evaluations for the three metrics on a fixed standardized data matrix."""

    def __init__(self, X, seed=0):
        self.X = X
        N, d = X.shape
        rng = np.random.default_rng(seed)
        i, j = rng.integers(0, N, 4000), rng.integers(0, N, 4000)
        self.b2 = float(np.median(((X[i] - X[j]) ** 2).sum(1)))
        S = np.cov(X, rowvar=False) + 1e-6 * np.eye(d)
        self.A = np.linalg.inv(S)
        self.norm = np.sqrt((X ** 2).sum(1))
        self.sq = (X ** 2).sum(1)

    def block(self, name, I, J):
        X = self.X
        if name == "mahal":
            return (X[I] @ self.A) @ X[J].T
        D2 = np.maximum(self.sq[I][:, None] + self.sq[J][None, :] - 2 * X[I] @ X[J].T, 0.0)
        if name == "rbf":
            return np.exp(-D2 / (2 * self.b2))
        D = np.sqrt(D2)                                          # distance-induced kernel
        return 0.5 * (self.norm[I][:, None] + self.norm[J][None, :] - D)

    def diag(self, name):
        if name == "mahal":
            return np.einsum("ij,jk,ik->i", self.X, self.A, self.X)
        if name == "rbf":
            return np.ones(len(self.X))
        return self.norm.copy()

    def rowmeans(self, name, chunk=1500):
        """g_i = mean_j k(X_i, X_j), and also q' K for leverage probabilities q (returned later)."""
        N = len(self.X)
        if name == "mahal":
            m = self.X.mean(0)
            return self.X @ self.A @ m
        g = np.empty(N)
        allj = np.arange(N)
        for s in range(0, N, chunk):
            I = np.arange(s, min(N, s + chunk))
            g[I] = self.block(name, I, allj).mean(1)
        return g

    def quad(self, name, u, chunk=1500):
        """u' K u for a weight vector u over all rows."""
        N = len(self.X)
        if name == "mahal":
            v = self.X.T @ u
            return float(v @ self.A @ v)
        tot = 0.0
        allj = np.arange(N)
        for s in range(0, N, chunk):
            I = np.arange(s, min(N, s + chunk))
            tot += float(u[I] @ (self.block(name, I, allj) @ u))
        return tot

    def within_sums(self, name, inv, H):
        """sum_{i,j in h} k_ij for every stratum h."""
        out = np.zeros(H)
        order = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(inv, minlength=H))]
        for h in range(H):
            I = order[bounds[h]:bounds[h + 1]]
            if len(I) == 0:
                continue
            if name == "mahal":
                v = self.X[I].sum(0)
                out[h] = float(v @ self.A @ v)
            else:
                tot = 0.0
                for s in range(0, len(I), 2000):
                    tot += float(self.block(name, I[s:s + 2000], I).sum())
                out[h] = tot
        return out


def stratified_prediction(kd, ws, inv, Nh, rh):
    """Exact E D^2 for stratified SRSWOR with design weights."""
    N = Nh.sum()
    mean_diag = np.bincount(inv, kd) / Nh
    S2 = np.where(Nh > 1, Nh / np.maximum(Nh - 1, 1) * (mean_diag - ws / Nh ** 2), 0.0)
    return float((((Nh / N) ** 2) * (1 - rh / Nh) * S2 / rh).sum())


def design_partition(X, delta, design):
    N = len(X)
    _, lam = pc_directions(X, min(X.shape[1], 50))
    cum = np.cumsum(lam) / lam.sum()
    if design == "grid":
        k = 3; B = grid_bins(delta * N, k); strat = "grid"
    else:
        k = int(min(np.searchsorted(cum, 0.70) + 1, 10)); B = 5; strat = "profile"
    V, _ = pc_directions(X, k)
    sc = X @ V
    key = strata_keys(sc, percentile_cuts(sc, B), strat)
    _, inv, Nh = np.unique(key, return_inverse=True, return_counts=True)
    rh = floor_alloc(Nh, delta)
    return dict(k=k, B=B, inv=inv, Nh=Nh, rh=rh, r=int(rh.sum()))


def one_rep(seed, parts, Kobj_X, kernels_data, p_lev, r_ref):
    """Observed D^2 for each design and kernel in one replicate."""
    X = Kobj_X
    rng = np.random.default_rng(seed)
    N = len(X)
    out = []
    samples = {}
    for name, P in parts.items():
        inv, Nh, rh = P["inv"], P["Nh"], P["rh"]
        order = np.argsort(inv + rng.random(N), kind="stable")
        starts = np.r_[0, np.cumsum(Nh)[:-1]]
        within = np.arange(N) - starts[inv[order]]
        pick = order[within < rh[inv[order]]]
        w = (Nh[inv[pick]] / N) / rh[inv[pick]]
        samples[name] = (pick, w / w.sum())
    pick = rng.choice(N, size=r_ref, replace=False)
    samples["SRS"] = (pick, np.full(r_ref, 1.0 / r_ref))
    idx = rng.choice(N, size=r_ref, replace=True, p=p_lev)
    samples["Leverage"] = (idx, np.full(r_ref, 1.0 / r_ref))
    w = 1.0 / (N * p_lev[idx]); samples["Coreset"] = (idx, w / r_ref)
    for meth, (I, w) in samples.items():
        for kname, (K, g, c) in kernels_data.items():
            Kss = K.block(kname, I, I)
            D2 = float(w @ Kss @ w - 2 * w @ g[I] + c)
            if kname == "energy":
                D2 *= 2
            out.append(dict(method=meth, kernel=kname, D2=max(D2, 0.0), r=len(I)))
    return out


def run(name, delta=0.05, reps=1000, jobs=10, seed=20261003):
    t0 = time.time()
    X = load_real(name)
    N = len(X)
    parts = {"PCA-QS grid": design_partition(X, delta, "grid"),
             "PCA-QS profile": design_partition(X, delta, "profile")}
    r_ref = parts["PCA-QS grid"]["r"]
    lev = leverage(X); p_lev = lev / lev.sum()
    K = Kern(X)
    kernels_data, pred = {}, []
    for kname in KERNELS:
        kd = K.diag(kname)
        g = K.rowmeans(kname)
        c = float(g.mean())
        kernels_data[kname] = (K, g, c)
        fac = 2.0 if kname == "energy" else 1.0
        tot_ss = c * N * N
        S2 = N / (N - 1) * (kd.mean() - c)
        for meth, P in parts.items():
            ws = K.within_sums(kname, P["inv"], len(P["Nh"]))
            pred.append(dict(method=meth, kernel=kname, r=P["r"],
                             predicted=fac * stratified_prediction(kd, ws, P["inv"], P["Nh"], P["rh"])))
        pred.append(dict(method="SRS", kernel=kname, r=r_ref, predicted=fac * (1 - r_ref / N) * S2 / r_ref))
        muN2 = c
        pred.append(dict(method="Coreset", kernel=kname, r=r_ref,
                         predicted=fac * ((kd / p_lev).sum() / N ** 2 - muN2) / r_ref))
        qKq = K.quad(kname, p_lev); qKu = float(p_lev @ g)
        bias2 = qKq - 2 * qKu + muN2
        pred.append(dict(method="Leverage", kernel=kname, r=r_ref,
                         predicted=fac * (bias2 + (float(p_lev @ kd) - qKq) / r_ref)))
        print(f"  {name} {kname}: predictions done ({time.time() - t0:.0f}s)", flush=True)
    obs = Parallel(n_jobs=jobs)(delayed(one_rep)(seed + i, parts, X, kernels_data, p_lev, r_ref)
                                for i in range(reps))
    O = pd.DataFrame([row for rep in obs for row in rep])
    S = O.groupby(["method", "kernel"]).agg(observed=("D2", "mean"), se=("D2", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                                            r_obs=("r", "mean")).reset_index()
    df = pd.DataFrame(pred).merge(S, on=["method", "kernel"])
    srs = df[df.method == "SRS"].set_index("kernel")["predicted"]
    # ratio to SRS of the same size: SRS expectation scales as (1 - r/N)/r
    def srs_at(row):
        r0 = r_ref
        return srs[row.kernel] * ((1 - row.r / N) / row.r) / ((1 - r0 / N) / r0)
    df["srs_same_size"] = df.apply(srs_at, axis=1)
    df["pred_ratio"] = df.predicted / df.srs_same_size
    df["obs_ratio"] = df.observed / df.srs_same_size
    df.insert(0, "dataset", name); df["N"] = N; df["d"] = X.shape[1]
    df["k_grid"] = parts["PCA-QS grid"]["k"]; df["k_profile"] = parts["PCA-QS profile"]["k"]
    df["H_grid"] = len(parts["PCA-QS grid"]["Nh"]); df["H_profile"] = len(parts["PCA-QS profile"]["Nh"])
    df.to_csv(os.path.join(FIG, f"metric_theory_{name}.csv"), index=False)
    print(df[["method", "kernel", "r", "predicted", "observed", "se", "pred_ratio", "obs_ratio"]].round(5).to_string(index=False), flush=True)
    print(f"{name} done in {time.time() - t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MAGIC", "EEG", "CreditCard", "Epileptic"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    for n in a.datasets:
        run(n, reps=a.reps, jobs=a.jobs)
