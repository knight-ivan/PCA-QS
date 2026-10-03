#!/usr/bin/env python3
"""Histogram divergences between a subset and the full data: second-order prediction vs simulation.

On a fixed partition of the score space into cells c with full-data masses p_c > 0, the divergence
KL(P_hat || P_N) = sum_c p_hat_c log(p_hat_c / p_c) has the second-order expansion
(1/2) sum_c (p_hat_c - p_c)^2 / p_c, so E KL ~ (1/2) sum_c Var_d(p_hat_c) / p_c, with Var_d the exact
design variance of the weighted cell frequency; Jensen-Shannon is about one quarter of that.
Data: anisotropic Gaussian mixture (d = 50), N = 50,000, sample principal directions, k = 2 scores,
histogram with 8 equal-width bins per score on the full-data range (empty cells dropped).
Designs: grid and profile strata on the k = 2 scores with m = 5, floor allocation and design weights,
and SRS without replacement of the same size; delta in {0.02, 0.05, 0.10}; 1000 replicates.
Output: ../figures/histogram_kl_check.csv
"""
import os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcaqs.data import anisotropic_gmm
from theory_validation import pc_directions, percentile_cuts, strata_keys, floor_alloc, FIG


def cells(sc, bins=8):
    lo, hi = sc.min(0), sc.max(0)
    idx = np.minimum(((sc - lo) / (hi - lo) * bins).astype(int), bins - 1)
    key = idx[:, 0] * bins + idx[:, 1]
    _, c = np.unique(key, return_inverse=True)
    return c


def predicted(cell, inv, Nh, rh):
    """(1/2) sum_c Var_d(p_hat_c)/p_c with the exact stratified variance."""
    N = len(cell); C = cell.max() + 1
    p = np.bincount(cell, minlength=C) / N
    H = len(Nh)
    cnt = np.zeros((H, C)); np.add.at(cnt, (inv, cell), 1)
    ph = cnt / Nh[:, None]
    S2 = np.where(Nh[:, None] > 1, Nh[:, None] / np.maximum(Nh[:, None] - 1, 1) * ph * (1 - ph), 0.0)
    var = (((Nh / N) ** 2 * (1 - rh / Nh))[:, None] * S2 / rh[:, None]).sum(0)
    return 0.5 * float((var / p).sum())


def kl_js(phat, p):
    m = phat > 0
    kl = float((phat[m] * np.log(phat[m] / p[m])).sum())
    q = 0.5 * (phat + p)
    js = 0.5 * float((phat[m] * np.log(phat[m] / q[m])).sum()) + 0.5 * float((p * np.log(p / q)).sum())
    return kl, js


def one(seed, cell, designs, N, C):
    rng = np.random.default_rng(seed)
    p = np.bincount(cell, minlength=C) / N
    out = []
    for name, (inv, Nh, rh) in designs.items():
        order = np.argsort(inv + rng.random(N), kind="stable")
        starts = np.r_[0, np.cumsum(Nh)[:-1]]
        within = np.arange(N) - starts[inv[order]]
        pick = order[within < rh[inv[order]]]
        w = (Nh[inv[pick]] / N) / rh[inv[pick]]
        phat = np.bincount(cell[pick], weights=w, minlength=C)
        kl, js = kl_js(phat, p)
        out.append(dict(design=name, KL=kl, JS=js))
    return out


def main(N=50_000, reps=1000, seed=20261003):
    X, _ = anisotropic_gmm(N, random_state=7)
    X = (X - X.mean(0)) / X.std(0)
    V, _ = pc_directions(X, 2)
    sc = X @ V
    cell = cells(sc); C = cell.max() + 1
    rows = []
    for delta in (0.02, 0.05, 0.10):
        designs = {}
        for strat in ("grid", "profile"):
            key = strata_keys(sc, percentile_cuts(sc, 5), strat)
            _, inv, Nh = np.unique(key, return_inverse=True, return_counts=True)
            designs[f"PCA-QS {strat}"] = (inv, Nh, floor_alloc(Nh, delta))
        r = int(designs["PCA-QS grid"][2].sum())
        designs["SRS"] = (np.zeros(N, dtype=int), np.array([N]), np.array([r]))
        res = Parallel(n_jobs=12)(delayed(one)(seed + i, cell, designs, N, C) for i in range(reps))
        D = pd.DataFrame([x for rr in res for x in rr])
        for name, (inv, Nh, rh) in designs.items():
            sub = D[D.design == name]
            pr = predicted(cell, inv, Nh, rh)
            rows.append(dict(delta=delta, design=name, r=int(rh.sum()), cells=C,
                             KL_pred=pr, KL_obs=sub.KL.mean(), KL_se=sub.KL.std(ddof=1) / np.sqrt(reps),
                             JS_pred=pr / 4, JS_obs=sub.JS.mean(), JS_se=sub.JS.std(ddof=1) / np.sqrt(reps),
                             srs_bound=(C - 1) / (2 * int(rh.sum())) * (1 - int(rh.sum()) / N)))
        print(pd.DataFrame(rows).tail(3).round(6).to_string(index=False), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "histogram_kl_check.csv"), index=False)
    return df


if __name__ == "__main__":
    main()
