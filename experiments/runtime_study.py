#!/usr/bin/env python3
"""Computing time of PCA-QS and the comparison designs on the six real data sets.

Single-threaded (BLAS threads = 1), median of 1000 runs, delta = 0.05, standardized data in memory.
PCA-QS steps: principal directions (covariance eigendecomposition, O(N d^2 + d^3)), scores and
cutoffs (O(N d k + k N log N)), stratum labels (O(N k m)), allocation and within-stratum sampling
(O(N)). Leverage/coreset: thin QR of the N x (d+1) design (O(N d^2)) and weighted draws.
Output: ../figures/runtime_study.csv
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_validation import load_real, REAL, percentile_cuts, strata_keys, floor_alloc, grid_bins, FIG


def pcaqs(X, delta, design, rng):
    t = {}
    t0 = time.perf_counter()
    evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    if design == "grid":
        k = 3; B = grid_bins(delta * len(X), k); strat = "grid"
    else:
        cum = np.cumsum(evals) / evals.sum(); k = int(min(np.searchsorted(cum, 0.70) + 1, 10)); B = 5; strat = "profile"
    V = evecs[:, :k] * np.sign(evecs[np.argmax(np.abs(evecs[:, :k]), axis=0), np.arange(k)])
    t["pca"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc = X @ V
    key = strata_keys(sc, percentile_cuts(sc, B), strat)
    _, inv, Nh = np.unique(key, return_inverse=True, return_counts=True)
    t["strata"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rh = floor_alloc(Nh, delta)
    N = len(X)
    order = np.argsort(inv + rng.random(N), kind="stable")
    starts = np.r_[0, np.cumsum(Nh)[:-1]]
    within = np.arange(N) - starts[inv[order]]
    pick = order[within < rh[inv[order]]]
    t["sample"] = time.perf_counter() - t0
    return t, len(pick)


def leverage_draw(X, r, rng, weighted):
    t0 = time.perf_counter()
    A = np.hstack([np.ones((len(X), 1)), X])
    Q, _ = np.linalg.qr(A)
    lev = (Q ** 2).sum(1); p = lev / lev.sum()
    idx = rng.choice(len(X), size=r, replace=True, p=p)
    if weighted:
        w = 1.0 / (len(X) * p[idx] * r)
    return time.perf_counter() - t0


def main(delta=0.05, runs=1000):
    rows = []
    for name in REAL:
        X = load_real(name); N, d = X.shape
        rng = np.random.default_rng(1)
        for design in ("grid", "profile"):
            ts = [pcaqs(X, delta, design, rng) for _ in range(runs)]
            r = ts[0][1]
            for part in ("pca", "strata", "sample"):
                rows.append(dict(dataset=name, N=N, d=d, method=f"PCA-QS {design}", part=part,
                                 seconds=float(np.median([t[0][part] for t in ts]))))
            rows.append(dict(dataset=name, N=N, d=d, method=f"PCA-QS {design}", part="total",
                             seconds=float(np.median([sum(t[0].values()) for t in ts]))))
        rr = int(round(delta * N))
        t = [time.perf_counter() for _ in range(1)]
        srs = []
        for _ in range(runs):
            t0 = time.perf_counter(); rng.choice(N, size=rr, replace=False); srs.append(time.perf_counter() - t0)
        rows.append(dict(dataset=name, N=N, d=d, method="SRS", part="total", seconds=float(np.median(srs))))
        for meth, wtd in (("Leverage", False), ("Coreset", True)):
            rows.append(dict(dataset=name, N=N, d=d, method=meth, part="total",
                             seconds=float(np.median([leverage_draw(X, rr, rng, wtd) for _ in range(runs)]))))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG, "runtime_study.csv"), index=False)
    print(df[df.part == "total"].pivot(index="dataset", columns="method", values="seconds").round(4).to_string())
    print(df[df.method.str.startswith("PCA-QS")].pivot_table(index=["dataset", "method"], columns="part", values="seconds").round(4).to_string())


if __name__ == "__main__":
    main()
