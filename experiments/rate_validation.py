#!/usr/bin/env python3
"""Reproduce the convergence-rate figure (Theorem 1) and slope table.

Sweeps the retained size n and measures, in the top-k PCA subspace:
  * marginal quantile error to the full data        -> theory O(n^{-1/2})
  * exact multivariate 2-Wasserstein to a fresh
    equal-size draw from the truth (small n)         -> theory O(n^{-1/k})

Usage:
    python experiments/rate_validation.py            # default settings
    python experiments/rate_validation.py --reps 20  # more replications
Outputs figures/rate_validation.png and figures/rate_validation_slopes.csv.
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices, choose_design
from pcaqs.data import anisotropic_gmm
from pcaqs.metrics import quantile_error, exact_w2
from joblib import Parallel, delayed

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


def _quant_rep(k, Nf, delta, seed):
    gen = np.random.default_rng(seed)
    X, _ = anisotropic_gmm(Nf, random_state=int(gen.integers(1 << 31)))
    qs = PCAQS(n_components=k, n_bins=5, retention=delta, random_state=int(gen.integers(1 << 31)),
               stratification="profile")
    sc = qs.fit(X).scores(X)
    qi = qs.sample_indices(X, allocation="floor")
    si = srs_indices(len(X), len(qi), random_state=int(gen.integers(1 << 31)))
    return quantile_error(sc, sc[qi]), quantile_error(sc, sc[si])


def _w2_rep(k, n, delta, seed):
    gen = np.random.default_rng(seed)
    Nf = int(n / delta)
    X, _ = anisotropic_gmm(Nf, random_state=int(gen.integers(1 << 31)))
    qs = PCAQS(n_components=k, n_bins=5, retention=delta, random_state=int(gen.integers(1 << 31)),
               stratification="profile")
    sc = qs.fit(X).scores(X)
    qi = qs.sample_indices(X, allocation="floor")
    Xr, _ = anisotropic_gmm(len(qi), random_state=int(gen.integers(1 << 31)))
    ref = qs.scores(Xr)
    return exact_w2(sc[qi], ref)


def run(reps=10, delta=0.10, ks=(2, 3), seed=20260806, jobs=-1):
    rec = []
    # (a) quantile error vs full data, sweep retained size via N (parallel over reps)
    for k in ks:
        for Nf in [5000, 10000, 20000, 40000, 80000, 160000]:
            n_ret = int(round(delta * Nf))
            outs = Parallel(n_jobs=jobs)(
                delayed(_quant_rep)(k, Nf, delta, seed + 101 * k + b) for b in range(reps))
            rec.append(dict(k=k, n=n_ret, method="PCA-QS", metric="quantile_error",
                            value=float(np.mean([o[0] for o in outs]))))
            rec.append(dict(k=k, n=n_ret, method="SRS", metric="quantile_error",
                            value=float(np.mean([o[1] for o in outs]))))
            print(f"[a] k={k} n~{n_ret} done", flush=True)
    # (b) exact W2 vs fresh equal-size draw, small n (parallel over reps)
    for k in ks:
        for n in [100, 200, 400, 800]:
            outs = Parallel(n_jobs=jobs)(
                delayed(_w2_rep)(k, n, delta, seed + 211 * k + b) for b in range(reps))
            rec.append(dict(k=k, n=n, method="PCA-QS", metric="W2_exact", value=float(np.mean(outs))))
            print(f"[b] k={k} n={n} done", flush=True)
    return pd.DataFrame(rec)


def fit_and_plot(res):
    theory = lambda m, k: {"W2_exact": -1.0 / k, "quantile_error": -0.5}[m]
    rows = []
    for (k, meth, m), g in res.groupby(["k", "method", "metric"]):
        g = g[g["value"] > 0]
        if len(g) >= 3:
            b = np.polyfit(np.log(g["n"]), np.log(g["value"]), 1)[0]
            rows.append(dict(metric=m, k=k, method=meth,
                             fitted=round(b, 3), theory=round(theory(m, k), 3)))
    slopes = pd.DataFrame(rows).sort_values(["metric", "k", "method"])
    slopes.to_csv(os.path.join(OUT, "rate_validation_slopes.csv"), index=False)
    print("\n", slopes.to_string(index=False))

    col = {2: "#4c72b0", 3: "#55a868"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    a = ax[0]
    for k in sorted(res.k.unique()):
        for meth, ls, mk in (("PCA-QS", "-", "o"), ("SRS", "--", "s")):
            g = res[(res.metric == "quantile_error") & (res.k == k) & (res.method == meth)]
            g = g[g.value > 0].sort_values("n")
            if len(g):
                a.plot(g.n, g.value, ls, marker=mk, ms=4, color=col.get(k, "gray"),
                       alpha=0.95 if meth == "PCA-QS" else 0.55, label=f"{meth}, k={k}")
    g0 = res[(res.metric == "quantile_error") & (res.k == min(res.k)) & (res.method == "PCA-QS")].sort_values("n")
    nn = g0.n.values
    a.plot(nn, g0.value.iloc[0] * (nn / nn[0]) ** -0.5, ":", color="gray", label="theory $r^{-1/2}$")
    a.set(xscale="log", yscale="log", xlabel="retained size $r$", ylabel="quantile error to full data")
    a.set_title("Retained-sample quantile discrepancy (empirical $r^{-1/2}$)"); a.grid(True, which="both", ls=":", alpha=0.35); a.legend(fontsize=7)
    a = ax[1]
    for k in sorted(res.k.unique()):
        g = res[(res.metric == "W2_exact") & (res.k == k) & (res.method == "PCA-QS")]
        g = g[g.value > 0].sort_values("n")
        if len(g):
            a.plot(g.n, g.value, "-o", ms=4, color=col.get(k, "gray"), label=f"PCA-QS, k={k}")
    g0 = res[(res.metric == "W2_exact") & (res.k == min(res.k)) & (res.method == "PCA-QS")].sort_values("n")
    if len(g0):
        nn = g0.n.values
        for k in sorted(res.k.unique()):
            a.plot(nn, g0.value.iloc[0] * (nn / nn[0]) ** (-1.0 / k), ":", color=col.get(k, "gray"),
                   label=f"theory $r^{{-1/{k}}}$")
    a.set(xscale="log", yscale="log", xlabel="retained size $r$", ylabel="exact $W_2$ to truth")
    a.set_title("Wasserstein geometric rate $r^{-1/k}$"); a.grid(True, which="both", ls=":", alpha=0.35); a.legend(fontsize=7)
    fig.suptitle("Empirical retained-subspace rates on an A2-satisfying generator (diagnostics)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "rate_validation.png"), dpi=150)
    print("wrote", os.path.join(OUT, "rate_validation.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    res = run(reps=a.reps, ks=tuple(a.ks), jobs=a.jobs)
    res.to_csv(os.path.join(OUT, "rate_validation.csv"), index=False)
    fit_and_plot(res)
