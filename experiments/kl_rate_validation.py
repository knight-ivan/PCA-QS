#!/usr/bin/env python3
"""Validate the KL exponent O(r^{-4/(k+4)}) against the ANALYTIC projected density.

The structure-fidelity data is a 2-component Gaussian mixture, so the top-k PCA-score
density f_Y is itself an exact k-dimensional Gaussian mixture whose parameters follow
from the class means/covariances and the (per-replicate) standardisation + PCA map.
Using this exact density as the target removes the finite-reference floor that made the
earlier KDE-vs-sample KL estimate too noisy. We fit a KDE f_hat to the retained PCA-QS
scores and estimate the REVERSE KL, D(f_hat || f_Y), by Monte-Carlo over draws from
f_hat, then fit the log-log slope in the retained size r. The reverse direction matches
the corrected Theorem 1(iii): the density in the chi^2 denominator (here f_Y, a Gaussian
mixture that is strictly positive everywhere) is the one bounded below, so the estimate
is finite with no clipping -- unlike the forward KL, whose bounded-support-KDE denominator
can vanish where f_Y>0 and send the divergence to +infinity.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.special import logsumexp
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS
from pcaqs.data import anisotropic_gmm, ANISO_EIGS, ANISO_M1, ANISO_VAR1, ANISO_W

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
# Anisotropic generator so assumption A2 (distinct leading eigenvalues) actually holds.
D = 50; MU1 = ANISO_M1; VAR0v = ANISO_EIGS; VAR1v = ANISO_VAR1 * ANISO_EIGS; W = ANISO_W


def projected_mixture(X):
    """Return (weights, [m_c], [C_c]) of the exact top-k projected density, and Vk/scaler."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    return mu, sd, Vt  # caller picks k


def one_rep(k, r, N, seed):
    gen = np.random.default_rng([seed, k, r])
    X, _ = anisotropic_gmm(N, random_state=int(gen.integers(1 << 31)))
    mu, sd, Vt = projected_mixture(X)
    Vk = Vt[:k].T                                   # (D, k)
    sc = ((X - mu) / sd) @ Vk
    # analytic projected components (within-class covariances are diagonal -> Vk' diag Vk)
    m0 = Vk.T @ ((np.zeros(D) - mu) / sd); m1 = Vk.T @ ((MU1 - mu) / sd)
    C0 = Vk.T @ np.diag(VAR0v / sd ** 2) @ Vk
    C1 = Vk.T @ np.diag(VAR1v / sd ** 2) @ Vk
    comps = [multivariate_normal(m0, C0, allow_singular=True),
             multivariate_normal(m1, C1, allow_singular=True)]

    qs = PCAQS(n_components=k, n_bins=max(2, int(np.floor(r ** (1.0 / k)))),
               retention=r / N, random_state=int(gen.integers(1 << 31)))
    qs.mean_ = np.zeros(k); qs.scale_ = np.ones(k); qs.components_ = np.eye(k)
    idx = qs.sample_indices(sc, exact_size=r)
    try:
        fhat = gaussian_kde(sc[idx].T)
    except Exception:
        return np.nan
    # REVERSE KL, D(fhat || f_Y), via MC over draws from fhat (Theorem 1(iii) direction).
    # f_Y is a strictly positive Gaussian mixture, so the denominator never vanishes.
    z = fhat.resample(1700, seed=int(gen.integers(1 << 31))).T      # (n, k) ~ fhat
    logfh = np.log(np.clip(fhat(z.T), 1e-300, None))
    logfY = logsumexp(np.column_stack([np.log(W[0]) + comps[0].logpdf(z),
                                       np.log(W[1]) + comps[1].logpdf(z)]), axis=1)
    return float(np.mean(logfh - logfY))


def run(ks, reps, delta, jobs, seed):
    rows = []
    for k in ks:
        for Nf in [4000, 8000, 16000, 32000, 64000]:
            r = int(round(delta * Nf))
            vals = Parallel(n_jobs=jobs)(delayed(one_rep)(k, r, Nf, seed + b) for b in range(reps))
            vals = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
            rows.append(dict(k=k, r=r, KL=np.mean(vals)))
            print(f"k={k} r={r} KL={np.mean(vals):.4g}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20260806)
    a = ap.parse_args()
    df = run(a.ks, a.reps, a.delta, a.jobs, a.seed)
    df.to_csv(os.path.join(OUT, "kl_rate_validation.csv"), index=False)
    slopes = []
    for k, g in df.groupby("k"):
        g = g[g.KL > 0]
        b = np.polyfit(np.log(g.r), np.log(g.KL), 1)[0]
        slopes.append(dict(k=k, fitted=round(b, 3), theory=round(-4.0 / (k + 4), 3)))
    sl = pd.DataFrame(slopes); sl.to_csv(os.path.join(OUT, "kl_rate_slopes.csv"), index=False)
    print("\n=== KL log-log slopes vs theory -4/(k+4) ===\n", sl.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.2, 4.2)); col = {2: "#4c72b0", 3: "#55a868"}
    for k, g in df.groupby("k"):
        g = g[g.KL > 0].sort_values("r")
        ax.plot(g.r, g.KL, "-o", ms=4, color=col.get(k, "gray"), label=f"KL, $k={k}$")
        n = g.r.values
        ax.plot(n, g.KL.iloc[0] * (n / n[0]) ** (-4.0 / (k + 4)), ":", color=col.get(k, "gray"),
                label=f"theory $r^{{-4/({k}+4)}}$")
    ax.set(xscale="log", yscale="log", xlabel="retained size $r$",
           ylabel=r"reverse KL $D(\hat f_r\,\|\,f_Y)$ to analytic density")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "kl_rate_validation.png"), dpi=150)
    print("wrote kl_rate_validation.png")
