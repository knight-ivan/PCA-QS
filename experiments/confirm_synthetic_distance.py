#!/usr/bin/env python3
"""Confirmation rerun of the synthetic distance-metric study, using the clean
`pcaqs` package and the CORRECTED metric definitions (distribution-level
Mahalanobis, 1-Wasserstein pairwise-distance discrepancy, histogram KL/JS on the
PCA-score marginals). Compares PCA-QS vs SRS at an identical exact retained size.

Replicates are independent, so they run in PARALLEL (joblib). Success criterion,
per the project's stance: PCA-QS should win on MOST metrics and stay competitive
on the rest --- not dominate every metric.

Outputs figures/confirm_synthetic_distance{,_summary}.csv and a printed
paired-difference summary (mean QS-SRS, 95% CI, fraction of runs with QS < SRS).
"""
import os, sys, argparse
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices, choose_design
from pcaqs.data import structure_fidelity_gmm
from pcaqs.metrics import quantile_error, energy_distance, mmd_rbf, mahalanobis_mean

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]


def marginal_hist_kl_js(A, B, bins=30):
    kl, js = [], []
    for j in range(A.shape[1]):
        lo = min(A[:, j].min(), B[:, j].min()); hi = max(A[:, j].max(), B[:, j].max())
        grid = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(A[:, j], grid); q, _ = np.histogram(B[:, j], grid)
        p = (p + 1.0) / (p.sum() + bins); q = (q + 1.0) / (q.sum() + bins)
        kl.append(np.sum(p * np.log(p / q)))
        m = 0.5 * (p + q)
        js.append(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return float(np.mean(kl)), float(np.mean(js))


def pairwise_w1(A, B, n_anchor, gen):
    def pdist_sample(M):
        S = M[gen.choice(len(M), size=min(n_anchor, len(M)), replace=False)]
        from scipy.spatial.distance import pdist
        return pdist(S)
    return float(wasserstein_distance(pdist_sample(A), pdist_sample(B)))


def occupancy_m(k, r, fixed=None):
    """Bins per component from the shared design rule (about 10 retained points per cell)."""
    if fixed:
        return fixed
    return 5                                                  # profile design: B = 5 quintile cutoffs


def one_run(k, b, N, delta, seed, fixed_m=None, space="pca"):
    """One independent replicate for component count k; returns two rows."""
    gen = np.random.default_rng([seed, k, b])
    X, _ = structure_fidelity_gmm(N, random_state=gen.integers(1 << 31))
    r = int(round(delta * N))
    m = occupancy_m(k, r, fixed=fixed_m)
    qs = PCAQS(n_components=k, n_bins=m, retention=delta, stratification="profile",
               random_state=int(gen.integers(1 << 31)))
    sc = qs.fit(X).scores(X)
    qi = qs.sample_indices(X, allocation="floor")
    H_N = qs.n_cells_
    si = srs_indices(len(X), len(qi), random_state=int(gen.integers(1 << 31)))
    # Strata come from the PCA scores; metrics are scored either in that PCA
    # subspace (default) or in the original standardized feature space Xs.
    Xs = (X - qs.mean_) / qs.scale_
    score_on = sc if space == "pca" else Xs
    rows = []
    for meth, idx in (("PCA-QS", qi), ("SRS", si)):
        A, B = score_on, score_on[idx]
        kl, js = marginal_hist_kl_js(A, B)
        rows.append(dict(k=k, run=b, method=meth, m=m, H_N=H_N, r=r, space=space,
                         quantile_error=quantile_error(A, B),
                         KL=kl, JS=js,
                         energy=energy_distance(A, B, random_state=0),
                         MMD=mmd_rbf(A, B, random_state=0),
                         Mahalanobis=mahalanobis_mean(A, B),
                         pairwise_W1=pairwise_w1(A, B, 400, gen)))
    return rows


def summarize(df):
    out = []
    for k in sorted(df.k.unique()):
        sub = df[df.k == k]
        m_used = int(sub["m"].iloc[0]); H = int(sub["H_N"].median()); r = int(sub["r"].iloc[0])
        for metric in METRICS:
            q = sub[sub.method == "PCA-QS"].sort_values("run")[metric].values
            s = sub[sub.method == "SRS"].sort_values("run")[metric].values
            d = q - s
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append(dict(k=k, m=m_used, H_N=H, r=r, metric=metric,
                            QS_mean=q.mean(), SRS_mean=s.mean(), diff=d.mean(),
                            ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se,
                            frac_QS_better=float(np.mean(q < s))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--N", type=int, default=100_000)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fixed-m", type=int, default=None,
                    help="fixed bins per PC (default: occupancy-limited m with m^k<=r)")
    ap.add_argument("--space", choices=["pca", "original"], default="pca",
                    help="score metrics in the top-k PCA subspace (default) or the original standardized feature space")
    a = ap.parse_args()
    # dynamic-k config: smallest k explaining >=70% variance (standardized), the same rule the
    # real-data / classification / regression studies use. On this near-isotropic generator the
    # spectrum is nearly flat, so the dynamic k is large and occupancy-limited -- an honest
    # illustration (consistent with the high-k real datasets) of why the variance rule needs an
    # occupancy cap.
    from sklearn.preprocessing import StandardScaler
    X0, _ = structure_fidelity_gmm(a.N, random_state=a.seed)
    Xs0 = StandardScaler().fit_transform(X0)
    sv0 = np.linalg.svd(Xs0 - Xs0.mean(0), full_matrices=False, compute_uv=False)
    k_dyn = int(np.searchsorted(np.cumsum(sv0 ** 2) / np.sum(sv0 ** 2), 0.70) + 1)
    ks = list(dict.fromkeys(list(a.ks) + [k_dyn]))   # append dynamic k, dedup, preserve order
    print(f"dynamic k (70% variance) = {k_dyn}; configs k = {ks}", flush=True)
    tasks = [(k, b) for k in ks for b in range(a.runs)]
    mode = f"fixed m={a.fixed_m}" if a.fixed_m else "occupancy-limited m (m^k<=r)"
    print(f"running {len(tasks)} replicates on {a.jobs} workers [{mode}] ...", flush=True)
    results = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(one_run)(k, b, a.N, 0.05, a.seed, a.fixed_m, a.space) for k, b in tasks)
    df = pd.DataFrame([r for pair in results for r in pair])
    tag = "" if a.space == "pca" else "_origspace"
    df.to_csv(os.path.join(OUT, f"confirm_synthetic_distance{tag}.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(OUT, f"confirm_synthetic_distance{tag}_summary.csv"), index=False)
    pd.set_option("display.width", 170, "display.max_columns", 20)
    print("\n=== paired PCA-QS - SRS (negative diff => PCA-QS closer to full data) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
