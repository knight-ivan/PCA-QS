#!/usr/bin/env python3
"""Reproduce the variance-reduction study (Theorem 2): PCA-QS vs SRS discrepancy.

Two modes:
  --from-csv PATH   reproduce the paper's figure from the saved 1000-run results
                    (results_combined_ordered.csv), the full 1M x 50 scale; or
  (default)         run a fresh, smaller synthetic study with this package so the
                    result is self-contained and quick.

Outputs figures/variance_reduction.png and figures/variance_reduction_summary.csv.
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices
from pcaqs.data import structure_fidelity_gmm
from pcaqs.metrics import quantile_error, energy_distance, mmd_rbf, mahalanobis_mean

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
METRICS = ["quantile_error", "energy_distance", "MMD", "Mahalanobis"]


def fresh_study(runs=50, N=20000, delta=0.05, ks=(3, 5, 10), seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for _ in range(runs):
            X, _ = structure_fidelity_gmm(N, random_state=rng.integers(1 << 31))
            qs = PCAQS(n_components=k, n_bins=10, retention=delta,
                       random_state=int(rng.integers(1 << 31)))
            sc_full = qs.fit(X).scores(X)
            qi = qs.sample_indices(X)
            si = srs_indices(len(X), len(qi), random_state=int(rng.integers(1 << 31)))
            for meth, idx in (("PCA-QS", qi), ("SRS", si)):
                A, B = sc_full, sc_full[idx]
                rows.append(dict(k=k, method=meth,
                                 quantile_error=quantile_error(A, B),
                                 energy_distance=energy_distance(A, B, random_state=0),
                                 MMD=mmd_rbf(A, B, random_state=0),
                                 Mahalanobis=mahalanobis_mean(A, B)))
        print(f"fresh study k={k} done", flush=True)
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["k", "method"], value_vars=METRICS, var_name="metric", value_name="value")
    return long, [str(k) for k in ks]


def from_csv(path):
    df = pd.read_csv(path)
    order = ["3", "5", "10", "dyn=21"]
    m = {"KL divergence": ("QS KL Divergence", "SRS KL Divergence"),
         "Energy distance": ("QS Energy Distance", "SRS Energy Distance"),
         "MMD": ("QS MMD", "SRS MMD"),
         "JS divergence": ("QS JS Divergence (labels)", "SRS JS Divergence (labels)")}
    rows = []
    for name, (q, s) in m.items():
        for _, r in df.iterrows():
            pc = str(r["PC Setting"])
            rows.append(dict(k=pc, metric=name, method="PCA-QS", value=r[q]))
            rows.append(dict(k=pc, metric=name, method="SRS", value=r[s]))
    return pd.DataFrame(rows), order


def plot(long, order, title):
    metrics = list(long["metric"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    x = np.arange(len(order)); w = 0.36; floor = 1e-11
    for ax, name in zip(axes.ravel(), metrics):
        qm = [max(long[(long.metric == name) & (long.k == k) & (long.method == "PCA-QS")]["value"].mean(), floor) for k in order]
        sm = [max(long[(long.metric == name) & (long.k == k) & (long.method == "SRS")]["value"].mean(), floor) for k in order]
        qs = [long[(long.metric == name) & (long.k == k) & (long.method == "PCA-QS")]["value"].std() for k in order]
        ss = [long[(long.metric == name) & (long.k == k) & (long.method == "SRS")]["value"].std() for k in order]
        ax.bar(x - w / 2, sm, w, yerr=ss, capsize=3, color="#c44e52", label="SRS")
        ax.bar(x + w / 2, qm, w, yerr=qs, capsize=3, color="#4c72b0", label="PCA-QS")
        ax.set_yscale("log"); ax.set_title(name)
        ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in order]); ax.grid(True, axis="y", ls=":", alpha=0.5)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005, "Lower is better; PCA-QS (blue) below SRS (red) validates the variance-reduction guarantee.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(os.path.join(OUT, "variance_reduction.png"), dpi=150)
    print("wrote", os.path.join(OUT, "variance_reduction.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", default=None, help="path to results_combined_ordered.csv")
    ap.add_argument("--runs", type=int, default=50)
    a = ap.parse_args()
    if a.from_csv:
        long, order = from_csv(a.from_csv)
        title = "PCA-QS vs SRS discrepancy to full data (paper's 1000-run results)"
    else:
        long, order = fresh_study(runs=a.runs)
        title = f"PCA-QS vs SRS discrepancy (fresh {a.runs}-run study, this package)"
    long.groupby(["metric", "k", "method"])["value"].agg(["mean", "std"]).to_csv(
        os.path.join(OUT, "variance_reduction_summary.csv"))
    plot(long, order, title)
