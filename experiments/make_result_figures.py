#!/usr/bin/env python3
"""Redraw the numerical-result figures from the rerun confirmation CSVs, with a
consistent colour scheme (PCA-QS = blue, SRS = red) and clear legends.

Produces (into ../figures/, and copies into ../manuscript/ for the paper):
  results_synthetic_distance.png  - PCA-QS vs SRS across 7 metrics x k (1000 runs)
  results_real_data_heatmap.png   - per-dataset win pattern + occupancy/k effect
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_CR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # code_release
FIG = os.path.join(_CR, "figures")
MAN = os.path.join(os.path.dirname(_CR), "manuscript")                     # ../manuscript
QS_C, SRS_C = "#4c72b0", "#c44e52"          # PCA-QS blue, SRS red (consistent everywhere)
METRIC_ORDER = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
METRIC_LAB = {"quantile_error": "quantile", "KL": "KL", "JS": "JS", "energy": "energy",
              "MMD": "MMD", "Mahalanobis": "Mahalanobis", "pairwise_W1": "pairwise-$W_1$"}


def fig_synthetic():
    df = pd.read_csv(os.path.join(FIG, "confirm_synthetic_distance_summary.csv"))
    ks = sorted(df.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.4), sharey=False)
    x = np.arange(len(METRIC_ORDER)); w = 0.38
    for ax, k in zip(np.atleast_1d(axes), ks):
        g = df[df.k == k].set_index("metric")
        qs = [max(g.loc[m, "QS_mean"], 1e-12) for m in METRIC_ORDER]
        sr = [max(g.loc[m, "SRS_mean"], 1e-12) for m in METRIC_ORDER]
        ax.bar(x - w/2, sr, w, color=SRS_C, label="SRS")
        ax.bar(x + w/2, qs, w, color=QS_C, label="PCA-QS")
        ax.set_yscale("log")
        Hn = int(g["H_N"].iloc[0]); rr = int(g["r"].iloc[0])
        occ = "$H_N{\\leq}r$" if Hn <= rr else f"$H_N{{\\approx}}{Hn}{{\\gg}}r$"
        tag = "dynamic " if k not in (3, 5, 10) else ""
        ax.set_title(f"{tag}$k={k}$  ($m={int(g['m'].iloc[0])}$, {occ})", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([METRIC_LAB[m] for m in METRIC_ORDER], rotation=40, ha="right", fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
    np.atleast_1d(axes)[0].set_ylabel("discrepancy to full data (log; lower better)")
    np.atleast_1d(axes)[0].legend(loc="upper left", framealpha=0.95, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1])
    for d in (FIG, MAN):
        fig.savefig(os.path.join(d, "results_synthetic_distance.png"), dpi=150)
    print("wrote results_synthetic_distance.png")


def _load(name):
    p = os.path.join(FIG, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fig_real():
    # all datasets from the two consolidated 1000-rep files
    dyn = _load("confirm_real_data_1000_dynamicK_summary.csv")   # all 7 at dynamic k
    k4 = _load("confirm_real_data_k4recovery_summary.csv")        # 3 high-d at k=4
    # (short_name, display_name, source_df) ; label shows k read from the data
    spec = [
        ("CreditCard", "CreditCard", dyn), ("MAGIC", "MAGIC", dyn), ("EEG", "EEG", dyn),
        ("HIGGS", "HIGGS", dyn),
        ("OnlineNews", "OnlineNews", dyn), ("OnlineNews", "OnlineNews", k4),
        ("Epileptic", "Epileptic", dyn),  ("Epileptic", "Epileptic", k4),
        ("YearPrediction", "YearPred.", dyn), ("YearPrediction", "YearPred.", k4),
    ]
    labels, M = [], []
    for ds, disp, src in spec:
        if src is None:
            continue
        g = src[src.dataset == ds]
        if g.empty:
            continue
        kval = int(g["k"].iloc[0])
        g = g.set_index("metric")
        labels.append(f"{disp} (k={kval})")
        M.append([g.loc[m, "frac_QS_better"] - 0.5 if m in g.index else 0.0 for m in METRIC_ORDER])
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(labels) + 1.4))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    im = ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")   # RdBu: blue=+ (PCA-QS), red=- (SRS)
    ax.set_xticks(range(len(METRIC_ORDER))); ax.set_xticklabels([METRIC_LAB[m] for m in METRIC_ORDER], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(METRIC_ORDER)):
            f = M[i, j] + 0.5
            ax.text(j, i, f"{f:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.32 else "black")
    # separators between groups
    for y in (2.5, 3.5, 5.5, 7.5):
        ax.axhline(y, color="k", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-0.5, 0, 0.5])
    cb.ax.set_yticklabels(["SRS\ncloser", "tie", "PCA-QS\ncloser"], fontsize=7.5)
    ax.set_xlabel("distributional discrepancy metric")
    ax.set_ylabel("dataset (retained dimension $k$)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    for d in (FIG, MAN):
        fig.savefig(os.path.join(d, "results_real_data_heatmap.png"), dpi=150)
    print("wrote results_real_data_heatmap.png")


if __name__ == "__main__":
    fig_synthetic()
    fig_real()
