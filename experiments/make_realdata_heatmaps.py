#!/usr/bin/env python3
"""Real-data distributional heat maps (original feature space) for the theory paper.

Left panel: profile design (companion paper's PCA-QS; k = min(k*, 10), B = 5), followed by
the uncapped variance-threshold rows that illustrate strata degeneration. Right panel: grid
design (k capped at floor(log2(r/10)), m = floor((r/10)^(1/k))). Each cell is the fraction of
the 1000 replicates on which PCA-QS is closer to the full data than SRS of the same realized
size. Row labels give k, the mean number of occupied strata H_N and the realized retention.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_CR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_CR, "figures")
MAN = os.path.join(os.path.dirname(_CR), "manuscript")
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
LAB = {"quantile_error": "quantile", "KL": "KL", "JS": "JS", "energy": "energy", "MMD": "MMD",
       "Mahalanobis": "Mahalanobis", "pairwise_W1": "pairwise-$W_1$"}
ORDER = ["MAGIC", "EEG", "CreditCard", "HIGGS", "YearPrediction", "Epileptic"]


def rows_from(csv, suffix=""):
    p = os.path.join(FIG, csv)
    if not os.path.exists(p):
        return [], []
    d = pd.read_csv(p)
    labels, M = [], []
    for ds in ORDER:
        g = d[d.dataset == ds]
        if g.empty:
            continue
        gi = g.set_index("metric")
        labels.append(f"{ds}{suffix}  $k$={int(g.k.iloc[0])}, $H_N$={g.H_N.iloc[0]:.0f}, "
                      f"{100 * g.realized_retention.iloc[0]:.1f}%")
        M.append([gi.loc[m, "frac_QS_better"] - 0.5 for m in METRICS])
    return labels, M


def panel(ax, labels, M, title):
    M = np.array(M)
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5), aspect="auto")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([LAB[m] for m in METRICS], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j] + 0.5:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.32 else "black")
    ax.set_title(title, fontsize=9.5)
    return im


def main():
    lp, Mp = rows_from("confirm_real_data_origspace_profile_summary.csv")
    lu, Mu = rows_from("confirm_real_data_origspace_profile_uncapped_summary.csv", suffix=" uncapped")
    lg, Mg = rows_from("confirm_real_data_origspace_grid_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 0.55 * max(len(lp) + len(lu), len(lg)) + 2.4),
                           gridspec_kw={"wspace": 0.55})
    im = panel(ax[0], lp + lu, Mp + Mu, "profile strata (companion design)")
    if lu:
        ax[0].axhline(len(lp) - 0.5, color="k", lw=1.2)
    panel(ax[1], lg, Mg, "grid strata")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, ticks=[-0.5, 0, 0.5])
    cb.ax.set_yticklabels(["SRS\ncloser", "tie", "PCA-QS\ncloser"], fontsize=7.5)
    for d in (FIG, MAN, os.path.join(os.path.dirname(_CR), "manuscript_ejs")):
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, "results_real_data_heatmap.png"), dpi=160, bbox_inches="tight")
    print("wrote results_real_data_heatmap.png")


if __name__ == "__main__":
    main()
