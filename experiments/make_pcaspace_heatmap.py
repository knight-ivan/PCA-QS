#!/usr/bin/env python3
"""PCA-score-space heat map (supplement companion to the original-space figure).

Same profile design, seeds and replicate count as the main text's figure; only the space in
which the discrepancies are measured differs. Input: the PCA-space run of confirm_real_data.py
(`--space pca --stratification profile --tag _pca_profile`).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_CR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_CR, "figures")
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
LAB = {"quantile_error": "quantile", "KL": "KL", "JS": "JS", "energy": "energy", "MMD": "MMD",
       "Mahalanobis": "Mahalanobis", "pairwise_W1": "pairwise-$W_1$"}
ORDER = ["MAGIC", "EEG", "CreditCard", "HIGGS", "YearPrediction", "Epileptic"]


def main():
    d = pd.read_csv(os.path.join(FIG, "confirm_real_data_pca_profile_summary.csv"))
    labels, M = [], []
    for ds in ORDER:
        g = d[d.dataset == ds]
        if g.empty:
            continue
        gi = g.set_index("metric")
        labels.append(f"{ds}  $k$={int(g.k.iloc[0])}, $H_N$={g.H_N.iloc[0]:.0f}, "
                      f"{100 * g.realized_retention.iloc[0]:.1f}%")
        M.append([gi.loc[m, "frac_QS_better"] - 0.5 for m in METRICS])
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(labels) + 2.2))
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5), aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([LAB[m] for m in METRICS], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j] + 0.5:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.32 else "black")
    ax.set_title("profile strata, discrepancies measured in the top-$k$ PCA-score space", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-0.5, 0, 0.5])
    cb.ax.set_yticklabels(["SRS\ncloser", "tie", "PCA-QS\ncloser"], fontsize=7.5)
    for out in (FIG, os.path.join(os.path.dirname(_CR), "manuscript_ejs")):
        if os.path.isdir(out):
            fig.savefig(os.path.join(out, "results_real_data_heatmap_pcaspace.png"),
                        dpi=160, bbox_inches="tight")
    print("wrote results_real_data_heatmap_pcaspace.png")


if __name__ == "__main__":
    main()
