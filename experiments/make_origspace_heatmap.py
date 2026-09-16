#!/usr/bin/env python3
"""Real-data structure-preservation heatmap in the ORIGINAL feature space
(restoring the original paper's scoring: metrics between the retained subset and
the full data on the original standardized features; PCA only builds the strata).

Reads whatever original-space 1000-rep summaries are present, so it works both
for the feasible-regime rows alone and for the full row set once the heavy rows
(HIGGS/YearPrediction/high-k) finish. Blue = PCA-QS closer to full data.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_CR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_CR, "figures")
MAN = os.path.join(os.path.dirname(_CR), "manuscript")
METRIC_ORDER = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
METRIC_LAB = {"quantile_error": "quantile", "KL": "KL", "JS": "JS", "energy": "energy",
              "MMD": "MMD", "Mahalanobis": "Mahalanobis", "pairwise_W1": "pairwise-$W_1$"}


def _load(name):
    p = os.path.join(FIG, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _cat(names):
    frames = [_load(n) for n in names]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    # dynamic-k rows: group A (CreditCard/MAGIC/EEG) + group C (Epileptic/OnlineNews/HIGGS/YearPred)
    dyn = _cat(["confirm_real_data_origspace_dynamic_summary.csv",
                "confirm_real_data_origspace_highk_summary.csv"])
    # k=4 recovery rows: group B (Epileptic/OnlineNews) + group D (YearPrediction)
    k4 = _cat(["confirm_real_data_origspace_k4_summary.csv",
               "confirm_real_data_origspace_ypk4_summary.csv"])

    spec = [
        ("CreditCard", "CreditCard", dyn), ("MAGIC", "MAGIC", dyn), ("EEG", "EEG", dyn),
        ("HIGGS", "HIGGS", dyn),
        # OnlineNews omitted: it is analysed in the companion empirical paper (Comm. Stat. R3)
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
        kval = int(g["k"].iloc[0]); g = g.set_index("metric")
        labels.append(f"{disp} (k={kval})")
        M.append([g.loc[m, "frac_QS_better"] - 0.5 if m in g.index else 0.0 for m in METRIC_ORDER])
    M = np.array(M)
    n = len(labels)
    print(f"rows present: {n}  ->  {labels}")

    fig, ax = plt.subplots(figsize=(8.6, 0.62 * n + 1.4))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    im = ax.imshow(M, cmap="RdBu", norm=norm, aspect="auto")     # blue=+ (PCA-QS), red=- (SRS)
    ax.set_xticks(range(len(METRIC_ORDER)))
    ax.set_xticklabels([METRIC_LAB[m] for m in METRIC_ORDER], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(len(METRIC_ORDER)):
            f = M[i, j] + 0.5
            ax.text(j, i, f"{f:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.32 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-0.5, 0, 0.5])
    cb.ax.set_yticklabels(["SRS\ncloser", "tie", "PCA-QS\ncloser"], fontsize=7.5)
    ax.set_xlabel("distributional discrepancy metric (original feature space)")
    ax.set_ylabel("dataset (retained dimension $k$)")
    ax.set_title("Structure preservation in the ORIGINAL feature space (1000 reps)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(FIG, "results_real_data_heatmap_origspace.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
