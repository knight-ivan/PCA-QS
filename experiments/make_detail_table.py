#!/usr/bin/env python3
"""Emit a corrected detailed-metric LaTeX table from the 1000-rep confirmation.

Replaces the superseded appendix tables (old point-to-center Mahalanobis, binned KL,
Ded x PC grid). Each cell shows PCA-QS / SRS mean discrepancy to the full data in the
top-k PCA-score space; the smaller (closer) value is bold-faced.
"""
import os, pandas as pd, numpy as np
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
HEAD = {"quantile_error": "quantile", "KL": "KL", "JS": "JS", "energy": "energy",
        "MMD": "MMD", "Mahalanobis": "Mahal.", "pairwise_W1": "pair-$W_1$"}
ROWS = [  # (dataset, source csv, display)
    ("CreditCard", "dyn", "Credit Card"), ("MAGIC", "dyn", "MAGIC"), ("EEG", "dyn", "EEG"),
    ("HIGGS", "dyn", "HIGGS"),
    ("OnlineNews", "dyn", "OnlineNews"), ("OnlineNews", "k4", "OnlineNews"),
    ("Epileptic", "dyn", "Epileptic"), ("Epileptic", "k4", "Epileptic"),
    ("YearPrediction", "dyn", "YearPred."), ("YearPrediction", "k4", "YearPred."),
]


def fmt(v):
    return f"{v:.3g}"


def cell(qs, sr):
    a, b = fmt(qs), fmt(sr)
    return (f"\\textbf{{{a}}}/{b}" if qs < sr else f"{a}/\\textbf{{{b}}}")


def main():
    dyn = pd.read_csv(os.path.join(FIG, "confirm_real_data_1000_dynamicK_summary.csv"))
    k4 = pd.read_csv(os.path.join(FIG, "confirm_real_data_k4recovery_summary.csv"))
    src = {"dyn": dyn, "k4": k4}
    lines = []
    for ds, s, disp in ROWS:
        g = src[s][src[s].dataset == ds]
        if g.empty:
            continue
        kval = int(g["k"].iloc[0]); gi = g.set_index("metric")
        cells = " & ".join(cell(gi.loc[m, "QS_mean"], gi.loc[m, "SRS_mean"]) for m in METRICS)
        lines.append(f"{disp} & {kval} & {cells} \\\\")
    header = "Dataset & $k$ & " + " & ".join(HEAD[m] for m in METRICS) + r" \\"
    body = "\n".join(lines)
    tex = ("\\begin{table}[h]\n\\centering\\small\n"
           "\\caption{Corrected detailed metrics (mean over $1000$ replicates, equal exact "
           "retained size, top-$k$ PCA-score space). Each cell is PCA-QS\\,/\\,SRS discrepancy "
           "to the full data; \\textbf{bold} marks the smaller (closer). Lower is better.}\n"
           "\\label{tab:detailed_corrected}\n"
           "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{ll" + "c" * len(METRICS) + "}\n"
           "\\toprule\n" + header + "\n\\midrule\n" + body + "\n\\bottomrule\n"
           "\\end{tabular}}\n\\end{table}\n")
    out = os.path.join(FIG, "detailed_metrics_table.tex")
    open(out, "w").write(tex)
    print(tex)
    print("wrote", out)


if __name__ == "__main__":
    main()
