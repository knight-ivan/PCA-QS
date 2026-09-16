#!/usr/bin/env python3
"""Per-dataset detailed metric tables for the supplement (PCA-score space).

Reads the PCA-space run of `confirm_real_data.py` --- the same profile design, seeds and
replicate count as the original-space results in the main text --- and writes one LaTeX table
per dataset with the mean discrepancies, paired 95% CIs and the fraction of replicates on
which PCA-QS is closer to the full data.

Regenerate the input with:
    python3 confirm_real_data.py --datasets MAGIC EEG CreditCard HIGGS YearPrediction Epileptic \
        --reps 1000 --stratification profile --space pca --tag _pca_profile
"""
import os, pandas as pd

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
SRC = "confirm_real_data_pca_profile_summary.csv"
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
MNAME = {"quantile_error": "Quantile error", "KL": "KL", "JS": "JS", "energy": "Energy",
         "MMD": "MMD", "Mahalanobis": "Mahalanobis", "pairwise_W1": r"Pairwise $W_1$"}
# (key, display name) in the order used by the main text
DATASETS = [("MAGIC", "MAGIC Gamma Telescope"), ("EEG", "EEG Eye State"),
            ("CreditCard", "Credit Card (UCI default)"), ("HIGGS", "HIGGS"),
            ("YearPrediction", "Year Prediction MSD"), ("Epileptic", "Epileptic Seizure")]


def mbody(v):
    """3-sig-fig math body, converting e-notation to \\times 10^{n}."""
    s = f"{v:.3g}"
    if "e" in s:
        mant, exp = s.split("e"); s = f"{mant}\\times 10^{{{int(exp)}}}"
    return s


def cell(v, bold):
    b = mbody(v)
    return f"$\\mathbf{{{b}}}$" if bold else f"${b}$"


def table(d, ds, disp):
    g = d[d.dataset == ds]
    if g.empty:
        return ""
    k, m = int(g["k"].iloc[0]), int(g["m"].iloc[0])
    H, ret = g["H_N"].iloc[0], 100 * g["realized_retention"].iloc[0]
    gi = g.set_index("metric")
    lines = []
    for mt in METRICS:
        if mt not in gi.index:
            continue
        qs, sr = gi.loc[mt, "QS_mean"], gi.loc[mt, "SRS_mean"]
        dd, lo, hi, fr = (gi.loc[mt, c] for c in ("diff", "ci_lo", "ci_hi", "frac_QS_better"))
        sig = lo > 0 or hi < 0                       # bold only a significant difference
        lines.append(f"{MNAME[mt]} & {cell(qs, sig and qs < sr)} & {cell(sr, sig and sr < qs)} & "
                     f"${mbody(dd)}\\;[{mbody(lo)},\\,{mbody(hi)}]$ & {fr:.3f} \\\\")
    cap = (f"{disp}: discrepancies to the full data in the top-$k$ PCA-score space "
           f"(profile design, $k{{=}}{k}$, $m{{=}}{m}$, $H_N={H:.0f}$ occupied strata, realized "
           f"retention ${ret:.1f}\\%$; mean over $1000$ replicates). Lower is better; "
           f"\\textbf{{bold}} marks the design closer to the full data when the paired interval "
           f"excludes zero. The last column is the fraction of replicates on which PCA-QS is closer.")
    return ("\\begin{table}[h]\n\\centering\\small\n\\caption{" + cap + "}\n"
            f"\\label{{tab:detail_{ds.lower()}}}\n\\resizebox{{\\textwidth}}{{!}}{{%\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n"
            "Metric & PCA-QS & SRS & diff $[95\\%$ CI$]$ & Frac.\\ PCA-QS closer \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


SHORT = {"MAGIC": "MAGIC", "EEG": "EEG", "CreditCard": "Credit Card", "HIGGS": "HIGGS",
         "YearPrediction": "YearPred.", "Epileptic": "Epileptic"}


def compact(d):
    """One-row-per-dataset overview table (PCA-QS / SRS per metric)."""
    rows = []
    for ds, _ in DATASETS:
        g = d[d.dataset == ds]
        if g.empty:
            continue
        gi = g.set_index("metric")
        cells = []
        for mt in METRICS:
            qs, sr = gi.loc[mt, "QS_mean"], gi.loc[mt, "SRS_mean"]
            sig = gi.loc[mt, "ci_lo"] > 0 or gi.loc[mt, "ci_hi"] < 0
            cells.append(f"{cell(qs, sig and qs < sr)[1:-1]}/{cell(sr, sig and sr < qs)[1:-1]}")
        rows.append(f"{SHORT[ds]} & {int(g.k.iloc[0])} & " + " & ".join(f"${c}$" for c in cells) + r" \\")
    cap = ("Discrepancies to the full data in the top-$k$ PCA-score space (profile design, $m=5$, "
           "$\\delta=0.05$, mean over $1000$ replicates, simple random sampling at the same realized "
           "size). Each cell is PCA-QS\\,/\\,SRS; \\textbf{bold} marks the closer design when the "
           "paired interval excludes zero. Lower is better.")
    return ("\\begin{table}[H]\n\\centering\\small\n\\caption{" + cap + "}\n"
            "\\label{tab:detailed_corrected}\n\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{llccccccc}\n\\toprule\n"
            "Dataset & $k$ & quantile & KL & JS & energy & MMD & Mahal. & pair-$W_1$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def main():
    d = pd.read_csv(os.path.join(FIG, SRC))
    tex = compact(d) + "\n" + "\n".join(table(d, ds, disp) for ds, disp in DATASETS)
    path = os.path.join(FIG, "detailed_metrics_tables_appendix.tex")
    open(path, "w").write(tex)
    print(tex)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
