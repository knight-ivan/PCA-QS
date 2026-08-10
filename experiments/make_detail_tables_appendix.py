#!/usr/bin/env python3
"""Regenerate the per-dataset detailed metric tables (appendix), corrected metrics.

Restores the granularity of the originally-submitted per-dataset tables (organised in the
same four categories) but with the CORRECTED metric definitions, 1000-replicate means,
95% CIs, and the fraction of replicates on which PCA-QS is closer to the full data.  For
the three datasets whose dynamic-k choice violates the occupancy budget m^k<=r
(Epileptic, OnlineNews, YearPrediction) we report the occupancy-feasible k=4 configuration
and note the degradation at the naive dynamic k.
"""
import os, pandas as pd, numpy as np

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
METRICS = ["quantile_error", "KL", "JS", "energy", "MMD", "Mahalanobis", "pairwise_W1"]
MNAME = {"quantile_error": "Quantile error", "KL": "KL", "JS": "JS", "energy": "Energy",
         "MMD": "MMD", "Mahalanobis": "Mahalanobis", "pairwise_W1": r"Pairwise $W_1$"}
# dataset -> (category, display, source, occupancy_note)
CATS = [
    ("Imbalanced Datasets", [
        ("CreditCard", "Credit Card (UCI default)", "dyn", None),
        ("MAGIC", "MAGIC Gamma Telescope", "dyn", None)]),
    ("Time-Series and Signal Data", [
        ("EEG", "EEG Eye State", "dyn", None),
        ("Epileptic", "Epileptic Seizure", "k4", 21)]),
    ("Large-Scale Data", [
        ("HIGGS", "HIGGS", "dyn", None),
        ("YearPrediction", "Year Prediction MSD", "k4", 28)]),
    ("Feature Complexity (Text/NLP)", [
        ("OnlineNews", "Online News Popularity", "k4", 17)]),
]


def mbody(v):
    """3-sig-fig math body, converting e-notation to \\times 10^{n}."""
    s = f"{v:.3g}"
    if "e" in s:
        mant, exp = s.split("e"); s = f"{mant}\\times 10^{{{int(exp)}}}"
    return s


def cell(v, bold):
    b = mbody(v)
    return f"$\\mathbf{{{b}}}$" if bold else f"${b}$"


def table(row_src, ds, disp, note_dynk):
    gi = row_src[row_src.dataset == ds]
    if gi.empty:
        return ""
    k = int(gi["k"].iloc[0]); m = int(gi["m"].iloc[0]); gi = gi.set_index("metric")
    lines = []
    for mt in METRICS:
        if mt not in gi.index:
            continue
        qs, sr = gi.loc[mt, "QS_mean"], gi.loc[mt, "SRS_mean"]
        d, lo, hi, fr = (gi.loc[mt, c] for c in ("diff", "ci_lo", "ci_hi", "frac_QS_better"))
        qcell, scell = cell(qs, qs < sr), cell(sr, sr < qs)
        diffcell = f"${mbody(d)}\\;[{mbody(lo)},\\,{mbody(hi)}]$"
        lines.append(f"{MNAME[mt]} & {qcell} & {scell} & {diffcell} & {fr:.3f} \\\\")
    note = ""
    if note_dynk is not None:
        note = (f"\n\\emph{{Occupancy note:}} the naive dynamic choice $k{{=}}{note_dynk}$ gives "
                f"$m^{{k}}\\gg r$ (most composite cells empty), and PCA-QS degrades to at best a tie; "
                f"the occupancy-feasible $k{{=}}4$ above is the fair comparison.")
    cap = (f"{disp}: corrected discrepancies to the full data (mean over $1000$ replicates, "
           f"top-$k$ PCA-score space, $k{{=}}{k}$, $m{{=}}{m}$). Lower is better; \\textbf{{bold}} marks "
           f"the method closer to the full data. The last column is the fraction of replicates on "
           f"which PCA-QS is closer.{note}")
    key = ds.lower()
    return ("\\begin{table}[h]\n\\centering\\small\n\\caption{" + cap + "}\n"
            f"\\label{{tab:detail_{key}}}\n\\begin{{tabular}}{{lcccc}}\n\\toprule\n"
            "Metric & PCA-QS & SRS & diff $[95\\%$ CI$]$ & Frac.\\ PCA-QS closer \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    dyn = pd.read_csv(os.path.join(FIG, "confirm_real_data_1000_dynamicK_summary.csv"))
    k4 = pd.read_csv(os.path.join(FIG, "confirm_real_data_k4recovery_summary.csv"))
    src = {"dyn": dyn, "k4": k4}
    out = []
    for cat, rows in CATS:
        out.append(f"\\subsection{{{cat}}}")
        for ds, disp, s, note in rows:
            out.append(table(src[s], ds, disp, note))
    tex = "\n".join(out)
    path = os.path.join(FIG, "detailed_metrics_tables_appendix.tex")
    open(path, "w").write(tex)
    print(tex)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
