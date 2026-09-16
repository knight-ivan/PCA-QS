#!/usr/bin/env python3
"""Coefficient-recovery study: does PCA-QS estimate the *effect of each variable*
better than SRS, even when predictive accuracy is tied?

Rationale (Theorem 2): a regression/logistic coefficient is, to leading order, an
average of per-observation influence functions --- a smooth functional of the sample
--- so the stratification variance-reduction guarantee applies. We therefore expect
coefficients from a PCA-QS subset to be (a) closer to the full-data coefficients and
(b) more stable across replicates than from an SRS subset of the same size.

For each dataset and replicate: split train/test, standardise on train, fit the model
on ALL of train (beta_full), then on equal-size PCA-QS and SRS subsets; record the
coefficient error ||beta_sub - beta_full||_2 (standardised-feature space, so it is a
statement about variable effects). Aggregate the paired PCA-QS vs SRS error and the
across-replicate coefficient stability.

Classification: logistic regression (Credit Card, APS/OpenML, MAGIC, Epileptic seizure).
Regression:     ridge (YearPredictionMSD).
(CASP, Bike Sharing and Appliances loaders are kept for reference; those datasets are
analysed in the companion empirical paper and are not used in the theory paper.)
"""
import os, sys, argparse
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices, choose_design

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CREDIT = os.path.join(_PROJ, "LaTeX", "Final Version", "Real Data Logistic Regression",
                      "crediccard", "UCI_Credit_Card.csv")


_REAL = os.path.join(_PROJ, "LaTeX", "Final Version", "Real Data Marix Comparions csv")


def load(name):
    if name == "MAGIC":
        df = pd.read_csv(os.path.join(_REAL, "magic+gamma+telescope", "magic04.csv"), header=None)
        return df.iloc[:, :-1].to_numpy(float), (df.iloc[:, -1] == "g").to_numpy().astype(int), "clf"
    if name == "Epileptic":
        df = pd.read_csv(os.path.join(_REAL, "Epileptic Seizure Recognition", "Epileptic_Seizure_Recognition.csv"))
        X = df[[c for c in df.columns if c.startswith("X") and c[1:].isdigit()]].to_numpy(float)
        return X, (df["y"] == 1).to_numpy().astype(int), "clf"          # seizure vs non-seizure
    if name == "YearPrediction":
        df = pd.read_csv(os.path.join(_REAL, "YearPredictionMSD", "YearPredictionMSD.csv"), header=None)
        return df.iloc[:, 1:].to_numpy(float), df.iloc[:, 0].to_numpy(float), "reg"
    if name == "CreditCard":
        df = pd.read_csv(CREDIT); y = df.iloc[:, -1].to_numpy().astype(int)
        X = df.iloc[:, :-1].select_dtypes(include=[np.number]).to_numpy(float); return X, y, "clf"
    if name == "APS":
        d = fetch_openml("APSFailure", version=1, as_frame=True)
        X = d.data.select_dtypes(include=[np.number]).to_numpy(float)
        return X, (d.target.to_numpy() == "pos").astype(int), "clf"
    if name == "CASP":
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv")
        return df.drop(columns=["RMSD"]).to_numpy(float), df["RMSD"].to_numpy(float), "reg"
    if name == "BikeSharing":
        d = fetch_openml(data_id=42712, as_frame=True)
        return d.data.select_dtypes(include=[np.number]).to_numpy(float), \
               pd.to_numeric(d.target, errors="coerce").to_numpy(), "reg"
    if name == "Appliances":
        d = fetch_openml("appliances_energy_prediction", version=1, as_frame=True)
        return d.data.select_dtypes(include=[np.number]).to_numpy(float), \
               pd.to_numeric(d.target, errors="coerce").to_numpy(), "reg"


def _model(task):
    return LogisticRegression(max_iter=1000) if task == "clf" else Ridge(alpha=1.0)


def _coef(m):
    c = m.coef_
    return c.ravel()


def one_rep(X, y, task, seed, retain, var_target):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed,
                                          stratify=y if task == "clf" else None)
    if not np.isfinite(Xtr).all():
        imp = SimpleImputer(strategy="median").fit(Xtr); Xtr = imp.transform(Xtr)
    sc = StandardScaler().fit(Xtr); Xtr_s = sc.transform(Xtr)
    n = len(Xtr); r = int(round(retain * n)); gen = np.random.default_rng(seed)

    beta_full = _coef(_model(task).fit(Xtr_s, ytr))            # target: full-train coefficients

    pca = PCA().fit(Xtr_s)
    k, m, _ = choose_design(r, pca.explained_variance_ratio_, var_target=var_target, stratification="profile")
    qs = PCAQS(n_components=k, n_bins=m, retention=retain, random_state=int(gen.integers(1 << 31)),
               stratification="profile").fit(Xtr)
    qi = qs.sample_indices(Xtr, allocation="floor")
    si = srs_indices(n, len(qi), random_state=int(gen.integers(1 << 31)))   # same realized size

    out = {}
    for meth, idx in (("PCA-QS", qi), ("SRS", si)):
        b = _coef(_model(task).fit(Xtr_s[idx], ytr[idx]))
        out[meth] = b
    return dict(seed=seed,
                err_QS=float(np.linalg.norm(out["PCA-QS"] - beta_full)),
                err_SRS=float(np.linalg.norm(out["SRS"] - beta_full)),
                relerr_QS=float(np.linalg.norm(out["PCA-QS"] - beta_full) / (np.linalg.norm(beta_full) + 1e-12)),
                relerr_SRS=float(np.linalg.norm(out["SRS"] - beta_full) / (np.linalg.norm(beta_full) + 1e-12)),
                coef_QS=out["PCA-QS"], coef_SRS=out["SRS"])


def run(name, reps, retain, var_target, base_seed, jobs):
    X, y, task = load(name)
    print(f"{name}: N={len(X)} d={X.shape[1]} task={task} reps={reps}", flush=True)
    res = Parallel(n_jobs=jobs)(delayed(one_rep)(X, y, task, base_seed + i, retain, var_target)
                                for i in range(reps))
    coefs_qs = np.array([r["coef_QS"] for r in res]); coefs_srs = np.array([r["coef_SRS"] for r in res])
    errQS = np.array([r["err_QS"] for r in res]); errSRS = np.array([r["err_SRS"] for r in res])
    # coefficient stability: mean over coefficients of the across-replicate SD
    stab_qs = coefs_qs.std(0).mean(); stab_srs = coefs_srs.std(0).mean()
    d = errQS - errSRS; se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(dataset=name, task=task, d=X.shape[1],
                err_QS=errQS.mean(), err_SRS=errSRS.mean(),
                relerr_QS=np.mean([r["relerr_QS"] for r in res]),
                relerr_SRS=np.mean([r["relerr_SRS"] for r in res]),
                err_diff=d.mean(), ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se,
                frac_QS_closer=float(np.mean(errQS < errSRS)),
                coef_stability_QS=stab_qs, coef_stability_SRS=stab_srs,
                stability_ratio_QS_over_SRS=stab_qs / stab_srs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["CreditCard", "APS", "MAGIC", "Epileptic", "YearPrediction"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--retain", type=float, default=0.1)
    ap.add_argument("--var", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    rows = [run(n, a.reps, a.retain, a.var, a.seed, a.jobs) for n in a.datasets]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "coef_recovery_summary.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    show = df[["dataset", "task", "d", "relerr_QS", "relerr_SRS", "frac_QS_closer",
               "stability_ratio_QS_over_SRS"]]
    print("\n=== coefficient recovery: PCA-QS vs SRS (lower rel.err / stability-ratio<1 => PCA-QS better) ===")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
