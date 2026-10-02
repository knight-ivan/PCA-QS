#!/usr/bin/env python3
"""Spectral geometry of the real datasets: effective rank (Roy and Vetterli 2007), leading-direction
concentration A_1 = lambda_1 / tr(Sigma), the variance shares rho_3 and rho_{k*}, and k*, the smallest
number of components reaching 70% of the variance. Deterministic; output ../figures/geometry_summary.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_validation import load_real, REAL, FIG

def main():
    rows = []
    for name in REAL:
        X = load_real(name)
        lam = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1], 0, None)
        q = lam / lam.sum()
        erank = float(np.exp(-(q[q > 0] * np.log(q[q > 0])).sum()))
        cum = np.cumsum(q)
        kstar = int(np.searchsorted(cum, 0.70) + 1)
        rows.append(dict(dataset=name, N=len(X), d=X.shape[1], erank=erank, erank_over_d=erank / X.shape[1],
                         A1=q[0], rho3=cum[2], kstar=kstar, rho_kstar=cum[kstar - 1]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG, "geometry_summary.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
