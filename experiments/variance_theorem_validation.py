#!/usr/bin/env python3
"""Direct numerical validation of Proposition (design) and Theorem 2 (variance reduction).

A table of smaller KL/MMD/energy distances does NOT test the variance theorem: that theorem
is about the SAMPLING VARIANCE of an estimator of mu = E[g(X)].  Here we test it head-on.

Fix a frame X_1..X_N and the PCA-QS partition into cells S_h (sizes N_h, weights
pi_h = N_h/N).  For several scalar functionals g:

  * Proposition:  the weighted estimator  Phat_w g = sum_h pi_h * mean_{s_h} g   is
    design-unbiased for the frame average P_N g, with design variance
        V_prop = sum_h (N_h/N)^2 (1 - r_h/N_h) S_{h,g}^2 / r_h.
    We check bias ~ 0 and Var_empirical / V_prop ~ 1 over B within-cell resamples.

  * Theorem 2:  R * [Var(Phat_SRS g) - Var(Phat_w g)]  ->  B_g := sum_h pi_h (mu_h - mu)^2
    (the between-stratum variance).  We plot observed vs predicted; theory is the 45-deg line.
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS, srs_indices
from pcaqs.data import anisotropic_gmm

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
BLUE, RED = "#4c72b0", "#c44e52"


def functionals(X, sc):
    # Light-tailed functionals spanning a range of between-stratum variance B, from a
    # near-zero anchor (a tail coordinate, orthogonal to the stratified top-3 PCs) to large
    # (linear/bounded functions of the leading PC scores).  Heavy-tailed squares (X1^2,
    # ||X||^2) are deliberately excluded: their difference-of-sample-variances estimator is
    # 8th-moment-dominated and needs far more replications to stabilise.
    cols = {"PC1": sc[:, 0], "PC2": sc[:, 1], "PC3": sc[:, 2],
            "tanh(PC1)": np.tanh(sc[:, 0]), "1{PC1>0}": (sc[:, 0] > 0).astype(float),
            "X1": X[:, 0], "X6(tail)": X[:, 5]}
    return np.column_stack(list(cols.values())), list(cols.keys())


def run(N=40000, k=3, m=5, delta=0.05, reps=1000, seed=20260806, stratification="profile"):
    rng = np.random.default_rng(seed)
    X, _ = anisotropic_gmm(N, random_state=int(rng.integers(1 << 31)))
    qs = PCAQS(n_components=k, n_bins=m, retention=delta, random_state=int(rng.integers(1 << 31)),
               stratification=stratification)
    sc = qs.fit(X).scores(X)
    lab = qs.strata(X)                                     # composite cell label per frame point
    uniq, lab = np.unique(lab, return_inverse=True)        # 0..L-1
    L = len(uniq); Nh = np.bincount(lab, minlength=L); pih = Nh / N
    G, names = functionals(X, sc); P = G.shape[1]
    mu = G.mean(0)                                         # frame averages P_N g

    # between-stratum variance B_g and within-cell S^2 (ddof=1), per functional
    cell_sum = np.zeros((L, P)); np.add.at(cell_sum, lab, G)
    cell_mean = cell_sum / Nh[:, None]
    B = (pih[:, None] * (cell_mean - mu) ** 2).sum(0)      # sum_h pi_h (mu_h - mu)^2
    sq_sum = np.zeros((L, P)); np.add.at(sq_sum, lab, G ** 2)
    S2 = (sq_sum - Nh[:, None] * cell_mean ** 2) / np.maximum(Nh[:, None] - 1, 1)  # within-cell var

    idx0 = qs.sample_indices(X, allocation="floor")
    R = len(idx0)                                          # realized size under max(1, floor(delta N_g))              # exact allocation is deterministic in r_h
    rh = np.bincount(lab[idx0], minlength=L).astype(float)
    fpc = np.where(Nh > 0, 1 - rh / np.maximum(Nh, 1), 0.0)
    V_prop = (((pih ** 2 * fpc)[:, None]) * np.where(rh[:, None] > 0, S2 / np.maximum(rh[:, None], 1), 0.0)).sum(0)

    muw = np.empty((reps, P)); mus = np.empty((reps, P))
    for b in range(reps):
        qi = qs.sample_indices(X, allocation="floor")
        lb = lab[qi]
        s = np.zeros((L, P)); np.add.at(s, lb, G[qi])
        c = np.bincount(lb, minlength=L)
        cm = np.where(c[:, None] > 0, s / np.maximum(c[:, None], 1), cell_mean)   # empty cell -> frame mean (rare)
        muw[b] = (pih[:, None] * cm).sum(0)                # weighted stratified estimate
        si = srs_indices(N, R, random_state=int(rng.integers(1 << 31)))
        mus[b] = G[si].mean(0)
    var_w = muw.var(0, ddof=1); var_s = mus.var(0, ddof=1)
    # exact SRSWOR design variance at the same realized size (no Monte Carlo error)
    var_s_exact = (1 - R / N) * G.var(0, ddof=1) / R
    bias = muw.mean(0) - mu
    df = pd.DataFrame(dict(functional=names, P_N_g=mu, mean_muw=muw.mean(0), abs_bias=np.abs(bias),
                           var_pcaqs=var_w, var_prop=V_prop, var_ratio=var_w / V_prop,
                           var_srs=var_s, var_srs_exact=var_s_exact,
                           obs_R_dVar=R * (var_s_exact - var_w), pred_B=(1 - R / N) * B,
                           pred_B_formula=R * (var_s_exact - V_prop)))
    return df, dict(N=N, k=k, m=m, R=R, L=L, reps=reps)


def plot(df, meta):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    a = ax[0]
    lim = max(df.pred_B.max(), (df.obs_R_dVar + 2 * meta["R"] * df.var_pcaqs * np.sqrt(2.0 / (meta["reps"] - 1))).max()) * 1.08
    a.plot([0, lim], [0, lim], ":", color="gray", label="theory ($45^\\circ$)")
    se = 2 * meta["R"] * df.var_pcaqs * np.sqrt(2.0 / (meta["reps"] - 1))    # +-2 Monte Carlo s.e.
    a.errorbar(df.pred_B, df.obs_R_dVar, yerr=se, fmt="o", ms=6, color=BLUE, ecolor=BLUE,
               elinewidth=1, capsize=3, zorder=3, label="observed $\\pm 2$ MC s.e.")
    for _, r in df.iterrows():
        a.annotate(r.functional, (r.pred_B, r.obs_R_dVar), fontsize=7,
                   xytext=(4, 3), textcoords="offset points")
    a.set(xlabel=r"predicted $(1-R/N)\sum_h\pi_h(\mu_h-\mu)^2$",
          ylabel=r"observed $R\,[\mathrm{Var}_{\mathrm{SRS}}-\widehat{\mathrm{Var}}_{\mathrm{QS}}]$",
          xlim=(0, lim), ylim=(0, lim))
    a.set_title("variance reduction $=$ between-stratum variance")
    a.grid(True, ls=":", alpha=0.4); a.legend(fontsize=8)
    a = ax[1]
    x = np.arange(len(df))
    a.bar(x, df.var_ratio, color=BLUE, alpha=0.85,
          yerr=2 * np.sqrt(2.0 / (meta["reps"] - 1)), capsize=3, ecolor="k")
    a.axhline(1.0, color=RED, ls="--", lw=1.2, label="exact (ratio $=1$)")
    a.set_xticks(x); a.set_xticklabels(df.functional, rotation=45, ha="right", fontsize=7)
    a.set(ylabel="empirical Var / Proposition formula", ylim=(0, 1.3))
    a.set_title("simulated design variance / exact formula ($\\pm2$ MC s.e.)"); a.legend(fontsize=8)
    a.grid(True, axis="y", ls=":", alpha=0.4)
    fig.suptitle(f"Profile strata: N={meta['N']}, k={meta['k']}, m={meta['m']}, realized r={meta['R']}, "
                 f"{meta['reps']} replicates", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "variance_theorem_validation.png"), dpi=150)
    print("wrote variance_theorem_validation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--N", type=int, default=40000)
    a = ap.parse_args()
    df, meta = run(N=a.N, reps=a.reps)
    df.to_csv(os.path.join(OUT, "variance_theorem_validation.csv"), index=False)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(df.to_string(index=False))
    print("\nmax |bias| =", df.abs_bias.max(), " median var_ratio =", df.var_ratio.median())
    plot(df, meta)
