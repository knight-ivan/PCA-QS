#!/usr/bin/env python3
"""Hybrid strata: grid labels on the leading J components, cutoff-count profile on the rest.

The hybrid family interpolates between the profile design (J = 0) and the full grid (J = k).
For Gaussian data and isotropically averaged linear targets, the fine-bin limit of the
target-averaged variance ratio is

    1 - rho_J - (rho_k - rho_J) / (k - J),        rho_J = sum_{j<=J} lambda_j / tr(Sigma),

(with the last term absent when J = k), and the number of possible strata is
B^J * C(k - J + B - 1, B - 1).

Studies
  sim    population-level (idealized proportional allocation) target-averaged variance ratio
         for J = 0..k on two Gaussian spectra, against the limit;
  real   the six real datasets at delta = 0.05 with the companion component count
         k = min(k*, 10) and B = 5: for J = 0..k (while the partition is not degenerate),
         occupied strata, realized retention, the unequal-weighting factor kappa_w, the exact
         design-variance ratio for the feature means (floor allocation) and the Gaussian limit.

No replication is needed: every variance is the exact (population or finite-population)
stratified formula. Outputs: ../figures/hybrid_{sim,real}.csv.
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_validation import (strata_keys, n_possible_strata, percentile_cuts, within_cell_moments,
                               floor_alloc, pc_directions, load_real, REAL, _haar, FIG)

SPECTRA = {
    # geometric decay: leading shares unequal throughout
    "geometric": lambda d: 10 * 0.7 ** np.arange(d),
    # two dominant directions followed by a nearly flat block: labels matter for J <= 2 only
    "two-spike": lambda d: np.r_[12.0, 8.0, 2.0, 1.9, 1.8, np.full(d - 5, 1.0)],
}


def hybrid_limit(lam, k, J):
    rho = np.cumsum(lam) / lam.sum()
    rJ = rho[J - 1] if J > 0 else 0.0
    rk = rho[k - 1]
    return 1 - rJ - ((rk - rJ) / (k - J) if J < k else 0.0)


def kappa_w(Nh, rh):
    N, r = Nh.sum(), rh.sum()
    return float(r / N ** 2 * (Nh ** 2 / rh).sum())


def study_sim(N=2_000_000, d=20, k=5, Bs=(5, 8), max_strata=50_000, seed=21):
    """Same population size and stratum cap as study_profile in theory_validation.py."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, spec in SPECTRA.items():
        lam = spec(d)
        Q = _haar(d, d, rng)
        X = rng.standard_normal((N, d)) * np.sqrt(lam) @ Q.T
        sc = X @ Q[:, :k]
        tot = X.var(0).sum()
        for B in Bs:
            cuts = percentile_cuts(sc, B)
            for J in range(k + 1):
                design = f"hybrid:{J}"
                M = n_possible_strata(k, B, design)
                if M > max_strata:
                    continue
                key = strata_keys(sc, cuts, design)
                Nh, _, V = within_cell_moments(key, X)
                ratio = float(((Nh / N)[:, None] * V).sum() / tot)
                rows.append(dict(spectrum=name, k=k, B=B, J=J, possible_strata=M, occupied=len(Nh),
                                 ratio=ratio, limit=hybrid_limit(lam, k, J)))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(FIG, "hybrid_sim.csv"), index=False)
    return df


def exact_mean_ratio(X, key, delta):
    """Exact design-variance ratio (trace over features) of the weighted PCA-QS mean under the
    floor allocation, against SRS without replacement of the same realized size."""
    N = len(X)
    Nh, _, Sh2 = within_cell_moments(key, X)
    rh = floor_alloc(Nh, delta)
    r = int(rh.sum())
    v_qs = (((Nh / N) ** 2 * (1 - rh / Nh))[:, None] * Sh2 / rh[:, None]).sum()
    v_srs = (1 - r / N) * X.var(0, ddof=1).sum() / r
    # proportional allocation on the same strata (no floor), for reference
    v_prop = (1 - r / N) / r * ((Nh / N)[:, None] * Sh2).sum()
    return float(v_qs / v_srs), float(v_prop / v_srs), Nh, rh


def study_real(delta=0.05, B=5, kw_max=1.05, datasets=None):
    rows = []
    for name in (datasets or list(REAL)):
        X = load_real(name)
        N, d = X.shape
        _, lam = pc_directions(X, min(d, 50))
        cum = np.cumsum(lam) / lam.sum()
        k = int(min(np.searchsorted(cum, 0.70) + 1, 10))
        V, _ = pc_directions(X, k)
        sc = X @ V
        cuts = percentile_cuts(sc, B)
        for J in range(k + 1):
            key = strata_keys(sc, cuts, f"hybrid:{J}")
            ratio, ratio_prop, Nh, rh = exact_mean_ratio(X, key, delta)
            kw = kappa_w(Nh, rh)
            rows.append(dict(dataset=name, N=N, d=d, k=k, B=B, J=J, delta=delta,
                             possible_strata=n_possible_strata(k, B, f"hybrid:{J}"),
                             H_N=len(Nh), occupancy=len(Nh) / (delta * N),
                             realized_retention=rh.sum() / N, kappa_w=kw,
                             exact_mean_ratio=ratio, proportional_ratio=ratio_prop,
                             gaussian_limit=hybrid_limit(lam, k, J), passes_check=kw <= kw_max))
            print({c: (round(v, 4) if isinstance(v, float) else v) for c, v in rows[-1].items()}, flush=True)
            if len(Nh) > delta * N:          # partition finer than the retained size: stop refining
                break
    df = pd.DataFrame(rows)
    # rule: the largest J whose partition passes the kappa_w check
    df["rule_J"] = df.groupby("dataset")["J"].transform(
        lambda s: s[df.loc[s.index, "passes_check"]].max() if df.loc[s.index, "passes_check"].any() else -1)
    df.to_csv(os.path.join(FIG, f"hybrid_real_d{int(round(100 * delta))}.csv"), index=False)
    return df


def select_design(X, delta, B=5, k_max=10, lam=None):
    """Choose (k, J) by the exact design variance of the feature means.

    Candidates: 1 <= k <= k_max and 0 <= J <= k - 1 (J = k - 1 already equals the grid on k
    components), keeping only partitions with at most delta*N occupied strata. The criterion is
    the exact finite-population variance of the weighted feature means under the floor
    allocation, summed over features: a target-agnostic summary computable from the frame
    before any point is drawn. Returns the candidate table sorted by the criterion."""
    N = len(X)
    Vall, lam = pc_directions(X, k_max)
    rows = []
    for k in range(1, k_max + 1):
        sc = X @ Vall[:, :k]
        cuts = percentile_cuts(sc, B)
        for J in range(0, max(k - 1, 0) + 1):
            key = strata_keys(sc, cuts, f"hybrid:{J}")
            H = len(np.unique(key))
            if H > delta * N:
                break
            ratio, _, Nh, rh = exact_mean_ratio(X, key, delta)
            rows.append(dict(k=k, J=J, H_N=H, kappa_w=kappa_w(Nh, rh), exact_mean_ratio=ratio,
                             realized_retention=rh.sum() / N, gaussian_limit=hybrid_limit(lam, k, J)))
    return pd.DataFrame(rows).sort_values("exact_mean_ratio").reset_index(drop=True)


def study_select(delta=0.05, datasets=None):
    out = []
    for name in (datasets or list(REAL)):
        X = load_real(name)
        cand = select_design(X, delta)
        best = cand.iloc[0].to_dict()
        best.update(dataset=name, N=len(X), delta=delta, n_candidates=len(cand))
        out.append(best)
        cand.assign(dataset=name).to_csv(os.path.join(FIG, f"hybrid_select_{name}_d{int(round(100 * delta))}.csv"), index=False)
        print(name, {c: (round(v, 4) if isinstance(v, float) else v) for c, v in best.items()}, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(FIG, f"hybrid_select_d{int(round(100 * delta))}.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="*", default=["sim", "real"])
    ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    if "sim" in a.studies:
        print(study_sim().round(4).to_string(index=False))
    if "real" in a.studies:
        print(study_real(delta=a.delta).round(4).to_string(index=False))
    if "select" in a.studies:
        print(study_select(delta=a.delta).round(4).to_string(index=False))
