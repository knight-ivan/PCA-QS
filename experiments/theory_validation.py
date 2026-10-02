#!/usr/bin/env python3
"""Numerical validation of the design-based theory of PCA-QS.

Two stratifications are studied throughout (see pcaqs/sampler.py):
  profile  cutoff-count profiles, the design of the companion Communications in
           Statistics paper (B = 5 quintile cutoffs);
  grid     the full cross-classification of the per-component bins, the finest
           PC-quantile stratification (every profile stratum is a union of grid cells).
Allocation is the companion paper's m_g = max(1, floor(delta N_g)); simple random
sampling is compared at the same realized size r = sum_g m_g.

Studies (one per result of Section 2.3 of the paper):
  rate        excess within-stratum variance of a Lipschitz functional vs the number of
              bins, bounded and Gaussian scores, grid vs profile (population cutoffs);
  estimated   realized PCA-QS/SRS design-variance ratio with strata estimated from a
              frame of size N vs the oracle (population) ratio;
  clt         coverage of 95% intervals (conditional and unconditional);
  directions  average design effect over isotropic linear targets for principal,
              random and coordinate-axis directions vs the Gaussian limit;
  realdirs    the same on the real datasets (exact finite-population design variances).

All design variances use the exact stratified formula (Proposition 1). Every study uses
fixed seeds and 1000 replicates where replication is involved. Outputs go to
../figures/theory_*.{csv,png}.
"""
import os, sys, argparse, glob
import numpy as np, pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pcaqs import PCAQS
from pcaqs.data import anisotropic_gmm

_CR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(_CR, "figures")
MAN = os.path.join(os.path.dirname(_CR), "manuscript")
os.makedirs(FIG, exist_ok=True)
DESIGNS = ("profile", "grid")


# ----------------------------------------------------------------------------- helpers
def pc_directions(X, k):
    """Top-k eigenvectors of the covariance with the sampler's sign convention."""
    evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
    V = evecs[:, ::-1][:, :k]
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(k)])
    return V, evals[::-1]


def percentile_cuts(scores, B):
    return np.percentile(scores, np.arange(1, B) * 100.0 / B, axis=0)      # (B-1, k)


def strata_keys(scores, cuts, design):
    """Stratum ids from cutoffs `cuts` (B-1, k).

    design = "profile"   cutoff-count profile of all k components;
             "grid"      full cross-classification of the k component bins;
             "hybrid:J"  grid labels of the leading J components combined with the
                         cutoff-count profile of the remaining k - J components
                         ("hybrid:0" = profile, "hybrid:k" = grid).
    """
    n, k = scores.shape
    above = scores[:, None, :] > cuts[None, :, :]
    B = cuts.shape[0] + 1
    if design == "profile":
        J = 0
    elif design == "grid":
        J = k
    elif design.startswith("hybrid:"):
        J = int(design.split(":")[1])
        if not 0 <= J <= k:
            raise ValueError(f"hybrid:J needs 0 <= J <= k, got J={J}, k={k}")
    else:
        raise ValueError(design)
    key = np.zeros(n, dtype=np.int64)
    if J > 0:                                       # labelled bins of the leading J components
        bins = above[:, :, :J].sum(axis=1)
        for j in range(J):
            key = key * B + bins[:, j]
    if J < k:                                       # profile of the remaining components
        cnt = above[:, :, J:].sum(axis=2)
        for b in range(B - 1):
            key = key * (k - J + 1) + cnt[:, b]
    return key


def n_possible_strata(k, B, design):
    """Number of possible strata: B^J * C(k-J+B-1, B-1)."""
    from math import comb
    J = 0 if design == "profile" else k if design == "grid" else int(design.split(":")[1])
    return B ** J * comb(k - J + B - 1, B - 1)


def floor_alloc(Nh, delta):
    """Companion paper's allocation m_g = max(1, floor(delta N_g)), capped at N_g."""
    return np.minimum(np.maximum(1, np.floor(delta * Nh).astype(np.int64)), Nh)


def largest_remainder(counts, r, min_per_cell=0):
    """Exact-size largest-remainder allocation (kept for sensitivity analyses)."""
    base = PCAQS._largest_remainder(np.asarray(counts), int(r), min_per_cell=min_per_cell)
    assert base.sum() == r
    return base


def within_cell_moments(key, Phi):
    """Per-stratum sizes, means and (N_h-1)-variances of the columns of Phi (n x q)."""
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    H, q = len(uniq), Phi.shape[1]
    s1 = np.zeros((H, q)); s2 = np.zeros((H, q))
    for c in range(q):
        s1[:, c] = np.bincount(inv, Phi[:, c], minlength=H)
        s2[:, c] = np.bincount(inv, Phi[:, c] ** 2, minlength=H)
    mean = s1 / counts[:, None]
    ss = s2 - counts[:, None] * mean ** 2
    var = np.where(counts[:, None] > 1, ss / np.maximum(counts[:, None] - 1, 1), 0.0)
    return counts, mean, var


def design_variances(key, Phi, delta):
    """Exact design variances (Prop. 1): weighted PCA-QS estimator vs SRSWOR of equal size."""
    N = len(key)
    Nh, _, Sh2 = within_cell_moments(key, Phi)
    rh = floor_alloc(Nh, delta)
    r = int(rh.sum())
    v_qs = (((Nh / N) ** 2 * (1 - rh / Nh))[:, None] * Sh2 / rh[:, None]).sum(0)
    v_srs = (1 - r / N) * Phi.var(0, ddof=1) / r
    return v_qs, v_srs, rh, Nh


# ----------------------------------------------------------------------------- rate
def study_rate(N=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    lam = np.array([4.0, 2.0, 1.0, 0.5, 0.5, 0.5]); k = 2
    Bs = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    rows = []
    for dist in ("uniform", "gaussian"):
        Z = (rng.uniform(-np.sqrt(3), np.sqrt(3), (N, 6)) if dist == "uniform"
             else rng.standard_normal((N, 6)))
        X = Z * np.sqrt(lam)
        psi = np.sin(X[:, 0]) + np.abs(X[:, 1])          # Lipschitz in the top-2 scores, L = sqrt(2)
        tail = 0.5 * X[:, 2] + 0.5 * X[:, 4]              # independent of the strata
        g = psi + tail
        tau2 = 0.25 * lam[2] + 0.25 * lam[4]
        var_g = g.var()
        for B in Bs:
            u = np.arange(1, B) / B
            if dist == "uniform":
                cuts = np.column_stack([np.sqrt(lam[j]) * np.sqrt(3) * (2 * u - 1) for j in range(k)])
            else:
                cuts = np.column_stack([np.sqrt(lam[j]) * norm.ppf(u) for j in range(k)])
            for design in DESIGNS:
                key = strata_keys(X[:, :k], cuts, design)
                Nh, _, V = within_cell_moments(key, np.column_stack([psi, g]))
                pi = Nh / N
                excess = float((pi * V[:, 0]).sum())
                total = float((pi * V[:, 1]).sum())
                bound = (2.0 * sum(12 * lam[j] for j in range(k)) / B ** 2) if dist == "uniform" else np.nan
                rows.append(dict(dist=dist, design=design, m=B, strata=len(Nh), excess=excess,
                                 bound=bound, total=total, tau2=tau2, var_g=var_g,
                                 efficiency=total / var_g))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_rate.csv"), index=False)
    for (dist, design), g in df[df.m >= 8].groupby(["dist", "design"]):
        slope = np.polyfit(np.log(g.m), np.log(g.excess), 1)[0]
        print(f"[rate] {dist:8s} {design:7s}: slope of excess vs m (m>=8) = {slope:.2f}; "
              f"efficiency at m=64 = {g.efficiency.iloc[-1]:.3f} (floor {g.tau2.iloc[0] / g.var_g.iloc[0]:.3f})")
    return df


# ----------------------------------------------------------------------------- estimated strata
def _phis(X):
    return np.column_stack([X[:, 0], X[:, 1] ** 2, np.sin(X[:, 0] + X[:, 1]),
                            (X[:, 0] > 0).astype(float), X[:, 5]])


PHI_NAMES = ["x1", "x2^2", "sin(x1+x2)", "1{x1>0}", "x6 (tail)"]


def _oracle(k, B, design, seed=99, Nref=4_000_000):
    Xr, _ = anisotropic_gmm(Nref, random_state=seed)
    V, _ = pc_directions(Xr, k)
    sc = (Xr - Xr.mean(0)) @ V
    key = strata_keys(sc, percentile_cuts(sc, B), design)
    Phi = _phis(Xr)
    Nh, _, Vh = within_cell_moments(key, Phi)
    ratio = (Nh[:, None] / Nref * Vh).sum(0) / Phi.var(0)
    return dict(ratio=ratio, Ephi=Phi.mean(0), strata=len(Nh))


def _estimated_ratio(N, k, B, delta, design, seed):
    X, _ = anisotropic_gmm(N, random_state=seed)
    qs = PCAQS(n_components=k, n_bins=B, retention=delta, standardize=False,
               stratification=design).fit(X)
    v_qs, v_srs, rh, Nh = design_variances(qs.strata(X), _phis(X), delta)
    return v_qs / v_srs


def study_estimated(k=3, B=5, delta=0.05, Ns=(2000, 8000, 32000, 128000, 512000), reps=1000, jobs=-1):
    rows = []
    for design in DESIGNS:
        orc = _oracle(k, B, design)
        for N in Ns:
            res = np.array(Parallel(n_jobs=jobs)(
                delayed(_estimated_ratio)(N, k, B, delta, design, 1000 + i) for i in range(reps)))
            for c, name in enumerate(PHI_NAMES):
                rows.append(dict(design=design, N=N, phi=name, oracle=orc["ratio"][c],
                                 mean_ratio=res[:, c].mean(), sd_ratio=res[:, c].std(ddof=1),
                                 mean_abs_dev=np.abs(res[:, c] - orc["ratio"][c]).mean(), reps=reps))
        print(design, "oracle ratios:", dict(zip(PHI_NAMES, np.round(orc["ratio"], 3))), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_estimated.csv"), index=False)
    print(df.pivot_table(index=["design", "phi"], columns="N", values="mean_abs_dev").round(4))
    return df


# ----------------------------------------------------------------------------- CLT / coverage
def _weighted_estimate(Phi, key_inv, Nh, rh, pick, N):
    """Weighted stratified estimate and its variance estimate from retained indices."""
    h = key_inv[pick]
    q = Phi.shape[1]
    cnt = np.bincount(h, minlength=len(Nh))
    s1 = np.column_stack([np.bincount(h, Phi[pick, c], minlength=len(Nh)) for c in range(q)])
    s2 = np.column_stack([np.bincount(h, Phi[pick, c] ** 2, minlength=len(Nh)) for c in range(q)])
    ok = cnt > 0
    est = ((Nh[ok] / N)[:, None] * (s1[ok] / cnt[ok, None])).sum(0)
    many = cnt > 1
    var = (s2[many] - cnt[many, None] * (s1[many] / cnt[many, None]) ** 2) / (cnt[many, None] - 1)
    v = (((Nh[many] / N) ** 2 * (1 - rh[many] / Nh[many]) / rh[many])[:, None] * var).sum(0)
    return est, v


def _cond_frame(N, k, B, delta, design, seed, draws):
    X, _ = anisotropic_gmm(N, random_state=seed)
    Phi = _phis(X); PN = Phi.mean(0)
    qs = PCAQS(n_components=k, n_bins=B, retention=delta, standardize=False,
               stratification=design).fit(X)
    uniq, inv, Nh = np.unique(qs.strata(X), return_inverse=True, return_counts=True)
    rh = floor_alloc(Nh, delta)
    starts = np.r_[0, np.cumsum(Nh)[:-1]]
    rng = np.random.default_rng(seed + 11)
    cover = np.zeros(Phi.shape[1]); z = norm.ppf(0.975)
    for _ in range(draws):
        order = np.argsort(inv + rng.random(N), kind="stable")
        within = np.arange(N) - starts[inv[order]]
        pick = order[within < rh[inv[order]]]
        est, v = _weighted_estimate(Phi, inv, Nh, rh, pick, N)
        cover += np.abs(est - PN) <= z * np.sqrt(v)
    return cover / draws


def _uncond_frame(N, k, B, delta, design, seed, Ephi):
    X, _ = anisotropic_gmm(N, random_state=seed)
    Phi = _phis(X)
    qs = PCAQS(n_components=k, n_bins=B, retention=delta, standardize=False,
               random_state=seed + 3, stratification=design).fit(X)
    pick = qs.sample_indices(X, allocation="floor")
    uniq, inv, Nh = np.unique(qs.strata(X), return_inverse=True, return_counts=True)
    rh = floor_alloc(Nh, delta)
    est, v = _weighted_estimate(Phi, inv, Nh, rh, pick, N)
    v_pop = v + Phi[pick].var(0, ddof=1) / N
    return (np.abs(est - Ephi) <= norm.ppf(0.975) * np.sqrt(v_pop)).astype(float)


def study_clt(k=3, B=5, N=40000, deltas=(0.02, 0.05, 0.20), frames=10, draws=1000,
              uframes=1000, jobs=-1):
    rows = []
    for design in DESIGNS:
        orc = _oracle(k, B, design)
        for delta in deltas:
            c = np.array(Parallel(n_jobs=jobs)(
                delayed(_cond_frame)(N, k, B, delta, design, 5000 + i, draws) for i in range(frames))).mean(0)
            u = np.array(Parallel(n_jobs=jobs)(
                delayed(_uncond_frame)(N, k, B, delta, design, 90000 + i, orc["Ephi"])
                for i in range(uframes))).mean(0)
            for j, name in enumerate(PHI_NAMES):
                rows.append(dict(design=design, delta=delta, phi=name, cover_conditional=c[j],
                                 cover_unconditional=u[j], n_cond=frames * draws, n_uncond=uframes))
            print(design, delta, np.round(c, 3), np.round(u, 3), flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_clt.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- directions
def _haar(d, k, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, k)))
    return Q * np.sign(np.diag(R))


def _avg_ratio_population(X, W, B, design):
    """tr(sum_h pi_h Cov_h(X)) / tr(Cov X): idealized average design effect along W."""
    sc = X @ W
    key = strata_keys(sc, percentile_cuts(sc, B), design)
    Nh, _, V = within_cell_moments(key, X)
    return float(((Nh / len(X))[:, None] * V).sum() / X.var(0).sum())


def study_directions(d=20, k=3, N=2_000_000, n_random=30, seed=3):
    rng = np.random.default_rng(seed)
    lam = 10 * 0.7 ** np.arange(d)
    Q = _haar(d, d, rng)                                     # random rotation: axes != PCs
    Sigma = Q @ np.diag(lam) @ Q.T
    X = rng.standard_normal((N, d)) * np.sqrt(lam) @ Q.T
    rho_k = lam[k:].sum() / lam.sum()

    def limit(W):
        M = Sigma @ W @ np.linalg.solve(W.T @ Sigma @ W, W.T @ Sigma)
        return 1 - np.trace(M) / np.trace(Sigma)

    Vk = Q[:, :k]
    E = np.eye(d)[:, np.argsort(-np.diag(Sigma))[:k]]
    Ws = [_haar(d, k, rng) for _ in range(n_random)]
    rows = []
    for design, B in (("grid", 16), ("profile", 5), ("grid", 5)):
        rows.append(dict(design=design, B=B, directions="PC", empirical=_avg_ratio_population(X, Vk, B, design), limit=limit(Vk)))
        rows.append(dict(design=design, B=B, directions="top-variance axes", empirical=_avg_ratio_population(X, E, B, design), limit=limit(E)))
        for W in Ws:
            rows.append(dict(design=design, B=B, directions="random", empirical=_avg_ratio_population(X, W, B, design), limit=limit(W)))
    for BB in (2, 4, 8, 16, 32):
        for design in DESIGNS:
            rows.append(dict(design=design, B=BB, directions="PC (sweep)",
                             empirical=_avg_ratio_population(X, Vk, BB, design), limit=limit(Vk)))
    df = pd.DataFrame(rows); df["rho_k"] = rho_k
    df.to_csv(os.path.join(FIG, "theory_directions.csv"), index=False)
    print(f"[directions] rho_k = {rho_k:.4f}")
    print(df.groupby(["design", "B", "directions"])[["empirical", "limit"]].agg(["mean", "min", "max"]).round(4))
    return df



# ----------------------------------------------------------------------------- profile limits
def study_profile(N=2_000_000, seed=11):
    """Profile vs grid: single-component targets (independent scores; limit 1 - 1/k), Gaussian
    linear targets averaged over directions (limit 1 - rho_k / k vs 1 - rho_k), and a
    symmetric target (squared radius; profile = grid)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in (1, 2, 3, 5):
        U = rng.random((N, k)); g = U[:, 0]
        for B in (5, 16, 32):
            cuts = np.tile((np.arange(1, B) / B)[:, None], (1, k))
            for design in DESIGNS:
                if design == "grid" and B ** k > 50_000:
                    continue
                key = strata_keys(U, cuts, design)
                Nh, _, V = within_cell_moments(key, g[:, None])
                rows.append(dict(study="single-component", k=k, B=B, design=design, strata=len(Nh),
                                 efficiency=float((Nh / N * V[:, 0]).sum() / g.var()),
                                 limit=(1 - 1 / k) if design == "profile" else 0.0))
    d = 20
    lam = 10 * 0.7 ** np.arange(d); Q = _haar(d, d, rng)
    X = rng.standard_normal((N, d)) * np.sqrt(lam) @ Q.T
    for k in (1, 2, 3, 5):
        rho = lam[:k].sum() / lam.sum()
        for B in (5, 16, 32):
            for design in DESIGNS:
                if design == "grid" and B ** k > 50_000:
                    continue
                rows.append(dict(study="gaussian-linear-average", k=k, B=B, design=design,
                                 efficiency=_avg_ratio_population(X, Q[:, :k], B, design),
                                 limit=(1 - rho / k) if design == "profile" else (1 - rho)))
        if k > 1:
            Y = X @ Q[:, :k]; g = ((Y / np.sqrt(lam[:k])) ** 2).sum(1)
            for B in (5, 16):
                cuts = percentile_cuts(Y, B)
                for design in DESIGNS:
                    key = strata_keys(Y, cuts, design)
                    Nh, _, V = within_cell_moments(key, g[:, None])
                    rows.append(dict(study="symmetric-radius", k=k, B=B, design=design,
                                     efficiency=float((Nh / N * V[:, 0]).sum() / g.var()), limit=np.nan))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_profile.csv"), index=False)
    print(df.round(4).to_string(index=False))
    return df


# ----------------------------------------------------------------------------- retention
def study_retention(N=10_000, d=500, delta=0.05, rho=0.2, seed=12):
    """Realized retention of the companion allocation max(1, floor(delta N_g)) as the number of
    profile strata grows with k (equicorrelated Gaussian, as in the companion paper), against
    the bounds of the retention proposition."""
    rng = np.random.default_rng(seed)
    X = np.sqrt(1 - rho) * rng.standard_normal((N, d)) + np.sqrt(rho) * rng.standard_normal((N, 1))
    X = (X - X.mean(0)) / X.std(0)
    V, _ = pc_directions(X, 300)
    rows = []
    for k in (1, 2, 5, 10, 20, 50, 100, 267):
        sc = X @ V[:, :k]
        for design in ("profile",):
            key = strata_keys(sc, percentile_cuts(sc, 5), design)
            Nh = np.unique(key, return_counts=True)[1]
            rh = floor_alloc(Nh, delta)
            H = len(Nh)
            rows.append(dict(k=k, design=design, H_N=H, singletons=int((Nh == 1).sum()),
                             realized_retention=rh.sum() / N, lower=max(delta - H / N, H / N),
                             upper=min(1.0, delta + H / N)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_retention.csv"), index=False)
    print(df.round(4).to_string(index=False))
    return df


# ----------------------------------------------------------------------------- outliers (counterexample)
def _trace_deff_and_bound(X, V, B, design, eps, zeta_norm, trZ):
    """Average design effect over isotropic linear targets and the contamination lower bound."""
    sc = X @ V
    key = strata_keys(sc, percentile_cuts(sc, B), design)
    Nh, _, Vv = within_cell_moments(key, X)
    deff = float(((Nh / len(X))[:, None] * Vv).sum() / X.var(0).sum())
    return deff


def study_outliers(N=400_000, d=10, k=3, seed=13):
    """Rare extreme points break the design-effect predictions (Proposition on contamination):
    synthetic contamination sweep against the lower bound, and EEG with/without its extreme rows."""
    rng = np.random.default_rng(seed)
    lam = 4.0 * 0.6 ** np.arange(d); trZ = lam.sum()
    Z = rng.standard_normal((N, d)) * np.sqrt(lam)
    direction = np.ones(d) / np.sqrt(d)
    rows = []
    for eps in (0.0005, 0.002):
        n_out = int(round(eps * N))
        for kappa2 in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0):
            zeta = direction * np.sqrt(kappa2 * trZ / eps) if kappa2 > 0 else direction * 0.0
            X = Z.copy(); X[:n_out] += zeta
            V, lamX = pc_directions(X, k)
            rhoX = lamX[:k].sum() / lamX.sum()
            for design, B in (("profile", 5), ("grid", 5)):
                sc = X @ V
                key = strata_keys(sc, percentile_cuts(sc, B), design)
                Nh, _, Vv = within_cell_moments(key, X)
                deff = float(((Nh / N)[:, None] * Vv).sum() / X.var(0).sum())
                # mass of the smallest stratum that contains an outlier
                uniq, inv = np.unique(key, return_inverse=True)
                pi_star = float(Nh[np.unique(inv[:n_out])].min() / N) if n_out else 1.0
                zn = np.linalg.norm(zeta); e = n_out / N
                eta = e * zn / np.sqrt(trZ * pi_star)
                bound = 1 - (1 + eta) ** 2 / (1 + (e * (1 - e) * zn ** 2) / trZ) if kappa2 > 0 else np.nan
                rows.append(dict(study="synthetic", eps=e, kappa2=kappa2, design=design, B=B, k=k,
                                 deff=deff, lower_bound=bound,
                                 gaussian_limit=(1 - rhoX / k) if design == "profile" else (1 - rhoX)))
    X = load_real("EEG")
    ss = ((X - X.mean(0)) ** 2).sum(1)
    top = np.argsort(-ss)[:4]
    share = float(ss[top].sum() / ss.sum())
    keep = np.ones(len(X), bool); keep[top] = False
    Xc = X[keep]; Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    V, lamX = pc_directions(X, k); rhoX = lamX[:k].sum() / lamX.sum()
    for design in DESIGNS:                                   # certainty stratum for the extreme rows
        B = 5 if design == "profile" else grid_bins(0.05 * len(X), k)
        sc = X @ V
        key = strata_keys(sc, percentile_cuts(sc, B), design).copy(); key[top] = -1
        Nh, _, S2 = within_cell_moments(key, X)
        rh = floor_alloc(Nh, 0.05); rh[0] = Nh[0]              # key -1 sorts first: take all
        r = int(rh.sum()); NN = len(X)
        vq = (((Nh / NN) ** 2 * (1 - rh / Nh))[:, None] * S2 / rh[:, None]).sum()
        vs = (1 - r / NN) * X.var(0, ddof=1).sum() / r
        rows.append(dict(study="EEG, 4 extreme rows in a certainty stratum", eps=4 / len(X), kappa2=np.nan,
                         design=design, B=B, k=k, deff=float(vq / vs), lower_bound=np.nan,
                         variance_share_4rows=share,
                         gaussian_limit=(1 - rhoX / k) if design == "profile" else (1 - rhoX)))
    for label, XX in (("EEG, all rows", X), ("EEG, 4 extreme rows removed", Xc)):
        V, lamX = pc_directions(XX, k); rhoX = lamX[:k].sum() / lamX.sum()
        for design in DESIGNS:
            B = 5 if design == "profile" else grid_bins(0.05 * len(XX), k)
            deff, H, r = _avg_design_ratio(XX, V, B, design, 0.05)
            rows.append(dict(study=label, eps=4 / len(X), kappa2=np.nan, design=design, B=B, k=k,
                             deff=deff, lower_bound=np.nan, variance_share_4rows=share,
                             gaussian_limit=(1 - rhoX / k) if design == "profile" else (1 - rhoX)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_outliers.csv"), index=False)
    print(df.round(4).to_string(index=False))
    return df


# ----------------------------------------------------------------------------- occupancy diagnostics
def study_occupancy(deltas=(0.05, 0.01)):
    """Pre-sampling diagnostics of the profile design on the real datasets: occupied strata H_N,
    H_N/(delta N), the unequal-weighting factor kappa_w of Proposition (retention)(c) and the
    realized retention, for k = min(k*, 10) (companion rule) and k = 10."""
    from math import comb
    rows = []
    for name in REAL:
        X = load_real(name); N, d = X.shape
        V, lam = pc_directions(X, min(d, 10))
        kstar = int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.70) + 1)   # lam: all eigenvalues
        for k in sorted({min(kstar, 10), 10}):
            sc = X @ V[:, :k]
            Nh = np.unique(strata_keys(sc, percentile_cuts(sc, 5), "profile"), return_counts=True)[1]
            for delta in deltas:
                rh = floor_alloc(Nh, delta); r = int(rh.sum())
                rows.append(dict(dataset=name, N=N, d=d, k_star=kstar, k=k, M=comb(k + 4, 4), H_N=len(Nh),
                                 delta=delta, delta_N=delta * N, H_over_deltaN=len(Nh) / (delta * N),
                                 kappa_w=r * np.sum(Nh ** 2 / rh) / N ** 2, realized_retention=r / N))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_occupancy.csv"), index=False)
    print(df.round(3).to_string(index=False))
    return df

# ----------------------------------------------------------------------------- real data
_PROJ = os.path.dirname(os.path.dirname(_CR))
DATA_ROOT = os.environ.get("PCAQS_DATA_ROOT",
                           os.path.join(_PROJ, "LaTeX", "Final Version", "Real Data Marix Comparions csv"))
REAL = {   # name -> (glob, header, drop-by-name-substring, drop-by-index)
    "CreditCard": ("**/CreditCard/UCI_Credit_Card.csv", "infer", ["ID", "default"], []),
    "MAGIC": ("**/magic+gamma+telescope/magic04.csv", None, [], []),
    "EEG": ("**/EEG Eye State/EEG_Eye_State.csv", None, [], []),           # local file has no label column
    "Epileptic": ("**/Epileptic Seizure Recognition/Epileptic_Seizure_Recognition.csv", "infer", ["Unnamed", "y"], []),
    "HIGGS": ("**/Higgs/HIGGS-2.csv", None, [], []),                       # local file has no label column
    "YearPrediction": ("**/YearPredictionMSD/YearPredictionMSD.csv", None, [], [0]),
}


def load_real(name):
    """Numeric features, complete rows, non-constant columns, standardized."""
    pat, header, drop_subs, drop_idx = REAL[name]
    path = sorted(glob.glob(os.path.join(DATA_ROOT, pat), recursive=True))[0]
    df = pd.read_csv(path, header=header)
    if drop_idx:
        df = df.drop(columns=df.columns[drop_idx])
    df = df[[c for c in df.columns if not any(s in str(c) for s in drop_subs)]]
    X = df.select_dtypes(include=[np.number]).to_numpy(float)
    X = X[np.isfinite(X).all(1)]
    X = X[:, X.std(0) > 0]
    return (X - X.mean(0)) / X.std(0)


def grid_bins(r, k, cell_target=10):
    return max(2, int(np.floor((r / cell_target) ** (1.0 / k))))


def _avg_design_ratio(X, W, B, design, delta):
    sc = X @ W
    key = strata_keys(sc, percentile_cuts(sc, B), design)
    v_qs, v_srs, rh, Nh = design_variances(key, X, delta)
    return float(v_qs.sum() / v_srs.sum()), int(len(Nh)), int(rh.sum())


def study_realdirs(k=3, delta=0.05, n_random=20, seed=5):
    rows = []
    for name in REAL:
        X = load_real(name); N, d = X.shape
        V, lam = pc_directions(X, k)
        rho_k = lam[k:].sum() / lam.sum()
        for design in DESIGNS:
            B = 5 if design == "profile" else grid_bins(delta * N, k)
            rng = np.random.default_rng(seed)
            pc, H, r = _avg_design_ratio(X, V, B, design, delta)
            rnd = [_avg_design_ratio(X, _haar(d, k, rng), B, design, delta)[0] for _ in range(n_random)]
            axes = [_avg_design_ratio(X, np.eye(d)[:, rng.choice(d, k, replace=False)], B, design, delta)[0]
                    for _ in range(n_random)]
            rows.append(dict(dataset=name, design=design, N=N, d=d, k=k, B=B, r=r, H_N=H, rho_k=rho_k,
                             PC=pc, random_mean=np.mean(rnd), random_min=np.min(rnd),
                             axes_mean=np.mean(axes), axes_min=np.min(axes)))
            print({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(FIG, "theory_realdirs.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- figures
def make_figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6)); ax = axes.ravel()
    df = pd.read_csv(os.path.join(FIG, "theory_rate.csv"))
    sty = {("uniform", "grid"): ("o-", "tab:blue"), ("gaussian", "grid"): ("s-", "tab:orange"),
           ("uniform", "profile"): ("o--", "tab:blue"), ("gaussian", "profile"): ("s--", "tab:orange")}
    for (dist, design), g in df.groupby(["dist", "design"]):
        mk, col = sty[(dist, design)]
        ax[0].loglog(g.m, g.excess, mk, color=col, ms=4, label=f"{dist} scores, {design}")
    g = df[(df.dist == "uniform") & (df.design == "grid")]
    ax[0].loglog(g.m, g.bound, "k:", lw=1, label="grid bound, Theorem (b)")
    ax[0].set_xlabel("bins per component $m$"); ax[0].set_ylabel(r"excess $\sum_\ell\pi_\ell\,\mathrm{Var}(\psi\mid S_\ell)$")
    ax[0].set_title("(a) excess variance: grid vanishes, profile plateaus", fontsize=10); ax[0].legend(fontsize=7)

    de = pd.read_csv(os.path.join(FIG, "theory_estimated.csv"))
    agg = de.groupby(["design", "N"]).mean_abs_dev.max().reset_index()
    for design, mk in (("profile", "o-"), ("grid", "s--")):
        g = agg[agg.design == design]
        ax[1].loglog(g.N, g.mean_abs_dev, mk, label=f"{design} (worst of five functionals)")
    Ns = np.array(sorted(de.N.unique()))
    ax[1].loglog(Ns, agg.mean_abs_dev.max() * (Ns / Ns[0]) ** -0.5, "k:", lw=1, label=r"slope $-1/2$")
    ax[1].set_xlabel("frame size $N$"); ax[1].set_ylabel("mean |realized ratio $-$ population ratio|")
    ax[1].set_title("(b) estimated strata", fontsize=10); ax[1].legend(fontsize=7.5)

    dd = pd.read_csv(os.path.join(FIG, "theory_directions.csv"))
    main = dd[(dd.design == "grid") & (dd.B == 16) & (dd.directions != "PC (sweep)")]
    for des, mk, col in (("random", "o", "tab:gray"), ("top-variance axes", "^", "tab:orange"), ("PC", "*", "tab:blue")):
        g = main[main.directions == des]
        ax[2].scatter(g.limit, g.empirical, marker=mk, color=col, s=80 if des == "PC" else 25, label=des, zorder=3)
    lo, hi = main[["limit", "empirical"]].min().min(), main[["limit", "empirical"]].max().max()
    ax[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[2].axvline(dd.rho_k.iloc[0], color="tab:blue", lw=0.8, ls=":")
    ax[2].text(dd.rho_k.iloc[0], hi, r" $\varrho_3$", color="tab:blue", va="top", fontsize=9)
    ax[2].set_xlabel(r"Gaussian limit $1-\mathrm{tr}(\Pi_W\Sigma)/\mathrm{tr}\,\Sigma$")
    ax[2].set_ylabel("average design effect (grid, $m=16$)")
    ax[2].set_title("(c) stratifying directions (grid)", fontsize=10); ax[2].legend(fontsize=7.5)

    dp = pd.read_csv(os.path.join(FIG, "theory_profile.csv"))
    big = dp[(dp.design == "profile") & (dp.B == 16)]
    for study, mk, col, lab in (("single-component", "o", "tab:purple", "single-component target: $1-1/k$"),
                                ("gaussian-linear-average", "s", "tab:green", "Gaussian linear targets: $1-\\rho_k/k$")):
        g = big[big.study == study]
        ax[3].scatter(g.limit, g.efficiency, marker=mk, color=col, s=45, label=lab, zorder=3)
        for _, r in g.iterrows():
            ax[3].annotate(f"k={int(r.k)}", (r.limit, r.efficiency), fontsize=7, xytext=(4, -9), textcoords="offset points")
    # k = 5 excluded: the m = 16 grid has 16^5 cells for 2e6 points (cells too sparse to estimate)
    rad = dp[(dp.study == "symmetric-radius") & (dp.B == 16) & (dp.k <= 3)].pivot_table(index="k", columns="design", values="efficiency")
    ax[3].scatter(rad["grid"], rad["profile"], marker="D", color="tab:red", s=40,
                  label="symmetric target: profile vs grid", zorder=3)
    ax[3].plot([0, 1], [0, 1], "k--", lw=1)
    ax[3].set_xlim(-0.03, 1.0); ax[3].set_ylim(-0.03, 1.0)
    ax[3].set_xlabel("theoretical limit (or grid efficiency for the symmetric target)")
    ax[3].set_ylabel("profile design effect ($m=16$)")
    ax[3].set_title("(d) what profiles preserve", fontsize=10); ax[3].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for d in (FIG, MAN, os.path.join(os.path.dirname(_CR), "manuscript_ejs")):
        if os.path.isdir(d):
            fig.savefig(os.path.join(d, "theory_validation.png"), dpi=160)
    print("wrote theory_validation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="*", default=["rate", "estimated", "clt", "directions", "realdirs", "profile", "retention", "figures"])
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    for s in a.studies:
        print(f"=== {s}", flush=True)
        if s == "rate": study_rate()
        elif s == "estimated": study_estimated(jobs=a.jobs)
        elif s == "clt": study_clt(jobs=a.jobs)
        elif s == "directions": study_directions()
        elif s == "realdirs": study_realdirs()
        elif s == "profile": study_profile()
        elif s == "retention": study_retention()
        elif s == "outliers": study_outliers()
        elif s == "occupancy": study_occupancy()
        elif s == "figures": make_figures()
