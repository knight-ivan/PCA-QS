"""Synthetic data generators used in the paper.

`structure_fidelity_gmm` is the generator for the distance-metric / rate studies;
`classification_gmm` is the separate generator for the downstream-classification
study.  Both match the descriptions in the manuscript appendix.
"""
from __future__ import annotations
import numpy as np


def structure_fidelity_gmm(n, d=50, nonlinear=False, random_state=None):
    """Two-component Gaussian mixture (pi = 0.9 / 0.1).

    Class 0 ~ N(0, I_d);  Class 1 ~ N(0.5 * 1_d, 1.2 I_d).
    If `nonlinear`, append all pairwise products and elementwise sines
    (d=50 -> 1325 features).  Returns (X, y).
    """
    rng = np.random.default_rng(random_state)
    y = (rng.random(n) < 0.1).astype(int)
    X = rng.standard_normal((n, d))
    m1 = 0.5 * np.ones(d)
    idx1 = y == 1
    X[idx1] = m1 + np.sqrt(1.2) * rng.standard_normal((idx1.sum(), d))
    if nonlinear:
        iu, ju = np.triu_indices(d, k=1)
        inter = X[:, iu] * X[:, ju]
        X = np.hstack([X, inter, np.sin(X)])
    return X, y


# --- Anisotropic generator with DISTINCT leading eigenvalues (assumption A2 holds) ---
# The near-isotropic structure_fidelity_gmm has population covariance 1.02*I + 0.0225*11',
# whose eigenvalues are lambda_1=2.145 and lambda_2=...=lambda_50=1.02 (tied), so A2
# (distinct leading eigenvalues) fails for every k>=2.  This generator instead uses an
# anisotropic diagonal within-class covariance, giving population eigenvalues
# ~[4.32, 3.18, 2.12, 1.58, 1.25, 1.02, ...] -- distinct and above the tail for k<=5.
# It is the generator for the Theorem-1 rate / KL validation, where A2 is invoked.
ANISO_D = 50
ANISO_EIGS = np.array([4.0, 3.0, 2.0, 1.5, 1.2] + [1.0] * (ANISO_D - 5))  # within-class variances
ANISO_M1 = np.concatenate([[1.5, 1.2, 1.0, 0.8, 0.6], np.zeros(ANISO_D - 5)])  # class-1 mean shift
ANISO_VAR1 = 1.2       # class-1 covariance = ANISO_VAR1 * diag(ANISO_EIGS)
ANISO_W = np.array([0.9, 0.1])


def anisotropic_gmm(n, random_state=None):
    """Two-component Gaussian mixture with anisotropic, DISTINCT leading eigenvalues.

    Class 0 ~ N(0, D); Class 1 ~ N(ANISO_M1, ANISO_VAR1 * D), D = diag(ANISO_EIGS),
    weights 0.9 / 0.1.  Population covariance 1.02*D + 0.09*ANISO_M1 ANISO_M1', whose
    leading eigenvalues are distinct and exceed the tail (A2 holds for k<=5), unlike the
    near-isotropic `structure_fidelity_gmm`.  Returns (X, y).
    """
    rng = np.random.default_rng(random_state)
    y = (rng.random(n) < ANISO_W[1]).astype(int)
    sd0 = np.sqrt(ANISO_EIGS)
    X = sd0 * rng.standard_normal((n, ANISO_D))                      # class 0 ~ N(0, D)
    idx1 = y == 1
    X[idx1] = ANISO_M1 + np.sqrt(ANISO_VAR1) * sd0 * rng.standard_normal((int(idx1.sum()), ANISO_D))
    return X, y


def classification_gmm(n, random_state=None):
    """30-feature imbalanced (90:10) classification generator.

    20 informative features (mean-separated by ||mu1-mu0|| = 1.5*sqrt(20)),
    5 redundant (random linear combinations of informative), 2 repeated, 3 noise.
    Returns (X, y) with 30 columns.
    """
    rng = np.random.default_rng(random_state)
    y = (rng.random(n) < 0.1).astype(int)
    n_inf = 20
    sep = 1.5 * np.sqrt(n_inf)
    direction = rng.standard_normal(n_inf)
    direction /= np.linalg.norm(direction)
    mu0 = np.zeros(n_inf)
    mu1 = mu0 + sep * direction
    Xinf = rng.standard_normal((n, n_inf)) + np.where(y[:, None] == 1, mu1, mu0)
    W = rng.standard_normal((n_inf, 5))                 # 5 redundant
    Xred = Xinf @ W
    Xrep = Xinf[:, :2]                                  # 2 repeated
    Xnoise = rng.standard_normal((n, 3))               # 3 noise
    X = np.hstack([Xinf, Xred, Xrep, Xnoise])          # 30 columns
    return X, y
