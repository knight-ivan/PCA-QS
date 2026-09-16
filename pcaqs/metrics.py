"""Distribution-discrepancy metrics used to score a subsample against reference data.

All functions take two point clouds A, B (rows = points) and return a scalar; lower
means the two distributions are closer.  These are the metrics reported in the
paper plus the exact 2-Wasserstein used for the geometric-rate check.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def quantile_error(A, B, ps=None):
    """Mean absolute marginal-quantile difference (theory: O(n^{-1/2}))."""
    if ps is None:
        ps = np.linspace(0.05, 0.95, 19)
    return float(np.mean([np.abs(np.quantile(A[:, j], ps) - np.quantile(B[:, j], ps)).mean()
                          for j in range(A.shape[1])]))


def sliced_w2(A, B, n_dir=200, n_grid=200, random_state=None):
    """Sliced 2-Wasserstein: mean over random 1-D projections (cheap surrogate)."""
    rng = np.random.default_rng(random_state)
    d = A.shape[1]
    u = np.linspace(0, 1, n_grid)
    dirs = rng.standard_normal((n_dir, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    acc = 0.0
    for w in dirs:
        acc += np.mean((np.quantile(A @ w, u) - np.quantile(B @ w, u)) ** 2)
    return float(np.sqrt(acc / n_dir))


def exact_w2(A, B):
    """Exact 2-Wasserstein between equal-size empirical clouds via optimal assignment.

    Cost is O(n^2) memory / ~O(n^3) time, so keep n small (<= ~1000).
    """
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    C = cdist(A, B, "sqeuclidean")                 # avoids an n x n x d intermediate
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r, c].mean()))


def energy_distance(A, B, max_n=2000, random_state=None):
    """Energy distance E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'|| (subsampled for cost)."""
    rng = np.random.default_rng(random_state)
    def sub(M):
        return M if len(M) <= max_n else M[rng.choice(len(M), max_n, replace=False)]
    A, B = sub(A), sub(B)
    def md(P, Q):
        return cdist(P, Q).mean()                  # O(n^2) memory, not O(n^2 d)
    return float(2 * md(A, B) - md(A, A) - md(B, B))


def mmd_rbf(A, B, gamma=1.0, max_n=2000, random_state=None):
    """Squared MMD with an RBF kernel (subsampled for cost)."""
    rng = np.random.default_rng(random_state)
    def sub(M):
        return M if len(M) <= max_n else M[rng.choice(len(M), max_n, replace=False)]
    A, B = sub(A), sub(B)
    def k(P, Q):
        return np.exp(-gamma * cdist(P, Q, "sqeuclidean")).mean()
    return float(k(A, A) + k(B, B) - 2 * k(A, B))


def mahalanobis_mean(A, B):
    """Mahalanobis distance between the two sample means (cov from A)."""
    cov = np.cov(A, rowvar=False) + 1e-9 * np.eye(A.shape[1])
    diff = A.mean(0) - B.mean(0)
    return float(np.sqrt(diff @ np.linalg.solve(cov, diff)))
