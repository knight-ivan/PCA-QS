"""Core PCA-Guided Quantile Sampling (PCA-QS) sampler.

PCA-QS keeps the original feature space and uses the leading principal components
only to *guide* a quantile-based stratification: each of the top-k PC scores is
split into m quantile bins, the composite (product) bin is the stratum, and points
are drawn from each stratum under proportional allocation.  See Foo & Chang,
"PCA-Guided Quantile Sampling".
"""
from __future__ import annotations
import numpy as np


class PCAQS:
    """PCA-Guided Quantile Sampling.

    Parameters
    ----------
    n_components : int
        Number of leading principal components used to guide stratification (k).
    n_bins : int
        Number of quantile bins per component (m).
    retention : float
        Fraction of points to retain within each stratum (delta), in (0, 1].
    standardize : bool
        Center/scale features to zero mean and unit variance before PCA.
    random_state : int | None
        Seed for the within-stratum random draws.

    Notes
    -----
    The retained subset lives in the *original* feature space; PCA is used only to
    define the strata.  Sampling within a stratum is uniform by default.
    """

    def __init__(self, n_components=5, n_bins=10, retention=0.05,
                 standardize=True, random_state=None):
        self.n_components = int(n_components)
        self.n_bins = int(n_bins)
        self.retention = float(retention)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    # -- fit -----------------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.standardize:
            self.mean_ = X.mean(0)
            self.scale_ = X.std(0) + 1e-12
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        Xs = (X - self.mean_) / self.scale_
        # top-k right singular vectors of the centered, scaled data
        _, sv, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
        self.components_ = Vt[: self.n_components].T          # (p, k)
        self.singular_values_ = sv[: self.n_components]
        return self

    # -- helpers -------------------------------------------------------------
    def scores(self, X):
        """Return the top-k PC scores of X under the fitted transform."""
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Xs @ self.components_

    def strata(self, X):
        """Composite quantile-stratum id for each row of X (0 .. m^k-1, occupied)."""
        sc = self.scores(X)
        n, k = sc.shape
        key = np.zeros(n, dtype=np.int64)
        self.bin_edges_ = []
        for j in range(k):
            edges = np.quantile(sc[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1])
            self.bin_edges_.append(edges)
            key = key * self.n_bins + np.digitize(sc[:, j], edges)
        return key

    # -- sample --------------------------------------------------------------
    def sample_indices(self, X, exact_size=None, min_per_cell=0):
        """Indices of the retained subset (fully vectorised; no per-cell loop).

        With ``exact_size=None`` (default) uses the per-cell ceiling rule
        ``ceil(retention * N_h)``; the realized size then satisfies
        ``delta*N <= R < delta*N + H_N`` (it can overshoot the nominal target).

        With ``exact_size=r`` uses a largest-remainder (Hamilton) allocation so the
        returned subset has size exactly ``r`` (requires ``r <= N``). Set
        ``min_per_cell=1`` to floor every occupied cell to a positive retained size, so
        that every inclusion probability is positive and the weighted estimator is
        design-unbiased (requires ``r >= H_N``). The default ``min_per_cell=0`` is plain
        proportional allocation, under which the smallest cells may receive 0.
        """
        key = self.strata(X)
        uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        H, n = len(counts), len(key)
        self.n_cells_ = H                               # H_N

        if exact_size is None:
            rh = np.minimum(np.ceil(self.retention * counts).astype(np.int64), counts)
        else:
            rh = self._largest_remainder(counts, int(exact_size), min_per_cell=min_per_cell)

        # pick rh[g] uniformly at random within each cell g, without a Python loop:
        # order points group-major with a random tiebreak, then keep the first rh per group.
        u = self._rng.random(n)
        order = np.lexsort((u, inv))
        grp = inv[order]
        start = np.zeros(H, dtype=np.int64)
        start[1:] = np.cumsum(counts)[:-1]              # first position of each group in `order`
        within_rank = np.arange(n) - start[grp]
        keep = order[within_rank < rh[grp]]
        return np.sort(keep)

    @staticmethod
    def _largest_remainder(counts, r, min_per_cell=0):
        """Hamilton allocation of exactly r units proportional to counts (r_h <= N_h).

        With ``min_per_cell >= 1`` every occupied cell is floored to at least that many
        retained units, giving each a positive inclusion probability so the weighted
        estimator (``design_weights``) is design-unbiased (see the paper's design-based
        theory); this requires ``r >= min_per_cell * H_N``. The default ``min_per_cell=0``
        is plain proportional allocation, under which the smallest cells may receive 0.

        Returns an int array aligned with `counts`.
        """
        counts = np.asarray(counts, dtype=np.int64)
        N = int(counts.sum()); H = len(counts)
        if r > N:
            raise ValueError(f"exact_size={r} exceeds N={N}")
        ideal = r * counts / N
        base = np.minimum(np.floor(ideal).astype(np.int64), counts)
        rem = int(r - base.sum())
        if rem > 0:                                     # give +1 to largest remainders with capacity
            frac = ideal - np.floor(ideal)
            order = np.argsort(-frac)
            elig = order[base[order] < counts]
            base[elig[:rem]] += 1
        floor = int(min_per_cell)
        if floor > 0 and (base < floor).any():
            if r < floor * H:
                raise ValueError(
                    f"exact_size={r} < min_per_cell({floor}) * H_N({H}): the partition is "
                    f"too fine to floor every occupied cell; reduce k/m or increase r.")
            base = np.minimum(np.maximum(base, floor), counts)   # raise deficient cells to floor
            excess = int(base.sum() - r)                          # reclaim so the sum stays == r
            if excess > 0:
                cap = base - floor                                # each cell can give down to floor
                for idx in np.argsort(-cap):                      # take from the largest surplus first
                    if excess <= 0:
                        break
                    take = min(int(cap[idx]), excess)
                    base[idx] -= take; excess -= take
        return base

    def design_weights(self, X, idx):
        """Design weights N_h/(N * r_h) for retained points `idx` (sum to 1).

        Use these for weighted, design-unbiased estimates of full-data functionals
        (see the design-based theory in the paper).
        """
        key = self.strata(X)
        N = len(key)
        ret_labels, ret_counts = np.unique(key[idx], return_counts=True)
        rh = dict(zip(ret_labels, ret_counts))
        Nh = dict(zip(*np.unique(key, return_counts=True)))
        w = np.array([(Nh[key[i]] / N) / rh[key[i]] for i in idx])
        return w / w.sum()

    def fit_sample_indices(self, X, exact_size=None):
        return self.fit(X).sample_indices(X, exact_size=exact_size)

    def fit_sample(self, X, exact_size=None):
        idx = self.fit_sample_indices(X, exact_size=exact_size)
        return np.asarray(X)[idx], idx


def srs_indices(n, size, random_state=None):
    """Simple random sampling: indices of `size` rows out of `n`, without replacement."""
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=size, replace=False))
