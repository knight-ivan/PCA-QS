"""Core PCA-Guided Quantile Sampling (PCA-QS) sampler.

PCA-QS keeps the original feature space and uses the leading principal components
only to *guide* a quantile-based stratification. Each of the top-k PC scores is cut at
its B-1 interior empirical quantiles, and the strata are either

* ``"profile"`` (default; the implementation of Foo & Chang's Communications in
  Statistics paper): observations sharing the cutoff-count profile
  g_i = (sum_j 1{Z_ij > tau_j1}, ..., sum_j 1{Z_ij > tau_j,B-1}) form one stratum; at most
  C(k+B-1, B-1) strata; or
* ``"grid"``: the full cross-classification of the per-component bins (at most B^k cells),
  the finest PC-quantile stratification, of which every profile stratum is a union; or
* ``"hybrid"``: the bins of the leading ``n_labeled`` = J components together with the
  cutoff-count profile of the remaining k - J (at most B^J * C(k-J+B-1, B-1) strata);
  J = 0 is the profile and J = k (or k - 1) the grid.

Points are drawn by simple random sampling within strata, with either the companion
paper's allocation max(1, floor(delta N_g)) (``allocation="floor"``) or an exact-size
largest-remainder allocation.
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
        Number of quantile bins per component (B; 5 = quintiles in the companion paper).
    stratification : {"profile", "grid", "hybrid"}
        Cutoff-count profiles (default), the full cross-classification, or hybrid strata.
    n_labeled : int
        Number of leading components whose bins are kept as labels (J); used only with
        ``stratification="hybrid"``.
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

    def __init__(self, n_components=5, n_bins=5, retention=0.05,
                 standardize=True, random_state=None, stratification="profile", n_labeled=0):
        if stratification not in ("profile", "grid", "hybrid"):
            raise ValueError("stratification must be 'profile', 'grid' or 'hybrid'")
        if stratification == "hybrid" and not 0 <= int(n_labeled) <= int(n_components):
            raise ValueError("n_labeled must satisfy 0 <= n_labeled <= n_components")
        self.n_labeled = int(n_labeled)
        self.stratification = stratification
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
        # deterministic sign convention (as scikit-learn's svd_flip with
        # u_based_decision=False): the largest-magnitude loading of each component is
        # positive. Profile strata depend on component signs, so this fixes them uniquely.
        signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
        signs[signs == 0] = 1.0
        Vt = Vt * signs[:, None]
        self.components_ = Vt[: self.n_components].T          # (p, k)
        self.singular_values_ = sv[: self.n_components]
        return self

    # -- helpers -------------------------------------------------------------
    def scores(self, X):
        """Return the top-k PC scores of X under the fitted transform."""
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.scale_
        return Xs @ self.components_

    def strata(self, X):
        """Stratum id for each row of X.

        Cutoffs are the empirical (100 b / B)% points of each score (``np.percentile``),
        and an observation is above a cutoff when its score is strictly greater.
        """
        sc = self.scores(X)
        n, k = sc.shape
        B = self.n_bins
        q = np.percentile(sc, np.arange(1, B) * 100.0 / B, axis=0)       # (B-1, k)
        self.bin_edges_ = [q[:, j] for j in range(k)]
        above = sc[:, None, :] > q[None, :, :]                             # (n, B-1, k)
        key = np.zeros(n, dtype=np.int64)
        J = {"profile": 0, "grid": k}.get(self.stratification, self.n_labeled)
        if J > 0:                                                          # labelled bins, 0..B-1
            bins = above[:, :, :J].sum(axis=1)                             # (n, J)
            for j in range(J):
                key = key * B + bins[:, j]
        if J < k:                                                          # profile of the rest
            counts = above[:, :, J:].sum(axis=2)                           # (n, B-1), values 0..k-J
            for b in range(B - 1):
                key = key * (k - J + 1) + counts[:, b]
        return key

    # -- sample --------------------------------------------------------------
    def sample_indices(self, X, exact_size=None, min_per_cell=0, allocation=None):
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

        if allocation == "floor":                       # companion paper: max(1, floor(delta N_g))
            rh = np.minimum(np.maximum(1, np.floor(self.retention * counts).astype(np.int64)), counts)
        elif exact_size is None:
            rh = np.minimum(np.ceil(self.retention * counts).astype(np.int64), counts)
        else:
            if min_per_cell == "auto":                  # floor at one point per cell when feasible
                min_per_cell = 1 if int(exact_size) >= H else 0
            rh = self._largest_remainder(counts, int(exact_size), min_per_cell=min_per_cell)
        self.cell_counts_, self.cell_alloc_ = counts, rh

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
                # remove one unit at a time from the cell that is most over-allocated relative
                # to its proportional share (base - ideal), never going below the floor, so the
                # allocation stays as close to proportional as the floor allows
                import heapq
                heap = [(-(base[i] - ideal[i]), i) for i in range(H) if base[i] > floor]
                heapq.heapify(heap)
                while excess > 0:
                    _, i = heapq.heappop(heap)
                    base[i] -= 1; excess -= 1
                    if base[i] > floor:
                        heapq.heappush(heap, (-(base[i] - ideal[i]), i))
        assert base.sum() == r
        return base

    def diagnostics(self):
        """Occupancy diagnostics of the last draw: occupied cells H_N, singleton cells,
        cells left unrepresented (r_h = 0) and their share of the frame, the smallest
        positive allocation, and the realized retention."""
        c, a = self.cell_counts_, self.cell_alloc_
        return dict(H_N=int(len(c)), singletons=int((c == 1).sum()),
                    empty_cells=int((a == 0).sum()),
                    unrepresented_share=float(c[a == 0].sum() / c.sum()),
                    min_alloc=int(a[a > 0].min()), realized_retention=float(a.sum() / c.sum()))

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
