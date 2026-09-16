"""Design-parameter rules for PCA-QS (Section 2.5 of the paper).

The same rule is used by every experiment:

* k: the requested k, or the smallest k whose leading components explain `var_target` of
  the variance, capped at ``min(k_max, floor(log2(r / cell_target)))`` so that cells stay
  populated (pass ``cap=False`` only to illustrate what happens without the cap);
* m: ``floor((r / cell_target) ** (1 / k))`` (at least 2), i.e. about `cell_target`
  retained points per cell;
* allocation: largest remainder with every occupied cell floored at one retained point
  (``min_per_cell="auto"`` in :meth:`PCAQS.sample_indices`), as Proposition 1 requires.
"""
from __future__ import annotations
import numpy as np

CELL_TARGET = 10
K_MAX = 10


N_BINS_PROFILE = 5


def choose_design(r, explained_ratio=None, k=None, var_target=0.70,
                  cell_target=CELL_TARGET, k_max=K_MAX, cap=True, stratification="grid"):
    """Return (k, m, k_requested) for retained size r.

    ``stratification="profile"`` follows the companion paper: B = 5 quintile cutoffs and
    k = min(k_requested, k_max) (profiles need no occupancy cap: at most C(k+4, 4) strata).
    ``stratification="grid"`` caps k at floor(log2(r / cell_target)) and sets
    m = floor((r / cell_target)^(1/k)).
    """
    if k is None:
        cum = np.cumsum(np.asarray(explained_ratio, dtype=float))
        k = int(np.searchsorted(cum, var_target) + 1)
    k_req = int(k)
    if explained_ratio is not None:
        k = min(k, len(explained_ratio))
    if stratification == "profile":
        return (int(min(k, k_max)) if cap else int(k)), N_BINS_PROFILE, k_req
    if cap:
        k_cap = max(1, min(k_max, int(np.floor(np.log2(max(r / cell_target, 2.0))))))
        k = min(k, k_cap)
    m = max(2, int(np.floor((r / cell_target) ** (1.0 / k))))
    return int(k), int(m), k_req
