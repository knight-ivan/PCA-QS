# PCA-Guided Quantile Sampling (PCA-QS)

Reference implementation and reproducibility code for

> **PCA-Guided Quantile Sampling: Preserving Data Structure in Large-Scale Subsampling**
> Hui-Mean Foo and Yuan-chin Ivan Chang, Institute of Statistical Science, Academia Sinica.

PCA-QS is a structure-preserving subsampling method. It keeps the **original
feature space** and uses the leading principal components only to *guide* a
quantile-based stratification: each of the top-`k` PC scores is split into `m`
quantile bins, the composite bin is the stratum, and points are drawn from each
stratum under proportional allocation. The result is a subsample that tracks the
full data's distribution more closely than simple random sampling (SRS), with a
provable variance-reduction guarantee — provided the number of composite cells
respects the **occupancy budget** `m^k <= r` (`r` = retained size).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pure NumPy / SciPy / scikit-learn (+ joblib for parallelism) — no compilation, no GPU.

## Quickstart

```python
import numpy as np
from pcaqs import PCAQS, srs_indices

X = np.random.randn(100_000, 50)          # your data (rows = samples)

qs = PCAQS(n_components=5, n_bins=10, retention=0.05, random_state=0)
idx = qs.fit_sample_indices(X)            # indices of the retained subset
X_small = X[idx]                          # subsample in the ORIGINAL feature space
```

Key knobs: `n_components` (k, PCs that guide stratification), `n_bins` (m, quantile
bins per component), `retention` (delta, fraction kept per stratum). Keep
`m^k <= r`, e.g. `m <= floor(r**(1/k))`.

## Reproduce the paper's results (1000 replications each)

Every study is parallelized (`--jobs -1`) and writes summary CSVs + figures to
`figures/`. The committed `figures/` contents are the exact records behind the
paper's tables and figures.

```bash
# --- Theory validation ---
python experiments/variance_theorem_validation.py --reps 4000   # Prop + Theorem 2 (variance reduction)
python experiments/kl_rate_validation.py       --reps 1000      # Thm 1(iii) reverse-KL O(r^{-4/(k+4)})
python experiments/rate_validation.py          --reps 1000      # quantile ~n^{-1/2}, exact W2 ~n^{-1/k} (diagnostics)

# --- Distributional fidelity ---
python experiments/confirm_synthetic_distance.py --runs 1000    # 7 metrics vs SRS, k = 3,5,10,dynamic
python experiments/confirm_real_data.py          --reps 1000    # 7 real datasets, dynamic k + k=4 recovery

# --- Downstream tasks ---
python experiments/confirm_classification.py      --runs 1000   # synthetic clf, leakage-free (logistic + RF)
python experiments/confirm_classification_real.py --reps 1000   # Credit Card, APS (leakage-free, logistic)
python experiments/confirm_regression.py          --reps 1000   # vs SRS / leverage-score / coreset
python experiments/coef_recovery.py               --reps 1000   # coefficient stability, 5 datasets (logistic + ridge)

# --- Figures / tables ---
python experiments/make_result_figures.py           # synthetic distance panels + real-data heatmap
python experiments/make_detail_table.py             # consolidated corrected metric table
python experiments/make_detail_tables_appendix.py   # per-dataset detailed metric tables (appendix)
```

See `SIMULATION_CONFIRMATION.md` for the full run log, seeds, and per-study findings.

## Repository layout

```
pcaqs/                 core library
  sampler.py             PCAQS sampler (fit / strata / sample_indices) + srs_indices
  metrics.py             quantile error, energy, MMD, Mahalanobis, exact/sliced W2, hist KL/JS
  data.py                synthetic generators: structure_fidelity_gmm, anisotropic_gmm
                         (distinct leading eigenvalues, so A2 holds), classification_gmm
experiments/
  variance_theorem_validation.py   Proposition + Theorem 2, head-on (45-degree check)
  rate_validation.py               quantile / exact-W2 rate diagnostics (parallel)
  kl_rate_validation.py            reverse-KL exponent vs the analytic projected-mixture density
  confirm_synthetic_distance.py    PCA-QS vs SRS across 7 metrics x k (incl. dynamic)
  confirm_real_data.py             real-data structure preservation (7 datasets)
  confirm_classification.py        synthetic downstream classification (leakage-free)
  confirm_classification_real.py   real-data classification (Credit Card, APS)
  confirm_regression.py            regression vs SRS / leverage-score / coreset
  coef_recovery.py                 coefficient-recovery / stability study
  make_result_figures.py           redraw the paper's figures from the summary CSVs
  make_detail_table*.py            generate the corrected metric tables (LaTeX)
  paper_original/                  the exact original scripts that produced the first-submission
                                   numbers, kept for archival reproducibility
figures/               summary CSVs + generated figures (the committed result records)
SIMULATION_CONFIRMATION.md   run log, seeds, and per-study findings
requirements.txt
LICENSE
```

## Data

The synthetic experiments are fully self-contained (`pcaqs.data`). The real-data
studies use public datasets — Credit Card Default, SCANIA APS Failure, EEG Eye
State, Epileptic Seizure, MAGIC Gamma, Online News Popularity, YearPredictionMSD,
HIGGS, CASP, Bike Sharing, Appliances Energy — fetched from OpenML / UCI by the
scripts (see each `confirm_*` / `coef_recovery` file for the source id). Raw
datasets are **not** redistributed here.

## Reproducibility notes

- All reported studies use **1000 replications** (the variance-theorem validation
  uses 4000 resamples, chosen for its difference-of-variances estimator).
- Seeds are fixed and recorded in each script; the real-data / classification runs
  use per-replicate seed `base + i` (base logged in `SIMULATION_CONFIRMATION.md`).
- The "dynamic" PC configuration selects the smallest `k` explaining 70% of the
  variance; on near-isotropic data this can exceed the occupancy budget, in which
  case PCA-QS degrades to (or below) SRS — cap `k` so that `m^k <= r`.

## Citation

```bibtex
@unpublished{FooChang_PCAQS,
  author = {Foo, Hui-Mean and Chang, Yuan-chin Ivan},
  title  = {PCA-Guided Quantile Sampling: Preserving Data Structure in Large-Scale Subsampling},
  note   = {Manuscript},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
