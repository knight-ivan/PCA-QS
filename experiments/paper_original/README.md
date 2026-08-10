# Original paper scripts (archival)

These are the exact scripts that produced the numbers/figures in the manuscript,
kept unmodified for reproducibility. They predate the refactored `pcaqs/` package
and each carries its own (ad-hoc) PCA-QS implementation. For new work prefer the
maintained `pcaqs` library and the drivers in `experiments/`.

- `synthetic_matrix_comparison/` — 1M x 50 GMM distance-metric study (Table/Fig).
- `real_data_matrix_comparison/`  — UCI distance-metric study.
- `classification_comparison/`    — downstream logistic/SVM/RF/kNN/XGBoost study.
- `data_generation/`              — synthetic data generation.

Real datasets are not bundled; download each from its original source.
