"""PCA-Guided Quantile Sampling (PCA-QS).

A structure-preserving subsampling method: use the leading principal components to
guide quantile stratification, keep the original feature space, and sample within
strata under proportional allocation.
"""
from .sampler import PCAQS, srs_indices
from . import metrics, data

__all__ = ["PCAQS", "srs_indices", "metrics", "data"]
__version__ = "0.1.0"
