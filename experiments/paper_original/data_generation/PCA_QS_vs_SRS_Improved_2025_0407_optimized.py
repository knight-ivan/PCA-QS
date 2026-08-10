
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.linalg import pinvh
from numpy.linalg import slogdet
from tqdm import tqdm
import time
from joblib import Parallel, delayed

CONFIG = {
    "n_trials": 1000,
    "n_samples": 1000000,
    "n_features": 50,
    "mixture_coefficient": 0.95,
    "sample_sizes": [10000, 20000, 50000],
    "fixed_n_components_list": [5, 10, 15, 20],
    "include_dynamic_baseline": True,
    "random_seed_base": 42,
    "results_path": "results_synthetic.csv"
}

def generate_extreme_imbalance_gmm(n_samples, n_features, mixture_coefficient, rng):
    mean_healthy = np.full(n_features, -2)
    mean_unhealthy = np.full(n_features, 5)
    cov_healthy = np.full((n_features, n_features), 0.8) + np.eye(n_features) * 0.2
    cov_unhealthy = np.full((n_features, n_features), 0.4) + np.eye(n_features) * 0.6
    n_healthy = int(n_samples * mixture_coefficient)
    n_unhealthy = n_samples - n_healthy
    X_h = rng.multivariate_normal(mean_healthy, cov_healthy, n_healthy)
    X_u = rng.multivariate_normal(mean_unhealthy, cov_unhealthy, n_unhealthy)
    X = np.vstack([X_h, X_u])
    return StandardScaler().fit_transform(X)

def sample_srs(X, size, rng):
    return X[rng.choice(len(X), size=size, replace=False)]

def pca_quantile_sampling(data, deduce_rate=0.1, num_quantiles=10, use_dynamic_pcs=True, num_pcs=None, variance_threshold=0.8, rng=None, pca_fit_cache=None):
    if rng is None:
        rng = np.random.default_rng()
    actual_num_pcs = num_pcs
    if use_dynamic_pcs:
        if pca_fit_cache is None:
            pca_full = PCA().fit(data)
        else:
            pca_full = pca_fit_cache
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        actual_num_pcs = np.searchsorted(cum_var, variance_threshold) + 1
    pca = PCA(n_components=actual_num_pcs)
    pca_scores = pca.fit_transform(data)
    bins = np.zeros_like(pca_scores[:, :actual_num_pcs], dtype=int)
    for i in range(actual_num_pcs):
        quantiles = np.unique(np.quantile(pca_scores[:, i], np.linspace(0, 1, num_quantiles + 1)))
        bins[:, i] = np.searchsorted(quantiles, pca_scores[:, i], side='right')
    _, bin_idx = np.unique(bins, axis=0, return_inverse=True)
    sampled_indices = []
    for group in np.unique(bin_idx):
        group_indices = np.where(bin_idx == group)[0]
        k = max(1, int(deduce_rate * len(group_indices)))
        sampled_indices.extend(rng.choice(group_indices, k, replace=False))
    final_sample_size = min(len(sampled_indices), int(deduce_rate * len(data)))
    return data[sampled_indices[:final_sample_size]], actual_num_pcs

def energy_distance_avg(original, sample, sample_size_limit=2000):
    def energy_1d(x, y):
        if len(x) > sample_size_limit:
            x = x[np.random.choice(len(x), sample_size_limit, replace=False)]
        if len(y) > sample_size_limit:
            y = y[np.random.choice(len(y), sample_size_limit, replace=False)]
        return (np.mean(pdist(x, 'euclidean')) +
                np.mean(pdist(y, 'euclidean')) -
                2 * np.mean(pdist(np.vstack([x, y]), 'euclidean')[:len(x)*len(y)]))
    return np.mean([energy_1d(original[:, i:i+1], sample[:, i:i+1]) for i in range(original.shape[1])])

def mahalanobis_distance(sample, population):
    mean_sample = np.mean(sample, axis=0)
    mean_pop = np.mean(population, axis=0)
    cov = np.cov(population, rowvar=False)
    inv_cov = pinvh(cov)
    diff = mean_sample - mean_pop
    return np.sqrt(diff.T @ inv_cov @ diff)

def kl_divergence_mvnorm(sample, population):
    mu_p = np.mean(population, axis=0)
    mu_q = np.mean(sample, axis=0)
    cov_p = np.cov(population, rowvar=False)
    cov_q = np.cov(sample, rowvar=False)
    inv_cov_q = pinvh(cov_q)
    trace_term = np.trace(inv_cov_q @ cov_p)
    diff = mu_q - mu_p
    mean_term = diff.T @ inv_cov_q @ diff
    sign_p, logdet_p = slogdet(cov_p)
    sign_q, logdet_q = slogdet(cov_q)
    logdet_term = logdet_q - logdet_p
    k = len(mu_p)
    return 0.5 * (trace_term + mean_term - k + logdet_term)

def mmd_rbf(X, Y, gamma=1.0):
    XX = np.exp(-gamma * pdist(X, 'sqeuclidean'))
    YY = np.exp(-gamma * pdist(Y, 'sqeuclidean'))
    XY = np.exp(-gamma * pdist(np.vstack([X, Y]), 'sqeuclidean'))
    n = len(X)
    m = len(Y)
    mmd = (np.sum(XX) / (n * (n - 1)) +
           np.sum(YY) / (m * (m - 1)) -
           2 * np.sum(XY[:n * m]) / (n * m))
    return mmd

def run_trial(trial_idx):
    rng = np.random.default_rng(CONFIG["random_seed_base"] + trial_idx)
    X = generate_extreme_imbalance_gmm(CONFIG["n_samples"], CONFIG["n_features"], CONFIG["mixture_coefficient"], rng)
    pca_fit_cache = PCA().fit(X) if CONFIG["include_dynamic_baseline"] else None
    results = []
    for size in CONFIG["sample_sizes"]:
        deduce_rate = size / len(X)
        for method_name, sample_func in {
            "SRS": lambda: sample_srs(X, size, rng),
            **{f"PCA-QS-{pcs}PC": lambda pcs=pcs: pca_quantile_sampling(X, deduce_rate=deduce_rate, use_dynamic_pcs=False, num_pcs=pcs, rng=rng)[0]
               for pcs in CONFIG["fixed_n_components_list"]}
        }.items():
            sample_data = sample_func()
            results.append({
                "trial": trial_idx,
                "method": method_name,
                "sample_size": size,
                "num_pcs": None if "SRS" in method_name else int(method_name.split('-')[-1].replace("PC", "")),
                "energy_distance": energy_distance_avg(X, sample_data),
                "mahalanobis": mahalanobis_distance(sample_data, X),
                "kl_divergence": kl_divergence_mvnorm(sample_data, X),
                "mmd": mmd_rbf(X, sample_data)
            })
        if CONFIG["include_dynamic_baseline"]:
            sample_dyn, used_pcs = pca_quantile_sampling(X, deduce_rate=deduce_rate, use_dynamic_pcs=True, rng=rng, pca_fit_cache=pca_fit_cache)
            results.append({
                "trial": trial_idx,
                "method": "PCA-QS-Dynamic",
                "sample_size": size,
                "num_pcs": used_pcs,
                "energy_distance": energy_distance_avg(X, sample_dyn),
                "mahalanobis": mahalanobis_distance(sample_dyn, X),
                "kl_divergence": kl_divergence_mvnorm(sample_dyn, X),
                "mmd": mmd_rbf(X, sample_dyn)
            })
    return results

def main():
    start_time = time.time()
    all_results = Parallel(n_jobs=6)(
        delayed(run_trial)(i) for i in tqdm(range(CONFIG["n_trials"]))
    )
    flat_results = [row for result in all_results for row in result]
    df = pd.DataFrame(flat_results)
    df.to_csv(CONFIG["results_path"], index=False)
    print(f"Saved results to {CONFIG['results_path']} in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
