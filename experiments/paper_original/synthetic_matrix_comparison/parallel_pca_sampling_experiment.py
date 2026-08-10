
# =====================
# Config Section
# =====================
CONFIG = {
    'random_seed': 123127,
    'num_samples': 1000000,
    'num_features': 50,
    'num_informative': 30,
    'n_redundant': 5,
    'n_repeated': 2,
    'class_sep': 1.5,
    'weights': [0.9, 0.1],
    'num_quantiles': 10,
    'use_dynamic_pcs': True,
    'num_pcs': None,
    'variance_threshold': 0.7,
    'mmd_gamma': 1.0,
    'kl_bins': 50,
    'pairwise_sample_size': 300
}

# =====================
# Execution Parameters
# =====================
RUN_OPTIONS = {
    'pc_options': ['3', '5', '10', 'dyn'],
    'repeats': 1000,
    'csv_path': None,
    'output_dir': 'results_1M_50F-parallel'
}

import numpy as np
import pandas as pd
import os
import argparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import multiprocessing


def load_data(csv_path=None):
    if csv_path:
        return pd.read_csv(csv_path)
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=CONFIG['num_samples'],
        n_features=CONFIG['num_features'],
        n_informative=CONFIG['num_informative'],
        n_redundant=CONFIG['n_redundant'],
        n_repeated=CONFIG['n_repeated'],
        n_classes=2,
        weights=CONFIG['weights'],
        class_sep=CONFIG['class_sep'],
        random_state=np.random.randint(0, 1e6)
    )
    return pd.DataFrame(X), y


def standardize_data(df):
    return StandardScaler().fit_transform(df)


def apply_pca(data, n_components):
    pca = TruncatedSVD(n_components=n_components, random_state=np.random.randint(0, 1e6))
    transformed = pca.fit_transform(data)
    return transformed, pca


def pca_quantile_sampling(data, deduce_rate=0.1, num_quantiles=5, use_dynamic_pcs=True, num_pcs=None, variance_threshold=0.7, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if use_dynamic_pcs:
        pca_full = TruncatedSVD(n_components=min(data.shape[1] - 1, 20), random_state=np.random.randint(0, 1e6))
        pca_full.fit(data)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        num_pcs = np.searchsorted(cumulative_variance, variance_threshold) + 1

    pca = TruncatedSVD(n_components=num_pcs, random_state=np.random.randint(0, 1e6))
    pca_scores = pca.fit_transform(data)

    bins = np.zeros_like(pca_scores[:, :num_pcs], dtype=int)
    for i in range(num_pcs):
        quantiles = np.unique(np.quantile(pca_scores[:, i], np.linspace(0, 1, num_quantiles + 1)))
        bins[:, i] = np.digitize(pca_scores[:, i], quantiles, right=True)

    _, bin_idx = np.unique(bins, axis=0, return_inverse=True)
    sampled = []
    for g in np.unique(bin_idx):
        group_indices = np.where(bin_idx == g)[0]
        k = max(1, int(deduce_rate * len(group_indices)))
        sampled.extend(rng.choice(group_indices, k, replace=False))

    return data[sampled[:int(len(sampled))]], sampled[:int(len(sampled))]


def simple_random_sample(data, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng(CONFIG['random_seed'])
    indices = rng.choice(len(data), size=n_samples, replace=False)
    return data[indices], indices


def reconstruction_error(original, reduced, pca):
    projected_back = reduced @ pca.components_
    return np.mean(np.square(original[:projected_back.shape[0]] - projected_back))


def pairwise_distance_diff(original, sampled):
    k = min(CONFIG['pairwise_sample_size'], len(original), len(sampled))
    idx_orig = np.random.choice(len(original), k, replace=False)
    idx_samp = np.random.choice(len(sampled), k, replace=False)
    d_orig = pairwise_distances(original[idx_orig])
    d_sample = pairwise_distances(sampled[idx_samp])
    return np.mean(np.abs(d_orig - d_sample))


def multivariate_energy_distance(X, Y, sample_size=300):
    n = min(len(X), len(Y), sample_size)
    rng = np.random.default_rng(CONFIG['random_seed'])
    X_sub = X[rng.choice(len(X), size=n, replace=False)]
    Y_sub = Y[rng.choice(len(Y), size=n, replace=False)]
    dist_XY = pairwise_distances(X_sub, Y_sub)
    dist_XX = pairwise_distances(X_sub, X_sub)
    dist_YY = pairwise_distances(Y_sub, Y_sub)
    return 2 * np.mean(dist_XY) - np.mean(dist_XX) - np.mean(dist_YY)


def marginal_energy_distance(original, sampled):
    min_len = min(original.shape[0], sampled.shape[0])
    return np.mean([np.linalg.norm(original[:min_len, i] - sampled[:min_len, i]) for i in range(original.shape[1])])


def normalized_energy_distance(original, sampled):
    base = marginal_energy_distance(original, sampled)
    norm = np.std(original) + np.std(sampled)
    return base / norm if norm > 0 else np.nan


def kl_divergence(original, sampled, bins=50):
    kl_total = 0
    for i in range(original.shape[1]):
        p_hist, _ = np.histogram(original[:, i], bins=bins, density=True)
        q_hist, _ = np.histogram(sampled[:, i], bins=bins, density=True)
        p_hist += 1e-8
        q_hist += 1e-8
        kl_total += entropy(p_hist, q_hist)
    return kl_total / original.shape[1]


def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=1.0):
    k = min(CONFIG['pairwise_sample_size'], len(X), len(Y))
    X_sub = X[:k]
    Y_sub = Y[:k]
    XX = np.exp(-cdist(X_sub, X_sub, 'sqeuclidean') * gamma)
    YY = np.exp(-cdist(Y_sub, Y_sub, 'sqeuclidean') * gamma)
    XY = np.exp(-cdist(X_sub, Y_sub, 'sqeuclidean') * gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mahalanobis_distance_avg(X, Y):
    try:
        VI = np.linalg.pinv(np.cov(X.T))
        return np.mean([mahalanobis(x, y, VI) for x, y in zip(X, Y)])
    except Exception as e:
        print(f"⚠️ Mahalanobis error: {e}")
        return np.nan


def jensen_shannon_divergence(p, q):
    p = np.asarray(p) + 1e-10
    q = np.asarray(q) + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def run_one_experiment(pc_opt, run_index, data, y):
    rng = np.random.default_rng(CONFIG['random_seed'] + run_index)
    is_dynamic = (str(pc_opt).lower() == 'dyn')
    CONFIG['use_dynamic_pcs'] = is_dynamic

    if is_dynamic:
        pca_full = TruncatedSVD(n_components=min(data.shape[1] - 1, 20), random_state=np.random.randint(0, 1e6))
        pca_full.fit(data)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        CONFIG['num_pcs'] = np.searchsorted(cumulative_variance, CONFIG['variance_threshold']) + 1
        n_components = CONFIG['num_pcs']
    else:
        n_components = int(pc_opt)

    pc_setting_label = f'dyn={n_components}' if is_dynamic else str(pc_opt)
    reduced_pca, pca_model = apply_pca(data, n_components)
    deduce_rate = n_components / data.shape[0]

    sampled_qs, qs_indices = pca_quantile_sampling(
        data,
        deduce_rate=deduce_rate,
        num_quantiles=CONFIG['num_quantiles'],
        use_dynamic_pcs=is_dynamic,
        num_pcs=n_components,
        variance_threshold=CONFIG['variance_threshold'],
        rng=rng
    )

    sampled_srs, srs_indices = simple_random_sample(data, n_components, rng=rng)

    return {
        'PC Setting': pc_setting_label,
        'Run': run_index,
        'QS JS Divergence (labels)': jensen_shannon_divergence(np.bincount(y, minlength=2) / len(y), np.bincount(y[qs_indices], minlength=2) / len(qs_indices)),
        'SRS JS Divergence (labels)': jensen_shannon_divergence(np.bincount(y, minlength=2) / len(y), np.bincount(y[srs_indices], minlength=2) / len(srs_indices)),
        'PCA Reconstruction Error': reconstruction_error(data, reduced_pca, pca_model),
        'QS Distance Difference': pairwise_distance_diff(data, sampled_qs),
        'QS Energy Distance': multivariate_energy_distance(data, sampled_qs),
        'QS KL Divergence': kl_divergence(data, sampled_qs, bins=CONFIG['kl_bins']),
        'QS MMD': maximum_mean_discrepancy(data, sampled_qs, gamma=CONFIG['mmd_gamma']),
        'QS Mahalanobis Distance': mahalanobis_distance_avg(data, sampled_qs),
        'SRS Distance Difference': pairwise_distance_diff(data, sampled_srs),
        'SRS Energy Distance': multivariate_energy_distance(data, sampled_srs),
        'QS Normalized Energy Distance': normalized_energy_distance(data, sampled_qs),
        'SRS Normalized Energy Distance': normalized_energy_distance(data, sampled_srs),
        'SRS KL Divergence': kl_divergence(data, sampled_srs, bins=CONFIG['kl_bins']),
        'SRS MMD': maximum_mean_discrepancy(data, sampled_srs, gamma=CONFIG['mmd_gamma']),
        'SRS Mahalanobis Distance': mahalanobis_distance_avg(data, sampled_srs)
    }


def main(args):
    df, y = load_data(args.csv_path if args.csv_path else None)
    data = standardize_data(df)

    pc_options = [int(opt) if opt.isdigit() else opt.strip() for opt in args.pc_options.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    jobs = [(pc_opt, run, data, y) for pc_opt in pc_options for run in range(args.repeats)]

    n_jobs = max(1, multiprocessing.cpu_count() - 5)
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_experiment)(*job) for job in jobs
    )

    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(args.output_dir, 'results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}")

        for pc_label, group_df in results_df.groupby("PC Setting"):
            safe_label = str(pc_label).replace("=", "_")
            file_path = os.path.join(args.output_dir, f'results_{safe_label}.csv')
            group_df.to_csv(file_path, index=False)
            print(f"✅ Saved: {file_path}")

        combined_path = os.path.join(args.output_dir, 'results_combined.csv')
        results_df.to_csv(combined_path, index=False)
        print(f"📦 Combined results saved to {combined_path}")
    else:
        print("⚠️ No results were generated. Please check earlier logs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=RUN_OPTIONS['csv_path'])
    parser.add_argument('--output_dir', type=str, default=RUN_OPTIONS['output_dir'])
    parser.add_argument('--pc_options', type=str, default=','.join(map(str, RUN_OPTIONS['pc_options'])))
    parser.add_argument('--repeats', type=int, default=RUN_OPTIONS['repeats'])
    args = parser.parse_args()
    main(args)
