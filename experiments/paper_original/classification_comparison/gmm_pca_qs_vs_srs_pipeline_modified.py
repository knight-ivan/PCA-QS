
# ======================== Configuration ========================
config = {
    'random_seed': 123,
    'n_samples': 100000,
    'n_features': 50,
    'n_informative': 30,
    'n_redundant': 15,
    'n_repeated': 10,
    'class_sep': 0.1,
    'flip_y': 0.1,
    'weights': [0.9, 0.1],
    'nonlinear': True,  # Set to False if you want the original GMM
    'test_size': 1000,
    'train_sample_sizes': [5000],
    'n_bins_pca_qs': 10,
    'repeat': 1000,
    'variance_threshold': 0.70,
    'classifier_list': [ 'xgboost', 'svm', 'random_forest', 'knn', 'logistic'],
    'knn_n_neighbors': 5,
    'csv_output_prefix': 'home_Correlated_pca_qs_vs_srs',
    'n_jobs': -2
}

# ======================== Imports ==============================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from joblib import Parallel, delayed
from tqdm import tqdm
import os

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

def generate_gmm_data(n_samples, weights, n_features, random_seed=42, nonlinear=False):
    np.random.seed(random_seed)
    centers = [np.zeros(n_features), np.ones(n_features) * 0.5]
    covariances = [np.identity(n_features), np.identity(n_features) * 1.2]
    X_list, y_list = [], []

    for class_idx, (weight, center, cov) in enumerate(zip(weights, centers, covariances)):
        n_class_samples = int(n_samples * weight)
        X_class = np.random.multivariate_normal(mean=center, cov=cov, size=n_class_samples)
        y_class = np.full(n_class_samples, class_idx)
        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if nonlinear:
        # Add interaction terms
        interaction_terms = np.array([X[:, i] * X[:, j]
                                      for i in range(n_features)
                                      for j in range(i + 1, n_features)]).T

        # Add sine transformation
        sine_terms = np.sin(X)

        # Combine all features
        X = np.hstack([X, interaction_terms, sine_terms])

    return X, y


# ======================== Utility Functions =====================
def compute_fp_fn_rates(cm):
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return fpr, fnr, tpr, tnr

def evaluate_results(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    fpr, fnr, tpr, tnr = compute_fp_fn_rates(cm)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
    recall = tpr
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return acc, auc, fpr, fnr, tpr, tnr, f1

def get_classifier(name, seed):
    if name == 'logistic':
        return LogisticRegression(solver='liblinear', random_state=seed)
    elif name == 'svm':
        return SVC(kernel='rbf', probability=True, random_state=seed)
    elif name == 'random_forest':
        return RandomForestClassifier(random_state=seed, n_jobs=config['n_jobs'])
    elif name == 'xgboost':
        if xgb_available:
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed, n_jobs=config['n_jobs'])
        else:
            raise ImportError("XGBoost is not installed.")
    elif name == 'knn':
        return KNeighborsClassifier(n_neighbors=config['knn_n_neighbors'], n_jobs=config['n_jobs'])
    else:
        raise ValueError(f"Unknown classifier: {name}")

def pca_qs_sample_indices(X_scaled, train_indices, sample_size, n_bins, variance_threshold):
    # Perform PCA
    X_pca_full = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(X_pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Use only training data
    X_pca_train = X_pca[train_indices]

    # Discretize each PC into quantile bins
    quantile_bins = []
    for i in range(n_components):
        quantiles = np.percentile(X_pca_train[:, i], np.linspace(0, 100, n_bins + 1))
        bins = np.digitize(X_pca_train[:, i], quantiles[1:-1], right=True)
        quantile_bins.append(bins)

    quantile_bins = np.stack(quantile_bins, axis=1)
    composite_keys = ['-'.join(map(str, row)) for row in quantile_bins]

    # Map keys to indices
    from collections import defaultdict
    bin_map = defaultdict(list)
    for idx, key in zip(train_indices, composite_keys):
        bin_map[key].append(idx)

    # Sample from each group proportionally
    pca_qs_indices = []
    for group in bin_map.values():
        size = max(1, int(len(group) * sample_size / len(train_indices)))
        sampled = np.random.choice(group, size=min(len(group), size), replace=False)
        pca_qs_indices.extend(sampled)

    return pca_qs_indices, n_components


def run_single_experiment(classifier_name, sample_size, repeat):
    np.random.seed(config['random_seed'] + repeat)

    X, y = generate_gmm_data(
        n_samples=config['n_samples'],
        weights=config['weights'],
        n_features=config['n_features'],
        random_seed=config['random_seed'] + repeat
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    test_indices = np.random.choice(len(X_scaled), size=config['test_size'], replace=False)
    train_indices = np.setdiff1d(np.arange(len(X_scaled)), test_indices)
    X_test = X_scaled[test_indices]
    y_test = y[test_indices]

    records = []

    srs_indices = np.random.choice(train_indices, size=sample_size, replace=False)
    X_srs = X_scaled[srs_indices]
    y_srs = y[srs_indices]
    if len(np.unique(y_srs)) >= 2:
        clf_srs = get_classifier(classifier_name, seed=config['random_seed'] + repeat)
        clf_srs.fit(X_srs, y_srs)
        y_prob_srs = clf_srs.predict_proba(X_test)[:, 1]
        threshold_srs = np.mean(y_srs)
        y_pred_srs = (y_prob_srs >= threshold_srs).astype(int)
        acc_srs, auc_srs, fpr_srs, fnr_srs, tpr_srs, tnr_srs, f1_srs = evaluate_results(y_test, y_pred_srs, y_prob_srs)
        records.append({
            'method': 'SRS', 'sample_size': sample_size, 'repeat': repeat,
            'accuracy': acc_srs, 'auc': auc_srs, 'fpr': fpr_srs, 'fnr': fnr_srs,
            'n_pca_components': None,
            'tpr': tpr_srs, 'tnr': tnr_srs, 'f1': f1_srs
        })

    pca_qs_indices, n_components_used = pca_qs_sample_indices(
        X_scaled, train_indices, sample_size,
        config['n_bins_pca_qs'], config['variance_threshold']
    )
    X_qs = X_scaled[pca_qs_indices]
    y_qs = y[pca_qs_indices]
    if len(np.unique(y_qs)) >= 2:
        clf_qs = get_classifier(classifier_name, seed=config['random_seed'] + repeat)
        clf_qs.fit(X_qs, y_qs)
        y_prob_qs = clf_qs.predict_proba(X_test)[:, 1]
        threshold_qs = np.mean(y_qs)
        y_pred_qs = (y_prob_qs >= threshold_qs).astype(int)
        acc_qs, auc_qs, fpr_qs, fnr_qs, tpr_qs, tnr_qs, f1_qs = evaluate_results(y_test, y_pred_qs, y_prob_qs)
        records.append({
            'method': 'PCA-QS', 'sample_size': sample_size, 'repeat': repeat,
            'n_pca_components': n_components_used,
            'accuracy': acc_qs, 'auc': auc_qs, 'fpr': fpr_qs, 'fnr': fnr_qs,
            'tpr': tpr_qs, 'tnr': tnr_qs, 'f1': f1_qs
        })

    return records

# ======================== Main Experiment ======================
def run_experiment_all_classifiers():
    for classifier_name in config['classifier_list']:
        if classifier_name == 'xgboost' and not xgb_available:
            continue

        output_path = f"{config['csv_output_prefix']}_{classifier_name}_results.csv"
        tasks = [
            (classifier_name, sample_size, repeat)
            for sample_size in config['train_sample_sizes']
            for repeat in range(config['repeat'])
        ]

        all_records = Parallel(n_jobs=config['n_jobs'])(
            delayed(run_single_experiment)(clf, size, rep)
            for clf, size, rep in tqdm(tasks, desc=f"Running {classifier_name}")
        )

        flat_records = [rec for sublist in all_records for rec in sublist]
        if flat_records:
            df_results = pd.DataFrame(flat_records)
            df_results.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

# ======================== Run ======================
if __name__ == '__main__':
    run_experiment_all_classifiers()
