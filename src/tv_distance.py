"""
TV Distance Estimation for Distribution Shift Detection

This module implements Total Variation (TV) distance estimation using:
1. Histogram-based TV for 1-D scalar signals (exact, no tuning needed)
2. k-NN density-ratio TV for multi-dimensional data (logit vectors)
   with automatic intrinsic-dimensionality estimation to avoid saturation.

Used to compare ID vs OOD logit distributions and to evaluate the 12
hand-crafted suitability signals from the ICML 2025 paper.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Optional, Tuple, Dict, List


# ---------------------------------------------------------------------------
# 1-D histogram-based TV  (preferred for individual scalar signals)
# ---------------------------------------------------------------------------

def histogram_tv_distance(
    P: np.ndarray,
    Q: np.ndarray,
    bins: int = 50,
    range_quantile: float = 0.001,
) -> float:
    """
    Estimate TV distance between 1-D samples using density histograms.

    TV(P,Q) = 0.5 * sum_i |p_i - q_i|  where p_i, q_i are bin densities.

    Args:
        P: (n,) samples from reference distribution
        Q: (m,) samples from test distribution
        bins: number of histogram bins
        range_quantile: clip extreme quantiles to stabilize bin edges

    Returns:
        TV distance in [0, 1]
    """
    P = np.asarray(P).ravel()
    Q = np.asarray(Q).ravel()

    # Shared bin edges (robust to outliers)
    combined = np.concatenate([P, Q])
    lo = np.quantile(combined, range_quantile)
    hi = np.quantile(combined, 1 - range_quantile)
    if hi <= lo:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)

    p_hist, _ = np.histogram(P, bins=edges, density=True)
    q_hist, _ = np.histogram(Q, bins=edges, density=True)

    bin_width = edges[1] - edges[0]
    tv = 0.5 * np.sum(np.abs(p_hist - q_hist)) * bin_width
    return float(np.clip(tv, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Intrinsic dimensionality estimation (MLE, Levina-Bickel 2004)
# ---------------------------------------------------------------------------

def _estimate_intrinsic_dim(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate intrinsic dimensionality via maximum-likelihood (Levina-Bickel).

    Uses the ratio of successive kNN distances to estimate the local
    dimensionality at each point, then averages.
    """
    n, dim = X.shape
    k = min(k, n - 2)
    if k < 2:
        return float(dim)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(X)
    dists, _ = nn.kneighbors(X)
    dists = dists[:, 1:]  # drop self

    # MLE: d_hat(x) = 1 / (1/(k-1) * sum_{j=1}^{k-1} log(T_k / T_j))
    eps = 1e-10
    T_k = dists[:, -1:] + eps  # (n, 1)
    log_ratios = np.log(T_k / (dists[:, :-1] + eps))  # (n, k-1)
    d_hat = 1.0 / (log_ratios.mean(axis=1) + eps)

    # Robust average (median to reduce outlier impact)
    return float(np.median(d_hat))


# ---------------------------------------------------------------------------
# k-NN density-ratio TV  (for multi-dimensional logit / feature vectors)
# ---------------------------------------------------------------------------

def knn_tv_distance(
    P: np.ndarray,
    Q: np.ndarray,
    k: Optional[int] = None,
    d: Optional[int] = None,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    auto_intrinsic_dim: bool = True,
    subsample: Optional[int] = 5000,
) -> float:
    """
    Estimate TV distance between samples P and Q using k-NN density ratios.

    The density ratio estimator is:
        r_k(x) = (d_Q(x) / d_P(x))^d

    The symmetrized TV estimator is:
        TV_hat = 0.5 * [ E_P |1 - r_k(x)| + E_Q |1 - 1/r_k(y)| ]

    Args:
        P: (n, dim) samples from reference distribution
        Q: (m, dim) samples from test distribution
        k: number of neighbors (default: sqrt(n))
        d: intrinsic dimensionality override.  If None and auto_intrinsic_dim
           is True, estimated from data.  Otherwise falls back to ambient dim.
        use_pca: whether to apply PCA for dimensionality reduction
        pca_components: number of PCA components (default: min(dim, 20))
        auto_intrinsic_dim: estimate intrinsic dim instead of using ambient dim
        subsample: cap sample size for speed (None to disable)

    Returns:
        Symmetrized TV distance estimate in [0, 1]
    """
    P = np.atleast_2d(P)
    Q = np.atleast_2d(Q)
    n, dim = P.shape
    m = Q.shape[0]

    assert P.shape[1] == Q.shape[1], "P and Q must have same dimensionality"
    assert n > 0 and m > 0, "P and Q must have at least one sample"

    # Optional subsampling for speed on large datasets
    if subsample and n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        P = P[idx]
        n = subsample
    if subsample and m > subsample:
        idx = np.random.choice(m, subsample, replace=False)
        Q = Q[idx]
        m = subsample

    # Apply PCA if requested
    if use_pca:
        pca_components = pca_components or min(dim, 20)
        pca = PCA(n_components=pca_components)
        combined = np.vstack([P, Q])
        combined_pca = pca.fit_transform(combined)
        P = combined_pca[:n]
        Q = combined_pca[n:]
        dim = pca_components

    # Set default k
    k = k or max(1, int(np.sqrt(n)))
    k = min(k, n - 1, m - 1)
    if k < 1:
        k = 1

    # Set intrinsic dimensionality
    if d is not None:
        pass  # caller override
    elif auto_intrinsic_dim and dim > 1:
        d = max(1, int(round(_estimate_intrinsic_dim(
            np.vstack([P, Q]), k=min(20, n + m - 2)
        ))))
    else:
        d = dim

    # Build k-NN indices
    nn_P = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(P)
    nn_Q = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(Q)

    dist_PP, _ = nn_P.kneighbors(P)
    d_PP = dist_PP[:, -1]

    dist_PQ, _ = nn_Q.kneighbors(P)
    d_PQ = dist_PQ[:, min(k - 1, m - 1)]

    dist_QP, _ = nn_P.kneighbors(Q)
    d_QP = dist_QP[:, min(k - 1, n - 1)]

    dist_QQ, _ = nn_Q.kneighbors(Q)
    d_QQ = dist_QQ[:, -1]

    eps = 1e-10
    r_P = np.clip((d_PQ / (d_PP + eps)) ** d, 1e-6, 1e6)
    r_Q = np.clip((d_QQ / (d_QP + eps)) ** d, 1e-6, 1e6)

    tv_from_P = np.mean(np.abs(1 - r_P))
    tv_from_Q = np.mean(np.abs(1 - 1.0 / r_Q))

    tv_estimate = 0.5 * (tv_from_P + tv_from_Q)
    return float(np.clip(tv_estimate, 0.0, 1.0))


def knn_tv_distance_batch(
    P: np.ndarray,
    Q_list: list,
    k: Optional[int] = None,
    d: Optional[int] = None,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    auto_intrinsic_dim: bool = True,
) -> np.ndarray:
    """
    Compute TV distance from reference P to multiple test distributions Q.

    Args:
        P: (n, dim) samples from reference distribution
        Q_list: list of (m_i, dim) arrays, each representing a test distribution

    Returns:
        Array of TV distances, one per Q in Q_list
    """
    return np.array([
        knn_tv_distance(P, Q, k=k, d=d, use_pca=use_pca,
                        pca_components=pca_components,
                        auto_intrinsic_dim=auto_intrinsic_dim)
        for Q in Q_list
    ])


# ---------------------------------------------------------------------------
# Per-sample kNN distance scores (building block for correctness prediction)
# ---------------------------------------------------------------------------

def knn_distance_scores(
    P_ref: np.ndarray,
    Q_test: np.ndarray,
    k: int = 10,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample kNN distance to a reference distribution.

    For each sample in Q_test: distance to its k-th nearest neighbor in P_ref.
    For each sample in P_ref: leave-one-out distance (k-th neighbor excluding self).

    These distances serve as a density proxy: samples far from the reference
    distribution are less likely to be correctly classified.

    Args:
        P_ref: (n, dim) reference samples (e.g., id_test logits)
        Q_test: (m, dim) test samples (e.g., ood_test logits)
        k: number of neighbors
        use_pca: whether to apply PCA before computing distances
        pca_components: number of PCA components (default: min(dim, 20))

    Returns:
        Tuple of:
            ref_scores: (n,) leave-one-out kNN distances for P_ref
            test_scores: (m,) kNN distances for Q_test to P_ref
    """
    P_ref = np.atleast_2d(P_ref).astype(np.float64)
    Q_test = np.atleast_2d(Q_test).astype(np.float64)
    n, dim = P_ref.shape

    assert P_ref.shape[1] == Q_test.shape[1], "P_ref and Q_test must have same dimensionality"

    # Optional PCA
    if use_pca:
        pca_components = pca_components or min(dim, 20)
        pca = PCA(n_components=pca_components)
        combined = np.vstack([P_ref, Q_test])
        combined_pca = pca.fit_transform(combined)
        P_ref = combined_pca[:n]
        Q_test = combined_pca[n:]

    # Clamp k to valid range
    k = min(k, n - 1)
    if k < 1:
        k = 1

    # Build kNN index on reference set
    # For leave-one-out on P_ref, we need k+1 neighbors (first is self)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(P_ref)

    # Leave-one-out distances for P_ref (exclude self, take k-th)
    dist_ref, _ = nn.kneighbors(P_ref)
    ref_scores = dist_ref[:, k]  # k-th neighbor (0-indexed: self=0, 1st=1, ..., k-th=k)

    # Distances for Q_test to P_ref (no self to exclude)
    dist_test, _ = nn.kneighbors(Q_test)
    test_scores = dist_test[:, k - 1]  # k-th neighbor (0-indexed, no self)

    return ref_scores, test_scores


# ---------------------------------------------------------------------------
# Per-sample TV correctness scores (density ratio framework)
# ---------------------------------------------------------------------------

def tv_correctness_scores(
    ref_logits: np.ndarray,
    ref_correct: np.ndarray,
    test_logits: np.ndarray,
    k: Optional[int] = None,
    d: Optional[int] = None,
    use_pca: bool = False,
    pca_components: Optional[int] = None,
    auto_intrinsic_dim: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate P(correct | x) per sample using kNN density ratio (TV framework).

    Splits reference into P (correct samples) and Q (incorrect samples),
    builds kNN indices on each, and estimates the density ratio per sample.

    The kNN density estimate at x is:  p_hat(x) = k / (n_P * c_d * d_P(x)^d)
    where d_P(x) = distance to k-th neighbor in P, d = intrinsic dimensionality.

    Applying Bayes' rule with empirical priors, the posterior simplifies to:
        P(correct | x) = d_Q(x)^d / (d_P(x)^d + d_Q(x)^d)
                        = 1 / (1 + (d_P(x) / d_Q(x))^d)

    where d_P(x), d_Q(x) are distances to the k-th neighbor in the correct
    and incorrect distributions respectively.  The sample-size priors cancel
    with the density normalization, so the score is purely geometric.

    For ref samples: leave-one-out (exclude self from whichever set they
    belong to).  For test samples: direct query against both indices.

    Args:
        ref_logits: (n, dim) reference logits
        ref_correct: (n,) boolean correctness labels
        test_logits: (m, dim) test logits
        k: number of neighbors (default: sqrt(min(n_P, n_Q)))
        d: intrinsic dimensionality override
        use_pca: apply PCA before distance computation
        pca_components: PCA components (default: min(dim, 20))
        auto_intrinsic_dim: estimate d from data if not provided

    Returns:
        ref_scores: (n,) estimated P(correct | x) for ref samples
        test_scores: (m,) estimated P(correct | x) for test samples
    """
    ref_logits = np.atleast_2d(ref_logits).astype(np.float64)
    test_logits = np.atleast_2d(test_logits).astype(np.float64)
    ref_correct = np.asarray(ref_correct).astype(bool)
    n, dim = ref_logits.shape
    m = test_logits.shape[0]

    assert ref_logits.shape[1] == test_logits.shape[1], \
        "ref and test must have same dimensionality"

    # Optional PCA
    if use_pca:
        pca_components = pca_components or min(dim, 20)
        pca = PCA(n_components=pca_components)
        combined = np.vstack([ref_logits, test_logits])
        combined_pca = pca.fit_transform(combined)
        ref_logits = combined_pca[:n]
        test_logits = combined_pca[n:]
        dim = pca_components

    # Split ref into P (correct) and Q (incorrect)
    P_mask = ref_correct
    Q_mask = ~ref_correct
    P = ref_logits[P_mask]
    Q = ref_logits[Q_mask]
    n_P = P.shape[0]
    n_Q = Q.shape[0]

    # Set default k
    k = k or max(1, int(np.sqrt(min(n_P, n_Q))))
    k = min(k, n_P - 1, n_Q - 1)
    if k < 1:
        k = 1

    # Estimate intrinsic dimensionality
    if d is not None:
        pass
    elif auto_intrinsic_dim and dim > 1:
        d = max(1, int(round(_estimate_intrinsic_dim(
            ref_logits, k=min(20, n - 2)
        ))))
    else:
        d = dim

    eps = 1e-10

    # Build kNN indices (k+1 for leave-one-out)
    nn_P = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(P)
    nn_Q = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(Q)

    # --- Ref scores (leave-one-out) ---
    ref_scores = np.zeros(n)

    # P samples (correct): self is in nn_P, not in nn_Q
    dist_PP, _ = nn_P.kneighbors(P)      # (n_P, k+1), col 0 is self
    dist_PQ, _ = nn_Q.kneighbors(P)      # (n_P, k+1), no self
    d_P_for_P = dist_PP[:, k] + eps      # k-th real neighbor (skip self at col 0)
    d_Q_for_P = dist_PQ[:, k - 1] + eps  # k-th neighbor in Q (no self)

    ratio_P = np.clip((d_P_for_P / d_Q_for_P) ** d, 1e-8, 1e8)
    ref_scores[P_mask] = 1.0 / (1.0 + ratio_P)

    # Q samples (incorrect): self is in nn_Q, not in nn_P
    dist_QP, _ = nn_P.kneighbors(Q)      # (n_Q, k+1), no self
    dist_QQ, _ = nn_Q.kneighbors(Q)      # (n_Q, k+1), col 0 is self
    d_P_for_Q = dist_QP[:, k - 1] + eps  # k-th neighbor in P (no self)
    d_Q_for_Q = dist_QQ[:, k] + eps      # k-th real neighbor (skip self)

    ratio_Q = np.clip((d_P_for_Q / d_Q_for_Q) ** d, 1e-8, 1e8)
    ref_scores[Q_mask] = 1.0 / (1.0 + ratio_Q)

    # --- Test scores ---
    dist_TP, _ = nn_P.kneighbors(test_logits)  # (m, k+1), no self
    dist_TQ, _ = nn_Q.kneighbors(test_logits)  # (m, k+1), no self
    d_P_for_T = dist_TP[:, k - 1] + eps        # k-th neighbor in P
    d_Q_for_T = dist_TQ[:, k - 1] + eps        # k-th neighbor in Q

    ratio_T = np.clip((d_P_for_T / d_Q_for_T) ** d, 1e-8, 1e8)
    test_scores = 1.0 / (1.0 + ratio_T)

    return ref_scores, test_scores


# ---------------------------------------------------------------------------
# High-level: compute TV across all signals + logits in one call
# ---------------------------------------------------------------------------

def compute_all_tv_distances(
    P_logits: np.ndarray,
    Q_logits: np.ndarray,
    pca_components: int = 10,
    histogram_bins: int = 50,
) -> Dict[str, float]:
    """
    Compute TV distances between ID (P) and OOD (Q) using multiple methods.

    Returns a dict with:
      - per-signal histogram TV  (12 entries)
      - kNN TV on raw logits (with auto intrinsic dim)
      - kNN TV on PCA-reduced logits
      - kNN TV on the 12-signal feature vector

    Args:
        P_logits: (n, C) logits from reference distribution
        Q_logits: (m, C) logits from test distribution
        pca_components: PCA dim for logit-space TV
        histogram_bins: bins for 1-D histogram TV

    Returns:
        Dictionary  method_name -> TV distance
    """
    results = {}

    # 1. Per-signal histogram TV
    sig_P = compute_suitability_signals(P_logits)
    sig_Q = compute_suitability_signals(Q_logits)
    for name in sig_P:
        results[f"hist_{name}"] = histogram_tv_distance(
            sig_P[name], sig_Q[name], bins=histogram_bins,
        )

    # 2. kNN TV on raw logits (auto intrinsic dim)
    results["knn_logits"] = knn_tv_distance(P_logits, Q_logits)

    # 3. kNN TV on PCA-reduced logits
    results[f"knn_logits_pca{pca_components}"] = knn_tv_distance(
        P_logits, Q_logits, use_pca=True, pca_components=pca_components,
    )

    # 4. kNN TV on the 12-signal feature vector
    feat_P = signals_to_features(sig_P)
    feat_Q = signals_to_features(sig_Q)
    results["knn_12signals"] = knn_tv_distance(feat_P, feat_Q)

    return results


def compute_suitability_signals(logits: np.ndarray) -> dict:
    """
    Compute all 12 suitability filter signals from logits.

    These are the signals used by the original suitability filter paper:
    1. conf_max: maximum softmax probability
    2. conf_std: standard deviation of softmax probabilities
    3. conf_entropy: Shannon entropy of softmax distribution
    4. logit_mean: mean of logit values
    5. logit_max: maximum logit value
    6. logit_std: standard deviation of logit values
    7. logit_diff_top2: difference between top 2 logits
    8. loss: cross-entropy loss (using argmax as pseudo-label)
    9. margin_loss: difference in log-prob between top 2 classes
    10. class_prob_ratio: ratio of top 2 softmax probabilities
    11. top_k_probs_sum: sum of top 10% class probabilities
    12. energy: negative log-sum-exp of logits

    Args:
        logits: (N, C) array of logits for N samples, C classes

    Returns:
        Dictionary with signal name -> (N,) array of values
    """
    N, C = logits.shape

    # Softmax probabilities
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)  # Numerically stable
    softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Predictions
    predictions = logits.argmax(axis=1)

    # 1. Confidence max
    conf_max = softmax.max(axis=1)

    # 2. Confidence std
    conf_std = softmax.std(axis=1)

    # 3. Confidence entropy
    eps = 1e-10
    conf_entropy = -np.sum(softmax * np.log(softmax + eps), axis=1)

    # 4. Logit mean
    logit_mean = logits.mean(axis=1)

    # 5. Logit max
    logit_max = logits.max(axis=1)

    # 6. Logit std
    logit_std = logits.std(axis=1)

    # 7. Logit diff top 2
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]  # Descending
    logit_diff_top2 = sorted_logits[:, 0] - sorted_logits[:, 1]

    # 8. Loss (cross-entropy with argmax as label)
    pred_probs = softmax[np.arange(N), predictions]
    loss = -np.log(pred_probs + eps)

    # 9. Margin loss
    sorted_softmax = np.sort(softmax, axis=1)[:, ::-1]
    pred_class_probs = sorted_softmax[:, 0]
    next_best_probs = sorted_softmax[:, 1]
    pred_class_loss = -np.log(pred_class_probs + eps)
    next_best_loss = -np.log(next_best_probs + eps)
    margin_loss = pred_class_loss - next_best_loss

    # 10. Class probability ratio
    class_prob_ratio = pred_class_probs / (next_best_probs + eps)

    # 11. Top-k probs sum (top 10%)
    top_k = max(1, int(C * 0.1))
    top_k_probs_sum = sorted_softmax[:, :top_k].sum(axis=1)

    # 12. Energy
    energy = -np.log(exp_logits.sum(axis=1)) - logits_max.squeeze()

    return {
        'conf_max': conf_max,
        'conf_std': conf_std,
        'conf_entropy': conf_entropy,
        'logit_mean': logit_mean,
        'logit_max': logit_max,
        'logit_std': logit_std,
        'logit_diff_top2': logit_diff_top2,
        'loss': loss,
        'margin_loss': margin_loss,
        'class_prob_ratio': class_prob_ratio,
        'top_k_probs_sum': top_k_probs_sum,
        'energy': energy,
    }


def signals_to_features(signals: dict) -> np.ndarray:
    """
    Convert signals dictionary to feature matrix.

    Args:
        signals: Dictionary of signal_name -> (N,) array

    Returns:
        (N, 12) feature matrix in the order used by the suitability filter
    """
    signal_order = [
        'conf_max', 'conf_std', 'conf_entropy',
        'logit_mean', 'logit_max', 'logit_std', 'logit_diff_top2',
        'loss', 'margin_loss', 'class_prob_ratio', 'top_k_probs_sum',
        'energy'
    ]
    return np.column_stack([signals[name] for name in signal_order])


def compare_tv_vs_signals(
    P_logits: np.ndarray,
    Q_logits: np.ndarray,
    labels_P: np.ndarray,
    labels_Q: np.ndarray,
) -> dict:
    """
    Compare TV distance against the 12 suitability signals for shift detection.

    Args:
        P_logits: (n, C) logits from reference distribution
        Q_logits: (m, C) logits from test distribution
        labels_P: (n,) true labels for P
        labels_Q: (m,) true labels for Q

    Returns:
        Dictionary with comparison metrics
    """
    from sklearn.metrics import roc_auc_score

    # Compute TV distance
    tv_dist = knn_tv_distance(P_logits, Q_logits)

    # Compute signals for both distributions
    signals_P = compute_suitability_signals(P_logits)
    signals_Q = compute_suitability_signals(Q_logits)

    # For each signal, compute how well it distinguishes P from Q
    # Using a simple threshold-based approach: can we tell if a sample is from P or Q?

    results = {
        'tv_distance': tv_dist,
        'signal_separability': {},
    }

    for signal_name in signals_P.keys():
        sig_P = signals_P[signal_name]
        sig_Q = signals_Q[signal_name]

        # Combine for AUC computation
        all_signals = np.concatenate([sig_P, sig_Q])
        all_labels = np.concatenate([np.zeros(len(sig_P)), np.ones(len(sig_Q))])

        # Handle edge cases
        if len(np.unique(all_labels)) < 2 or np.all(all_signals == all_signals[0]):
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(all_labels, all_signals)
                # Take max(auc, 1-auc) since direction doesn't matter
                auc = max(auc, 1 - auc)
            except:
                auc = 0.5

        results['signal_separability'][signal_name] = auc

    # Also compute accuracy metrics
    preds_P = P_logits.argmax(axis=1)
    preds_Q = Q_logits.argmax(axis=1)

    results['accuracy_P'] = (preds_P == labels_P).mean() if labels_P is not None else None
    results['accuracy_Q'] = (preds_Q == labels_Q).mean() if labels_Q is not None else None

    return results


if __name__ == "__main__":
    np.random.seed(42)

    print("=== Histogram TV (1-D) ===")
    P1 = np.random.randn(5000)
    Q1 = np.random.randn(5000)
    print(f"Same dist:      {histogram_tv_distance(P1, Q1):.4f}  (expect ~0)")
    Q1s = np.random.randn(5000) + 1.0
    print(f"Shifted +1:     {histogram_tv_distance(P1, Q1s):.4f}  (expect ~0.5)")
    Q1b = np.random.randn(5000) + 3.0
    print(f"Shifted +3:     {histogram_tv_distance(P1, Q1b):.4f}  (expect ~0.9)")

    print("\n=== kNN TV (multi-D, auto intrinsic dim) ===")
    P = np.random.randn(2000, 62)
    Q = np.random.randn(2000, 62)
    tv = knn_tv_distance(P, Q)
    print(f"Same dist (62-D):      {tv:.4f}  (expect ~0)")

    Q2 = np.random.randn(2000, 62) + 0.5
    tv2 = knn_tv_distance(P, Q2)
    print(f"Shifted +0.5 (62-D):   {tv2:.4f}  (expect moderate)")

    Q3 = np.random.randn(2000, 62) * 2 + 2
    tv3 = knn_tv_distance(P, Q3)
    print(f"Shifted +2, *2 (62-D): {tv3:.4f}  (expect high)")

    print("\n=== Intrinsic dim estimate ===")
    d_hat = _estimate_intrinsic_dim(P, k=20)
    print(f"62-D Gaussian intrinsic dim: {d_hat:.1f}")

    print("\n=== Signal computation ===")
    logits = np.random.randn(200, 62)
    signals = compute_suitability_signals(logits)
    print(f"Computed {len(signals)} signals")
    for name, values in signals.items():
        print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}")
