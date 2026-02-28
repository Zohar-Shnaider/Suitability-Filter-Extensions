"""
Alternative TV Distance Approximations for Correctness Prediction

Method 1: Sliced TV — project onto random 1D directions, estimate density
           ratio via KDE, average across projections.

Method 2: Neural Witness — train a small MLP to approximate the variational
           (dual) TV: TV(P,Q) = sup_{||f||_inf <= 1} 0.5*(E_P[f] - E_Q[f]).
           Output in [-1,1] via tanh, convert to P(correct|x).

Both methods produce per-sample correctness scores comparable to
tv_correctness_scores() in src/tv_distance.py.
"""

import numpy as np
from scipy.stats import gaussian_kde
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.tv_distance import histogram_tv_distance


# ============================================================================
# Utilities
# ============================================================================

def _random_unit_vectors(n_projections: int, dim: int,
                         rng: np.random.RandomState) -> np.ndarray:
    """Sample n_projections unit vectors uniformly from S^{dim-1}."""
    raw = rng.randn(n_projections, dim)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / (norms + 1e-12)


# ============================================================================
# Method 1: Sliced TV
# ============================================================================

def sliced_tv_distance(
    P: np.ndarray,
    Q: np.ndarray,
    n_projections: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Sliced TV distance between two multidimensional distributions.

    Projects P and Q onto random 1D directions and computes histogram TV
    in each projection.  Returns (mean TV, max TV) across projections.

    Args:
        P: (n, dim) samples from distribution P
        Q: (m, dim) samples from distribution Q
        n_projections: number of random projection directions
        seed: random seed

    Returns:
        (mean_tv, max_tv)
    """
    P = np.atleast_2d(P).astype(np.float64)
    Q = np.atleast_2d(Q).astype(np.float64)
    dim = P.shape[1]
    assert P.shape[1] == Q.shape[1]

    rng = np.random.RandomState(seed)
    thetas = _random_unit_vectors(n_projections, dim, rng)

    tvs = np.empty(n_projections)
    for i, theta in enumerate(thetas):
        p_proj = P @ theta
        q_proj = Q @ theta
        tvs[i] = histogram_tv_distance(p_proj, q_proj)

    return float(tvs.mean()), float(tvs.max())


def _histogram_density_ratio(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    eval_points: np.ndarray,
    bins: int = 100,
) -> np.ndarray:
    """
    Estimate P(correct|x) = p(x) / (p(x) + q(x)) via histograms on 1D data.

    Much faster than KDE for large datasets (O(n+m+eval) vs O(n*eval)).
    """
    combined = np.concatenate([p_samples, q_samples])
    lo, hi = np.quantile(combined, 0.001), np.quantile(combined, 0.999)
    if hi <= lo:
        return np.full(len(eval_points), 0.5)

    edges = np.linspace(lo, hi, bins + 1)
    bin_width = edges[1] - edges[0]

    p_hist, _ = np.histogram(p_samples, bins=edges)
    q_hist, _ = np.histogram(q_samples, bins=edges)

    # Normalize to density (add smoothing)
    p_density = (p_hist + 1.0) / ((len(p_samples) + bins) * bin_width)
    q_density = (q_hist + 1.0) / ((len(q_samples) + bins) * bin_width)

    # Find bin for each eval point
    bin_idx = np.digitize(eval_points, edges) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)

    p_hat = p_density[bin_idx]
    q_hat = q_density[bin_idx]

    return p_hat / (p_hat + q_hat)


def sliced_tv_correctness_scores(
    ref_logits: np.ndarray,
    ref_correct: np.ndarray,
    test_logits: np.ndarray,
    n_projections: int = 500,
    bins: int = 100,
    method: str = 'histogram',
    bandwidth: str = 'scott',
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-sample P(correct|x) via sliced density ratio estimation.

    For each random 1D projection:
      - Estimate 1D density ratio p(x)/(p(x)+q(x)) for P (correct) vs Q (incorrect)
      - Use either histogram (fast) or KDE (slow but smoother)
    Average scores across all projections.

    Args:
        ref_logits: (n, dim) reference logits
        ref_correct: (n,) boolean correctness labels
        test_logits: (m, dim) test logits
        n_projections: number of random projection directions
        bins: number of histogram bins (for method='histogram')
        method: 'histogram' (fast) or 'kde' (slow, smoother)
        bandwidth: KDE bandwidth method, only for method='kde'
        seed: random seed

    Returns:
        ref_scores: (n,) estimated P(correct|x) for ref samples
        test_scores: (m,) estimated P(correct|x) for test samples
    """
    ref_logits = np.atleast_2d(ref_logits).astype(np.float64)
    test_logits = np.atleast_2d(test_logits).astype(np.float64)
    ref_correct = np.asarray(ref_correct).astype(bool)

    n, dim = ref_logits.shape
    m = test_logits.shape[0]

    P = ref_logits[ref_correct]    # correct samples
    Q = ref_logits[~ref_correct]   # incorrect samples

    rng = np.random.RandomState(seed)
    thetas = _random_unit_vectors(n_projections, dim, rng)

    # Vectorized projection: (n_proj, dim) @ (dim, n) -> (n_proj, n)
    ref_projs = thetas @ ref_logits.T   # (n_proj, n)
    test_projs = thetas @ test_logits.T  # (n_proj, m)
    p_projs = thetas @ P.T              # (n_proj, n_P)
    q_projs = thetas @ Q.T              # (n_proj, n_Q)

    ref_score_accum = np.zeros(n)
    test_score_accum = np.zeros(m)
    n_valid = 0

    for i in range(n_projections):
        p_proj = p_projs[i]
        q_proj = q_projs[i]

        if p_proj.std() < 1e-12 or q_proj.std() < 1e-12:
            continue

        if method == 'histogram':
            ref_scores_i = _histogram_density_ratio(
                p_proj, q_proj, ref_projs[i], bins=bins)
            test_scores_i = _histogram_density_ratio(
                p_proj, q_proj, test_projs[i], bins=bins)
        else:  # kde
            eps = 1e-10
            try:
                kde_p = gaussian_kde(p_proj, bw_method=bandwidth)
                kde_q = gaussian_kde(q_proj, bw_method=bandwidth)
            except np.linalg.LinAlgError:
                continue

            p_hat_ref = kde_p(ref_projs[i]) + eps
            q_hat_ref = kde_q(ref_projs[i]) + eps
            ref_scores_i = p_hat_ref / (p_hat_ref + q_hat_ref)

            p_hat_test = kde_p(test_projs[i]) + eps
            q_hat_test = kde_q(test_projs[i]) + eps
            test_scores_i = p_hat_test / (p_hat_test + q_hat_test)

        ref_score_accum += ref_scores_i
        test_score_accum += test_scores_i
        n_valid += 1

    if n_valid == 0:
        return np.full(n, 0.5), np.full(m, 0.5)

    ref_scores = ref_score_accum / n_valid
    test_scores = test_score_accum / n_valid
    return ref_scores, test_scores


# ============================================================================
# Method 2: Neural Witness
# ============================================================================

class WitnessNetwork(nn.Module):
    """
    Small MLP with tanh output for variational TV estimation.

    Output in [-1, 1], satisfying the witness constraint ||f||_inf <= 1.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def neural_witness_correctness_scores(
    ref_logits: np.ndarray,
    ref_correct: np.ndarray,
    test_logits: np.ndarray,
    hidden_dims: List[int] = [128, 64],
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    patience: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Per-sample P(correct|x) via neural witness (variational TV).

    Trains an MLP f: R^d -> [-1,1] to maximize E_P[f(x)] - E_Q[f(x)]
    (the variational representation of TV distance).

    Converts witness output to correctness score: score = (f(x) + 1) / 2.

    Args:
        ref_logits: (n, dim) reference logits
        ref_correct: (n,) boolean correctness labels
        test_logits: (m, dim) test logits
        hidden_dims: hidden layer sizes for the witness MLP
        epochs: maximum training epochs
        lr: learning rate
        weight_decay: L2 regularization
        batch_size: mini-batch size (sampled equally from P and Q)
        patience: early stopping patience (epochs without improvement)
        seed: random seed
        verbose: print training progress

    Returns:
        ref_scores: (n,) estimated P(correct|x)
        test_scores: (m,) estimated P(correct|x)
        info: dict with training metadata (losses, final_tv_estimate, etc.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ref_logits = np.atleast_2d(ref_logits).astype(np.float32)
    test_logits = np.atleast_2d(test_logits).astype(np.float32)
    ref_correct = np.asarray(ref_correct).astype(bool)

    n, dim = ref_logits.shape

    # Standardize using ref statistics
    mean = ref_logits.mean(axis=0)
    std = ref_logits.std(axis=0) + 1e-8
    ref_normed = (ref_logits - mean) / std
    test_normed = (test_logits - mean) / std

    P = ref_normed[ref_correct]
    Q = ref_normed[~ref_correct]
    n_P, n_Q = len(P), len(Q)

    # Hold out 10% of ref for early stopping
    rng = np.random.RandomState(seed)
    n_val_p = max(1, int(0.1 * n_P))
    n_val_q = max(1, int(0.1 * n_Q))
    perm_p = rng.permutation(n_P)
    perm_q = rng.permutation(n_Q)

    P_val, P_train = P[perm_p[:n_val_p]], P[perm_p[n_val_p:]]
    Q_val, Q_train = Q[perm_q[:n_val_q]], Q[perm_q[n_val_q:]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P_train_t = torch.from_numpy(P_train).to(device)
    Q_train_t = torch.from_numpy(Q_train).to(device)
    P_val_t = torch.from_numpy(P_val).to(device)
    Q_val_t = torch.from_numpy(Q_val).to(device)

    model = WitnessNetwork(dim, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    n_P_train = len(P_train_t)
    n_Q_train = len(Q_train_t)
    half_batch = batch_size // 2

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        # Number of mini-batches per epoch
        n_batches = max(1, max(n_P_train, n_Q_train) // half_batch)

        for _ in range(n_batches):
            # Sample equal-sized mini-batches
            idx_p = torch.randint(0, n_P_train, (half_batch,), device=device)
            idx_q = torch.randint(0, n_Q_train, (half_batch,), device=device)

            f_p = model(P_train_t[idx_p])
            f_q = model(Q_train_t[idx_q])

            # Maximize E_P[f] - E_Q[f]  =>  minimize -(E_P[f] - E_Q[f])
            loss = -f_p.mean() + f_q.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            f_p_val = model(P_val_t)
            f_q_val = model(Q_val_t)
            val_loss = (-f_p_val.mean() + f_q_val.mean()).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 50 == 0:
            tv_est = 0.5 * (-val_loss)
            print(f"  Epoch {epoch+1:4d}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  TV_est={tv_est:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Score all samples
    ref_t = torch.from_numpy(ref_normed).to(device)
    test_t = torch.from_numpy(test_normed).to(device)

    with torch.no_grad():
        ref_f = model(ref_t).cpu().numpy()
        test_f = model(test_t).cpu().numpy()

    # Convert [-1, 1] -> [0, 1]: higher = more "P-like" = more likely correct
    ref_scores = (ref_f + 1.0) / 2.0
    test_scores = (test_f + 1.0) / 2.0

    # TV estimate from validation set
    with torch.no_grad():
        tv_estimate = 0.5 * (model(P_val_t).mean() - model(Q_val_t).mean()).item()

    info = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': len(train_losses) - epochs_no_improve,
        'tv_estimate': tv_estimate,
        'n_params': sum(p.numel() for p in model.parameters()),
    }

    return ref_scores, test_scores, info


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    dim = 62

    # Synthetic data: correct samples are centered, incorrect are shifted
    n_correct = 3000
    n_incorrect = 1000
    n_test = 500

    P = np.random.randn(n_correct, dim) * 0.5
    Q = np.random.randn(n_incorrect, dim) * 0.5 + 0.3

    ref_logits = np.vstack([P, Q])
    ref_correct = np.array([True] * n_correct + [False] * n_incorrect)
    test_logits = np.random.randn(n_test, dim) * 0.5 + 0.15

    print("=== Sliced TV Distance ===")
    mean_tv, max_tv = sliced_tv_distance(P, Q, n_projections=200)
    print(f"Mean TV = {mean_tv:.4f}, Max TV = {max_tv:.4f}")

    print("\n=== Sliced TV Correctness Scores ===")
    ref_scores, test_scores = sliced_tv_correctness_scores(
        ref_logits, ref_correct, test_logits, n_projections=200)
    print(f"Ref scores: mean={ref_scores.mean():.4f}, std={ref_scores.std():.4f}")
    print(f"  Correct mean={ref_scores[ref_correct].mean():.4f}, "
          f"Incorrect mean={ref_scores[~ref_correct].mean():.4f}")
    print(f"Test scores: mean={test_scores.mean():.4f}, std={test_scores.std():.4f}")

    print("\n=== Neural Witness Correctness Scores ===")
    ref_scores_nw, test_scores_nw, info = neural_witness_correctness_scores(
        ref_logits, ref_correct, test_logits,
        hidden_dims=[128, 64], epochs=100, verbose=True)
    print(f"Ref scores: mean={ref_scores_nw.mean():.4f}, std={ref_scores_nw.std():.4f}")
    print(f"  Correct mean={ref_scores_nw[ref_correct].mean():.4f}, "
          f"Incorrect mean={ref_scores_nw[~ref_correct].mean():.4f}")
    print(f"Test scores: mean={test_scores_nw.mean():.4f}")
    print(f"TV estimate: {info['tv_estimate']:.4f}")
    print(f"Params: {info['n_params']}")
    print(f"Best epoch: {info['best_epoch']}")
