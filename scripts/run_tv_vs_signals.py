"""
Experiment: TV Density Ratio Correctness Predictor vs 12-Signal Suitability Filter

The TV framework splits logits into two distributions — P (correct) and Q
(incorrect) — and uses kNN density ratios to estimate P(correct|x) per sample.
The kNN is just the approximation tool for the density ratio; the conceptual
framework is Total Variation distance.

Experiment 1 (Method Comparison):
  Compare methods with proper fold-internal CV:
    1. Paper's 12 signals + LR
    2. TV density ratio (raw score, no fitting)
    3. TV density ratio + LR (calibration)
  Also sweep k for the raw TV score.

Experiment 2 (Grid Search):
  Systematic search over k, PCA, intrinsic dimensionality.
  All CV builds the density ratio inside each fold (no leakage).

Usage:
    python scripts/run_tv_vs_signals.py
"""

import sys
import json
import pickle
import itertools
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project imports
PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.tv_distance import tv_correctness_scores
from src.synthetic_holdout import compute_sf_features_from_logits


# ============================================================================
# Utilities
# ============================================================================

def load_logits(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_eval_lr(X_train, y_train, X_test, y_test):
    """Train LR, evaluate on test. Returns dict with auc, acc, brier, proba."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_s, y_train)
    proba = model.predict_proba(X_test_s)[:, 1]
    return {
        'auc': roc_auc_score(y_test, proba),
        'acc': ((proba >= 0.5) == y_test).mean(),
        'brier': brier_score_loss(y_test, proba),
        'proba': proba,
    }


def cross_val_auc_precomputed(X, y, n_folds=5):
    """5-fold CV AUC for pre-computed features (no leakage concern)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        result = train_eval_lr(X[train_idx], y[train_idx],
                               X[val_idx], y[val_idx])
        aucs.append(result['auc'])
    return np.mean(aucs), np.std(aucs)


def cross_val_tv(logits, correct, k, n_folds=5, d=None,
                 use_pca=False, pca_components=None, auto_intrinsic_dim=True):
    """Proper CV: build TV density ratio INSIDE each fold.

    Returns mean_auc, std_auc for raw TV score (no LR).
    Also returns mean_auc_lr, std_auc_lr for TV score + LR.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs_raw = []
    aucs_lr = []
    briers_raw = []

    for train_idx, val_idx in skf.split(logits, correct):
        train_logits = logits[train_idx]
        train_correct = correct[train_idx]
        val_logits = logits[val_idx]
        val_correct = correct[val_idx]

        # Build density ratio on train fold, query val fold
        _, val_scores = tv_correctness_scores(
            train_logits, train_correct, val_logits, k=k, d=d,
            use_pca=use_pca, pca_components=pca_components,
            auto_intrinsic_dim=auto_intrinsic_dim)

        # Raw TV score — already estimates P(correct|x)
        auc_raw = roc_auc_score(val_correct, val_scores)
        brier_raw = brier_score_loss(val_correct, np.clip(val_scores, 0, 1))
        aucs_raw.append(auc_raw)
        briers_raw.append(brier_raw)

        # TV score + LR (calibration)
        train_scores, _ = tv_correctness_scores(
            train_logits, train_correct, train_logits, k=k, d=d,
            use_pca=use_pca, pca_components=pca_components,
            auto_intrinsic_dim=auto_intrinsic_dim)
        lr_res = train_eval_lr(
            train_scores.reshape(-1, 1), train_correct,
            val_scores.reshape(-1, 1), val_correct)
        aucs_lr.append(lr_res['auc'])

    return {
        'raw_auc': np.mean(aucs_raw), 'raw_std': np.std(aucs_raw),
        'raw_brier': np.mean(briers_raw),
        'lr_auc': np.mean(aucs_lr), 'lr_std': np.std(aucs_lr),
    }


# ============================================================================
# Experiment 1: Method Comparison
# ============================================================================

def experiment1(id_logits, id_correct, ood_logits, ood_correct):
    """Compare TV density ratio vs 12 signals, with proper CV."""
    print("=" * 70)
    print("EXPERIMENT 1: TV Density Ratio vs 12 Signals (Proper CV)")
    print("=" * 70)

    results = {}

    # ---- 1. Paper's method: 12 signals + LR ----
    print("\n[1] Paper's method: 12 signals + LR...")
    id_features = compute_sf_features_from_logits(id_logits)
    ood_features = compute_sf_features_from_logits(ood_logits)

    cv_auc, cv_std = cross_val_auc_precomputed(id_features, id_correct)
    ood_res = train_eval_lr(id_features, id_correct, ood_features, ood_correct)

    results['12_signals_lr'] = {
        'cv_auc': float(cv_auc), 'cv_std': float(cv_std),
        'ood_auc': float(ood_res['auc']), 'ood_acc': float(ood_res['acc']),
        'ood_brier': float(ood_res['brier']),
    }
    print(f"  CV AUC  = {cv_auc:.4f} +/- {cv_std:.4f}")
    print(f"  OOD AUC = {ood_res['auc']:.4f}  |  Brier = {ood_res['brier']:.4f}")

    # ---- 2 & 3. TV density ratio (raw + LR) for various k ----
    print("\n[2/3] TV density ratio for various k...")
    k_values = [3, 5, 10, 20, 50, 100]

    for k in k_values:
        # Proper CV
        cv = cross_val_tv(id_logits, id_correct, k=k)

        # OOD evaluation (train on full id_test)
        ref_scores, ood_scores = tv_correctness_scores(
            id_logits, id_correct, ood_logits, k=k)

        ood_auc_raw = roc_auc_score(ood_correct, ood_scores)
        ood_brier_raw = brier_score_loss(ood_correct, np.clip(ood_scores, 0, 1))
        ood_acc_raw = ((ood_scores >= 0.5) == ood_correct).mean()

        # TV + LR for OOD
        ood_lr = train_eval_lr(
            ref_scores.reshape(-1, 1), id_correct,
            ood_scores.reshape(-1, 1), ood_correct)

        results[f'tv_k{k}_raw'] = {
            'k': k,
            'cv_auc': float(cv['raw_auc']), 'cv_std': float(cv['raw_std']),
            'cv_brier': float(cv['raw_brier']),
            'ood_auc': float(ood_auc_raw), 'ood_acc': float(ood_acc_raw),
            'ood_brier': float(ood_brier_raw),
        }
        results[f'tv_k{k}_lr'] = {
            'k': k,
            'cv_auc': float(cv['lr_auc']), 'cv_std': float(cv['lr_std']),
            'ood_auc': float(ood_lr['auc']), 'ood_acc': float(ood_lr['acc']),
            'ood_brier': float(ood_lr['brier']),
        }

        print(f"  k={k:3d}  raw: CV={cv['raw_auc']:.4f}  OOD={ood_auc_raw:.4f}  Brier={ood_brier_raw:.4f}")
        print(f"  k={k:3d}  +LR: CV={cv['lr_auc']:.4f}  OOD={ood_lr['auc']:.4f}  Brier={ood_lr['brier']:.4f}")

    # ---- Comparison table ----
    print("\n" + "=" * 80)
    print(f"{'Method':<35} {'CV AUC':>10} {'OOD AUC':>10} {'Brier':>10}")
    print("-" * 80)
    r = results['12_signals_lr']
    print(f"{'12 signals + LR (paper)':<35} {r['cv_auc']:>10.4f} {r['ood_auc']:>10.4f} {r['ood_brier']:>10.4f}")
    print("-" * 80)
    for k in k_values:
        r_raw = results[f'tv_k{k}_raw']
        r_lr = results[f'tv_k{k}_lr']
        print(f"{'TV ratio (k=' + str(k) + ') raw':<35} {r_raw['cv_auc']:>10.4f} {r_raw['ood_auc']:>10.4f} {r_raw['ood_brier']:>10.4f}")
        print(f"{'TV ratio (k=' + str(k) + ') + LR':<35} {r_lr['cv_auc']:>10.4f} {r_lr['ood_auc']:>10.4f} {r_lr['ood_brier']:>10.4f}")
    print("=" * 80)

    return results


# ============================================================================
# Experiment 2: Grid Search
# ============================================================================

def experiment2(id_logits, id_correct, ood_logits, ood_correct):
    """Grid search over k, PCA, and intrinsic dimensionality."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Grid Search Over TV Density Ratio Parameters")
    print("=" * 70)

    k_values = [3, 5, 10, 20, 50, 100]
    pca_values = [None, 10, 20, 30]
    d_values = [None, 5, 10, 20]  # None = auto estimate

    grid_results = []
    best_cv_auc = -1
    best_config = None
    total = len(k_values) * len(pca_values) * len(d_values)
    i = 0

    for k, pca_comp, d_val in itertools.product(k_values, pca_values, d_values):
        i += 1
        use_pca = pca_comp is not None
        auto_dim = d_val is None

        # Proper CV
        cv = cross_val_tv(id_logits, id_correct, k=k, d=d_val,
                          use_pca=use_pca, pca_components=pca_comp,
                          auto_intrinsic_dim=auto_dim)

        # OOD eval
        ref_scores, ood_scores = tv_correctness_scores(
            id_logits, id_correct, ood_logits, k=k, d=d_val,
            use_pca=use_pca, pca_components=pca_comp,
            auto_intrinsic_dim=auto_dim)

        ood_auc = roc_auc_score(ood_correct, ood_scores)
        ood_brier = brier_score_loss(ood_correct, np.clip(ood_scores, 0, 1))

        config = {
            'k': k,
            'pca': pca_comp,
            'd': d_val if d_val is not None else 'auto',
            'cv_auc': float(cv['raw_auc']),
            'cv_std': float(cv['raw_std']),
            'cv_brier': float(cv['raw_brier']),
            'ood_auc': float(ood_auc),
            'ood_brier': float(ood_brier),
        }
        grid_results.append(config)

        if cv['raw_auc'] > best_cv_auc:
            best_cv_auc = cv['raw_auc']
            best_config = config

        if i % 12 == 0 or i == total:
            d_str = f"{d_val}" if d_val is not None else "auto"
            print(f"  [{i:3d}/{total}] k={k:3d} pca={str(pca_comp):>4s} d={d_str:>4s} "
                  f"CV={cv['raw_auc']:.4f}  OOD={ood_auc:.4f}")

    # Sort by CV AUC
    grid_results.sort(key=lambda x: x['cv_auc'], reverse=True)

    print(f"\n{'='*70}")
    print("Top 15 Configurations (by CV AUC):")
    print(f"{'='*70}")
    print(f"{'k':>5} {'PCA':>5} {'d':>5} {'CV AUC':>8} {'CV Std':>8} {'OOD AUC':>8} {'Brier':>8}")
    print("-" * 55)
    for r in grid_results[:15]:
        print(f"{r['k']:>5} {str(r['pca']):>5} {str(r['d']):>5} "
              f"{r['cv_auc']:>8.4f} {r['cv_std']:>8.4f} {r['ood_auc']:>8.4f} {r['ood_brier']:>8.4f}")

    print(f"\nBest config: k={best_config['k']}, pca={best_config['pca']}, "
          f"d={best_config['d']}")
    print(f"  CV AUC:  {best_config['cv_auc']:.4f} +/- {best_config['cv_std']:.4f}")
    print(f"  OOD AUC: {best_config['ood_auc']:.4f}")

    return {'grid_results': grid_results, 'best_config': best_config}


# ============================================================================
# Visualization
# ============================================================================

def make_figure(exp1_results, exp2_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Experiment 1 — TV score vs k ---
    ax = axes[0]
    k_values = [3, 5, 10, 20, 50, 100]

    raw_ood = [exp1_results[f'tv_k{k}_raw']['ood_auc'] for k in k_values]
    lr_ood = [exp1_results[f'tv_k{k}_lr']['ood_auc'] for k in k_values]
    raw_cv = [exp1_results[f'tv_k{k}_raw']['cv_auc'] for k in k_values]

    ax.plot(k_values, raw_ood, 'o-', label='TV raw (OOD)', color='#ff7f0e', linewidth=2)
    ax.plot(k_values, lr_ood, 's--', label='TV + LR (OOD)', color='#2ca02c', linewidth=1.5)
    ax.plot(k_values, raw_cv, '^:', label='TV raw (CV)', color='#1f77b4', linewidth=1.5, alpha=0.7)

    paper_auc = exp1_results['12_signals_lr']['ood_auc']
    ax.axhline(y=paper_auc, color='red', linestyle='--', alpha=0.7,
               label=f'12 signals + LR = {paper_auc:.3f}')

    ax.set_xlabel('k (neighbors)')
    ax.set_ylabel('AUC')
    ax.set_title('TV Density Ratio: AUC vs k')
    ax.set_xscale('log')
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)
    ax.legend(fontsize=8)
    ax.set_ylim(0.65, 0.90)
    ax.grid(True, alpha=0.3)

    # --- Right: Experiment 2 heatmap (k vs d, no PCA) ---
    ax = axes[1]
    grid = exp2_results['grid_results']

    no_pca = [r for r in grid if r['pca'] is None]
    d_vals = ['auto', 5, 10, 20]
    d_labels = ['auto', '5', '10', '20']
    k_vals = sorted(set(r['k'] for r in no_pca))

    heatmap = np.full((len(k_vals), len(d_vals)), np.nan)
    for r in no_pca:
        if r['d'] in d_vals:
            ki = k_vals.index(r['k'])
            di = d_vals.index(r['d'])
            heatmap[ki, di] = r['ood_auc']

    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0.65, vmax=0.90)
    ax.set_xticks(range(len(d_vals)))
    ax.set_xticklabels(d_labels)
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels(k_vals)
    ax.set_xlabel('Intrinsic dim (d)')
    ax.set_ylabel('k (neighbors)')
    ax.set_title('Grid Search OOD AUC (no PCA)')

    for i in range(len(k_vals)):
        for j in range(len(d_vals)):
            v = heatmap[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if v > 0.82 else 'black')

    fig.colorbar(im, ax=ax, label='OOD AUC')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    logits_path = PROJ_ROOT / 'results' / 'logits' / 'fmow_ERM_best_0_main.pkl'
    results_path = PROJ_ROOT / 'results' / 'tv_vs_signals_results.json'
    figure_path = PROJ_ROOT / 'results' / 'figures' / 'tv_vs_signals.png'

    print(f"Loading logits from {logits_path}...")
    data = load_logits(logits_path)

    id_logits = data['id_test']['logits']
    id_correct = data['id_test']['correct']
    ood_logits = data['test']['logits']
    ood_correct = data['test']['correct']

    print(f"ID test:  {id_logits.shape[0]} samples, {id_logits.shape[1]}-D logits, "
          f"accuracy={id_correct.mean():.4f}")
    print(f"OOD test: {ood_logits.shape[0]} samples, accuracy={ood_correct.mean():.4f}")

    exp1 = experiment1(id_logits, id_correct, ood_logits, ood_correct)
    exp2 = experiment2(id_logits, id_correct, ood_logits, ood_correct)

    all_results = {'experiment1': exp1, 'experiment2': exp2}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    make_figure(exp1, exp2, figure_path)

    # Summary
    best = exp2['best_config']
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<40} {'OOD AUC':>10}")
    print("-" * 52)
    print(f"{'12 signals + LR (paper)':<40} {exp1['12_signals_lr']['ood_auc']:>10.4f}")
    for k in [10, 20, 50]:
        if f'tv_k{k}_raw' in exp1:
            print(f"{'TV ratio (k=' + str(k) + ') raw':<40} {exp1[f'tv_k{k}_raw']['ood_auc']:>10.4f}")
    print(f"{'Best grid search':<40} {best['ood_auc']:>10.4f}")
    print(f"  Config: k={best['k']}, pca={best['pca']}, d={best['d']}")


if __name__ == '__main__':
    main()
