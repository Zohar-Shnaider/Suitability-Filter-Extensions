"""
Experiment: Alternative TV Approximations — Sliced TV and Neural Witness

Compares against baselines from run_tv_vs_signals.py:
  1. Paper's 12 signals + LR  (OOD AUC ~0.820)
  2. kNN density ratio k=10   (OOD AUC ~0.813)

New methods:
  3. Sliced TV density ratio (sweep n_projections, bandwidth)
  4. Neural witness MLP      (sweep architecture, epochs, weight_decay)

All methods use fold-internal CV on id_test, then full id_test -> ood_test eval.

Usage:
    python scripts/run_alternative_tv.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.alternative_tv import (
    sliced_tv_correctness_scores,
    sliced_tv_distance,
    neural_witness_correctness_scores,
)


# ============================================================================
# Utilities
# ============================================================================

def load_logits(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def cross_val_sliced(logits, correct, n_folds=5, **kwargs):
    """CV for sliced TV: build density ratio inside each fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    briers = []

    for train_idx, val_idx in skf.split(logits, correct):
        _, val_scores = sliced_tv_correctness_scores(
            logits[train_idx], correct[train_idx],
            logits[val_idx], **kwargs)
        aucs.append(roc_auc_score(correct[val_idx], val_scores))
        briers.append(brier_score_loss(correct[val_idx],
                                       np.clip(val_scores, 0, 1)))

    return {
        'cv_auc': float(np.mean(aucs)),
        'cv_std': float(np.std(aucs)),
        'cv_brier': float(np.mean(briers)),
    }


def cross_val_witness(logits, correct, n_folds=5, **kwargs):
    """CV for neural witness: train fresh model inside each fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    briers = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(logits, correct)):
        # Use different seed per fold for diversity
        fold_kwargs = dict(kwargs)
        fold_kwargs['seed'] = 42 + fold_i

        _, val_scores, _ = neural_witness_correctness_scores(
            logits[train_idx], correct[train_idx],
            logits[val_idx], **fold_kwargs)
        aucs.append(roc_auc_score(correct[val_idx], val_scores))
        briers.append(brier_score_loss(correct[val_idx],
                                       np.clip(val_scores, 0, 1)))

    return {
        'cv_auc': float(np.mean(aucs)),
        'cv_std': float(np.std(aucs)),
        'cv_brier': float(np.mean(briers)),
    }


# ============================================================================
# Experiment 1: Sliced TV Sweep
# ============================================================================

def experiment_sliced(id_logits, id_correct, ood_logits, ood_correct):
    """Sweep over n_projections, bins, and method for sliced TV."""
    print("=" * 70)
    print("SLICED TV: Hyperparameter Sweep")
    print("=" * 70)

    results = {}
    best_cv_auc = -1
    best_key = None

    # Configs: (n_projections, method, bins_or_bw)
    configs = [
        # Histogram method — fast, sweep projections and bins
        (50, 'histogram', 50),
        (100, 'histogram', 50),
        (200, 'histogram', 50),
        (500, 'histogram', 50),
        (1000, 'histogram', 50),
        (200, 'histogram', 100),
        (500, 'histogram', 100),
        (1000, 'histogram', 100),
        (500, 'histogram', 200),
        # KDE method — slower, fewer configs
        (100, 'kde', 'scott'),
        (200, 'kde', 'scott'),
        (100, 'kde', 'silverman'),
    ]
    total = len(configs)

    for i, (n_proj, method, param) in enumerate(configs, 1):
        if method == 'histogram':
            key = f'sliced_proj{n_proj}_{method}_bins{param}'
            kwargs = dict(n_projections=n_proj, method=method, bins=param)
            label = f"proj={n_proj}, hist bins={param}"
        else:
            key = f'sliced_proj{n_proj}_{method}_bw{param}'
            kwargs = dict(n_projections=n_proj, method=method, bandwidth=param)
            label = f"proj={n_proj}, kde bw={param}"

        print(f"\n  [{i}/{total}] {label}...")

        # CV
        cv = cross_val_sliced(id_logits, id_correct, **kwargs)

        # OOD
        ref_scores, ood_scores = sliced_tv_correctness_scores(
            id_logits, id_correct, ood_logits, **kwargs)

        ood_auc = roc_auc_score(ood_correct, ood_scores)
        ood_brier = brier_score_loss(ood_correct,
                                     np.clip(ood_scores, 0, 1))
        ood_acc = ((ood_scores >= 0.5) == ood_correct).mean()

        # Sliced TV distance (P=correct vs Q=incorrect from id)
        P_id = id_logits[id_correct]
        Q_id = id_logits[~id_correct]
        mean_tv, max_tv = sliced_tv_distance(P_id, Q_id,
                                              n_projections=n_proj)

        results[key] = {
            'n_projections': n_proj,
            'method': method,
            'bins_or_bw': param,
            'cv_auc': cv['cv_auc'],
            'cv_std': cv['cv_std'],
            'cv_brier': cv['cv_brier'],
            'ood_auc': float(ood_auc),
            'ood_acc': float(ood_acc),
            'ood_brier': float(ood_brier),
            'mean_tv': float(mean_tv),
            'max_tv': float(max_tv),
        }

        if cv['cv_auc'] > best_cv_auc:
            best_cv_auc = cv['cv_auc']
            best_key = key

        print(f"    CV AUC={cv['cv_auc']:.4f} +/- {cv['cv_std']:.4f}  "
              f"OOD AUC={ood_auc:.4f}  Brier={ood_brier:.4f}  "
              f"TV(mean={mean_tv:.4f}, max={max_tv:.4f})")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Config':<40} {'CV AUC':>8} {'OOD AUC':>8} {'Brier':>8}")
    print("-" * 70)
    sorted_keys = sorted([k for k in results if not k.startswith('_')],
                         key=lambda k: results[k]['cv_auc'], reverse=True)
    for key in sorted_keys:
        r = results[key]
        if r['method'] == 'histogram':
            label = f"proj={r['n_projections']}, hist bins={r['bins_or_bw']}"
        else:
            label = f"proj={r['n_projections']}, kde bw={r['bins_or_bw']}"
        marker = " <--" if key == best_key else ""
        print(f"{label:<40} {r['cv_auc']:>8.4f} {r['ood_auc']:>8.4f} "
              f"{r['ood_brier']:>8.4f}{marker}")
    print("=" * 80)

    results['_best'] = best_key
    return results


# ============================================================================
# Experiment 2: Neural Witness Sweep
# ============================================================================

def experiment_witness(id_logits, id_correct, ood_logits, ood_correct):
    """Sweep over architecture, epochs, and weight_decay for neural witness."""
    print("\n" + "=" * 70)
    print("NEURAL WITNESS: Hyperparameter Sweep")
    print("=" * 70)

    results = {}
    hidden_configs = [[64], [128, 64], [256, 128]]
    epoch_values = [100, 200, 500]
    wd_values = [0, 1e-4, 1e-3]

    best_cv_auc = -1
    best_key = None
    total = len(hidden_configs) * len(epoch_values) * len(wd_values)
    i = 0

    for hidden in hidden_configs:
        for ep in epoch_values:
            for wd in wd_values:
                i += 1
                h_str = 'x'.join(map(str, hidden))
                key = f'witness_h{h_str}_ep{ep}_wd{wd}'
                print(f"\n  [{i}/{total}] hidden={hidden}, epochs={ep}, wd={wd}...")

                # CV
                cv = cross_val_witness(id_logits, id_correct,
                                       hidden_dims=hidden, epochs=ep,
                                       weight_decay=wd, verbose=False)

                # OOD
                ref_scores, ood_scores, info = neural_witness_correctness_scores(
                    id_logits, id_correct, ood_logits,
                    hidden_dims=hidden, epochs=ep,
                    weight_decay=wd, verbose=False)

                ood_auc = roc_auc_score(ood_correct, ood_scores)
                ood_brier = brier_score_loss(ood_correct,
                                             np.clip(ood_scores, 0, 1))
                ood_acc = ((ood_scores >= 0.5) == ood_correct).mean()

                results[key] = {
                    'hidden_dims': hidden,
                    'epochs': ep,
                    'weight_decay': wd,
                    'cv_auc': cv['cv_auc'],
                    'cv_std': cv['cv_std'],
                    'cv_brier': cv['cv_brier'],
                    'ood_auc': float(ood_auc),
                    'ood_acc': float(ood_acc),
                    'ood_brier': float(ood_brier),
                    'tv_estimate': info['tv_estimate'],
                    'n_params': info['n_params'],
                    'best_epoch': info['best_epoch'],
                }

                if cv['cv_auc'] > best_cv_auc:
                    best_cv_auc = cv['cv_auc']
                    best_key = key

                print(f"    CV AUC={cv['cv_auc']:.4f} +/- {cv['cv_std']:.4f}  "
                      f"OOD AUC={ood_auc:.4f}  Brier={ood_brier:.4f}  "
                      f"TV_est={info['tv_estimate']:.4f}  "
                      f"params={info['n_params']}  stopped@{info['best_epoch']}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Config':<45} {'CV AUC':>8} {'OOD AUC':>8} {'Brier':>8} {'Params':>7}")
    print("-" * 80)
    sorted_keys = sorted([k for k in results if not k.startswith('_')],
                         key=lambda k: results[k]['cv_auc'], reverse=True)
    for key in sorted_keys:
        r = results[key]
        h_str = 'x'.join(map(str, r['hidden_dims']))
        label = f"h={h_str}, ep={r['epochs']}, wd={r['weight_decay']}"
        marker = " <--" if key == best_key else ""
        print(f"{label:<45} {r['cv_auc']:>8.4f} {r['ood_auc']:>8.4f} "
              f"{r['ood_brier']:>8.4f} {r['n_params']:>7d}{marker}")
    print("=" * 90)

    results['_best'] = best_key
    return results


# ============================================================================
# Comparison Figure
# ============================================================================

def make_figure(sliced_results, witness_results, baselines, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Sliced TV — n_projections vs AUC ---
    ax = axes[0]
    for label_filter, label_name in [('histogram_bins50', 'hist b=50'),
                                      ('histogram_bins100', 'hist b=100'),
                                      ('kde_bwscott', 'kde scott')]:
        keys = [k for k in sliced_results
                if not k.startswith('_') and label_filter in k]
        keys.sort(key=lambda k: sliced_results[k]['n_projections'])
        if not keys:
            continue
        n_projs = [sliced_results[k]['n_projections'] for k in keys]
        cv_aucs = [sliced_results[k]['cv_auc'] for k in keys]
        ood_aucs = [sliced_results[k]['ood_auc'] for k in keys]

        ax.plot(n_projs, ood_aucs, 'o-', label=f'OOD ({label_name})',
                linewidth=2)
        ax.plot(n_projs, cv_aucs, 's--', label=f'CV ({label_name})',
                linewidth=1.5, alpha=0.7)

    ax.axhline(y=baselines['paper_ood_auc'], color='red', linestyle='--',
               alpha=0.7, label=f"12 signals = {baselines['paper_ood_auc']:.3f}")
    ax.axhline(y=baselines['knn_ood_auc'], color='blue', linestyle=':',
               alpha=0.7, label=f"kNN k=10 = {baselines['knn_ood_auc']:.3f}")

    ax.set_xlabel('Number of Projections')
    ax.set_ylabel('AUC')
    ax.set_title('Sliced TV: AUC vs Projections')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.60, 0.90)

    # --- Panel 2: Neural Witness — architecture comparison ---
    ax = axes[1]
    hidden_configs = [[64], [128, 64], [256, 128]]

    for hidden in hidden_configs:
        h_str = 'x'.join(map(str, hidden))
        # Pick wd=1e-4 configs for this architecture
        keys = [k for k in witness_results
                if not k.startswith('_') and f'h{h_str}_' in k
                and '_wd0.0001' in k]
        if not keys:
            # Fall back to any wd
            all_keys = [k for k in witness_results
                        if not k.startswith('_') and f'h{h_str}_' in k]
            # Group by epochs, pick one wd per epoch
            seen_epochs = set()
            keys = []
            for k in sorted(all_keys):
                ep = witness_results[k]['epochs']
                if ep not in seen_epochs:
                    keys.append(k)
                    seen_epochs.add(ep)

        keys.sort(key=lambda k: witness_results[k]['epochs'])
        if keys:
            epochs = [witness_results[k]['epochs'] for k in keys]
            ood_aucs = [witness_results[k]['ood_auc'] for k in keys]
            ax.plot(epochs, ood_aucs, 'o-', label=f'h={h_str}', linewidth=2)

    ax.axhline(y=baselines['paper_ood_auc'], color='red', linestyle='--',
               alpha=0.7, label=f"12 signals = {baselines['paper_ood_auc']:.3f}")
    ax.axhline(y=baselines['knn_ood_auc'], color='blue', linestyle=':',
               alpha=0.7, label=f"kNN k=10 = {baselines['knn_ood_auc']:.3f}")

    ax.set_xlabel('Epochs')
    ax.set_ylabel('OOD AUC')
    ax.set_title('Neural Witness: OOD AUC vs Epochs (wd=1e-4)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.60, 0.90)

    # --- Panel 3: Final comparison bar chart ---
    ax = axes[2]

    methods = ['12 signals + LR\n(paper)']
    aucs = [baselines['paper_ood_auc']]
    colors = ['#d62728']

    methods.append('kNN density ratio\nk=10')
    aucs.append(baselines['knn_ood_auc'])
    colors.append('#1f77b4')

    # Best sliced
    best_sliced_key = sliced_results.get('_best')
    if best_sliced_key and best_sliced_key in sliced_results:
        r = sliced_results[best_sliced_key]
        methods.append(f"Sliced TV\np={r['n_projections']},{r['method']}")
        aucs.append(r['ood_auc'])
        colors.append('#ff7f0e')

    # Best witness
    best_witness_key = witness_results.get('_best')
    if best_witness_key and best_witness_key in witness_results:
        r = witness_results[best_witness_key]
        h_str = 'x'.join(map(str, r['hidden_dims']))
        methods.append(f"Neural witness\nh={h_str}")
        aucs.append(r['ood_auc'])
        colors.append('#2ca02c')

    bars = ax.bar(range(len(methods)), aucs, color=colors, alpha=0.85)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel('OOD AUC')
    ax.set_title('Method Comparison')
    ax.set_ylim(0.60, 0.90)
    ax.grid(True, alpha=0.3, axis='y')

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
    existing_path = PROJ_ROOT / 'results' / 'tv_vs_signals_results.json'
    results_path = PROJ_ROOT / 'results' / 'alternative_tv_results.json'
    figure_path = PROJ_ROOT / 'results' / 'figures' / 'alternative_tv.png'

    # Load data
    print(f"Loading logits from {logits_path}...")
    data = load_logits(logits_path)

    id_logits = data['id_test']['logits']
    id_correct = data['id_test']['correct']
    ood_logits = data['test']['logits']
    ood_correct = data['test']['correct']

    print(f"ID test:  {id_logits.shape[0]} samples, {id_logits.shape[1]}-D logits, "
          f"accuracy={id_correct.mean():.4f}")
    print(f"OOD test: {ood_logits.shape[0]} samples, accuracy={ood_correct.mean():.4f}")

    # Load existing baselines
    baselines = {}
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        exp1 = existing.get('experiment1', {})
        baselines['paper_cv_auc'] = exp1.get('12_signals_lr', {}).get('cv_auc', 0.846)
        baselines['paper_ood_auc'] = exp1.get('12_signals_lr', {}).get('ood_auc', 0.820)
        baselines['paper_brier'] = exp1.get('12_signals_lr', {}).get('ood_brier', 0.174)
        baselines['knn_cv_auc'] = exp1.get('tv_k10_raw', {}).get('cv_auc', 0.838)
        baselines['knn_ood_auc'] = exp1.get('tv_k10_raw', {}).get('ood_auc', 0.813)
        baselines['knn_brier'] = exp1.get('tv_k10_raw', {}).get('ood_brier', 0.189)
        print(f"\nBaselines loaded from {existing_path}")
    else:
        baselines = {
            'paper_cv_auc': 0.846, 'paper_ood_auc': 0.820, 'paper_brier': 0.174,
            'knn_cv_auc': 0.838, 'knn_ood_auc': 0.813, 'knn_brier': 0.189,
        }
        print("\nUsing hardcoded baselines (no existing results file)")

    print(f"  Paper (12 signals + LR): CV={baselines['paper_cv_auc']:.4f}  "
          f"OOD={baselines['paper_ood_auc']:.4f}")
    print(f"  kNN density ratio k=10:  CV={baselines['knn_cv_auc']:.4f}  "
          f"OOD={baselines['knn_ood_auc']:.4f}")

    # Run experiments
    sliced_results = experiment_sliced(id_logits, id_correct,
                                       ood_logits, ood_correct)
    witness_results = experiment_witness(id_logits, id_correct,
                                         ood_logits, ood_correct)

    # ---- Final comparison table ----
    print("\n" + "=" * 85)
    print("FINAL COMPARISON")
    print("=" * 85)
    print(f"{'Method':<45} {'CV AUC':>8} {'OOD AUC':>8} {'Brier':>8} {'Params':>7}")
    print("-" * 85)

    print(f"{'12 signals + LR (paper)':<45} "
          f"{baselines['paper_cv_auc']:>8.4f} "
          f"{baselines['paper_ood_auc']:>8.4f} "
          f"{baselines['paper_brier']:>8.4f} "
          f"{'13':>7}")

    print(f"{'kNN density ratio k=10':<45} "
          f"{baselines['knn_cv_auc']:>8.4f} "
          f"{baselines['knn_ood_auc']:>8.4f} "
          f"{baselines['knn_brier']:>8.4f} "
          f"{'0':>7}")

    best_sliced_key = sliced_results.get('_best')
    if best_sliced_key and best_sliced_key in sliced_results:
        r = sliced_results[best_sliced_key]
        label = f"Sliced TV (proj={r['n_projections']}, {r['method']})"
        print(f"{label:<45} {r['cv_auc']:>8.4f} {r['ood_auc']:>8.4f} "
              f"{r['ood_brier']:>8.4f} {'0':>7}")

    best_witness_key = witness_results.get('_best')
    if best_witness_key and best_witness_key in witness_results:
        r = witness_results[best_witness_key]
        h_str = 'x'.join(map(str, r['hidden_dims']))
        label = f"Neural witness (h={h_str}, ep={r['epochs']}, wd={r['weight_decay']})"
        print(f"{label:<45} {r['cv_auc']:>8.4f} {r['ood_auc']:>8.4f} "
              f"{r['ood_brier']:>8.4f} {r['n_params']:>7}")

    print("=" * 85)

    # Save results
    all_results = {
        'baselines': baselines,
        'sliced_tv': {k: v for k, v in sliced_results.items()
                      if not k.startswith('_')},
        'sliced_tv_best': best_sliced_key,
        'neural_witness': {k: v for k, v in witness_results.items()
                           if not k.startswith('_')},
        'neural_witness_best': best_witness_key,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Make figure
    make_figure(sliced_results, witness_results, baselines, figure_path)


if __name__ == '__main__':
    main()
