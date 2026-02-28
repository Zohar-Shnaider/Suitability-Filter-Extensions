"""
Run paper's 12 signals + LR and kNN TV density ratio on 3 different splits.

Split 1 (paper default): id_test -> test       (the original experiment)
Split 2:                  id_val  -> val        (validation sets)
Split 3:                  id_val  -> test       (different ref, same OOD eval)

Reports CV AUC (on ref), OOD AUC, and Brier score for each method x split.

Usage:
    python scripts/run_splits_comparison.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.tv_distance import tv_correctness_scores
from src.synthetic_holdout import compute_sf_features_from_logits


def load_logits(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def cross_val_auc_lr(X, y, n_folds=5):
    """5-fold CV for 12 signals + LR."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, briers = [], []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_tr, y[train_idx])
        proba = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y[val_idx], proba))
        briers.append(brier_score_loss(y[val_idx], proba))
    return np.mean(aucs), np.std(aucs), np.mean(briers)


def cross_val_tv(logits, correct, k=10, n_folds=5):
    """5-fold CV for kNN TV density ratio (built inside each fold)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, briers = [], []
    for train_idx, val_idx in skf.split(logits, correct):
        _, val_scores = tv_correctness_scores(
            logits[train_idx], correct[train_idx],
            logits[val_idx], k=k)
        aucs.append(roc_auc_score(correct[val_idx], val_scores))
        briers.append(brier_score_loss(correct[val_idx],
                                       np.clip(val_scores, 0, 1)))
    return np.mean(aucs), np.std(aucs), np.mean(briers)


def eval_split(ref_logits, ref_correct, ood_logits, ood_correct, split_name):
    """Run both methods on one split."""
    print(f"\n{'='*70}")
    print(f"SPLIT: {split_name}")
    print(f"  Ref:  {ref_logits.shape[0]} samples, acc={ref_correct.mean():.4f}")
    print(f"  OOD:  {ood_logits.shape[0]} samples, acc={ood_correct.mean():.4f}")
    print(f"{'='*70}")

    results = {
        'split': split_name,
        'ref_n': int(ref_logits.shape[0]),
        'ref_acc': float(ref_correct.mean()),
        'ood_n': int(ood_logits.shape[0]),
        'ood_acc': float(ood_correct.mean()),
    }

    # --- 12 signals + LR ---
    print("\n  [1] 12 signals + LR...")
    ref_features = compute_sf_features_from_logits(ref_logits)
    ood_features = compute_sf_features_from_logits(ood_logits)

    cv_auc, cv_std, cv_brier = cross_val_auc_lr(ref_features, ref_correct)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(ref_features)
    X_te = scaler.transform(ood_features)
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_tr, ref_correct)
    proba = model.predict_proba(X_te)[:, 1]
    ood_auc = roc_auc_score(ood_correct, proba)
    ood_brier = brier_score_loss(ood_correct, proba)

    results['signals_lr'] = {
        'cv_auc': float(cv_auc), 'cv_std': float(cv_std),
        'cv_brier': float(cv_brier),
        'ood_auc': float(ood_auc), 'ood_brier': float(ood_brier),
    }
    print(f"      CV AUC  = {cv_auc:.4f} +/- {cv_std:.4f}  Brier = {cv_brier:.4f}")
    print(f"      OOD AUC = {ood_auc:.4f}  Brier = {ood_brier:.4f}")

    # --- kNN TV density ratio for k = 3, 5, 10, 20, 50 ---
    print("\n  [2] kNN TV density ratio...")
    k_values = [3, 5, 10, 20, 50]
    results['knn_tv'] = {}

    for k in k_values:
        cv_auc_k, cv_std_k, cv_brier_k = cross_val_tv(
            ref_logits, ref_correct, k=k)

        ref_scores, ood_scores = tv_correctness_scores(
            ref_logits, ref_correct, ood_logits, k=k)
        ood_auc_k = roc_auc_score(ood_correct, ood_scores)
        ood_brier_k = brier_score_loss(ood_correct,
                                        np.clip(ood_scores, 0, 1))

        results['knn_tv'][f'k={k}'] = {
            'k': k,
            'cv_auc': float(cv_auc_k), 'cv_std': float(cv_std_k),
            'cv_brier': float(cv_brier_k),
            'ood_auc': float(ood_auc_k), 'ood_brier': float(ood_brier_k),
        }
        print(f"      k={k:3d}: CV={cv_auc_k:.4f}+/-{cv_std_k:.4f}  "
              f"OOD={ood_auc_k:.4f}  Brier={ood_brier_k:.4f}")

    return results


def main():
    data_path = PROJ_ROOT / 'results' / 'logits' / 'fmow_ERM_best_0_main.pkl'
    results_path = PROJ_ROOT / 'results' / 'splits_comparison_results.json'

    print(f"Loading data from {data_path}...")
    data = load_logits(data_path)

    splits = [
        ('Split 1: id_test -> test (paper)',
         data['id_test']['logits'], data['id_test']['correct'],
         data['test']['logits'], data['test']['correct']),
        ('Split 2: id_val -> val',
         data['id_val']['logits'], data['id_val']['correct'],
         data['val']['logits'], data['val']['correct']),
        ('Split 3: id_val -> test',
         data['id_val']['logits'], data['id_val']['correct'],
         data['test']['logits'], data['test']['correct']),
    ]

    all_results = []
    for split_name, ref_l, ref_c, ood_l, ood_c in splits:
        r = eval_split(ref_l, ref_c, ood_l, ood_c, split_name)
        all_results.append(r)

    # === Final comparison table ===
    print("\n\n" + "=" * 95)
    print("FINAL COMPARISON ACROSS ALL SPLITS")
    print("=" * 95)

    # Header
    print(f"\n{'Method':<30}", end="")
    for r in all_results:
        short = r['split'].split(':')[0]
        print(f"  {short:>20}", end="")
    print()
    print("-" * 95)

    # 12 signals + LR
    print(f"{'12 signals + LR':<30}", end="")
    for r in all_results:
        s = r['signals_lr']
        print(f"  {s['ood_auc']:.4f} (B={s['ood_brier']:.3f})", end="")
    print()

    print(f"{'  (CV AUC)':<30}", end="")
    for r in all_results:
        s = r['signals_lr']
        print(f"  {s['cv_auc']:.4f}+/-{s['cv_std']:.3f}  ", end="")
    print()

    # kNN TV for each k
    for k in [3, 5, 10, 20, 50]:
        print(f"{f'kNN TV k={k}':<30}", end="")
        for r in all_results:
            s = r['knn_tv'][f'k={k}']
            print(f"  {s['ood_auc']:.4f} (B={s['ood_brier']:.3f})", end="")
        print()

        print(f"{'  (CV AUC)':<30}", end="")
        for r in all_results:
            s = r['knn_tv'][f'k={k}']
            print(f"  {s['cv_auc']:.4f}+/-{s['cv_std']:.3f}  ", end="")
        print()

    print("=" * 95)

    # Summary: best kNN k per split
    print(f"\n{'Best kNN k per split:':<30}")
    for r in all_results:
        best_k = max(r['knn_tv'].values(), key=lambda x: x['ood_auc'])
        short = r['split'].split(':')[0]
        print(f"  {short}: k={best_k['k']}, OOD AUC={best_k['ood_auc']:.4f}")

    # Gap analysis
    print(f"\n{'Gap: paper - best kNN TV:':<30}")
    for r in all_results:
        paper_ood = r['signals_lr']['ood_auc']
        best_knn = max(v['ood_auc'] for v in r['knn_tv'].values())
        gap = paper_ood - best_knn
        short = r['split'].split(':')[0]
        print(f"  {short}: {gap:+.4f} ({'paper better' if gap > 0 else 'kNN better'})")

    # Save
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
