"""
Synthetic-data scaling experiment: extract logits from generated images and
evaluate all filter variants.

For each of 6 synthetic scales (5, 10, 25, 50, 100, 180 images/class):
  - Extract logits via DenseNet-121 (cached as scale_*/logits.pkl)
  - Evaluate 3 training variants × 2 methods on the OOD test set

Training variants:
  1. Real-only    — id_test logits (11,327 samples, baseline)
  2. Synthetic-only — synthetic logits at this scale
  3. Real + Synthetic — concatenation of both

Methods:
  1. 12 signals + LR
  2. kNN TV k=10

Outputs:
  - results/synthetic_scaling_results.json
  - results/figures/synthetic_scaling.png

Usage:
    python scripts/run_synthetic_scaling.py
    python scripts/run_synthetic_scaling.py --skip-extraction    # logits already cached
    python scripts/run_synthetic_scaling.py --scales 5 10 25
"""

import sys
import json
import pickle
import time
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.synthetic_holdout import (
    SyntheticFMoWDataset,
    extract_synthetic_features,
    compute_sf_features_from_logits,
)
from src.tv_distance import tv_correctness_scores
from src.extract_logits import load_wilds_fmow_model

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNTH_ROOT = PROJ_ROOT / "results" / "synthetic_scaling"
LOGITS_PATH = PROJ_ROOT / "results" / "logits" / "fmow_ERM_best_0_main.pkl"
RESULTS_JSON = PROJ_ROOT / "results" / "synthetic_scaling_results.json"
FIGURE_PATH = PROJ_ROOT / "results" / "figures" / "synthetic_scaling.png"

ALL_SCALES = [5, 10, 25, 50, 100, 180]
KNN_K = 10


def scale_dir(scale: int) -> Path:
    return SYNTH_ROOT / f"scale_{scale:03d}"


# ---------------------------------------------------------------------------
# Cross-validation helpers (match run_splits_comparison.py)
# ---------------------------------------------------------------------------

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
    """5-fold CV for kNN TV density ratio."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, briers = [], []
    for train_idx, val_idx in skf.split(logits, correct):
        _, val_scores = tv_correctness_scores(
            logits[train_idx], correct[train_idx],
            logits[val_idx], k=k,
        )
        aucs.append(roc_auc_score(correct[val_idx], val_scores))
        briers.append(brier_score_loss(correct[val_idx],
                                       np.clip(val_scores, 0, 1)))
    return np.mean(aucs), np.std(aucs), np.mean(briers)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_lr(train_logits, train_correct, ood_logits, ood_correct):
    """12 signals + LR: fit on train, score on OOD."""
    train_feat = compute_sf_features_from_logits(train_logits)
    ood_feat = compute_sf_features_from_logits(ood_logits)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_feat)
    X_te = scaler.transform(ood_feat)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_tr, train_correct)
    proba = model.predict_proba(X_te)[:, 1]

    ood_auc = roc_auc_score(ood_correct, proba)
    ood_brier = brier_score_loss(ood_correct, proba)

    # 5-fold CV on training set
    cv_auc, cv_std, cv_brier = cross_val_auc_lr(train_feat, train_correct)

    return {
        'ood_auc': float(ood_auc),
        'ood_brier': float(ood_brier),
        'cv_auc': float(cv_auc),
        'cv_std': float(cv_std),
        'cv_brier': float(cv_brier),
    }


def eval_tv(train_logits, train_correct, ood_logits, ood_correct, k=KNN_K):
    """kNN TV density ratio: fit on train, score on OOD."""
    _, ood_scores = tv_correctness_scores(
        train_logits, train_correct, ood_logits, k=k,
    )

    ood_auc = roc_auc_score(ood_correct, ood_scores)
    ood_brier = brier_score_loss(ood_correct, np.clip(ood_scores, 0, 1))

    # 5-fold CV on training set
    cv_auc, cv_std, cv_brier = cross_val_tv(train_logits, train_correct, k=k)

    return {
        'ood_auc': float(ood_auc),
        'ood_brier': float(ood_brier),
        'cv_auc': float(cv_auc),
        'cv_std': float(cv_std),
        'cv_brier': float(cv_brier),
    }


# ---------------------------------------------------------------------------
# Phase A: logit extraction
# ---------------------------------------------------------------------------

def extract_logits_for_scale(model, device, scale):
    """Extract and cache logits for one synthetic scale."""
    sdir = scale_dir(scale)
    cache = sdir / "logits.pkl"

    if cache.exists():
        print(f"  [cached] {cache}")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    dataset = SyntheticFMoWDataset(sdir)
    print(f"  {sdir.name}: {len(dataset)} images")

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    result = extract_synthetic_features(model, loader, device, verbose=True)

    with open(cache, 'wb') as f:
        pickle.dump(result, f)
    print(f"  saved → {cache}")

    return result


# ---------------------------------------------------------------------------
# Phase B: evaluation
# ---------------------------------------------------------------------------

def run_evaluation(scales, real_data):
    """Evaluate all variants for all scales."""
    real_logits = real_data['id_test']['logits']
    real_correct = real_data['id_test']['correct']
    ood_logits = real_data['test']['logits']
    ood_correct = real_data['test']['correct']

    print(f"\nReal ref:  {real_logits.shape[0]:,} samples, "
          f"acc={real_correct.mean():.4f}")
    print(f"OOD test:  {ood_logits.shape[0]:,} samples, "
          f"acc={ood_correct.mean():.4f}")

    # ---- Baselines (real-only) ----
    print("\n--- Real-only baselines ---")
    t0 = time.time()
    baseline_lr = eval_lr(real_logits, real_correct, ood_logits, ood_correct)
    print(f"  12 sig+LR  OOD AUC={baseline_lr['ood_auc']:.4f}  "
          f"CV={baseline_lr['cv_auc']:.4f}  [{time.time()-t0:.0f}s]")

    t0 = time.time()
    baseline_tv = eval_tv(real_logits, real_correct, ood_logits, ood_correct)
    print(f"  kNN TV     OOD AUC={baseline_tv['ood_auc']:.4f}  "
          f"CV={baseline_tv['cv_auc']:.4f}  [{time.time()-t0:.0f}s]")

    # ---- Per-scale evaluation ----
    per_scale = {}
    for scale in scales:
        sdir = scale_dir(scale)
        cache = sdir / "logits.pkl"
        if not cache.exists():
            print(f"\n[WARN] {cache} not found, skipping scale {scale}")
            continue

        with open(cache, 'rb') as f:
            synth = pickle.load(f)

        synth_logits = synth['logits']
        synth_correct = synth['correct']
        synth_acc = float(synth_correct.mean())

        print(f"\n{'='*60}")
        print(f"Scale {scale}/cls  ({synth_logits.shape[0]:,} synth images, "
              f"acc={synth_acc:.4f})")
        print(f"{'='*60}")

        result = {
            'scale': scale,
            'n_synth': int(synth_logits.shape[0]),
            'synth_acc': synth_acc,
        }

        # -- Synthetic-only --
        print("  [synth-only] 12 sig+LR ...")
        t0 = time.time()
        result['synth_lr'] = eval_lr(
            synth_logits, synth_correct, ood_logits, ood_correct)
        print(f"    OOD AUC={result['synth_lr']['ood_auc']:.4f}  "
              f"CV={result['synth_lr']['cv_auc']:.4f}  [{time.time()-t0:.0f}s]")

        print("  [synth-only] kNN TV ...")
        t0 = time.time()
        result['synth_tv'] = eval_tv(
            synth_logits, synth_correct, ood_logits, ood_correct)
        print(f"    OOD AUC={result['synth_tv']['ood_auc']:.4f}  "
              f"CV={result['synth_tv']['cv_auc']:.4f}  [{time.time()-t0:.0f}s]")

        # -- Augmented (real + synthetic) --
        aug_logits = np.concatenate([real_logits, synth_logits])
        aug_correct = np.concatenate([real_correct, synth_correct])

        print("  [augmented]  12 sig+LR ...")
        t0 = time.time()
        result['aug_lr'] = eval_lr(
            aug_logits, aug_correct, ood_logits, ood_correct)
        print(f"    OOD AUC={result['aug_lr']['ood_auc']:.4f}  "
              f"CV={result['aug_lr']['cv_auc']:.4f}  [{time.time()-t0:.0f}s]")

        print("  [augmented]  kNN TV ...")
        t0 = time.time()
        result['aug_tv'] = eval_tv(
            aug_logits, aug_correct, ood_logits, ood_correct)
        print(f"    OOD AUC={result['aug_tv']['ood_auc']:.4f}  "
              f"CV={result['aug_tv']['cv_auc']:.4f}  [{time.time()-t0:.0f}s]")

        per_scale[scale] = result

    return {
        'baseline_lr': baseline_lr,
        'baseline_tv': baseline_tv,
        'per_scale': per_scale,
        'real_n': int(real_logits.shape[0]),
        'real_acc': float(real_correct.mean()),
        'ood_n': int(ood_logits.shape[0]),
        'ood_acc': float(ood_correct.mean()),
    }


# ---------------------------------------------------------------------------
# Summary table + figure
# ---------------------------------------------------------------------------

def print_summary_table(results):
    """Print the summary table from the plan."""
    scales = sorted(results['per_scale'].keys())
    bl_lr = results['baseline_lr']['ood_auc']
    bl_tv = results['baseline_tv']['ood_auc']

    header_scales = "".join(f"{s:>8}/cls" for s in scales)
    print(f"\n{'Method / Variant':<35}{header_scales}{'Real':>10}")
    print("-" * (35 + 11 * len(scales) + 10))

    rows = [
        ("12 sig+LR, synth-only",  'synth_lr', bl_lr),
        ("12 sig+LR, augmented",   'aug_lr',   bl_lr),
        ("kNN TV, synth-only",     'synth_tv', bl_tv),
        ("kNN TV, augmented",      'aug_tv',   bl_tv),
    ]

    for label, key, baseline in rows:
        vals = ""
        for s in scales:
            if s in results['per_scale']:
                v = results['per_scale'][s][key]['ood_auc']
                vals += f"{v:>11.3f}"
            else:
                vals += f"{'---':>11}"
        vals += f"{baseline:>10.3f}"
        print(f"{label:<35}{vals}")

    print()


def make_figure(results):
    """Create the scaling figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scales = sorted(results['per_scale'].keys())
    bl_lr = results['baseline_lr']['ood_auc']
    bl_tv = results['baseline_tv']['ood_auc']

    synth_lr = [results['per_scale'][s]['synth_lr']['ood_auc'] for s in scales]
    synth_tv = [results['per_scale'][s]['synth_tv']['ood_auc'] for s in scales]
    aug_lr   = [results['per_scale'][s]['aug_lr']['ood_auc'] for s in scales]
    aug_tv   = [results['per_scale'][s]['aug_tv']['ood_auc'] for s in scales]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(scales, synth_lr, 'o-',  label=f'Synth-only, 12 sig+LR', color='#1f77b4')
    ax.plot(scales, synth_tv, 's-',  label=f'Synth-only, kNN TV',    color='#ff7f0e')
    ax.plot(scales, aug_lr,   'o--', label=f'Augmented, 12 sig+LR',  color='#2ca02c')
    ax.plot(scales, aug_tv,   's--', label=f'Augmented, kNN TV',     color='#d62728')

    ax.axhline(bl_lr, color='#1f77b4', ls=':', alpha=0.6,
               label=f'Real-only, 12 sig+LR ({bl_lr:.3f})')
    ax.axhline(bl_tv, color='#ff7f0e', ls=':', alpha=0.6,
               label=f'Real-only, kNN TV ({bl_tv:.3f})')

    ax.set_xlabel('Synthetic images per class')
    ax.set_ylabel('OOD AUC')
    ax.set_title('Synthetic Data Scaling: Filter Performance')
    ax.set_xscale('log')
    ax.set_xticks(scales)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {FIGURE_PATH}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic scaling experiment: logit extraction + evaluation"
    )
    parser.add_argument(
        "--scales", type=int, nargs="*", default=None,
        help="Subset of scales (default: all available)",
    )
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip logit extraction (use cached logits.pkl)",
    )
    parser.add_argument(
        "--skip-figure", action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--model-root", type=str, default="suitability",
        help="Root dir for load_wilds_fmow_model (default: suitability)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    scales = args.scales if args.scales else ALL_SCALES

    # Filter to scales whose directories actually exist
    available = [s for s in scales if scale_dir(s).exists()]
    if not available:
        print(f"No scale directories found under {SYNTH_ROOT}. "
              "Run run_synthetic_generation.py first.")
        return
    print(f"Available scales: {available}")

    # ---- Phase A: logit extraction ----
    if not args.skip_extraction:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nLoading DenseNet-121 on {device}...")
        model = load_wilds_fmow_model(args.model_root, device=device)

        print("\n--- Extracting synthetic logits ---")
        for scale in available:
            extract_logits_for_scale(model, device, scale)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ---- Load real data ----
    print(f"\nLoading real logits from {LOGITS_PATH}...")
    with open(LOGITS_PATH, 'rb') as f:
        real_data = pickle.load(f)

    # ---- Phase B: evaluation ----
    print("\n--- Running evaluation ---")
    results = run_evaluation(available, real_data)

    # ---- Print summary ----
    print_summary_table(results)

    # ---- Save JSON ----
    # Convert integer keys to strings for JSON serialization
    json_results = results.copy()
    json_results['per_scale'] = {
        str(k): v for k, v in results['per_scale'].items()
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {RESULTS_JSON}")

    # ---- Figure ----
    if not args.skip_figure and len(available) >= 2:
        make_figure(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
