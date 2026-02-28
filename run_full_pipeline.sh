#!/bin/bash
set -e

echo "============================================"
echo "  Full Experiment Pipeline"
echo "  $(date)"
echo "============================================"
echo

# --- Prerequisite checks ---

check_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: $1 not found."
        echo "  $2"
        exit 1
    fi
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: $1 not found."
        echo "  $2"
        exit 1
    fi
}

echo "Checking prerequisites..."

check_dir "suitability" \
    "Clone with: git clone https://github.com/cleverhans-lab/suitability.git"

check_file "suitability/experiments/fmow/fmow_seed:0_epoch:best_model.pth" \
    "FMoW classifier checkpoint missing from suitability repo."

check_dir "data/fmow_v1.1" \
    "Download with: python -c \"from wilds import get_dataset; get_dataset(dataset='fmow', download=True, root_dir='data')\""

check_dir "DiffusionSat" \
    "Clone with: git clone https://github.com/samar-khanna/DiffusionSat.git"

check_dir "checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64" \
    "Download from https://zenodo.org/records/13751498 and extract to checkpoints/"

echo "All prerequisites found."
echo

# --- Run experiments ---

echo "=== Step 0/5: Extract logits ==="
python scripts/run_extract_logits.py
echo "Done. Output: results/logits/"
echo

echo "=== Step 1/5: TV distance vs 12 signals ==="
python scripts/run_tv_vs_signals.py
echo "Done. Output: results/tv_vs_signals_results.json, results/figures/tv_vs_signals.png"
echo

echo "=== Step 2/5: Splits comparison ==="
python scripts/run_splits_comparison.py
echo "Done. Output: results/splits_comparison_results.json"
echo

echo "=== Step 3/5: Alternative TV methods ==="
python scripts/run_alternative_tv.py
echo "Done. Output: results/alternative_tv_results.json, results/figures/alternative_tv.png"
echo

echo "=== Step 4/5: Synthetic image generation ==="
if [ -d "results/synthetic_scaling" ]; then
    echo "Found existing synthetic images in results/synthetic_scaling/, continuing..."
else
    echo "SKIPPED: Synthetic generation takes hours, requires GPU + DiffusionSat checkpoint."
    echo "  Run manually if needed: python scripts/run_synthetic_generation.py"
fi
echo

echo "=== Step 5/5: Synthetic scaling experiment ==="
if [ -d "results/synthetic_scaling" ]; then
    python scripts/run_synthetic_scaling.py
    echo "Done. Output: results/synthetic_scaling_results.json, results/figures/synthetic_scaling.png"
else
    echo "SKIPPED: Requires synthetic images from Step 4."
    echo "  Run Step 4 first, then: python scripts/run_synthetic_scaling.py"
fi
echo

echo "============================================"
echo "  Pipeline complete: $(date)"
echo "  Results in results/ and results/figures/"
echo "============================================"
