# Challenging the Suitability Filter

## Overview

This project critically examines the ICML 2025 paper *"Suitability Filter for Dataset Distribution Shift"* ([cleverhans-lab/suitability](https://github.com/cleverhans-lab/suitability)), which proposes a suitability filter to detect distribution shift in the WILDS benchmark.

We identify two fundamental weaknesses:

1. **Redundant signal engineering:** The paper's 12 hand-crafted logit signals (softmax confidence, entropy, margin, energy, etc.) are highly correlated. We show that a single TV distance computed directly on raw logit distributions achieves comparable shift detection performance, making the 12-signal approach unnecessarily complex.

2. **Circular hold-out assumption:** The paper requires a hold-out set that "reflects" the inference-time distribution. This is circular: if you knew the deployment distribution, the problem would already be solved. We replace this assumption with DiffusionSat conditional generation, creating synthetic hold-out sets that cover diverse shift scenarios without prior knowledge of the deployment distribution.

## Repository Structure

```
Submission/
├── src/                           # Core source modules
│   ├── __init__.py                # Package exports
│   ├── tv_distance.py             # k-NN based TV distance estimation
│   ├── extract_logits.py          # Logit extraction from DenseNet-121
│   ├── alternative_tv.py          # Alternative TV estimation methods
│   ├── generate_synthetic.py      # DiffusionSat image generation
│   └── synthetic_holdout.py       # Synthetic hold-out suitability filter
├── scripts/                       # Experiment execution scripts
│   ├── run_extract_logits.py      # Step 0: Extract logits from FMoW
│   ├── run_tv_vs_signals.py       # Exp 1: TV distance vs 12 signals
│   ├── run_splits_comparison.py   # Exp 2: Cross-split comparisons
│   ├── run_alternative_tv.py      # Exp 3: Alternative TV methods
│   ├── run_synthetic_generation.py# Exp 4: Synthetic image generation
│   └── run_synthetic_scaling.py   # Exp 5: Synthetic scaling experiment
├── notebooks/                     # Interactive analysis
│   ├── tv_analysis.ipynb          # TV distance exploration
│   ├── tv_analysis_executed.ipynb # Pre-executed version with outputs
│   └── diffusion_experiments.ipynb# DiffusionSat experiments
├── results/                       # Pre-computed results (included)
│   ├── tv_vs_signals_results.json
│   ├── splits_comparison_results.json
│   ├── alternative_tv_results.json
│   ├── synthetic_scaling_results.json
│   ├── full_analysis_results.json
│   ├── tv_analysis_results.json
│   └── figures/                   # Generated plots (included)
│       ├── tv_vs_signals.png
│       ├── signal_correlation.png
│       ├── signal_shift_detection.png
│       ├── alternative_tv.png
│       ├── synthetic_scaling.png
│       ├── shift_magnitude_analysis.png
│       └── full_ablation.png
├── requirements.txt               # Python dependencies
├── run_full_pipeline.sh           # Run all experiments end-to-end
└── README.md                      # This file
```

**Not included in submission** (too large; download separately):
- `data/` (~55GB FMoW dataset)
- `checkpoints/` (~5GB DiffusionSat checkpoint)
- `DiffusionSat/` (cloned repository)
- `suitability/` (cloned baseline repository)
- `results/synthetic_scaling/` (~11GB generated images)
- `results/logits/` (~50MB extracted logits)

## Setup

### 1. Python Environment

```bash
conda create -n suitability python=3.10
conda activate suitability
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with the appropriate CUDA version for your system. See [pytorch.org](https://pytorch.org/get-started/locally/) for instructions. The versions in `requirements.txt` were tested with CUDA 12.1.

### 3. Clone External Repositories

```bash
# Suitability filter baseline (contains FMoW classifier checkpoint)
git clone https://github.com/cleverhans-lab/suitability.git

# DiffusionSat (conditional satellite image generation)
git clone https://github.com/samar-khanna/DiffusionSat.git
```

### 4. Download Data

**FMoW dataset** (~55GB):
```bash
python -c "from wilds import get_dataset; get_dataset(dataset='fmow', download=True, root_dir='data')"
```

**DiffusionSat checkpoint** (~5GB):
Download from [Zenodo](https://zenodo.org/records/13751498) and extract:
```bash
# Download finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64.zip from the Zenodo link
unzip finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64.zip -d checkpoints/
```

**FMoW classifier checkpoint:**
Already included in the cloned `suitability/` repository at `suitability/experiments/fmow/fmow_seed:0_epoch:best_model.pth`.

## Reproducing Experiments

All experiments can be run individually or end-to-end via `./run_full_pipeline.sh`. Each script prints progress and saves results to `results/`.

### Step 0: Extract Logits (prerequisite for all experiments)

```bash
python scripts/run_extract_logits.py
```
- **Produces:** `results/logits/*.pkl`
- **Requires:** FMoW dataset, suitability repo (with classifier checkpoint)
- **Runtime:** ~15 minutes on GPU

### Experiment 1: TV Distance vs 12 Signals

```bash
python scripts/run_tv_vs_signals.py
```
- **Produces:** `results/tv_vs_signals_results.json`, `results/figures/tv_vs_signals.png`
- **Requires:** Extracted logits from Step 0

### Experiment 2: Splits Comparison

```bash
python scripts/run_splits_comparison.py
```
- **Produces:** `results/splits_comparison_results.json`
- **Requires:** Extracted logits from Step 0

### Experiment 3: Alternative TV Methods

```bash
python scripts/run_alternative_tv.py
```
- **Produces:** `results/alternative_tv_results.json`, `results/figures/alternative_tv.png`
- **Requires:** Extracted logits from Step 0

### Experiment 4: Synthetic Image Generation

```bash
python scripts/run_synthetic_generation.py
```
- **Produces:** `results/synthetic_scaling/scale_*/` (images + metadata)
- **Requires:** DiffusionSat repo, DiffusionSat checkpoint, GPU with 24GB VRAM
- **Runtime:** ~1 second per image

### Experiment 5: Synthetic Scaling Experiment

```bash
python scripts/run_synthetic_scaling.py
```
- **Produces:** `results/synthetic_scaling_results.json`, `results/figures/synthetic_scaling.png`
- **Requires:** Generated synthetic images from Experiment 4, extracted logits from Step 0

### Interactive Analysis

```bash
jupyter notebook notebooks/tv_analysis.ipynb
```

The notebook includes a synthetic data fallback, so it can run without the full FMoW dataset for quick exploration.

## Pre-computed Results

All result JSON files and figures are included in the submission. Reviewers can inspect the results without re-running any experiments:

- `results/tv_vs_signals_results.json` — TV distance vs 12 signals comparison
- `results/splits_comparison_results.json` — Cross-split shift detection
- `results/alternative_tv_results.json` — Comparison of TV estimation methods
- `results/synthetic_scaling_results.json` — Synthetic hold-out scaling curves
- `results/full_analysis_results.json` — Combined analysis results
- `results/figures/*.png` — All plots

## Hardware Requirements

- **GPU:** NVIDIA RTX 3090 (24GB VRAM) or equivalent recommended
  - Required for DiffusionSat generation (Experiments 4-5)
  - Experiments 1-3 run on CPU (logits are pre-extracted)
- **Disk:** ~65GB for FMoW dataset + DiffusionSat checkpoint
- **RAM:** 16GB+ recommended

## References

- **Suitability Filter Paper:** [cleverhans-lab/suitability](https://github.com/cleverhans-lab/suitability)
- **DiffusionSat:** [samar-khanna/DiffusionSat](https://github.com/samar-khanna/DiffusionSat) — [arXiv:2312.03606](https://arxiv.org/abs/2312.03606)
- **WILDS-FMoW Benchmark:** [wilds.stanford.edu](https://wilds.stanford.edu/)
- **DiffusionSat Checkpoint:** [Zenodo 13751498](https://zenodo.org/records/13751498)
