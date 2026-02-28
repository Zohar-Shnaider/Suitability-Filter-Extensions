"""
Suitability Filter Challenge - Source Code

This package contains implementations for challenging the ICML 2025
Suitability Filter paper on WILDS-FMoW:

1. TV Distance Analysis (tv_distance.py)
   - k-NN based TV distance estimation
   - 12 suitability signal computation
   - Comparison utilities

2. Logit Extraction (extract_logits.py)
   - Extract logits from FMoW classifier
   - Support for multiple splits and filters

3. Synthetic Generation (generate_synthetic.py)
   - Generate FMoW images with DiffusionSat
   - Diverse metadata variations

4. Synthetic Hold-Out (synthetic_holdout.py)
   - Create synthetic hold-out sets
   - Train suitability filter without circular assumption
"""

from .tv_distance import (
    knn_tv_distance,
    knn_tv_distance_batch,
    compute_suitability_signals,
    signals_to_features,
    compare_tv_vs_signals,
)

from .extract_logits import (
    load_wilds_fmow_model,
    load_wilds_fmow_dataset,
    extract_logits,
    save_logits,
    load_logits,
)

from .generate_synthetic import (
    DiffusionSatGenerator,
    generate_synthetic_fmow_dataset,
    generate_for_single_class,
    FMOW_CATEGORIES,
    REGIONS,
    MetadataConfig,
)

from .synthetic_holdout import (
    SyntheticFMoWDataset,
    SyntheticSuitabilityFilter,
    create_synthetic_holdout,
    compute_sf_features_from_logits,
)

__all__ = [
    # TV Distance
    'knn_tv_distance',
    'knn_tv_distance_batch',
    'compute_suitability_signals',
    'signals_to_features',
    'compare_tv_vs_signals',
    # Logit Extraction
    'load_wilds_fmow_model',
    'load_wilds_fmow_dataset',
    'extract_logits',
    'save_logits',
    'load_logits',
    # Synthetic Generation
    'DiffusionSatGenerator',
    'generate_synthetic_fmow_dataset',
    'generate_for_single_class',
    'FMOW_CATEGORIES',
    'REGIONS',
    'MetadataConfig',
    # Synthetic Hold-Out
    'SyntheticFMoWDataset',
    'SyntheticSuitabilityFilter',
    'create_synthetic_holdout',
    'compute_sf_features_from_logits',
]
