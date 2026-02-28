"""
Logit Extraction Utilities for WILDS-FMoW

This module provides utilities for extracting logits from the pretrained
FMoW classifier on various data splits. It directly handles model loading
and dataset creation without depending on the suitability repo's broken
import chain.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
from tqdm import tqdm

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader

PROJ_ROOT = Path(__file__).parent.parent


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _remove_prefix_from_state_dict(state_dict, prefix="model."):
    """Remove prefix from state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_wilds_fmow_model(
    root_dir: str,
    algorithm: str = "ERM",
    model_type: str = "best",
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load pretrained FMoW model (DenseNet-121, 62 classes).

    Expects checkpoint at:
      {root_dir}/experiments/fmow/fmow_seed:{seed}_epoch:{model_type}_model.pth
    or for IRM/groupDRO:
      {root_dir}/experiments/{algorithm}/fmow/fmow_seed:{seed}_epoch:{model_type}_model.pth

    Args:
        root_dir: Root directory containing experiments/ folder
        algorithm: Training algorithm (ERM, IRM, groupDRO)
        model_type: 'best' or 'last' checkpoint
        seed: Random seed used during training
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    device = device or get_device()

    # Create DenseNet-121 with 62 output classes (FMoW)
    model = torchvision_models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 62)

    # Build checkpoint path (matching suitability repo convention)
    experiments_dir = Path(root_dir) / "experiments"
    if algorithm == "ERM":
        ckpt_path = experiments_dir / "fmow" / f"fmow_seed:{seed}_epoch:{model_type}_model.pth"
    else:
        ckpt_path = experiments_dir / algorithm / "fmow" / f"fmow_seed:{seed}_epoch:{model_type}_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_path}. "
            f"Download from CodaLab: https://worksheets.codalab.org/rest/bundles/"
            f"0x63a3f824ac6745ea8e9061f736671304/contents/blob/best_model.pth"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = _remove_prefix_from_state_dict(checkpoint["algorithm"])
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def load_wilds_fmow_dataset(
    root_dir: str,
    split: str,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
    pre_filter: Optional[Dict] = None,
) -> Tuple[DataLoader, Optional[list]]:
    """
    Load FMoW dataset split from WILDS.

    Args:
        root_dir: Root directory containing the fmow_v1.1 data
        split: Split name ('train', 'id_val', 'id_test', 'val', 'test')
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle data
        num_workers: Number of dataloader workers
        pre_filter: Filter dict (e.g., {'year': [2013], 'region': ['Asia']})

    Returns:
        Tuple of (DataLoader, filtered_indices or None)
    """
    from torchvision import transforms

    dataset = get_dataset(dataset="fmow", download=False, root_dir=root_dir)

    # Standard FMoW transform (224x224, ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    subset = dataset.get_subset(split, transform=transform)

    filtered_indices = None
    if pre_filter:
        # FMoW metadata columns: [region, year]
        valid_regions = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
        all_indices = list(range(len(subset)))
        filtered = all_indices

        for key, value in pre_filter.items():
            if not isinstance(value, list):
                value = [value]
            if key == "region":
                val_inds = [valid_regions.index(r) for r in value]
                filtered = [i for i in filtered if subset[i][2][0].item() in val_inds]
            elif key == "year":
                val_inds = [y - 2002 for y in value]
                filtered = [i for i in filtered if subset[i][2][1].item() in val_inds]

        filtered_indices = filtered
        subset = torch.utils.data.Subset(subset, filtered_indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader, filtered_indices


def extract_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    return_labels: bool = True,
    return_predictions: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract logits from model for all samples in dataloader.

    Returns:
        Dictionary with 'logits', 'labels', 'predictions', 'correct'
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    all_logits = []
    all_labels = []
    all_predictions = []

    iterator = tqdm(dataloader, desc="Extracting logits") if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)

            all_logits.append(outputs.cpu().numpy())

            if return_labels:
                all_labels.append(labels.cpu().numpy())

            if return_predictions:
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu().numpy())

    results = {
        'logits': np.vstack(all_logits),
    }

    if return_labels:
        results['labels'] = np.concatenate(all_labels)

    if return_predictions:
        results['predictions'] = np.concatenate(all_predictions)

    if return_labels and return_predictions:
        results['correct'] = (results['predictions'] == results['labels']).astype(bool)

    return results


def extract_logits_for_splits(
    model: torch.nn.Module,
    root_dir: str,
    splits: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract logits for multiple splits."""
    device = device or get_device()
    results = {}

    for split in splits:
        if verbose:
            print(f"Processing split: {split}")

        dataloader, _ = load_wilds_fmow_dataset(
            root_dir, split, batch_size=batch_size, num_workers=num_workers,
        )

        results[split] = extract_logits(
            model, dataloader, device=device, verbose=verbose
        )

    return results


def extract_logits_for_filtered_splits(
    model: torch.nn.Module,
    root_dir: str,
    filtered_splits: List[Tuple[str, Dict]],
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract logits for filtered splits (e.g., specific years/regions)."""
    device = device or get_device()
    results = {}

    for split_name, split_filter in filtered_splits:
        filter_str = str(split_filter)
        key = f"{split_name}_{filter_str}"

        if verbose:
            print(f"Processing: {key}")

        dataloader, indices = load_wilds_fmow_dataset(
            root_dir, split_name, batch_size=batch_size,
            num_workers=num_workers, pre_filter=split_filter,
        )

        extraction = extract_logits(
            model, dataloader, device=device, verbose=verbose
        )
        extraction['indices'] = indices
        results[key] = extraction

    return results


def save_logits(
    logits_dict: Dict,
    output_path: Union[str, Path],
    format: str = "npz",
) -> None:
    """Save extracted logits to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        flat_dict = {}
        for key, value in logits_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_dict[f"{key}_{subkey}"] = subvalue
            else:
                flat_dict[key] = value
        np.savez(output_path, **flat_dict)
    elif format == "pkl":
        with open(output_path, "wb") as f:
            pickle.dump(logits_dict, f)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_logits(input_path: Union[str, Path]) -> Dict:
    """Load extracted logits from file."""
    input_path = Path(input_path)

    if input_path.suffix == ".npz":
        data = np.load(input_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    elif input_path.suffix == ".pkl":
        with open(input_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file format: {input_path.suffix}")


# Standard FMoW splits
ID_SPLITS = ["id_val", "id_test"]
OOD_SPLITS = ["val", "test"]

ID_FILTERED_SPLITS = [
    ("id_val", {"year": [2002, 2003, 2004, 2005, 2006]}),
    ("id_val", {"year": [2007, 2008, 2009]}),
    ("id_val", {"year": [2010]}),
    ("id_val", {"year": [2011]}),
    ("id_val", {"year": [2012]}),
    ("id_val", {"region": ["Asia"]}),
    ("id_val", {"region": ["Europe"]}),
    ("id_val", {"region": ["Americas"]}),
    ("id_test", {"year": [2002, 2003, 2004, 2005, 2006]}),
    ("id_test", {"year": [2007, 2008, 2009]}),
    ("id_test", {"year": [2010]}),
    ("id_test", {"year": [2011]}),
    ("id_test", {"year": [2012]}),
    ("id_test", {"region": ["Asia"]}),
    ("id_test", {"region": ["Europe"]}),
    ("id_test", {"region": ["Americas"]}),
]

OOD_FILTERED_SPLITS = [
    ("val", {"year": [2013]}),
    ("val", {"year": [2014]}),
    ("val", {"year": [2015]}),
    ("val", {"region": ["Asia"]}),
    ("val", {"region": ["Europe"]}),
    ("val", {"region": ["Africa"]}),
    ("val", {"region": ["Americas"]}),
    ("val", {"region": ["Oceania"]}),
    ("test", {"year": [2016]}),
    ("test", {"year": [2017]}),
    ("test", {"region": ["Asia"]}),
    ("test", {"region": ["Europe"]}),
    ("test", {"region": ["Africa"]}),
    ("test", {"region": ["Americas"]}),
    ("test", {"region": ["Oceania"]}),
]


def main():
    """Example usage and extraction script."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract logits from FMoW classifier")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory with data/ and experiments/")
    parser.add_argument("--output_dir", type=str, default="results/logits", help="Output directory")
    parser.add_argument("--algorithm", type=str, default="ERM", help="Training algorithm")
    parser.add_argument("--model_type", type=str, default="best", help="best or last")
    parser.add_argument("--seed", type=int, default=0, help="Training seed")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_wilds_fmow_model(
        args.root_dir,
        algorithm=args.algorithm,
        model_type=args.model_type,
        seed=args.seed,
        device=device,
    )

    # Extract for main splits
    print("\nExtracting logits for main splits...")
    main_results = extract_logits_for_splits(
        model, args.root_dir, ID_SPLITS + OOD_SPLITS,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )

    output_path = Path(args.output_dir) / f"fmow_{args.algorithm}_{args.model_type}_{args.seed}_main.pkl"
    save_logits(main_results, output_path, format="pkl")
    print(f"Saved main results to {output_path}")

    # Extract for filtered splits
    print("\nExtracting logits for filtered ID splits...")
    id_filtered_results = extract_logits_for_filtered_splits(
        model, args.root_dir, ID_FILTERED_SPLITS,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )

    output_path = Path(args.output_dir) / f"fmow_{args.algorithm}_{args.model_type}_{args.seed}_id_filtered.pkl"
    save_logits(id_filtered_results, output_path, format="pkl")
    print(f"Saved ID filtered results to {output_path}")

    print("\nExtracting logits for filtered OOD splits...")
    ood_filtered_results = extract_logits_for_filtered_splits(
        model, args.root_dir, OOD_FILTERED_SPLITS,
        batch_size=args.batch_size, num_workers=args.num_workers, device=device,
    )

    output_path = Path(args.output_dir) / f"fmow_{args.algorithm}_{args.model_type}_{args.seed}_ood_filtered.pkl"
    save_logits(ood_filtered_results, output_path, format="pkl")
    print(f"Saved OOD filtered results to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
