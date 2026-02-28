"""
Synthetic Hold-Out Set Creation for Suitability Filter

This module creates synthetic hold-out sets using DiffusionSat-generated images,
replacing the original paper's assumption of having access to labeled data that
"reflects" the inference-time distribution. This addresses the circular assumption
problem: if you knew the inference distribution, you wouldn't need the filter.

The synthetic hold-out approach:
1. Generate diverse synthetic images with DiffusionSat
2. Run the classifier on synthetic images to get predictions
3. Determine correctness labels (predicted class == generation class)
4. Train the correctness estimator on this synthetic D_sf

This removes the need for any hold-out data that matches the deployment distribution.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm

# Add paths
PROJ_ROOT = Path(__file__).parent.parent
SUITABILITY_PATH = PROJ_ROOT / "suitability"
if str(SUITABILITY_PATH) not in sys.path:
    sys.path.insert(0, str(SUITABILITY_PATH))


# FMoW class names
FMOW_CATEGORIES = [
    "airport", "airport_hangar", "airport_terminal", "amusement_park",
    "aquaculture", "archaeological_site", "barn", "border_checkpoint",
    "burial_site", "car_dealership", "construction_site", "crop_field",
    "dam", "debris_or_rubble", "educational_institution", "electric_substation",
    "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
    "gas_station", "golf_course", "ground_transportation_station", "helipad",
    "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
    "lighthouse", "military_facility", "multi-unit_residential",
    "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
    "parking_lot_or_garage", "place_of_worship", "police_station", "port",
    "prison", "race_track", "railway_bridge", "recreational_facility",
    "road_bridge", "runway", "shipyard", "shopping_mall",
    "single-unit_residential", "smokestack", "solar_farm", "space_facility",
    "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
    "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
    "wind_farm", "zoo"
]


class SyntheticFMoWDataset(Dataset):
    """Dataset for synthetic FMoW images with associated metadata."""

    # ImageNet normalization (used by FMoW classifier)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root_dir: Union[str, Path],
        metadata_file: Optional[str] = "metadata.json",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
    ):
        """
        Initialize synthetic FMoW dataset.

        Args:
            root_dir: Directory containing synthetic images
            metadata_file: Name of metadata JSON file
            transform: Optional custom transform
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # Load metadata
        metadata_path = self.root_dir / metadata_file
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Scan directory for images
            self.metadata = self._scan_directory()

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.MEAN, std=self.STD),
            ])
        else:
            self.transform = transform

    def _scan_directory(self) -> List[Dict]:
        """Scan directory for images if no metadata file exists."""
        metadata = []

        for class_name in FMOW_CATEGORIES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = FMOW_CATEGORIES.index(class_name)

            for img_file in class_dir.glob("*.png"):
                metadata.append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'filename': str(img_file),
                })

            for img_file in class_dir.glob("*.jpg"):
                metadata.append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'filename': str(img_file),
                })

        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get item from dataset.

        Returns:
            Tuple of (image tensor, class index, metadata dict)
        """
        item = self.metadata[idx]

        # Load image
        img_path = item['filename']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        class_idx = item['class_idx']

        return image, class_idx, item


def compute_sf_features_from_logits(logits: np.ndarray) -> np.ndarray:
    """
    Compute suitability filter features (12 signals) from logits.

    This mirrors the get_sf_features function from the original paper.

    Args:
        logits: (N, C) array of logits

    Returns:
        (N, 12) array of features
    """
    N, C = logits.shape

    # Softmax (numerically stable)
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Predictions
    predictions = logits.argmax(axis=1)

    eps = 1e-10

    # 1. conf_max
    conf_max = softmax.max(axis=1)

    # 2. conf_std
    conf_std = softmax.std(axis=1)

    # 3. conf_entropy
    conf_entropy = -np.sum(softmax * np.log(softmax + eps), axis=1)

    # 4. logit_mean
    logit_mean = logits.mean(axis=1)

    # 5. logit_max
    logit_max = logits.max(axis=1)

    # 6. logit_std
    logit_std = logits.std(axis=1)

    # 7. logit_diff_top2
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    logit_diff_top2 = sorted_logits[:, 0] - sorted_logits[:, 1]

    # 8. loss
    pred_probs = softmax[np.arange(N), predictions]
    loss = -np.log(pred_probs + eps)

    # 9. margin_loss
    sorted_softmax = np.sort(softmax, axis=1)[:, ::-1]
    pred_class_probs = sorted_softmax[:, 0]
    next_best_probs = sorted_softmax[:, 1]
    pred_class_loss = -np.log(pred_class_probs + eps)
    next_best_loss = -np.log(next_best_probs + eps)
    margin_loss = pred_class_loss - next_best_loss

    # 10. class_prob_ratio
    class_prob_ratio = pred_class_probs / (next_best_probs + eps)

    # 11. top_k_probs_sum (top 10%)
    top_k = max(1, int(C * 0.1))
    top_k_probs_sum = sorted_softmax[:, :top_k].sum(axis=1)

    # 12. energy
    energy = -np.log(exp_logits.sum(axis=1)) - logits_max.squeeze()

    features = np.column_stack([
        conf_max, conf_std, conf_entropy,
        logit_mean, logit_max, logit_std, logit_diff_top2,
        loss, margin_loss, class_prob_ratio, top_k_probs_sum,
        energy
    ])

    return features


def extract_synthetic_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract features and correctness labels from synthetic dataset.

    Args:
        model: FMoW classifier model
        dataloader: DataLoader for synthetic images
        device: Device to use
        verbose: Show progress bar

    Returns:
        Dictionary with 'features', 'logits', 'predictions', 'labels', 'correct'
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_predictions = []

    iterator = tqdm(dataloader, desc="Extracting features") if verbose else dataloader

    with torch.no_grad():
        for images, labels, metadata in iterator:
            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_predictions.append(predictions.cpu().numpy())

    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels)
    predictions = np.concatenate(all_predictions)

    # Compute features
    features = compute_sf_features_from_logits(logits)

    # Correctness (prediction matches generation label)
    correct = (predictions == labels).astype(bool)

    return {
        'features': features,
        'logits': logits,
        'predictions': predictions,
        'labels': labels,
        'correct': correct,
    }


class SyntheticSuitabilityFilter:
    """
    Suitability filter trained on synthetic hold-out data.

    This replaces the original paper's approach which requires hold-out data
    that "reflects" the inference distribution - a circular assumption.
    """

    def __init__(
        self,
        normalize: bool = True,
        feature_subset: Optional[List[int]] = None,
    ):
        """
        Initialize the filter.

        Args:
            normalize: Whether to normalize features
            feature_subset: Optional subset of feature indices to use
        """
        self.normalize = normalize
        self.feature_subset = feature_subset
        self.scaler = StandardScaler()
        self.classifier = None

    def train(
        self,
        features: np.ndarray,
        correct: np.ndarray,
        calibrated: bool = True,
        cv_folds: int = 5,
    ):
        """
        Train the correctness estimator.

        Args:
            features: (N, 12) feature matrix from synthetic hold-out
            correct: (N,) boolean correctness labels
            calibrated: Whether to calibrate classifier
            cv_folds: Number of CV folds for calibration
        """
        X = features.copy()

        if self.feature_subset is not None:
            X = X[:, self.feature_subset]

        if self.normalize:
            X = self.scaler.fit_transform(X)

        base_model = LogisticRegression(max_iter=1000)

        if calibrated:
            self.classifier = CalibratedClassifierCV(
                estimator=base_model,
                method='sigmoid',
                cv=cv_folds,
            ).fit(X, correct)
        else:
            self.classifier = base_model.fit(X, correct)

    def predict_correctness_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Predict probability of correct prediction.

        Args:
            features: (N, 12) feature matrix

        Returns:
            (N,) array of correctness probabilities
        """
        X = features.copy()

        if self.feature_subset is not None:
            X = X[:, self.feature_subset]

        if self.normalize:
            X = self.scaler.transform(X)

        return self.classifier.predict_proba(X)[:, 1]

    def predict_correctness(
        self,
        features: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict correctness (binary).

        Args:
            features: (N, 12) feature matrix
            threshold: Decision threshold

        Returns:
            (N,) boolean array of predicted correctness
        """
        proba = self.predict_correctness_proba(features)
        return proba >= threshold

    def evaluate(
        self,
        features: np.ndarray,
        correct: np.ndarray,
    ) -> Dict:
        """
        Evaluate the filter on held-out data.

        Args:
            features: (N, 12) feature matrix
            correct: (N,) boolean correctness labels

        Returns:
            Dictionary with evaluation metrics
        """
        proba = self.predict_correctness_proba(features)
        predictions = proba >= 0.5

        accuracy = accuracy_score(correct, predictions)

        try:
            auc = roc_auc_score(correct, proba)
        except:
            auc = 0.5

        # Detection rates
        tp = ((predictions == True) & (correct == True)).sum()
        fp = ((predictions == True) & (correct == False)).sum()
        tn = ((predictions == False) & (correct == False)).sum()
        fn = ((predictions == False) & (correct == True)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'accuracy': accuracy,
            'auc': auc,
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'mean_predicted_correctness': proba.mean(),
            'actual_accuracy': correct.mean(),
        }

    def save(self, path: Union[str, Path]):
        """Save the trained filter."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler,
                'normalize': self.normalize,
                'feature_subset': self.feature_subset,
            }, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SyntheticSuitabilityFilter':
        """Load a trained filter."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        filter = cls(
            normalize=data['normalize'],
            feature_subset=data['feature_subset'],
        )
        filter.classifier = data['classifier']
        filter.scaler = data['scaler']

        return filter


def create_synthetic_holdout(
    model: torch.nn.Module,
    synthetic_dir: Union[str, Path],
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic hold-out set from generated images.

    Args:
        model: FMoW classifier model
        synthetic_dir: Directory containing synthetic images
        device: Device to use
        batch_size: Batch size for inference
        num_workers: Number of dataloader workers
        verbose: Show progress

    Returns:
        Tuple of (features, correct) arrays
    """
    # Create dataset and dataloader
    dataset = SyntheticFMoWDataset(synthetic_dir)

    if verbose:
        print(f"Loaded {len(dataset)} synthetic images")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Extract features
    results = extract_synthetic_features(model, dataloader, device, verbose)

    return results['features'], results['correct']


def compare_holdout_methods(
    model: torch.nn.Module,
    synthetic_dir: Union[str, Path],
    real_holdout_features: np.ndarray,
    real_holdout_correct: np.ndarray,
    test_features: np.ndarray,
    test_correct: np.ndarray,
    device: torch.device,
    verbose: bool = True,
) -> Dict:
    """
    Compare synthetic vs real hold-out for training the suitability filter.

    Args:
        model: FMoW classifier
        synthetic_dir: Directory with synthetic images
        real_holdout_features: Features from real hold-out (their approach)
        real_holdout_correct: Correctness from real hold-out
        test_features: Test features to evaluate on
        test_correct: Test correctness labels
        device: Device to use
        verbose: Show progress

    Returns:
        Dictionary with comparison results
    """
    results = {}

    # Create synthetic hold-out
    if verbose:
        print("Creating synthetic hold-out...")
    syn_features, syn_correct = create_synthetic_holdout(
        model, synthetic_dir, device, verbose=verbose
    )

    # Train with synthetic hold-out
    if verbose:
        print("\nTraining with synthetic hold-out...")
    syn_filter = SyntheticSuitabilityFilter()
    syn_filter.train(syn_features, syn_correct)
    results['synthetic'] = syn_filter.evaluate(test_features, test_correct)

    # Train with real hold-out (their approach)
    if verbose:
        print("\nTraining with real hold-out...")
    real_filter = SyntheticSuitabilityFilter()
    real_filter.train(real_holdout_features, real_holdout_correct)
    results['real'] = real_filter.evaluate(test_features, test_correct)

    # Summary
    if verbose:
        print("\n" + "=" * 50)
        print("Comparison Results:")
        print("=" * 50)
        print(f"\nSynthetic Hold-out:")
        for k, v in results['synthetic'].items():
            print(f"  {k}: {v:.4f}")
        print(f"\nReal Hold-out (their approach):")
        for k, v in results['real'].items():
            print(f"  {k}: {v:.4f}")

    return results


def main():
    """Example usage and command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Create and evaluate synthetic hold-out")
    parser.add_argument("--synthetic_dir", type=str, required=True, help="Synthetic images directory")
    parser.add_argument("--model_dir", type=str, required=True, help="FMoW model directory")
    parser.add_argument("--output", type=str, default="results/synthetic_filter.pkl", help="Output path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    from src.extract_logits import load_wilds_fmow_model
    model = load_wilds_fmow_model(args.model_dir, device=device)

    # Create synthetic hold-out
    print("\nCreating synthetic hold-out set...")
    features, correct = create_synthetic_holdout(
        model,
        args.synthetic_dir,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Created hold-out with {len(features)} samples")
    print(f"Correctness rate: {correct.mean():.4f}")

    # Train filter
    print("\nTraining suitability filter...")
    filter = SyntheticSuitabilityFilter()
    filter.train(features, correct)

    # Save
    filter.save(args.output)
    print(f"\nSaved filter to {args.output}")

    # Self-evaluation (for sanity check)
    eval_results = filter.evaluate(features, correct)
    print("\nSelf-evaluation (training data):")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
