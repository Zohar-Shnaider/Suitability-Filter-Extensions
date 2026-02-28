"""
Synthetic FMoW Image Generation with DiffusionSat

This module generates synthetic FMoW images using the DiffusionSat model,
conditioned on class labels and metadata. The synthetic images are used
to create a hold-out set that doesn't require knowledge of the inference
distribution (addressing the circular assumption in the original paper).
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add DiffusionSat to path
DIFFUSIONSAT_PATH = Path(__file__).parent.parent / "DiffusionSat"
if str(DIFFUSIONSAT_PATH) not in sys.path:
    sys.path.insert(0, str(DIFFUSIONSAT_PATH))

from diffusionsat import SatUNet, DiffusionSatPipeline, metadata_normalize


# FMoW class names (62 classes)
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

# Region mappings for diversity
REGIONS = ["Asia", "Europe", "Africa", "Americas", "Oceania"]

# Sample countries per region for caption generation
REGION_COUNTRIES = {
    "Asia": ["China", "India", "Japan", "South Korea", "Thailand", "Indonesia", "Vietnam"],
    "Europe": ["Germany", "France", "United Kingdom", "Spain", "Italy", "Poland", "Netherlands"],
    "Africa": ["South Africa", "Egypt", "Nigeria", "Kenya", "Morocco", "Ethiopia", "Ghana"],
    "Americas": ["United States", "Brazil", "Canada", "Mexico", "Argentina", "Colombia", "Chile"],
    "Oceania": ["Australia", "New Zealand", "Fiji", "Papua New Guinea"],
}


@dataclass
class MetadataConfig:
    """Configuration for generating diverse metadata."""
    # Years: ID (2002-2012) vs OOD (2013-2017)
    years_id: List[int] = field(default_factory=lambda: list(range(2002, 2013)))
    years_ood: List[int] = field(default_factory=lambda: list(range(2013, 2018)))

    # Regions
    regions: List[str] = field(default_factory=lambda: REGIONS.copy())

    # GSD range (ground sample distance in meters)
    gsd_range: Tuple[float, float] = (0.3, 1.5)

    # Cloud cover range (0-1)
    cloud_cover_range: Tuple[float, float] = (0.0, 0.3)

    # Months/days
    months: List[int] = field(default_factory=lambda: list(range(1, 13)))
    days: List[int] = field(default_factory=lambda: list(range(1, 29)))  # Safe for all months


# metadata_normalize is imported from diffusionsat.data_util


def generate_caption(
    class_name: str,
    country: Optional[str] = None,
    include_fmow_prefix: bool = True,
) -> str:
    """
    Generate a text caption for DiffusionSat.

    Args:
        class_name: FMoW class name (e.g., 'airport', 'hospital')
        country: Optional country name for geographic context
        include_fmow_prefix: Whether to include 'fmow' in caption

    Returns:
        Caption string
    """
    # Convert underscores to spaces
    class_display = ' '.join(class_name.split('_'))

    prefix = "a fmow" if include_fmow_prefix else "a"
    caption = f"{prefix} satellite image of a {class_display}"

    if country:
        caption += f" in {country}"

    return caption


def generate_metadata_variations(
    config: MetadataConfig,
    include_ood: bool = True,
    samples_per_variation: int = 1,
) -> List[Dict]:
    """
    Generate diverse metadata variations for synthetic image generation.

    Args:
        config: Metadata configuration
        include_ood: Whether to include OOD years
        samples_per_variation: Number of samples per unique combination

    Returns:
        List of metadata dictionaries
    """
    variations = []

    years = config.years_id + (config.years_ood if include_ood else [])

    for _ in range(samples_per_variation):
        for year in years:
            for region in config.regions:
                # Sample random values within ranges
                gsd = random.uniform(*config.gsd_range)
                cloud_cover = random.uniform(*config.cloud_cover_range)
                month = random.choice(config.months)
                day = random.choice(config.days)

                # Sample a country from the region
                country = random.choice(REGION_COUNTRIES[region])

                # Generate approximate lon/lat based on region
                lon, lat = _sample_coordinates_for_region(region)

                variations.append({
                    'year': year,
                    'month': month,
                    'day': day,
                    'region': region,
                    'country': country,
                    'gsd': gsd,
                    'cloud_cover': cloud_cover,
                    'lon': lon,
                    'lat': lat,
                })

    return variations


def _sample_coordinates_for_region(region: str) -> Tuple[float, float]:
    """Sample approximate coordinates for a region."""
    # Rough bounding boxes for regions
    region_bounds = {
        "Asia": {"lon": (60, 145), "lat": (5, 55)},
        "Europe": {"lon": (-10, 40), "lat": (35, 70)},
        "Africa": {"lon": (-20, 55), "lat": (-35, 35)},
        "Americas": {"lon": (-130, -35), "lat": (-55, 70)},
        "Oceania": {"lon": (110, 180), "lat": (-50, 0)},
    }

    bounds = region_bounds.get(region, {"lon": (-180, 180), "lat": (-90, 90)})
    lon = random.uniform(*bounds["lon"])
    lat = random.uniform(*bounds["lat"])

    return lon, lat


class DiffusionSatGenerator:
    """
    Generator for synthetic FMoW images using DiffusionSat.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the generator.

        Args:
            checkpoint_path: Path to DiffusionSat checkpoint
            device: Device to use
            dtype: Model dtype (float16 recommended for GPU)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or self._get_device()
        self.dtype = dtype
        self.pipe = None

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _find_best_checkpoint(self) -> Path:
        """Find the highest-numbered checkpoint-* subfolder."""
        ckpt_dirs = sorted(
            self.checkpoint_path.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not ckpt_dirs:
            raise FileNotFoundError(
                f"No checkpoint-* subdirectories found in {self.checkpoint_path}"
            )
        return ckpt_dirs[-1]

    def load_model(self):
        """Load the DiffusionSat pipeline (matches notebook/single-image.ipynb pattern)."""
        best_ckpt = self._find_best_checkpoint()
        print(f"Loading DiffusionSat from {self.checkpoint_path}")
        print(f"  UNet checkpoint: {best_ckpt.name}")

        # Load UNet from the best training checkpoint subfolder
        unet = SatUNet.from_pretrained(
            str(best_ckpt),
            subfolder="unet",
            torch_dtype=self.dtype,
        )

        # Load full pipeline (VAE, tokenizer, scheduler, etc.) from root
        self.pipe = DiffusionSatPipeline.from_pretrained(
            str(self.checkpoint_path),
            unet=unet,
            torch_dtype=self.dtype,
        )
        self.pipe = self.pipe.to(self.device)

        print(f"Model loaded on {self.device}")

    def generate_image(
        self,
        caption: str,
        metadata: List[float],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a single synthetic image.

        Args:
            caption: Text caption for the image
            metadata: Normalized metadata [lon, lat, gsd, cloud_cover, year, month, day]
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            self.load_model()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            caption,
            metadata=metadata,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    def generate_batch(
        self,
        captions: List[str],
        metadata_list: List[List[float]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seeds: Optional[List[int]] = None,
    ) -> List[Image.Image]:
        """
        Generate a batch of synthetic images.

        Args:
            captions: List of text captions
            metadata_list: List of normalized metadata
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seeds: Random seeds for reproducibility

        Returns:
            List of generated PIL Images
        """
        images = []
        for i, (caption, metadata) in enumerate(zip(captions, metadata_list)):
            seed = seeds[i] if seeds else None
            img = self.generate_image(
                caption, metadata, height, width,
                num_inference_steps, guidance_scale, seed
            )
            images.append(img)
        return images


def generate_synthetic_fmow_dataset(
    generator: DiffusionSatGenerator,
    output_dir: Union[str, Path],
    num_samples_per_class: int = 10,
    include_ood_years: bool = True,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    save_metadata: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Generate a complete synthetic FMoW-like dataset.

    Args:
        generator: DiffusionSatGenerator instance
        output_dir: Directory to save generated images
        num_samples_per_class: Number of samples per class
        include_ood_years: Whether to include OOD temporal shifts
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Base random seed
        save_metadata: Whether to save metadata JSON
        verbose: Show progress

    Returns:
        Dictionary with generation statistics and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    config = MetadataConfig()
    metadata_variations = generate_metadata_variations(
        config,
        include_ood=include_ood_years,
        samples_per_variation=1,
    )

    all_metadata = []
    generated_count = 0

    total_images = len(FMOW_CATEGORIES) * num_samples_per_class
    pbar = tqdm(total=total_images, desc="Generating images") if verbose else None

    for class_idx, class_name in enumerate(FMOW_CATEGORIES):
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        for sample_idx in range(num_samples_per_class):
            # Sample a metadata variation
            md_var = random.choice(metadata_variations)

            # Generate caption
            caption = generate_caption(
                class_name,
                country=md_var['country'],
                include_fmow_prefix=True,
            )

            # Prepare raw metadata
            raw_metadata = [
                md_var['lon'] + 180,  # Add base_lon
                md_var['lat'] + 90,   # Add base_lat
                md_var['gsd'],
                md_var['cloud_cover'],
                md_var['year'] - 1980,  # Subtract base_year
                md_var['month'],
                md_var['day'],
            ]

            # Normalize metadata
            norm_metadata = metadata_normalize(raw_metadata).tolist()

            # Generate unique seed for this image
            img_seed = seed + class_idx * 1000 + sample_idx

            try:
                # Generate image
                image = generator.generate_image(
                    caption=caption,
                    metadata=norm_metadata,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=img_seed,
                )

                # Save image
                img_filename = f"{class_name}_{sample_idx:04d}.png"
                image.save(class_dir / img_filename)

                # Store metadata
                sample_metadata = {
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'sample_idx': sample_idx,
                    'filename': str(class_dir / img_filename),
                    'caption': caption,
                    'metadata_raw': raw_metadata,
                    'metadata_normalized': norm_metadata,
                    **md_var,
                }
                all_metadata.append(sample_metadata)

                generated_count += 1

            except Exception as e:
                print(f"Error generating {class_name}_{sample_idx}: {e}")

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    # Save metadata
    if save_metadata:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    stats = {
        'total_generated': generated_count,
        'total_classes': len(FMOW_CATEGORIES),
        'samples_per_class': num_samples_per_class,
        'include_ood': include_ood_years,
        'image_size': (height, width),
    }

    return {'stats': stats, 'metadata': all_metadata}


def generate_for_single_class(
    generator: DiffusionSatGenerator,
    class_name: str,
    output_dir: Union[str, Path],
    num_samples: int = 10,
    include_ood_years: bool = True,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> List[Image.Image]:
    """
    Generate synthetic images for a single class (useful for testing).

    Args:
        generator: DiffusionSatGenerator instance
        class_name: FMoW class name
        output_dir: Output directory
        num_samples: Number of samples to generate
        include_ood_years: Include OOD temporal shifts
        height: Image height
        width: Image width
        num_inference_steps: Denoising steps
        guidance_scale: Guidance scale
        seed: Random seed

    Returns:
        List of generated images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    config = MetadataConfig()
    metadata_variations = generate_metadata_variations(
        config, include_ood=include_ood_years
    )

    images = []
    class_idx = FMOW_CATEGORIES.index(class_name) if class_name in FMOW_CATEGORIES else 0

    for i in range(num_samples):
        md_var = random.choice(metadata_variations)

        caption = generate_caption(class_name, country=md_var['country'])

        raw_metadata = [
            md_var['lon'] + 180,
            md_var['lat'] + 90,
            md_var['gsd'],
            md_var['cloud_cover'],
            md_var['year'] - 1980,
            md_var['month'],
            md_var['day'],
        ]
        norm_metadata = metadata_normalize(raw_metadata).tolist()

        img_seed = seed + class_idx * 1000 + i

        image = generator.generate_image(
            caption=caption,
            metadata=norm_metadata,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=img_seed,
        )

        images.append(image)
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)
        image.save(class_dir / f"{class_name}_{i:04d}.png")

    return images


def main():
    """Command-line interface for synthetic generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic FMoW images with DiffusionSat")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DiffusionSat checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/synthetic", help="Output directory")
    parser.add_argument("--num_samples_per_class", type=int, default=10, help="Samples per class")
    parser.add_argument("--single_class", type=str, default=None, help="Generate for single class only")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--num_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include_ood", action="store_true", help="Include OOD years")
    args = parser.parse_args()

    # Initialize generator
    generator = DiffusionSatGenerator(args.checkpoint)
    generator.load_model()

    if args.single_class:
        # Generate for single class
        images = generate_for_single_class(
            generator,
            args.single_class,
            args.output_dir,
            num_samples=args.num_samples_per_class,
            include_ood_years=args.include_ood,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        print(f"Generated {len(images)} images for class '{args.single_class}'")
    else:
        # Generate full dataset
        result = generate_synthetic_fmow_dataset(
            generator,
            args.output_dir,
            num_samples_per_class=args.num_samples_per_class,
            include_ood_years=args.include_ood,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        print(f"\nGeneration complete!")
        print(f"Stats: {result['stats']}")


if __name__ == "__main__":
    main()
