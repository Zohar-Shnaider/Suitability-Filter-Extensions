"""
Generate synthetic FMoW images at multiple per-class scales for the scaling
experiment.

Scales: 5, 10, 25, 50, 100, 180 images/class  (180/cls ≈ 11,160 total).

Output layout:
    results/synthetic_scaling/
        scale_005/  scale_010/  scale_025/  scale_050/  scale_100/  scale_180/

Each directory follows the generate_synthetic_fmow_dataset() convention:
    {class_name}/{class}_{idx:04d}.png  +  metadata.json

Resume support: scales whose metadata.json already exists are skipped.

Usage:
    python scripts/run_synthetic_generation.py                          # all scales
    python scripts/run_synthetic_generation.py --scales 5 10            # subset
    python scripts/run_synthetic_generation.py --checkpoint /path/to/ckpt
"""

import sys
import time
import argparse
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.generate_synthetic import DiffusionSatGenerator, generate_synthetic_fmow_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = str(
    PROJ_ROOT / "checkpoints" / "finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
)
OUTPUT_ROOT = PROJ_ROOT / "results" / "synthetic_scaling"
ALL_SCALES = [5, 10, 25, 50, 100, 180]

# Seeding: each scale gets an independent base seed so that
# adding a new scale doesn't change existing images.
BASE_SEED = 42
SEED_STRIDE = 10000  # seed_i = BASE_SEED + i * SEED_STRIDE


def scale_dir_name(scale: int) -> str:
    return f"scale_{scale:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic FMoW images at multiple scales"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="Path to DiffusionSat checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_ROOT),
        help="Root output directory",
    )
    parser.add_argument(
        "--scales", type=int, nargs="*", default=None,
        help="Subset of scales to generate (default: all)",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate even if metadata.json exists",
    )
    args = parser.parse_args()

    scales = args.scales if args.scales else ALL_SCALES
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- Plan what to generate ----
    to_generate = []
    for idx, scale in enumerate(ALL_SCALES):
        if scale not in scales:
            continue
        sdir = output_root / scale_dir_name(scale)
        meta = sdir / "metadata.json"
        if meta.exists() and not args.force:
            print(f"[skip] {sdir.name}: metadata.json exists (use --force to overwrite)")
            continue
        to_generate.append((idx, scale, sdir))

    if not to_generate:
        print("Nothing to generate. Done.")
        return

    total_images = sum(s * 62 for _, s, _ in to_generate)
    print(f"\nWill generate {len(to_generate)} scale(s), {total_images:,} images total.")
    print(f"Checkpoint: {args.checkpoint}\n")

    # ---- Load model once ----
    generator = DiffusionSatGenerator(args.checkpoint)
    generator.load_model()

    # ---- Generate each scale ----
    for idx, scale, sdir in to_generate:
        seed = BASE_SEED + idx * SEED_STRIDE
        n_images = scale * 62

        print(f"\n{'='*60}")
        print(f"Scale {scale}/cls  ({n_images:,} images)  seed={seed}")
        print(f"Output: {sdir}")
        print(f"{'='*60}")

        t0 = time.time()

        result = generate_synthetic_fmow_dataset(
            generator=generator,
            output_dir=sdir,
            num_samples_per_class=scale,
            include_ood_years=True,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=seed,
            save_metadata=True,
            verbose=True,
        )

        elapsed = time.time() - t0
        stats = result["stats"]
        print(
            f"\nDone: {stats['total_generated']:,}/{n_images:,} images in "
            f"{elapsed/60:.1f} min ({elapsed/max(stats['total_generated'],1):.1f}s/img)"
        )

    print("\nAll scales complete.")


if __name__ == "__main__":
    main()
