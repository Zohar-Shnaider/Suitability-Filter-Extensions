"""Run full logit extraction for all FMoW splits."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extract_logits import (
    load_wilds_fmow_model,
    extract_logits_for_splits,
    extract_logits_for_filtered_splits,
    save_logits,
    get_device,
    ID_SPLITS, OOD_SPLITS,
    ID_FILTERED_SPLITS, OOD_FILTERED_SPLITS,
)

MODEL_ROOT = "suitability"       # has experiments/fmow/checkpoint
DATA_ROOT  = "data"              # has fmow_v1.1/
OUTPUT_DIR = Path("results/logits")
BATCH_SIZE = 64
NUM_WORKERS = 4

device = get_device()
print(f"Device: {device}")
print(f"Output: {OUTPUT_DIR}\n")

# ── Load model ──────────────────────────────────────────────
print("Loading FMoW model...")
model = load_wilds_fmow_model(MODEL_ROOT, device=device)
print("Model loaded.\n")

# ── 1. Main splits (id_val, id_test, val, test) ────────────
t0 = time.time()
print("=" * 60)
print("Phase 1: Main splits")
print("=" * 60)

main_results = extract_logits_for_splits(
    model, DATA_ROOT, ID_SPLITS + OOD_SPLITS,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device,
)
out = OUTPUT_DIR / "fmow_ERM_best_0_main.pkl"
save_logits(main_results, out, format="pkl")
print(f"Saved → {out}")
for split, data in main_results.items():
    acc = data["correct"].mean()
    print(f"  {split}: {data['logits'].shape[0]} samples, accuracy={acc:.4f}")
print(f"Phase 1 done in {(time.time()-t0)/60:.1f} min\n")

# ── 2. ID filtered splits ──────────────────────────────────
t1 = time.time()
print("=" * 60)
print("Phase 2: ID filtered splits")
print("=" * 60)

id_filtered = extract_logits_for_filtered_splits(
    model, DATA_ROOT, ID_FILTERED_SPLITS,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device,
)
out = OUTPUT_DIR / "fmow_ERM_best_0_id_filtered.pkl"
save_logits(id_filtered, out, format="pkl")
print(f"Saved → {out}")
print(f"Phase 2 done in {(time.time()-t1)/60:.1f} min\n")

# ── 3. OOD filtered splits ─────────────────────────────────
t2 = time.time()
print("=" * 60)
print("Phase 3: OOD filtered splits")
print("=" * 60)

ood_filtered = extract_logits_for_filtered_splits(
    model, DATA_ROOT, OOD_FILTERED_SPLITS,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device=device,
)
out = OUTPUT_DIR / "fmow_ERM_best_0_ood_filtered.pkl"
save_logits(ood_filtered, out, format="pkl")
print(f"Saved → {out}")
print(f"Phase 3 done in {(time.time()-t2)/60:.1f} min\n")

# ── Summary ─────────────────────────────────────────────────
total = time.time() - t0
print("=" * 60)
print(f"ALL DONE in {total/60:.1f} min")
print("=" * 60)
print(f"Output files:")
for f in sorted(OUTPUT_DIR.glob("*.pkl")):
    print(f"  {f}  ({f.stat().st_size / 1e6:.1f} MB)")
