"""
SilentCare - Build Unified Benchmark Dataset
==============================================
Consolidates FER-2013 test, RAF-DB test, and AffectNet val into a single
balanced benchmark manifest. Per-source per-class equal sampling.

Datasets:
  - FER-2013 test:   local at data/FER2013/test/ (7 class folders)
  - RAF-DB test:     local at Proto_use_case/data/image/RAF-DB/aligned/
  - AffectNet val:   downloaded via HuggingFace datasets lib
  - ExpW:            skipped (not available on HuggingFace)

Usage:
    python scripts/build_unified_benchmark.py
"""

import json
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================
# Configuration
# ============================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "unified_benchmark"

CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4
SEED = 42

# --- FER-2013 ---
FER2013_DIR = PROJECT_DIR / "data" / "FER2013" / "test"
FER_TO_SC = {
    "angry": 1, "disgust": 0, "fear": 0, "sad": 0,
    "happy": 3, "neutral": 3, "surprise": 2,
}

# --- RAF-DB ---
RAFDB_IMAGE_DIR = PROJECT_DIR.parent / "data" / "image" / "RAF-DB" / "aligned"
RAFDB_LABEL_FILE = PROJECT_DIR.parent / "data" / "image" / "EmoLabel" / "list_patition_label.txt"
RAFDB_TO_SC = {1: 2, 2: 0, 3: 0, 4: 3, 5: 0, 6: 1, 7: 3}

# --- AffectNet ---
AFFECTNET_HF_NAME = "Mauregato/affectnet_short"
AFFECTNET_DIR = OUTPUT_DIR / "affectnet"
# AffectNet labels: 0=anger, 1=surprise, 2=contempt, 3=happy, 4=neutral,
#                   5=fear, 6=sad, 7=disgust
AFFECTNET_TO_SC = {0: 1, 1: 2, 2: 0, 3: 3, 4: 3, 5: 0, 6: 0, 7: 0}


# ============================================
# Loaders
# ============================================
def load_fer2013():
    """Load FER-2013 test set paths and labels."""
    entries = []  # list of (path, sc_label, original_label)
    for folder_name, sc_idx in FER_TO_SC.items():
        folder = FER2013_DIR / folder_name
        if not folder.exists():
            print(f"  WARNING: FER-2013 folder not found: {folder}")
            continue
        for f in sorted(folder.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                entries.append((str(f), sc_idx, folder_name))
    return entries


def load_rafdb():
    """Load RAF-DB test partition paths and labels."""
    if not RAFDB_LABEL_FILE.exists():
        print(f"  WARNING: RAF-DB label file not found: {RAFDB_LABEL_FILE}")
        return []
    if not RAFDB_IMAGE_DIR.exists():
        print(f"  WARNING: RAF-DB image dir not found: {RAFDB_IMAGE_DIR}")
        return []

    entries = []
    with open(RAFDB_LABEL_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            filename, raf_label = parts[0], int(parts[1])

            # Only test partition
            if not filename.startswith("test_"):
                continue

            if raf_label not in RAFDB_TO_SC:
                continue

            sc_label = RAFDB_TO_SC[raf_label]
            base = filename.replace(".jpg", "_aligned.jpg")
            img_path = RAFDB_IMAGE_DIR / base

            if img_path.exists():
                entries.append((str(img_path), sc_label, f"raf_{raf_label}"))
    return entries


def load_affectnet():
    """Download and load AffectNet val set from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WARNING: 'datasets' library not installed. Skipping AffectNet.")
        return []

    print("  Downloading AffectNet val from HuggingFace...")
    try:
        ds = load_dataset(AFFECTNET_HF_NAME, split="val")
    except Exception as e:
        print(f"  WARNING: Could not download AffectNet: {e}")
        return []

    AFFECTNET_DIR.mkdir(parents=True, exist_ok=True)

    label_names = ["anger", "surprise", "contempt", "happy", "neutral",
                    "fear", "sad", "disgust"]
    entries = []
    total = len(ds)
    for i, sample in enumerate(ds):
        img = sample["image"]
        label_idx = sample["label"]

        if label_idx not in AFFECTNET_TO_SC:
            continue

        sc_label = AFFECTNET_TO_SC[label_idx]
        orig_label = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)

        # Save image
        img_path = AFFECTNET_DIR / f"affectnet_{i:05d}.png"
        if not img_path.exists():
            img = img.convert("RGB")
            img.save(str(img_path))

        entries.append((str(img_path), sc_label, orig_label))

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{total} images processed...")

    print(f"  AffectNet: {len(entries)} images saved to {AFFECTNET_DIR}")
    return entries


# ============================================
# Balanced Sampling
# ============================================
def balanced_sample(source_data, seed=SEED):
    """
    Per-source per-class balanced sampling.

    For each SilentCare class, find the minimum count across all sources,
    then sample exactly that minimum from EACH source for that class.
    This ensures each source contributes equally within every class.

    Args:
        source_data: dict of {source_name: [(path, sc_label, orig_label), ...]}

    Returns:
        list of (path, sc_label, source_name, orig_label) tuples
    """
    rng = random.Random(seed)
    sources = list(source_data.keys())

    # Group by (source, class)
    by_source_class = defaultdict(list)
    for src in sources:
        for path, sc_label, orig_label in source_data[src]:
            by_source_class[(src, sc_label)].append((path, orig_label))

    # For each class, find minimum count across sources
    result = []
    for cls_idx in range(NUM_CLASSES):
        counts = []
        for src in sources:
            key = (src, cls_idx)
            counts.append(len(by_source_class[key]))

        min_count = min(counts) if counts else 0
        if min_count == 0:
            print(f"  WARNING: Class {CLASSES[cls_idx]} has 0 samples in at least one source. "
                  f"Counts: {dict(zip(sources, counts))}")
            # Use whatever is available from sources that have samples
            for src in sources:
                key = (src, cls_idx)
                items = by_source_class[key]
                for path, orig in items:
                    result.append((path, cls_idx, src, orig))
            continue

        print(f"  {CLASSES[cls_idx]}: min across sources = {min_count} "
              f"(counts: {dict(zip(sources, counts))})")

        # Sample exactly min_count from each source
        for src in sources:
            key = (src, cls_idx)
            items = by_source_class[key]
            sampled = rng.sample(items, min_count)
            for path, orig in sampled:
                result.append((path, cls_idx, src, orig))

    rng.shuffle(result)
    return result


# ============================================
# Main
# ============================================
def main():
    print("=" * 60)
    print("SilentCare - Build Unified Benchmark Dataset")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load each source
    source_data = {}

    print("\n[1/3] Loading FER-2013 test set...")
    fer_entries = load_fer2013()
    if fer_entries:
        source_data["FER-2013"] = fer_entries
        print(f"  FER-2013: {len(fer_entries)} images")
    else:
        print("  FER-2013: NOT FOUND, skipping")

    print("\n[2/3] Loading RAF-DB test partition...")
    raf_entries = load_rafdb()
    if raf_entries:
        source_data["RAF-DB"] = raf_entries
        print(f"  RAF-DB: {len(raf_entries)} images")
    else:
        print("  RAF-DB: NOT FOUND, skipping")

    print("\n[3/3] Loading AffectNet val set...")
    aff_entries = load_affectnet()
    if aff_entries:
        source_data["AffectNet"] = aff_entries
        print(f"  AffectNet: {len(aff_entries)} images")
    else:
        print("  AffectNet: SKIPPED")

    if not source_data:
        print("\nERROR: No datasets available. Cannot build benchmark.")
        sys.exit(1)

    # Print per-source per-class distribution before sampling
    print(f"\n{'='*60}")
    print("Raw distribution (before balancing):")
    for src, entries in source_data.items():
        counts = defaultdict(int)
        for _, sc_label, _ in entries:
            counts[sc_label] += 1
        row = ", ".join(f"{CLASSES[i]}: {counts[i]}" for i in range(NUM_CLASSES))
        print(f"  {src}: total={len(entries)} -> {row}")

    # Balanced sampling
    print(f"\n{'='*60}")
    print("Balanced sampling (per-source per-class equal):")
    benchmark = balanced_sample(source_data)

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("image_path,label,source,original_label\n")
        for path, label, src, orig in benchmark:
            f.write(f"{path},{label},{src},{orig}\n")
    print(f"\nManifest saved: {manifest_path} ({len(benchmark)} entries)")

    # Compute and save stats
    stats = {
        "total_images": len(benchmark),
        "sources": list(source_data.keys()),
        "classes": CLASSES,
        "seed": SEED,
        "per_class": {},
        "per_source": {},
        "per_source_per_class": {},
    }
    class_counts = defaultdict(int)
    source_counts = defaultdict(int)
    src_cls_counts = defaultdict(lambda: defaultdict(int))

    for _, label, src, _ in benchmark:
        class_counts[label] += 1
        source_counts[src] += 1
        src_cls_counts[src][label] += 1

    for i, cls in enumerate(CLASSES):
        stats["per_class"][cls] = class_counts[i]
    for src in source_data.keys():
        stats["per_source"][src] = source_counts[src]
        stats["per_source_per_class"][src] = {
            CLASSES[i]: src_cls_counts[src][i] for i in range(NUM_CLASSES)
        }

    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved: {stats_path}")

    # Print final distribution
    print(f"\n{'='*60}")
    print("Final benchmark distribution:")
    print(f"  Total: {len(benchmark)} images from {len(source_data)} sources")
    print(f"\n  Per class:")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {class_counts[i]}")
    print(f"\n  Per source:")
    for src in source_data.keys():
        print(f"    {src}: {source_counts[src]}")
    print(f"\n  Per source x class:")
    header = f"{'Source':<15}" + "".join(f"{cls:>12}" for cls in CLASSES) + f"{'Total':>12}"
    print(f"    {header}")
    print(f"    {'-' * len(header)}")
    for src in source_data.keys():
        row = f"{src:<15}"
        total = 0
        for i in range(NUM_CLASSES):
            c = src_cls_counts[src][i]
            row += f"{c:>12}"
            total += c
        row += f"{total:>12}"
        print(f"    {row}")

    print(f"\n{'='*60}")
    print("Unified benchmark built successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
