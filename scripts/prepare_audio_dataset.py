"""
SilentCare - Audio Dataset Preparation Script
==============================================
Downloads, normalizes, and augments audio data from multiple sources
into 4 classes: DISTRESS, ANGRY, ALERT, CALM.

Sources:
1. Donate-a-Cry Corpus (GitHub) -> DISTRESS
2. ESC-50 (GitHub) -> crying_baby->DISTRESS, laughing->CALM, silence->CALM
3. VIVAE (local) -> Fear->DISTRESS, Angry->ANGRY, Surprise->ALERT, Happy->CALM
4. AudioSet via ERTK (optional)

Output: data/audio_dataset/{DISTRESS,ANGRY,ALERT,CALM}/*.wav
All normalized to 22050Hz mono PCM 16-bit, duration 1-10s.
"""

import os
import sys
import shutil
import zipfile
import tempfile
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests
import soundfile as sf
import librosa
from tqdm import tqdm
from tabulate import tabulate

# ============================================
# Configuration
# ============================================
TARGET_SR = 22050
TARGET_CHANNELS = 1  # mono
MIN_DURATION_S = 1.0
MAX_DURATION_S = 10.0
MIN_FILES_PER_CLASS = 400

CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]

# Source URLs
DONATE_A_CRY_URL = "https://github.com/gveres/donateacry-corpus/archive/refs/heads/master.zip"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

# ESC-50 category mapping (category name -> SilentCare class)
ESC50_MAPPING = {
    "crying_baby": "DISTRESS",
    "laughing": "CALM",
    # Silence is not a named category in ESC-50, we skip it
}

# VIVAE mapping (folder name -> SilentCare class)
VIVAE_MAPPING = {
    "Fear": "DISTRESS",
    "Angry": "ANGRY",
    "Surprise": "ALERT",
    "Happy": "CALM",
}


def download_file(url, dest_path, description="Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def normalize_audio(audio, sr):
    """Normalize audio to target sample rate, mono, and clip to 1-10s."""
    # Resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Clip duration
    max_samples = int(MAX_DURATION_S * TARGET_SR)
    min_samples = int(MIN_DURATION_S * TARGET_SR)

    if len(audio) > max_samples:
        audio = audio[:max_samples]

    if len(audio) < min_samples:
        return None  # Too short, skip

    # Normalize amplitude to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio


def save_wav(audio, path):
    """Save audio as 16-bit PCM WAV."""
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, TARGET_SR, subtype="PCM_16")


def process_donate_a_cry(output_dir, temp_dir):
    """Download and process Donate-a-Cry corpus -> all DISTRESS."""
    print("\n" + "=" * 60)
    print("SOURCE 1: Donate-a-Cry Corpus")
    print("=" * 60)

    zip_path = Path(temp_dir) / "donateacry.zip"
    extract_dir = Path(temp_dir) / "donateacry"

    # Download
    print("Downloading Donate-a-Cry corpus...")
    download_file(DONATE_A_CRY_URL, zip_path, "Donate-a-Cry")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find audio files
    count = 0
    dest_dir = output_dir / "DISTRESS"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Search for wav/ogg files in the extracted directory
    for ext in ["*.wav", "*.ogg"]:
        for audio_file in extract_dir.rglob(ext):
            try:
                audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
                audio = normalize_audio(audio, sr)
                if audio is not None:
                    out_name = f"donateacry_{count:04d}.wav"
                    save_wav(audio, dest_dir / out_name)
                    count += 1
            except Exception as e:
                print(f"  Warning: Could not process {audio_file.name}: {e}")

    print(f"  Processed {count} files -> DISTRESS")
    return {"DISTRESS": count}


def process_esc50(output_dir, temp_dir):
    """Download and process ESC-50 -> selected categories."""
    print("\n" + "=" * 60)
    print("SOURCE 2: ESC-50")
    print("=" * 60)

    zip_path = Path(temp_dir) / "esc50.zip"
    extract_dir = Path(temp_dir) / "esc50"

    # Download
    print("Downloading ESC-50 dataset...")
    download_file(ESC50_URL, zip_path, "ESC-50")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Read metadata to find category mappings
    # ESC-50 filename format: {fold}-{clip_id}-{take}-{target}.wav
    # We need the meta/esc50.csv for category names
    meta_files = list(extract_dir.rglob("esc50.csv"))
    if not meta_files:
        print("  ERROR: Could not find esc50.csv metadata file")
        return {}

    import csv
    meta_path = meta_files[0]

    # Parse metadata: filename -> category
    file_to_category = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_to_category[row["filename"]] = row["category"]

    counts = defaultdict(int)

    # Find all audio files
    audio_dir_candidates = list(extract_dir.rglob("audio"))
    if not audio_dir_candidates:
        # Try to find wav files directly
        wav_files = list(extract_dir.rglob("*.wav"))
    else:
        wav_files = []
        for audio_dir in audio_dir_candidates:
            if audio_dir.is_dir():
                wav_files.extend(audio_dir.glob("*.wav"))

    for audio_file in tqdm(wav_files, desc="Processing ESC-50"):
        fname = audio_file.name
        category = file_to_category.get(fname)

        if category not in ESC50_MAPPING:
            continue

        target_class = ESC50_MAPPING[category]

        try:
            audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
            audio = normalize_audio(audio, sr)
            if audio is not None:
                dest_dir = output_dir / target_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                idx = counts[target_class]
                out_name = f"esc50_{category}_{idx:04d}.wav"
                save_wav(audio, dest_dir / out_name)
                counts[target_class] += 1
        except Exception as e:
            print(f"  Warning: Could not process {fname}: {e}")

    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} files from ESC-50")

    return dict(counts)


def process_vivae(output_dir, vivae_dir):
    """Process local VIVAE dataset with class remapping."""
    print("\n" + "=" * 60)
    print("SOURCE 3: VIVAE (local)")
    print("=" * 60)

    vivae_path = Path(vivae_dir)
    if not vivae_path.exists():
        print(f"  ERROR: VIVAE directory not found: {vivae_dir}")
        return {}

    counts = defaultdict(int)

    for folder_name, target_class in VIVAE_MAPPING.items():
        folder_path = vivae_path / folder_name
        if not folder_path.exists():
            print(f"  Warning: VIVAE folder not found: {folder_name}")
            continue

        wav_files = list(folder_path.glob("*.wav"))
        dest_dir = output_dir / target_class
        dest_dir.mkdir(parents=True, exist_ok=True)

        for audio_file in tqdm(wav_files, desc=f"VIVAE {folder_name}->{target_class}"):
            try:
                audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
                audio = normalize_audio(audio, sr)
                if audio is not None:
                    idx = counts[target_class]
                    out_name = f"vivae_{folder_name.lower()}_{idx:04d}.wav"
                    save_wav(audio, dest_dir / out_name)
                    counts[target_class] += 1
            except Exception as e:
                print(f"  Warning: Could not process {audio_file.name}: {e}")

    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} files from VIVAE")

    return dict(counts)


def augment_audio(audio, sr, method):
    """Apply a single augmentation method to audio."""
    if method == "time_stretch_fast":
        return librosa.effects.time_stretch(audio, rate=1.1)
    elif method == "time_stretch_slow":
        return librosa.effects.time_stretch(audio, rate=0.9)
    elif method == "pitch_up":
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    elif method == "pitch_down":
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
    elif method == "add_noise":
        noise = np.random.randn(len(audio)) * 0.01  # -20dB approx
        return audio + noise
    return audio


def augment_class(output_dir, class_name, target_count):
    """Augment a class to reach target_count files."""
    class_dir = output_dir / class_name
    existing_files = list(class_dir.glob("*.wav"))
    current_count = len(existing_files)

    if current_count >= target_count:
        return 0

    needed = target_count - current_count
    print(f"  {class_name}: need {needed} more files (have {current_count}, target {target_count})")

    augmentation_methods = [
        "time_stretch_fast", "time_stretch_slow",
        "pitch_up", "pitch_down", "add_noise"
    ]

    aug_count = 0
    file_idx = 0

    while aug_count < needed:
        # Cycle through existing files
        source_file = existing_files[file_idx % len(existing_files)]
        method = augmentation_methods[aug_count % len(augmentation_methods)]

        try:
            audio, sr = librosa.load(str(source_file), sr=TARGET_SR, mono=True)
            augmented = augment_audio(audio, sr, method)

            # Re-normalize after augmentation
            augmented = normalize_audio(augmented, TARGET_SR)
            if augmented is not None:
                out_name = f"aug_{method}_{aug_count:04d}.wav"
                save_wav(augmented, class_dir / out_name)
                aug_count += 1
        except Exception as e:
            print(f"    Warning: Augmentation failed for {source_file.name} ({method}): {e}")

        file_idx += 1

        # Safety: prevent infinite loop
        if file_idx > len(existing_files) * len(augmentation_methods) * 2:
            print(f"    Warning: Could not generate enough augmented files for {class_name}")
            break

    return aug_count


def generate_report(output_dir, source_counts):
    """Generate final report with file counts per class and source."""
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    # Count per class
    class_counts = {}
    for class_name in CLASSES:
        class_dir = output_dir / class_name
        if class_dir.exists():
            files = list(class_dir.glob("*.wav"))
            class_counts[class_name] = len(files)

            # Count by source
            source_breakdown = defaultdict(int)
            for f in files:
                name = f.name
                if name.startswith("donateacry_"):
                    source_breakdown["Donate-a-Cry"] += 1
                elif name.startswith("esc50_"):
                    source_breakdown["ESC-50"] += 1
                elif name.startswith("vivae_"):
                    source_breakdown["VIVAE"] += 1
                elif name.startswith("aug_"):
                    source_breakdown["Augmented"] += 1
                else:
                    source_breakdown["Other"] += 1

            class_counts[class_name] = {
                "total": len(files),
                "breakdown": dict(source_breakdown),
            }
        else:
            class_counts[class_name] = {"total": 0, "breakdown": {}}

    # Print summary table
    headers = ["Class", "Total", "Donate-a-Cry", "ESC-50", "VIVAE", "Augmented"]
    rows = []
    for cls in CLASSES:
        info = class_counts[cls]
        bd = info["breakdown"]
        rows.append([
            cls,
            info["total"],
            bd.get("Donate-a-Cry", 0),
            bd.get("ESC-50", 0),
            bd.get("VIVAE", 0),
            bd.get("Augmented", 0),
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    total_files = sum(info["total"] for info in class_counts.values())
    print(f"\nTotal files: {total_files}")
    print(f"Target per class: {MIN_FILES_PER_CLASS}")

    # Check if all classes meet minimum
    all_ok = all(class_counts[cls]["total"] >= MIN_FILES_PER_CLASS for cls in CLASSES)
    if all_ok:
        print("All classes meet the minimum file count.")
    else:
        for cls in CLASSES:
            if class_counts[cls]["total"] < MIN_FILES_PER_CLASS:
                print(f"WARNING: {cls} has only {class_counts[cls]['total']} files "
                      f"(minimum: {MIN_FILES_PER_CLASS})")

    return class_counts


def main():
    parser = argparse.ArgumentParser(description="Prepare SilentCare audio dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--vivae-dir",
        type=str,
        default=None,
        help="Path to local VIVAE dataset",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading external datasets (use existing)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip data augmentation",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "data" / "audio_dataset"
    vivae_dir = args.vivae_dir or str(
        project_dir.parent / "data" / "audio" / "vivae_excluded"
    )

    print("=" * 60)
    print("SilentCare - Audio Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"VIVAE directory: {vivae_dir}")
    print(f"Target sample rate: {TARGET_SR} Hz")
    print(f"Target: mono PCM 16-bit")
    print(f"Duration range: {MIN_DURATION_S}-{MAX_DURATION_S}s")
    print(f"Min files per class: {MIN_FILES_PER_CLASS}")

    # Create output directories
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    source_counts = defaultdict(lambda: defaultdict(int))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Source 1: Donate-a-Cry
        if not args.skip_download:
            counts = process_donate_a_cry(output_dir, temp_dir)
            for cls, cnt in counts.items():
                source_counts["Donate-a-Cry"][cls] = cnt

            # Source 2: ESC-50
            counts = process_esc50(output_dir, temp_dir)
            for cls, cnt in counts.items():
                source_counts["ESC-50"][cls] = cnt
        else:
            print("\nSkipping downloads (--skip-download)")

        # Source 3: VIVAE (local)
        counts = process_vivae(output_dir, vivae_dir)
        for cls, cnt in counts.items():
            source_counts["VIVAE"][cls] = cnt

    # Augmentation
    if not args.no_augment:
        print("\n" + "=" * 60)
        print("DATA AUGMENTATION")
        print("=" * 60)

        for cls in CLASSES:
            cls_dir = output_dir / cls
            current = len(list(cls_dir.glob("*.wav")))
            if current < MIN_FILES_PER_CLASS:
                aug_count = augment_class(output_dir, cls, MIN_FILES_PER_CLASS)
                print(f"  {cls}: augmented +{aug_count} files")
            else:
                print(f"  {cls}: {current} files, no augmentation needed")
    else:
        print("\nSkipping augmentation (--no-augment)")

    # Final report
    report = generate_report(output_dir, source_counts)

    print("\nDataset preparation complete!")
    print(f"Dataset location: {output_dir}")

    return report


if __name__ == "__main__":
    main()
