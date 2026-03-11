"""
SilentCare - Generate Fusion Ablation Study Results
====================================================
Evaluates fusion strategy using synthetic intra-class pairing:
  - Audio test predictions (from Step 1)
  - Video test predictions (from Step 2, ResNet50)
  - Random pairing within same class (random_state=42)

Three configurations compared:
  1. Audio only (weight 1.0 / 0.0)
  2. Video only (weight 0.0 / 1.0)
  3. SilentCare Adaptive Fusion (0.35 / 0.65 + agreement boost)

Usage:
    python scripts/generate_fusion_results.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ============================================
# Configuration
# ============================================
CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4
RANDOM_STATE = 42

# Fusion weights (matching production config.py)
AUDIO_WEIGHT = 0.30
VIDEO_WEIGHT = 0.70
AGREEMENT_BOOST = 1.3
UNCERTAINTY_THRESHOLD = 0.3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
AUDIO_PREDS = PROJECT_DIR / "results" / "audio" / "test_predictions.npz"
VIDEO_PREDS = PROJECT_DIR / "results" / "video" / "test_predictions.npz"
RESULTS_DIR = PROJECT_DIR / "results" / "fusion"

CLASS_COLORS = {
    "DISTRESS": "#e74c3c",
    "ANGRY": "#e67e22",
    "ALERT": "#f1c40f",
    "CALM": "#2ecc71",
}


def load_predictions():
    """Load predictions from audio and video steps."""
    print("Loading predictions...")

    audio_data = np.load(str(AUDIO_PREDS))
    video_data = np.load(str(VIDEO_PREDS))

    audio_y_true = audio_data["y_true"]
    audio_y_prob = audio_data["y_prob"]

    video_y_true = video_data["y_true"]
    video_y_prob = video_data["resnet_prob"]  # Use ResNet50 (best model)

    print(f"  Audio test set: {len(audio_y_true)} samples")
    print(f"  Video test set: {len(video_y_true)} samples")

    return audio_y_true, audio_y_prob, video_y_true, video_y_prob


def create_synthetic_pairs(audio_y_true, audio_y_prob, video_y_true, video_y_prob):
    """
    Create synthetic intra-class pairs.
    For each class, randomly pair audio and video predictions.
    The ground truth label is the shared class.
    """
    rng = np.random.RandomState(RANDOM_STATE)

    paired_audio_prob = []
    paired_video_prob = []
    paired_labels = []

    for cls_idx in range(NUM_CLASSES):
        audio_mask = audio_y_true == cls_idx
        video_mask = video_y_true == cls_idx

        audio_indices = np.where(audio_mask)[0]
        video_indices = np.where(video_mask)[0]

        n_audio = len(audio_indices)
        n_video = len(video_indices)

        if n_audio == 0 or n_video == 0:
            print(f"  WARNING: No samples for class {CLASSES[cls_idx]}")
            continue

        # Determine number of pairs: min of both sets
        n_pairs = min(n_audio, n_video)

        # Shuffle and take first n_pairs
        rng.shuffle(audio_indices)
        rng.shuffle(video_indices)

        for i in range(n_pairs):
            paired_audio_prob.append(audio_y_prob[audio_indices[i]])
            paired_video_prob.append(video_y_prob[video_indices[i]])
            paired_labels.append(cls_idx)

    paired_audio_prob = np.array(paired_audio_prob)
    paired_video_prob = np.array(paired_video_prob)
    paired_labels = np.array(paired_labels)

    print(f"\n  Total synthetic pairs: {len(paired_labels)}")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls}: {np.sum(paired_labels == i)}")

    return paired_audio_prob, paired_video_prob, paired_labels


def fuse_audio_only(audio_prob, video_prob):
    """Audio-only prediction."""
    return np.argmax(audio_prob, axis=1)


def fuse_video_only(audio_prob, video_prob):
    """Video-only prediction."""
    return np.argmax(video_prob, axis=1)


def fuse_silentcare(audio_prob, video_prob):
    """
    SilentCare adaptive fusion:
      1. Weighted sum: 0.35 * audio + 0.65 * video
      2. Agreement boost: if both agree, multiply by 1.3
      3. Uncertainty: if max confidence < 0.3, fall back to video
    """
    n = len(audio_prob)
    fused = AUDIO_WEIGHT * audio_prob + VIDEO_WEIGHT * video_prob

    audio_pred = np.argmax(audio_prob, axis=1)
    video_pred = np.argmax(video_prob, axis=1)

    # Agreement boost
    for i in range(n):
        if audio_pred[i] == video_pred[i]:
            fused[i, audio_pred[i]] *= AGREEMENT_BOOST

    # Normalize
    row_sums = fused.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    fused = fused / row_sums

    # Uncertainty fallback
    max_conf = np.max(fused, axis=1)
    predictions = np.argmax(fused, axis=1)
    for i in range(n):
        if max_conf[i] < UNCERTAINTY_THRESHOLD:
            predictions[i] = video_pred[i]

    return predictions


def compute_config_metrics(y_true, y_pred, config_name):
    """Compute metrics for a fusion config."""
    metrics = {
        "config": config_name,
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {},
    }

    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, cls in enumerate(CLASSES):
        metrics["per_class"][cls] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(np.sum(y_true == i)),
        }

    return metrics


def plot_fusion_comparison(results):
    """Bar plot comparing 3 fusion configurations."""
    configs = [r["config"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1_macro"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = np.arange(len(configs))
    width = 0.35
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy",
                   color=colors, alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1 Macro",
                   color=colors, alpha=0.55, edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", fontsize=10)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Ablation Study - Fusion Strategy Impact\n(Synthetic Intra-Class Pairing, random_state=42)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Highlight best
    best_idx = int(np.argmax(accs))
    ax.annotate("BEST", xy=(x[best_idx], accs[best_idx] + 0.04),
                fontsize=10, fontweight="bold", color="#27ae60", ha="center")

    plt.tight_layout()
    out_path = RESULTS_DIR / "fusion_comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nFusion comparison saved to {out_path}")


def plot_fusion_per_class(results):
    """Per-class F1 comparison for each fusion config."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = np.arange(NUM_CLASSES)
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, r in enumerate(results):
        f1_vals = [r["per_class"][cls]["f1"] for cls in CLASSES]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, f1_vals, width, label=r["config"],
                      color=colors[i], alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", fontsize=8)

    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1 by Fusion Strategy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "fusion_per_class.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Per-class fusion plot saved to {out_path}")


def plot_fusion_confusion_matrix(y_true, y_pred):
    """Confusion matrix for best fusion strategy."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Greens",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        vmin=0, vmax=100, square=True,
        cbar_kws={"label": "Percentage (%)"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Fusion (SilentCare Adaptive) - Confusion Matrix\n(Synthetic Intra-Class Pairing)",
                 fontsize=13, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j + 0.5, i + 0.75, f"(n={cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="gray")

    plt.tight_layout()
    out_path = RESULTS_DIR / "fusion_confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Fusion confusion matrix saved to {out_path}")


def main():
    print("=" * 60)
    print("SilentCare - Fusion Ablation Study")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions
    audio_y_true, audio_y_prob, video_y_true, video_y_prob = load_predictions()

    # Create synthetic pairs
    paired_audio, paired_video, paired_labels = create_synthetic_pairs(
        audio_y_true, audio_y_prob, video_y_true, video_y_prob
    )

    # Three fusion configurations
    configs = [
        ("Audio Only", fuse_audio_only),
        ("Video Only", fuse_video_only),
        ("SilentCare Fusion\n(0.30/0.70 + adaptive)", fuse_silentcare),
    ]

    results = []
    for name, fuse_fn in configs:
        y_pred = fuse_fn(paired_audio, paired_video)
        metrics = compute_config_metrics(paired_labels, y_pred, name)
        results.append(metrics)

        print(f"\n--- {name} ---")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        for cls in CLASSES:
            m = metrics["per_class"][cls]
            print(f"    {cls}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    # Plots
    plot_fusion_comparison(results)
    plot_fusion_per_class(results)

    # Confusion matrix for fusion strategy
    fusion_pred = fuse_silentcare(paired_audio, paired_video)
    plot_fusion_confusion_matrix(paired_labels, fusion_pred)

    # Save classification report for fusion
    report = classification_report(paired_labels, fusion_pred,
                                   target_names=CLASSES, digits=4)
    report_path = RESULTS_DIR / "classification_report_fusion.txt"
    with open(report_path, "w") as f:
        f.write("SilentCare Fusion - Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write("Strategy: Adaptive weighted fusion (audio=0.35, video=0.65)\n")
        f.write("Data: Synthetic intra-class pairing (random_state=42)\n")
        f.write(f"Total paired samples: {len(paired_labels)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"\nClassification report saved to {report_path}")

    # Save metrics JSON
    fusion_data = {
        "methodology": "synthetic intra-class pairing",
        "random_state": RANDOM_STATE,
        "note": (
            "Audio and video predictions are paired within the same class "
            "(intra-class random pairing). This simulates simultaneous audio+video "
            "capture where both modalities observe the same emotional state. "
            "Ground truth labels are shared (same class). This is methodologically "
            "valid as an ablation study to compare fusion strategies, but does NOT "
            "represent real paired audio-video recordings."
        ),
        "audio_source": "Audio_SilentCare_model.h5 on audio_dataset test split (20%, rs=42)",
        "video_source": "Video_SilentCare_model.pth (ResNet50) on RAF-DB test partition",
        "fusion_weights": {
            "audio": AUDIO_WEIGHT,
            "video": VIDEO_WEIGHT,
            "agreement_boost": AGREEMENT_BOOST,
            "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        },
        "num_paired_samples": int(len(paired_labels)),
        "pairs_per_class": {
            cls: int(np.sum(paired_labels == i))
            for i, cls in enumerate(CLASSES)
        },
        "results": results,
    }

    out_path = RESULTS_DIR / "fusion_metrics.json"
    with open(out_path, "w") as f:
        json.dump(fusion_data, f, indent=2)
    print(f"Fusion metrics saved to {out_path}")

    print(f"\n{'='*60}")
    print("Fusion results generation complete!")
    print(f"All outputs in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
