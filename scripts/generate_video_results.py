"""
SilentCare - Generate Video Experimental Results
=================================================
Evaluates TWO video models on RAF-DB test set:
  1. ResNet50 local (Video_SilentCare_model.pth)
  2. ViT HuggingFace (trpakov/vit-face-expression)

Produces metrics, confusion matrices, model comparison,
distribution plots, and classification reports.

Usage:
    python scripts/generate_video_results.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score
)
from tqdm import tqdm

# ============================================
# Configuration
# ============================================
CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 32

# RAF-DB 1-indexed label -> SilentCare class index
RAFDB_TO_SILENTCARE = {
    1: 2,  # Surprise -> ALERT
    2: 0,  # Fear -> DISTRESS
    3: 0,  # Disgust -> DISTRESS
    4: 3,  # Happy -> CALM
    5: 0,  # Sad -> DISTRESS
    6: 1,  # Angry -> ANGRY
    7: 3,  # Neutral -> CALM
}

# HuggingFace FER labels -> SilentCare class index
FER_TO_SILENTCARE = {
    "angry": 1,
    "disgust": 0,
    "fear": 0,
    "happy": 3,
    "neutral": 3,
    "sad": 0,
    "surprise": 2,
}

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent / "data" / "image"
RAFDB_IMAGE_DIR = DATA_DIR / "RAF-DB" / "aligned"
RAFDB_LABEL_FILE = DATA_DIR / "EmoLabel" / "list_patition_label.txt"
MODEL_DIR = PROJECT_DIR / "model"
RESNET_MODEL_PATH = MODEL_DIR / "Video_SilentCare_model.pth"
HISTORY_PATH = MODEL_DIR / "video_training_history.json"
RESULTS_DIR = PROJECT_DIR / "results" / "video"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = {
    "DISTRESS": "#e74c3c",
    "ANGRY": "#e67e22",
    "ALERT": "#f1c40f",
    "CALM": "#2ecc71",
}

FEATURE_DIM = 512


# ============================================
# ResNet50 Model (same architecture as training)
# ============================================
class SilentCareVideoModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.projection(x)
        x = self.classifier(x)
        return x


# ============================================
# Data Loading
# ============================================
def load_rafdb_test_set():
    """Load RAF-DB test partition with SilentCare class mapping."""
    print(f"Loading RAF-DB labels from {RAFDB_LABEL_FILE}")

    image_paths = []
    labels = []
    skipped = 0

    with open(RAFDB_LABEL_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue

            filename = parts[0]
            rafdb_label = int(parts[1])

            # Only test partition
            if not filename.startswith("test_"):
                continue

            if rafdb_label not in RAFDB_TO_SILENTCARE:
                skipped += 1
                continue

            silentcare_label = RAFDB_TO_SILENTCARE[rafdb_label]
            base = filename.replace(".jpg", "_aligned.jpg")
            img_path = RAFDB_IMAGE_DIR / base

            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(silentcare_label)
            else:
                skipped += 1

    labels = np.array(labels)
    print(f"Test set: {len(image_paths)} images, skipped {skipped}")
    print("Class distribution:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {np.sum(labels == i)}")

    return image_paths, labels


# ============================================
# ResNet50 Evaluation
# ============================================
def evaluate_resnet50(image_paths, labels):
    """Evaluate the local ResNet50 model on RAF-DB test set."""
    print(f"\n{'='*50}")
    print("Evaluating ResNet50 (Video_SilentCare_model.pth)")
    print(f"{'='*50}")

    # Load model
    model = SilentCareVideoModel()
    state_dict = torch.load(str(RESNET_MODEL_PATH), map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    # Transform (same as training validation)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_probs = []
    all_preds = []
    inference_times = []

    with torch.no_grad():
        for i in tqdm(range(len(image_paths)), desc="ResNet50 inference"):
            img = Image.open(image_paths[i]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            start = time.perf_counter()
            logits = model(tensor)
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed * 1000)  # ms

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            all_preds.append(int(np.argmax(probs)))

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    avg_time = np.mean(inference_times)

    print(f"Average inference time: {avg_time:.2f} ms/image")

    return y_pred, y_prob, avg_time


# ============================================
# ViT HuggingFace Evaluation
# ============================================
def evaluate_vit_hf(image_paths, labels):
    """Evaluate HuggingFace ViT (trpakov/vit-face-expression) on RAF-DB test set."""
    print(f"\n{'='*50}")
    print("Evaluating ViT HuggingFace (trpakov/vit-face-expression)")
    print(f"{'='*50}")

    from transformers import pipeline as hf_pipeline

    print("Loading HuggingFace model...")
    classifier = hf_pipeline(
        "image-classification",
        model="trpakov/vit-face-expression",
        top_k=7,
        device=0 if torch.cuda.is_available() else -1,
    )
    print("Model loaded.")

    all_probs = []
    all_preds = []
    inference_times = []

    for i in tqdm(range(len(image_paths)), desc="ViT inference"):
        img = Image.open(image_paths[i]).convert("RGB")

        start = time.perf_counter()
        results = classifier(img)
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed * 1000)

        # Map 7 FER labels to 4 SilentCare classes
        probs = np.zeros(NUM_CLASSES, dtype=np.float64)
        for r in results:
            label = r["label"]
            if label in FER_TO_SILENTCARE:
                idx = FER_TO_SILENTCARE[label]
                probs[idx] += r["score"]

        all_probs.append(probs)
        all_preds.append(int(np.argmax(probs)))

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    avg_time = np.mean(inference_times)

    print(f"Average inference time: {avg_time:.2f} ms/image")

    return y_pred, y_prob, avg_time


# ============================================
# Metrics and Plots
# ============================================
def compute_metrics(y_true, y_pred, model_name, avg_time=None):
    """Compute metrics dict for a model."""
    metrics = {
        "model": model_name,
        "num_test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {},
    }
    if avg_time is not None:
        metrics["avg_inference_time_ms"] = float(avg_time)

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


def print_metrics(metrics):
    """Print metrics summary."""
    print(f"\n  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    if "avg_inference_time_ms" in metrics:
        print(f"  Avg inference: {metrics['avg_inference_time_ms']:.2f} ms")
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*54}")
    for cls in CLASSES:
        m = metrics["per_class"][cls]
        print(f"  {cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generate normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        vmin=0, vmax=100, square=True,
        cbar_kws={"label": "Percentage (%)"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j + 0.5, i + 0.75, f"(n={cm[i, j]})",
                    ha="center", va="center", fontsize=7, color="gray")

    plt.tight_layout()
    out_path = RESULTS_DIR / filename
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Confusion matrix saved to {out_path}")


def plot_dataset_distribution(labels):
    """Plot test set class distribution."""
    counts = [int(np.sum(labels == i)) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = [CLASS_COLORS[c] for c in CLASSES]
    bars = ax.bar(CLASSES, counts, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10, str(int(h)),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("RAF-DB Test Set - Mapped to SilentCare Classes", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    out_path = RESULTS_DIR / "dataset_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Distribution plot saved to {out_path}")


def plot_model_comparison(metrics_resnet, metrics_vit):
    """Side-by-side comparison of both models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    # --- Panel 1: Overall metrics comparison ---
    ax = axes[0]
    metric_names = ["Accuracy", "F1 Macro", "F1 Weighted"]
    resnet_vals = [metrics_resnet["accuracy"], metrics_resnet["f1_macro"], metrics_resnet["f1_weighted"]]
    vit_vals = [metrics_vit["accuracy"], metrics_vit["f1_macro"], metrics_vit["f1_weighted"]]

    x = np.arange(len(metric_names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, resnet_vals, width, label="ResNet50", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, vit_vals, width, label="ViT (HF)", color="#e74c3c", alpha=0.85)

    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Overall Metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: Per-class F1 comparison ---
    ax = axes[1]
    resnet_f1 = [metrics_resnet["per_class"][c]["f1"] for c in CLASSES]
    vit_f1 = [metrics_vit["per_class"][c]["f1"] for c in CLASSES]

    x = np.arange(NUM_CLASSES)
    bars1 = ax.bar(x - width / 2, resnet_f1, width, label="ResNet50", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, vit_f1, width, label="ViT (HF)", color="#e74c3c", alpha=0.85)

    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=8)
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=8)

    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title("Per-Class F1 Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: Inference time comparison ---
    ax = axes[2]
    times = []
    labels = []
    if "avg_inference_time_ms" in metrics_resnet:
        times.append(metrics_resnet["avg_inference_time_ms"])
        labels.append("ResNet50")
    if "avg_inference_time_ms" in metrics_vit:
        times.append(metrics_vit["avg_inference_time_ms"])
        labels.append("ViT (HF)")

    colors = ["#3498db", "#e74c3c"][:len(times)]
    bars = ax.barh(labels, times, color=colors, alpha=0.85, height=0.5)
    ax.axvline(x=100, color="red", linestyle="--", linewidth=1.5, label="100ms threshold")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} ms", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Inference Time (ms)", fontsize=11)
    ax.set_title("Inference Speed", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Video Model Comparison: ResNet50 vs ViT (HuggingFace)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = RESULTS_DIR / "model_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Model comparison saved to {out_path}")


def plot_training_curves():
    """Plot video training curves if history exists."""
    if not HISTORY_PATH.exists():
        print("  Video training history not found, skipping.")
        return

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-s", markersize=3, label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Video ResNet50 - Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], "r-s", markersize=3, label="Val Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Video ResNet50 - Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "training_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Training curves saved to {out_path}")


def save_classification_report(y_true, y_pred, model_name, filename):
    """Save sklearn classification report to text file."""
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    out_path = RESULTS_DIR / filename
    with open(out_path, "w") as f:
        f.write(f"SilentCare Video Model - Classification Report (Test Set)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: RAF-DB test partition (mapped to 4 SilentCare classes)\n")
        f.write(f"Total test samples: {len(y_true)}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)
    print(f"  Report saved to {out_path}")


def main():
    print("=" * 60)
    print("SilentCare - Video Results Generation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test set
    image_paths, y_true = load_rafdb_test_set()

    # ===== ResNet50 =====
    resnet_pred, resnet_prob, resnet_time = evaluate_resnet50(image_paths, y_true)
    metrics_resnet = compute_metrics(y_true, resnet_pred, "ResNet50 (Video_SilentCare_model.pth)", resnet_time)
    print_metrics(metrics_resnet)
    plot_confusion_matrix(y_true, resnet_pred,
                          "ResNet50 - Confusion Matrix (Test Set, RAF-DB)",
                          "confusion_matrix_resnet50.png")
    save_classification_report(y_true, resnet_pred,
                               "ResNet50 (Video_SilentCare_model.pth)",
                               "classification_report_resnet50.txt")

    # ===== ViT HuggingFace =====
    vit_pred, vit_prob, vit_time = evaluate_vit_hf(image_paths, y_true)
    metrics_vit = compute_metrics(y_true, vit_pred, "ViT (trpakov/vit-face-expression)", vit_time)
    print_metrics(metrics_vit)
    plot_confusion_matrix(y_true, vit_pred,
                          "ViT (HuggingFace) - Confusion Matrix (Test Set, RAF-DB)",
                          "confusion_matrix_vit.png")
    save_classification_report(y_true, vit_pred,
                               "ViT (trpakov/vit-face-expression)",
                               "classification_report_vit.txt")

    # ===== Combined outputs =====
    # Primary confusion matrix = best model
    best_model = "resnet50" if metrics_resnet["accuracy"] > metrics_vit["accuracy"] else "vit"
    best_pred = resnet_pred if best_model == "resnet50" else vit_pred
    best_name = metrics_resnet["model"] if best_model == "resnet50" else metrics_vit["model"]
    plot_confusion_matrix(y_true, best_pred,
                          f"Video Model - Confusion Matrix (Test Set, RAF-DB)\n[Best: {best_name}]",
                          "confusion_matrix.png")
    save_classification_report(y_true, best_pred, best_name, "classification_report.txt")

    # Comparison plot
    plot_model_comparison(metrics_resnet, metrics_vit)

    # Dataset distribution
    plot_dataset_distribution(y_true)

    # Training curves
    plot_training_curves()

    # Save metrics JSON
    all_metrics = {
        "resnet50": metrics_resnet,
        "vit_huggingface": metrics_vit,
        "best_model": best_model,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved to {RESULTS_DIR / 'metrics.json'}")

    # Save predictions for fusion step
    np.savez(
        RESULTS_DIR / "test_predictions.npz",
        y_true=y_true,
        resnet_pred=resnet_pred,
        resnet_prob=resnet_prob,
        vit_pred=vit_pred,
        vit_prob=vit_prob,
    )
    print(f"  Test predictions saved for fusion step.")

    print(f"\n{'='*60}")
    print("Video results generation complete!")
    print(f"All outputs in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
