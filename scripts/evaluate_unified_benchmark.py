"""
SilentCare - Unified Benchmark Evaluation
===========================================
Evaluates all 5 video models on the unified benchmark dataset.

Models:
  1. ResNet50 (locally trained on RAF-DB)
  2. EfficientNet-B2 (locally trained on RAF-DB)
  3. MobileNetV3-Large (locally trained on RAF-DB)
  4. ViT trpakov (HuggingFace, production)
  5. ViT dima806 (HuggingFace)

Outputs to results/benchmark/:
  - unified_benchmark_metrics.json
  - confusion matrices (individual + grid)
  - model comparison charts
  - per-source accuracy heatmap
  - inference benchmark

Usage:
    python scripts/evaluate_unified_benchmark.py
    python scripts/evaluate_unified_benchmark.py --models resnet50 vit_trpakov
"""

import csv
import json
import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

# ============================================
# Configuration
# ============================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "benchmark"
MANIFEST_PATH = PROJECT_DIR / "data" / "unified_benchmark" / "manifest.csv"

sys.path.insert(0, str(PROJECT_DIR))

CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4

MODEL_REGISTRY = {
    "resnet50": {
        "display_name": "ResNet50",
        "type": "local",
        "backend": "resnet50",
        "color": "#e74c3c",
        "training_domain": "RAF-DB",
    },
    "efficientnet_b2": {
        "display_name": "EfficientNet-B2",
        "type": "local",
        "backend": "efficientnet_b2",
        "color": "#e67e22",
        "training_domain": "RAF-DB",
    },
    "mobilenet_v3": {
        "display_name": "MobileNetV3",
        "type": "local",
        "backend": "mobilenet_v3",
        "color": "#2ecc71",
        "training_domain": "RAF-DB",
    },
    "vit_trpakov": {
        "display_name": "ViT trpakov",
        "type": "vit",
        "backend": "vit_trpakov",
        "color": "#9b59b6",
        "training_domain": "FER-2013",
    },
    "vit_dima806": {
        "display_name": "ViT dima806",
        "type": "vit",
        "backend": "vit_dima806",
        "color": "#3498db",
        "training_domain": "FER+ (in21k)",
    },
}


# ============================================
# Data Loading
# ============================================
def load_manifest():
    """Load benchmark manifest CSV."""
    entries = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "path": row["image_path"],
                "label": int(row["label"]),
                "source": row["source"],
                "original_label": row["original_label"],
            })
    return entries


# ============================================
# Model Loading & Inference
# ============================================
def load_model(model_key):
    """Load a model by its registry key."""
    from silentcare.ml.video_model import VideoModel, FER_TO_SILENTCARE
    info = MODEL_REGISTRY[model_key]
    model = VideoModel(backend=info["backend"])
    return model, info


def predict_local(model, image_path):
    """Run local model (ResNet50/EfficientNet/MobileNet) on an image."""
    import cv2
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return np.array([0.0, 0.0, 0.0, 1.0])  # default CALM on error
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    probs = model._classify_face_local(img_bgr)
    return probs


def predict_vit(model, image_path):
    """Run HuggingFace ViT on an image."""
    from silentcare.ml.video_model import FER_TO_SILENTCARE
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return np.array([0.0, 0.0, 0.0, 1.0])
    results = model._vit_classifier(img)
    probs = np.zeros(NUM_CLASSES, dtype=np.float64)
    for r in results:
        if r["label"] in FER_TO_SILENTCARE:
            probs[FER_TO_SILENTCARE[r["label"]]] += r["score"]
    return probs


def evaluate_model(model_key, entries):
    """Evaluate a single model on all benchmark entries."""
    model, info = load_model(model_key)
    predict_fn = predict_vit if info["type"] == "vit" else predict_local

    predictions = []
    true_labels = []
    sources = []
    times = []

    total = len(entries)
    print(f"\n  Evaluating {info['display_name']} on {total} images...")

    for i, entry in enumerate(entries):
        t0 = time.time()
        probs = predict_fn(model, entry["path"])
        elapsed = time.time() - t0

        pred_idx = int(np.argmax(probs))
        predictions.append(pred_idx)
        true_labels.append(entry["label"])
        sources.append(entry["source"])
        times.append(elapsed * 1000)  # ms

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{total} done... (avg {np.mean(times):.1f} ms/image)")

    return {
        "predictions": np.array(predictions),
        "true_labels": np.array(true_labels),
        "sources": sources,
        "times": np.array(times),
    }


# ============================================
# Metrics Computation
# ============================================
def compute_metrics(result, model_key):
    """Compute comprehensive metrics for one model."""
    info = MODEL_REGISTRY[model_key]
    preds = result["predictions"]
    labels = result["true_labels"]
    sources = result["sources"]
    times = result["times"]

    # Overall metrics
    acc = float(accuracy_score(labels, preds))
    f1_m = float(f1_score(labels, preds, average="macro"))
    f1_w = float(f1_score(labels, preds, average="weighted"))

    # Per-class
    report = classification_report(labels, preds, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    per_class = {}
    for cls in CLASSES:
        if cls in report:
            per_class[cls] = {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }

    # Per-source metrics
    unique_sources = sorted(set(sources))
    per_source = {}
    per_source_per_class = {}
    for src in unique_sources:
        mask = np.array([s == src for s in sources])
        src_preds = preds[mask]
        src_labels = labels[mask]
        src_acc = float(accuracy_score(src_labels, src_preds))
        src_f1 = float(f1_score(src_labels, src_preds, average="macro", zero_division=0))
        per_source[src] = {
            "accuracy": round(src_acc, 4),
            "f1_macro": round(src_f1, 4),
            "num_samples": int(mask.sum()),
        }
        # Per-source per-class F1
        src_report = classification_report(src_labels, src_preds, target_names=CLASSES,
                                           output_dict=True, zero_division=0)
        per_source_per_class[src] = {
            cls: round(src_report[cls]["f1-score"], 4) for cls in CLASSES if cls in src_report
        }

    # Inference timing
    avg_ms = float(np.mean(times))
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))

    return {
        "display_name": info["display_name"],
        "training_domain": info["training_domain"],
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "avg_inference_ms": round(avg_ms, 1),
        "p50_inference_ms": round(p50, 1),
        "p95_inference_ms": round(p95, 1),
        "per_class": per_class,
        "per_source": per_source,
        "per_source_per_class": per_source_per_class,
    }


# ============================================
# Visualization
# ============================================
def plot_confusion_matrices(all_results, model_keys):
    """Plot individual and grid confusion matrices."""
    # Individual matrices
    for key in model_keys:
        r = all_results[key]
        cm = confusion_matrix(r["true_labels"], r["predictions"], labels=list(range(NUM_CLASSES)))
        info = MODEL_REGISTRY[key]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        metrics = compute_metrics(r, key)
        ax.set_title(f"{info['display_name']} - Confusion Matrix\n"
                     f"Accuracy: {metrics['accuracy']:.1%} | F1 Macro: {metrics['f1_macro']:.4f}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_matrix_{key}.png")
        plt.close(fig)

    # Grid (2x3 or 1x5 depending on count)
    n = len(model_keys)
    if n <= 3:
        rows, cols = 1, n
    else:
        rows, cols = 2, 3

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), dpi=150)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, key in enumerate(model_keys):
        r = all_results[key]
        cm = confusion_matrix(r["true_labels"], r["predictions"], labels=list(range(NUM_CLASSES)))
        info = MODEL_REGISTRY[key]
        metrics = compute_metrics(r, key)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[i])
        axes[i].set_title(f"{info['display_name']}\nAcc: {metrics['accuracy']:.1%} | F1: {metrics['f1_macro']:.4f}",
                          fontsize=10)
        axes[i].set_ylabel("True" if i % cols == 0 else "")
        axes[i].set_xlabel("Predicted")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("SilentCare - Unified Benchmark Confusion Matrices",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices_grid.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved confusion matrices")


def plot_model_comparison(all_metrics, model_keys):
    """Grouped bar chart comparing all models (accuracy + F1)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # Overall: accuracy and F1
    names = [MODEL_REGISTRY[k]["display_name"] for k in model_keys]
    colors = [MODEL_REGISTRY[k]["color"] for k in model_keys]
    accs = [all_metrics[k]["accuracy"] for k in model_keys]
    f1s = [all_metrics[k]["f1_macro"] for k in model_keys]

    x = np.arange(len(names))
    w = 0.35
    bars1 = axes[0].bar(x - w / 2, accs, w, label="Accuracy", color=colors, alpha=0.85, edgecolor="white")
    bars2 = axes[0].bar(x + w / 2, f1s, w, label="F1 Macro", color=colors, alpha=0.55, edgecolor="white")
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{bar.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{bar.get_height():.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=9, rotation=15)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Metrics", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(axis="y", alpha=0.3)

    # Per-class F1
    x = np.arange(len(CLASSES))
    w = 0.15
    for i, key in enumerate(model_keys):
        f1_vals = [all_metrics[key]["per_class"][cls]["f1"] for cls in CLASSES]
        axes[1].bar(x + i * w - (len(model_keys) - 1) * w / 2, f1_vals, w,
                    label=MODEL_REGISTRY[key]["display_name"],
                    color=MODEL_REGISTRY[key]["color"], alpha=0.8, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASSES, fontsize=10)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("SilentCare - 5-Model Unified Benchmark Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison_overall.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved model comparison chart")


def plot_per_class_comparison(all_metrics, model_keys):
    """Dedicated per-class F1 grouped bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    x = np.arange(len(CLASSES))
    w = 0.15
    for i, key in enumerate(model_keys):
        f1_vals = [all_metrics[key]["per_class"][cls]["f1"] for cls in CLASSES]
        offset = i * w - (len(model_keys) - 1) * w / 2
        bars = ax.bar(x + offset, f1_vals, w,
                      label=MODEL_REGISTRY[key]["display_name"],
                      color=MODEL_REGISTRY[key]["color"], alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Per-Class F1 Comparison (Unified Benchmark)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison_per_class.png")
    plt.close(fig)
    print("  Saved per-class comparison chart")


def plot_source_heatmap(all_metrics, model_keys):
    """Model x source accuracy heatmap."""
    sources = sorted(set(
        src for m in all_metrics.values() for src in m.get("per_source", {}).keys()
    ))
    if not sources:
        return

    data = []
    labels = []
    for key in model_keys:
        row = []
        for src in sources:
            acc = all_metrics[key].get("per_source", {}).get(src, {}).get("accuracy", 0)
            row.append(acc)
        data.append(row)
        labels.append(MODEL_REGISTRY[key]["display_name"])

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=sources, yticklabels=labels, ax=ax,
                vmin=0.3, vmax=1.0)
    ax.set_title("Per-Source Accuracy (Unified Benchmark)\nHigher = better generalization",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset Source", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "source_accuracy_heatmap.png")
    plt.close(fig)
    print("  Saved source accuracy heatmap")


def plot_inference_benchmark(all_metrics, model_keys):
    """Horizontal bar chart of inference times."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    names = [MODEL_REGISTRY[k]["display_name"] for k in model_keys]
    times = [all_metrics[k]["avg_inference_ms"] for k in model_keys]
    colors = [MODEL_REGISTRY[k]["color"] for k in model_keys]

    bars = ax.barh(names, times, color=colors, alpha=0.85, height=0.5, edgecolor="white")

    # 100ms real-time threshold
    ax.axvline(x=100, color="red", linestyle="--", linewidth=2,
               label="100 ms (real-time limit)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 2, bar.get_y() + bar.get_height() / 2,
                f"{w:.1f} ms", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Inference Time (ms)", fontsize=12)
    ax.set_title("Inference Benchmark (Unified Benchmark, CPU)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    max_time = max(times) * 1.3 if times else 150
    ax.set_xlim(0, max(max_time, 120))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "inference_benchmark.png")
    plt.close(fig)
    print("  Saved inference benchmark chart")


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser(description="SilentCare Unified Benchmark Evaluation")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY.keys()),
                        default=None,
                        help="Models to evaluate (default: all available)")
    args = parser.parse_args()

    print("=" * 60)
    print("SilentCare - Unified Benchmark Evaluation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        print("Run scripts/build_unified_benchmark.py first.")
        sys.exit(1)

    entries = load_manifest()
    print(f"\nLoaded {len(entries)} benchmark images")
    source_counts = defaultdict(int)
    class_counts = defaultdict(int)
    for e in entries:
        source_counts[e["source"]] += 1
        class_counts[e["label"]] += 1
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {class_counts[i]}")

    # Determine which models to evaluate
    model_keys = args.models or list(MODEL_REGISTRY.keys())

    # Check which models have .pth files available
    available_keys = []
    for key in model_keys:
        info = MODEL_REGISTRY[key]
        if info["type"] == "vit":
            available_keys.append(key)
        else:
            from silentcare.ml.video_model import BACKEND_MODEL_FILES
            model_file = PROJECT_DIR / "model" / BACKEND_MODEL_FILES[key]
            if model_file.exists():
                available_keys.append(key)
            else:
                print(f"  SKIP {info['display_name']}: {model_file.name} not found")

    if not available_keys:
        print("\nERROR: No models available for evaluation.")
        sys.exit(1)

    print(f"\nModels to evaluate: {[MODEL_REGISTRY[k]['display_name'] for k in available_keys]}")

    # Evaluate each model
    all_results = {}
    all_metrics = {}
    for key in available_keys:
        result = evaluate_model(key, entries)
        all_results[key] = result
        metrics = compute_metrics(result, key)
        all_metrics[key] = metrics

        print(f"\n  {MODEL_REGISTRY[key]['display_name']}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    F1 Macro:  {metrics['f1_macro']:.4f}")
        print(f"    Avg time:  {metrics['avg_inference_ms']:.1f} ms")

    # Save metrics JSON
    benchmark_info = {
        "total_images": len(entries),
        "sources": sorted(source_counts.keys()),
        "classes": CLASSES,
        "per_class_count": {CLASSES[i]: class_counts[i] for i in range(NUM_CLASSES)},
        "per_source_count": dict(source_counts),
    }

    ranking = sorted(all_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True)
    ranking_list = [
        {"model": k, "display_name": v["display_name"],
         "accuracy": v["accuracy"], "f1_macro": v["f1_macro"], "rank": i + 1}
        for i, (k, v) in enumerate(ranking)
    ]

    output = {
        "benchmark_info": benchmark_info,
        "models": all_metrics,
        "ranking": ranking_list,
    }

    json_path = RESULTS_DIR / "unified_benchmark_metrics.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved: {json_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrices(all_results, available_keys)
    plot_model_comparison(all_metrics, available_keys)
    plot_per_class_comparison(all_metrics, available_keys)
    plot_source_heatmap(all_metrics, available_keys)
    plot_inference_benchmark(all_metrics, available_keys)

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'DISTRESS':>10} {'Inference':>10} {'Domain':<15}"
    print(header)
    print("-" * 80)
    for key in available_keys:
        m = all_metrics[key]
        distress_f1 = m["per_class"].get("DISTRESS", {}).get("f1", 0)
        print(f"{MODEL_REGISTRY[key]['display_name']:<20} "
              f"{m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{distress_f1:>10.4f} {m['avg_inference_ms']:>8.1f}ms "
              f"{m['training_domain']:<15}")

    # Per-source breakdown
    print(f"\n{'='*80}")
    print("PER-SOURCE ACCURACY")
    print(f"{'='*80}")
    sources = sorted(source_counts.keys())
    header = f"{'Model':<20}" + "".join(f"{src:>15}" for src in sources)
    print(header)
    print("-" * (20 + 15 * len(sources)))
    for key in available_keys:
        m = all_metrics[key]
        row = f"{MODEL_REGISTRY[key]['display_name']:<20}"
        for src in sources:
            acc = m.get("per_source", {}).get(src, {}).get("accuracy", 0)
            row += f"{acc:>15.4f}"
        print(row)

    print(f"\nBest model: {ranking_list[0]['display_name']} "
          f"(F1 Macro: {ranking_list[0]['f1_macro']:.4f})")

    print(f"\n{'='*60}")
    print("Unified benchmark evaluation complete!")
    print(f"All outputs in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
