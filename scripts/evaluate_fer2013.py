"""
SilentCare - FER-2013 Evaluation Script
=========================================
Evaluates both ViT (HuggingFace) and ResNet50 on FER-2013 test set.

Usage:
    python scripts/evaluate_fer2013.py --model vit
    python scripts/evaluate_fer2013.py --model resnet50
    python scripts/evaluate_fer2013.py --model compare   (generates chart from saved JSONs)
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FER2013_DIR = PROJECT_ROOT / "data" / "FER2013" / "test"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]

FER_TO_SC = {
    "angry": 1, "disgust": 0, "fear": 0, "sad": 0,
    "happy": 3, "neutral": 3, "surprise": 2,
}


def get_fer2013_paths():
    """Return (file_paths, labels) without loading images into memory."""
    paths = []
    labels = []
    for folder_name, sc_idx in FER_TO_SC.items():
        folder = FER2013_DIR / folder_name
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*")):
            paths.append(f)
            labels.append(sc_idx)
    return paths, np.array(labels)


def run_vit(paths, labels):
    """Evaluate ViT HuggingFace on FER-2013 (stream images one at a time)."""
    from silentcare.ml.video_model import VideoModel, FER_TO_SILENTCARE
    print("[ViT] Loading model...")
    model = VideoModel(use_resnet=False)

    predictions = []
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            predictions.append(3)  # default CALM on error
            continue
        results = model._vit_classifier(img)
        probs = np.zeros(4, dtype=np.float64)
        for r in results:
            if r["label"] in FER_TO_SILENTCARE:
                probs[FER_TO_SILENTCARE[r["label"]]] += r["score"]
        predictions.append(int(np.argmax(probs)))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)} done...")
    elapsed = time.time() - t0
    return np.array(predictions), elapsed / len(paths) * 1000


def run_resnet(paths, labels):
    """Evaluate ResNet50 on FER-2013."""
    import cv2
    from silentcare.ml.video_model import VideoModel
    print("[ResNet50] Loading model...")
    model = VideoModel(use_resnet=True)

    predictions = []
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            predictions.append(3)
            continue
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        probs = model._classify_face_resnet(img_bgr)
        predictions.append(int(np.argmax(probs)))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(paths)} done...")
    elapsed = time.time() - t0
    return np.array(predictions), elapsed / len(paths) * 1000


def save_results(tag, predictions, labels, avg_ms):
    acc = float(accuracy_score(labels, predictions))
    f1_m = float(f1_score(labels, predictions, average="macro"))
    report = classification_report(labels, predictions, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])

    metrics = {
        "model": tag, "dataset": "FER-2013 test",
        "num_samples": int(len(labels)),
        "accuracy": round(acc, 4), "f1_macro": round(f1_m, 4),
        "avg_inference_ms": round(avg_ms, 1),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            } for cls in CLASSES if cls in report
        },
    }

    json_path = RESULTS_DIR / f"fer2013_{tag}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(f"{tag} - FER-2013 Confusion Matrix\nAccuracy: {acc:.1%} | F1 Macro: {f1_m:.4f}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        png_path = RESULTS_DIR / f"fer2013_{tag}_confusion_matrix.png"
        plt.savefig(str(png_path), dpi=150)
        plt.close()
        print(f"  Saved: {png_path}")
    except Exception as e:
        print(f"  WARNING: Could not save PNG: {e}")

    return metrics


def generate_comparison():
    """Generate comparison chart from saved JSON files."""
    vit_json = RESULTS_DIR / "fer2013_vit_metrics.json"
    res_json = RESULTS_DIR / "fer2013_resnet50_metrics.json"
    if not vit_json.exists() or not res_json.exists():
        print("Both fer2013_vit_metrics.json and fer2013_resnet50_metrics.json required.")
        return
    with open(vit_json) as f:
        vit_m = json.load(f)
    with open(res_json) as f:
        res_m = json.load(f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(CLASSES))
        w = 0.35
        vit_f1 = [vit_m["per_class"][c]["f1"] for c in CLASSES]
        res_f1 = [res_m["per_class"][c]["f1"] for c in CLASSES]
        axes[0].bar(x - w / 2, vit_f1, w, label="ViT HuggingFace", color="#4A90D9")
        axes[0].bar(x + w / 2, res_f1, w, label="ResNet50", color="#E67E22")
        axes[0].set_xticks(x); axes[0].set_xticklabels(CLASSES)
        axes[0].set_ylabel("F1 Score"); axes[0].set_title("Per-Class F1 (FER-2013)")
        axes[0].legend(); axes[0].set_ylim(0, 1)

        names = ["Accuracy", "F1 Macro"]
        vv = [vit_m["accuracy"], vit_m["f1_macro"]]
        rv = [res_m["accuracy"], res_m["f1_macro"]]
        x2 = np.arange(2)
        axes[1].bar(x2 - w / 2, vv, w, label="ViT HuggingFace", color="#4A90D9")
        axes[1].bar(x2 + w / 2, rv, w, label="ResNet50", color="#E67E22")
        axes[1].set_xticks(x2); axes[1].set_xticklabels(names)
        axes[1].set_ylabel("Score"); axes[1].set_title("Overall Metrics (FER-2013)")
        axes[1].legend(); axes[1].set_ylim(0, 1)

        plt.suptitle("SilentCare Video Model Comparison on FER-2013", fontsize=14, y=1.02)
        plt.tight_layout()
        path = RESULTS_DIR / "fer2013_comparison.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  WARNING: {e}")

    print(f"\n{'Metric':<20} {'ViT':>10} {'ResNet50':>10} {'Delta':>10}")
    print("-" * 50)
    da = vit_m["accuracy"] - res_m["accuracy"]
    df = vit_m["f1_macro"] - res_m["f1_macro"]
    print(f"{'Accuracy':<20} {vit_m['accuracy']:>10.1%} {res_m['accuracy']:>10.1%} {da:>+10.1%}")
    print(f"{'F1 Macro':<20} {vit_m['f1_macro']:>10.4f} {res_m['f1_macro']:>10.4f} {df:>+10.4f}")
    for cls in CLASSES:
        v = vit_m["per_class"][cls]["f1"]
        r = res_m["per_class"][cls]["f1"]
        print(f"  {cls} F1: ViT={v:.4f}  ResNet50={r:.4f}  delta={v - r:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "resnet50", "compare"], required=True)
    args = parser.parse_args()

    if args.model == "compare":
        generate_comparison()
        return

    print("=" * 60)
    print(f"SilentCare - FER-2013 Evaluation ({args.model})")
    print("=" * 60)

    paths, labels = get_fer2013_paths()
    print(f"Loaded {len(paths)} image paths")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {int(np.sum(labels == i))}")

    if args.model == "vit":
        preds, ms = run_vit(paths, labels)
    else:
        preds, ms = run_resnet(paths, labels)

    metrics = save_results(args.model, preds, labels, ms)
    print(f"\nAccuracy: {metrics['accuracy']:.1%}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"Avg inference: {ms:.1f} ms/image")


if __name__ == "__main__":
    main()
