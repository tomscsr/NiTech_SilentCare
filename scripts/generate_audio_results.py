"""
SilentCare - Generate Audio Experimental Results
=================================================
Evaluates Audio_SilentCare_model.h5 on the audio dataset test split.
Produces metrics, confusion matrix, training curves, distribution plot,
and classification report for the research report.

Usage:
    python scripts/generate_audio_results.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TFHUB_CACHE_DIR"] = os.path.join(
    os.environ.get("LOCALAPPDATA", os.environ.get("TEMP", ".")),
    "tfhub_modules2",
)

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
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
TARGET_SR = 22050
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "data" / "audio_dataset"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "Audio_SilentCare_model.h5"
HISTORY_PATH = MODEL_DIR / "audio_training_history.json"
RESULTS_DIR = PROJECT_DIR / "results" / "audio"

# Colors for each class
CLASS_COLORS = {
    "DISTRESS": "#e74c3c",
    "ANGRY": "#e67e22",
    "ALERT": "#f1c40f",
    "CALM": "#2ecc71",
}


def setup_output_dir():
    """Create results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")


def load_yamnet():
    """Load frozen YAMNet model from TensorFlow Hub."""
    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet loaded.")
    return yamnet_model


def extract_yamnet_embeddings(yamnet_model, audio, sr=TARGET_SR):
    """Extract YAMNet embeddings -> 3072-dim aggregated vector."""
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio

    audio_16k = audio_16k.astype(np.float32)
    scores, embeddings, spectrogram = yamnet_model(audio_16k)
    embeddings_np = embeddings.numpy()

    if len(embeddings_np) == 0:
        return np.zeros(3072, dtype=np.float32)

    mean_pool = np.mean(embeddings_np, axis=0)
    max_pool = np.max(embeddings_np, axis=0)
    std_pool = np.std(embeddings_np, axis=0)

    return np.concatenate([mean_pool, max_pool, std_pool]).astype(np.float32)


def load_dataset(yamnet_model):
    """Load all audio files and extract features."""
    print(f"\nLoading dataset from {DATASET_DIR}")

    X, y, filenames = [], [], []
    errors = 0

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_name} directory not found!")
            continue

        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"  {class_name}: {len(wav_files)} files")

        for wav_file in tqdm(wav_files, desc=f"  {class_name}", leave=True):
            try:
                audio, sr = librosa.load(str(wav_file), sr=TARGET_SR, mono=True)
                if len(audio) < TARGET_SR * 0.5:
                    continue

                features = extract_yamnet_embeddings(yamnet_model, audio, sr)
                X.append(features)
                y.append(class_idx)
                filenames.append(wav_file.name)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error: {wav_file.name}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nDataset loaded: {len(X)} samples, {errors} errors")
    return X, y, filenames


def generate_metrics(y_true, y_pred, y_prob):
    """Compute all metrics and save to JSON."""
    metrics = {
        "model": "Audio_SilentCare (YAMNet + Classification Head)",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "num_test_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {},
    }

    precision_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, cls in enumerate(CLASSES):
        metrics["per_class"][cls] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(np.sum(y_true == i)),
        }

    # Check training history
    if HISTORY_PATH.exists():
        metrics["training_history_available"] = True
    else:
        metrics["training_history_available"] = False
        metrics["note"] = "training history not available"

    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"AUDIO MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)
    for cls in CLASSES:
        m = metrics["per_class"][cls]
        print(f"{cls:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10d}")

    return metrics


def plot_confusion_matrix(y_true, y_pred):
    """Generate normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        vmin=0,
        vmax=100,
        square=True,
        cbar_kws={"label": "Percentage (%)"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Audio Model - Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")

    # Add raw counts as secondary annotation
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j + 0.5, i + 0.75,
                f"(n={cm[i, j]})",
                ha="center", va="center",
                fontsize=7, color="gray",
            )

    plt.tight_layout()
    out_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def plot_training_curves():
    """Plot training curves from history JSON if available."""
    if not HISTORY_PATH.exists():
        print("Training history not found, skipping training curves.")
        return

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss
    ax1.plot(epochs, history["loss"], "b-o", markersize=3, label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-s", markersize=3, label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(history["loss"]))

    # Accuracy
    ax2.plot(epochs, history["accuracy"], "b-o", markersize=3, label="Train Accuracy", linewidth=1.5)
    ax2.plot(epochs, history["val_accuracy"], "r-s", markersize=3, label="Val Accuracy", linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(history["accuracy"]))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    out_path = RESULTS_DIR / "training_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


def plot_dataset_distribution(y_train, y_test):
    """Plot class distribution for train and test sets."""
    train_counts = [np.sum(y_train == i) for i in range(NUM_CLASSES)]
    test_counts = [np.sum(y_test == i) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = np.arange(NUM_CLASSES)
    width = 0.35

    colors_train = [CLASS_COLORS[c] for c in CLASSES]
    colors_test = [CLASS_COLORS[c] for c in CLASSES]

    bars1 = ax.bar(x - width / 2, train_counts, width, label="Train",
                   color=colors_train, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, test_counts, width, label="Test",
                   color=colors_test, alpha=0.55, edgecolor="white", linewidth=0.8,
                   hatch="//")

    # Add count labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, str(int(h)),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, str(int(h)),
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Audio Dataset - Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(train_counts), max(test_counts)) * 1.15)

    plt.tight_layout()
    out_path = RESULTS_DIR / "dataset_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Dataset distribution saved to {out_path}")


def save_classification_report(y_true, y_pred):
    """Save sklearn classification report to text file."""
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
    )
    out_path = RESULTS_DIR / "classification_report.txt"
    with open(out_path, "w") as f:
        f.write("SilentCare Audio Model - Classification Report (Test Set)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: Audio_SilentCare_model.h5 (YAMNet + Dense Head)\n")
        f.write(f"Test split: {TEST_SIZE*100:.0f}%, random_state={RANDOM_STATE}\n")
        f.write(f"Total test samples: {len(y_true)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Classification report saved to {out_path}")
    print(f"\n{report}")


def main():
    print("=" * 60)
    print("SilentCare - Audio Results Generation")
    print("=" * 60)

    setup_output_dir()

    # Load YAMNet
    yamnet_model = load_yamnet()

    # Load and extract features from full dataset
    X, y, filenames = load_dataset(yamnet_model)

    # Split with same random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Load trained model
    print(f"\nLoading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"Model loaded. Parameters: {model.count_params():,}")

    # Predict on test set
    print("\nRunning predictions on test set...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Generate all outputs
    metrics = generate_metrics(y_test, y_pred, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_training_curves()
    plot_dataset_distribution(y_train, y_test)
    save_classification_report(y_test, y_pred)

    # Save test predictions for fusion step
    np.savez(
        RESULTS_DIR / "test_predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    print(f"\nTest predictions saved for fusion step.")

    print(f"\n{'='*60}")
    print("Audio results generation complete!")
    print(f"All outputs in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
