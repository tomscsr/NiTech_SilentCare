"""
SilentCare - Audio Model Training Script
=========================================
YAMNet (frozen) + classification head (TensorFlow/Keras)
4 classes: DISTRESS, ANGRY, ALERT, CALM

Pipeline:
  Input audio 1-10s (22050Hz mono)
  -> YAMNet frozen -> embeddings 1024-dim per frame (0.48s)
  -> Aggregation: mean + max + std pooling -> 3072-dim
  -> Dense(256, relu) + BatchNorm + Dropout(0.4)
  -> Dense(128, relu) + Dropout(0.3)
  -> Dense(4, softmax)

Output:
  model/Audio_SilentCare_model.h5
  model/audio_silentcare_classes.npy
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

# ============================================
# Configuration
# ============================================
CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4
TARGET_SR = 22050
YAMNET_FRAME_LEN = 0.48  # seconds per YAMNet frame

# Training params
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10  # early stopping
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "audio_dataset"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "Audio_SilentCare_model.h5"
CLASSES_PATH = MODEL_DIR / "audio_silentcare_classes.npy"


def load_yamnet():
    """Load frozen YAMNet model from TensorFlow Hub."""
    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet loaded successfully.")
    return yamnet_model


def extract_yamnet_embeddings(yamnet_model, audio, sr=TARGET_SR):
    """
    Extract YAMNet embeddings from audio.
    Returns aggregated feature vector (3072-dim): mean + max + std pooling.
    """
    # YAMNet expects float32 waveform at 16kHz
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio

    audio_16k = audio_16k.astype(np.float32)

    # Run YAMNet - returns (scores, embeddings, spectrogram)
    scores, embeddings, spectrogram = yamnet_model(audio_16k)

    # embeddings shape: (num_frames, 1024)
    embeddings_np = embeddings.numpy()

    if len(embeddings_np) == 0:
        return np.zeros(3072, dtype=np.float32)

    # Temporal aggregation: mean + max + std pooling
    mean_pool = np.mean(embeddings_np, axis=0)   # 1024
    max_pool = np.max(embeddings_np, axis=0)      # 1024
    std_pool = np.std(embeddings_np, axis=0)      # 1024

    # Concatenate -> 3072-dim
    aggregated = np.concatenate([mean_pool, max_pool, std_pool])

    return aggregated.astype(np.float32)


def load_dataset(yamnet_model, dataset_dir):
    """Load audio dataset and extract YAMNet embeddings for all files."""
    print(f"\nLoading dataset from {dataset_dir}")

    X = []
    y = []
    errors = 0

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_name} directory not found!")
            continue

        wav_files = list(class_dir.glob("*.wav"))
        print(f"  {class_name}: {len(wav_files)} files")

        for wav_file in tqdm(wav_files, desc=f"  {class_name}", leave=True):
            try:
                audio, sr = librosa.load(str(wav_file), sr=TARGET_SR, mono=True)

                # Skip very short audio
                if len(audio) < TARGET_SR * 0.5:  # min 0.5s
                    continue

                features = extract_yamnet_embeddings(yamnet_model, audio, sr)
                X.append(features)
                y.append(class_idx)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error processing {wav_file.name}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nDataset loaded: {len(X)} samples, {errors} errors")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {dict(zip(CLASSES, np.bincount(y, minlength=NUM_CLASSES)))}")

    return X, y


def build_classification_head(input_dim=3072, num_classes=NUM_CLASSES):
    """Build the classification head on top of YAMNet embeddings."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def compute_class_weights(y):
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def train():
    """Main training function."""
    print("=" * 60)
    print("SilentCare - Audio Model Training")
    print("YAMNet (frozen) + Classification Head")
    print("=" * 60)

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load YAMNet
    yamnet_model = load_yamnet()

    # Load and process dataset
    X, y = load_dataset(yamnet_model, DATASET_DIR)

    if len(X) == 0:
        print("ERROR: No data loaded!")
        return

    # Split: train / val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=42, stratify=y_temp
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")

    # Build model
    model = build_classification_head()
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Detailed classification report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    report = classification_report(
        y_test, y_pred_classes,
        target_names=CLASSES,
        digits=4,
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)

    # F1 scores
    f1_per_class = f1_score(y_test, y_pred_classes, average=None)
    f1_macro = f1_score(y_test, y_pred_classes, average="macro")
    f1_weighted = f1_score(y_test, y_pred_classes, average="weighted")

    print(f"\nF1 per class:")
    for cls, f1 in zip(CLASSES, f1_per_class):
        print(f"  {cls}: {f1:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")

    # Save model
    print(f"\nSaving model to {MODEL_PATH}")
    model.save(str(MODEL_PATH))

    # Save class names
    np.save(str(CLASSES_PATH), np.array(CLASSES))
    print(f"Saved class names to {CLASSES_PATH}")

    # Save training history
    history_path = MODEL_DIR / "audio_training_history.json"
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved training history to {history_path}")

    print("\n" + "=" * 60)
    print("Audio model training complete!")
    print("=" * 60)

    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": {cls: float(f1) for cls, f1 in zip(CLASSES, f1_per_class)},
    }


if __name__ == "__main__":
    results = train()
    if results:
        print("\n=== SUMMARY ===")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"F1 Macro: {results['f1_macro']:.4f}")
        for cls, f1 in results["f1_per_class"].items():
            print(f"  {cls}: F1={f1:.4f}")
