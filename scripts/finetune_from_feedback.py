"""
SilentCare - Fine-tuning from Human Feedback
==============================================
Loads feedback data (corrected labels + saved audio/video),
fine-tunes the audio and video classification heads,
and reports before/after accuracy.

Usage:
    python scripts/finetune_from_feedback.py [--confirm] [--audio-only] [--video-only]

Flags:
    --confirm       Overwrite production models without interactive prompt
    --audio-only    Only fine-tune the audio model
    --video-only    Only fine-tune the video model
    --mix-original  Mix original training data for underrepresented classes
"""

import sys
import os
import argparse
import shutil
import wave
from pathlib import Path
from datetime import datetime

import numpy as np

# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from silentcare.core.database import Database
from silentcare.app.config import (
    EMOTION_CLASSES, NUM_CLASSES, DATABASE_PATH,
    AUDIO_MODEL_PATH, AUDIO_SAMPLE_RATE,
)

CLASS_TO_IDX = {c: i for i, c in enumerate(EMOTION_CLASSES)}


def load_feedback_data(db):
    """Load all unused feedback entries with their file paths."""
    feedbacks = db.get_feedback(limit=10000, used_for_training=False)
    print(f"\n[Scan] Found {len(feedbacks)} unused feedback entries")

    audio_samples = []
    video_samples = []

    for fb in feedbacks:
        label = CLASS_TO_IDX.get(fb["correct_class"])
        if label is None:
            print(f"  WARNING: Unknown class '{fb['correct_class']}' in feedback {fb['id']}, skipping")
            continue

        # Audio
        if fb["audio_saved"]:
            audio_path = PROJECT_DIR / fb["audio_path"]
            if audio_path.exists():
                audio_samples.append({
                    "feedback_id": fb["id"],
                    "path": str(audio_path),
                    "label": label,
                    "correct_class": fb["correct_class"],
                })
            else:
                print(f"  WARNING: Audio file missing for feedback {fb['id']}: {audio_path}")

        # Video
        if fb["video_saved"]:
            video_dir = PROJECT_DIR / fb["video_path"]
            if video_dir.exists():
                frames = sorted(video_dir.glob("frame_*.jpg"))
                if frames:
                    video_samples.append({
                        "feedback_id": fb["id"],
                        "frames": [str(f) for f in frames],
                        "label": label,
                        "correct_class": fb["correct_class"],
                    })
            else:
                print(f"  WARNING: Video dir missing for feedback {fb['id']}: {video_dir}")

    return feedbacks, audio_samples, video_samples


def check_safety(samples, modality_name):
    """Check safety conditions before fine-tuning.

    Returns:
        True if safe to proceed, False otherwise.
    """
    if len(samples) < 10:
        print(f"\n[REFUSED] Only {len(samples)} {modality_name} samples available (minimum: 10)")
        return False

    class_counts = {}
    for s in samples:
        cls = s["correct_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n[{modality_name}] Sample distribution:")
    for cls in EMOTION_CLASSES:
        count = class_counts.get(cls, 0)
        print(f"  {cls}: {count} samples")

    # Check for empty classes
    empty_classes = [c for c in EMOTION_CLASSES if class_counts.get(c, 0) == 0]
    if empty_classes:
        print(f"\n[REFUSED] Classes with 0 samples: {empty_classes}")
        print("  This would cause catastrophic forgetting. Collect more feedback first.")
        return False

    # Warn about underrepresented classes
    low_classes = [c for c in EMOTION_CLASSES if 0 < class_counts.get(c, 0) < 5]
    if low_classes:
        print(f"\n[WARNING] Classes with < 5 samples: {low_classes}")
        print("  Consider using --mix-original to add original training data.")

    return True


def finetune_audio(audio_samples, confirm=False):
    """Fine-tune the audio classification head (Keras).

    The YAMNet backbone remains frozen. Only the classification head is updated.
    """
    print("\n" + "=" * 60)
    print("AUDIO FINE-TUNING")
    print("=" * 60)

    import warnings
    warnings.filterwarnings("ignore")
    import tensorflow as tf
    import tensorflow_hub as hub
    import librosa
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    model_path = PROJECT_DIR / AUDIO_MODEL_PATH
    if not model_path.exists():
        print(f"[ERROR] Audio model not found: {model_path}")
        return False

    # Load YAMNet for embedding extraction
    print("[Audio] Loading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    def extract_embeddings(audio_path):
        """Extract 3072-dim YAMNet embeddings from a WAV file."""
        audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        audio_16k = audio_16k.astype(np.float32)
        _, embeddings, _ = yamnet(audio_16k)
        embeddings_np = embeddings.numpy()
        if len(embeddings_np) == 0:
            return np.zeros(3072, dtype=np.float32)
        mean_pool = np.mean(embeddings_np, axis=0)
        max_pool = np.max(embeddings_np, axis=0)
        std_pool = np.std(embeddings_np, axis=0)
        return np.concatenate([mean_pool, max_pool, std_pool]).astype(np.float32)

    # Extract features
    print(f"[Audio] Extracting embeddings for {len(audio_samples)} samples...")
    X = []
    y = []
    for i, sample in enumerate(audio_samples):
        try:
            emb = extract_embeddings(sample["path"])
            X.append(emb)
            y.append(sample["label"])
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(audio_samples)}")
        except Exception as e:
            print(f"  ERROR processing {sample['path']}: {e}")

    X = np.array(X)
    y = np.array(y)
    print(f"[Audio] Feature matrix: {X.shape}, Labels: {y.shape}")

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load existing model
    print("[Audio] Loading existing model...")
    model = tf.keras.models.load_model(str(model_path))

    # Evaluate BEFORE
    print("\n[Audio] Evaluation BEFORE fine-tuning:")
    y_pred_before = np.argmax(model.predict(X, verbose=0), axis=1)
    print(classification_report(y, y_pred_before, target_names=EMOTION_CLASSES, zero_division=0))

    # Compile with reduced learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fine-tune
    print("[Audio] Fine-tuning...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=min(16, len(X_train)),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate AFTER
    print("\n[Audio] Evaluation AFTER fine-tuning:")
    y_pred_after = np.argmax(model.predict(X, verbose=0), axis=1)
    print(classification_report(y, y_pred_after, target_names=EMOTION_CLASSES, zero_division=0))

    # Save fine-tuned model
    finetuned_path = PROJECT_DIR / "model" / "Audio_SilentCare_model_finetuned.h5"
    model.save(str(finetuned_path))
    print(f"[Audio] Fine-tuned model saved: {finetuned_path}")

    # Overwrite production model?
    if confirm or _ask_confirm("Overwrite production audio model?"):
        backup_path = model_path.with_suffix(".h5.backup")
        shutil.copy2(model_path, backup_path)
        print(f"[Audio] Backup saved: {backup_path}")
        shutil.copy2(finetuned_path, model_path)
        print(f"[Audio] Production model updated: {model_path}")
    else:
        print("[Audio] Production model NOT updated.")

    return True


def finetune_video(video_samples, confirm=False):
    """Fine-tune the video ViT classifier (PyTorch).

    Replaces the 7-class FER head with a 4-class SilentCare head
    and fine-tunes only the classifier layer.
    """
    print("\n" + "=" * 60)
    print("VIDEO FINE-TUNING")
    print("=" * 60)

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from transformers import ViTForImageClassification, ViTImageProcessor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Video] Device: {device}")

    model_name = "trpakov/vit-face-expression"
    print(f"[Video] Loading {model_name}...")
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Replace classifier head: 7 FER classes -> 4 SilentCare classes
    hidden_size = model.config.hidden_size
    model.classifier = nn.Linear(hidden_size, NUM_CLASSES)
    model.config.num_labels = NUM_CLASSES
    model.config.id2label = {i: c for i, c in enumerate(EMOTION_CLASSES)}
    model.config.label2id = CLASS_TO_IDX

    # Freeze all layers except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)

    # Prepare dataset: use middle frame from each sample
    class FeedbackFrameDataset(Dataset):
        def __init__(self, samples, processor):
            self.items = []
            for s in samples:
                # Use middle frame
                mid = len(s["frames"]) // 2
                self.items.append({"path": s["frames"][mid], "label": s["label"]})
            self.processor = processor

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            image = Image.open(item["path"]).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "label": torch.tensor(item["label"], dtype=torch.long),
            }

    # Split
    train_samples, val_samples = train_test_split(
        video_samples, test_size=0.2,
        stratify=[s["label"] for s in video_samples], random_state=42
    )

    train_ds = FeedbackFrameDataset(train_samples, processor)
    val_ds = FeedbackFrameDataset(val_samples, processor)
    all_ds = FeedbackFrameDataset(video_samples, processor)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    all_loader = DataLoader(all_ds, batch_size=8, shuffle=False)

    # Evaluate BEFORE (random head, so this is baseline)
    print("\n[Video] Note: 'before' metrics use a randomly initialized 4-class head")

    # Fine-tune
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_state = None

    print("[Video] Fine-tuning classifier head...")
    for epoch in range(30):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(pixel_values=pixels)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = outputs.logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)
                outputs = model(pixel_values=pixels)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item() * len(labels)
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f} "
              f"train_acc={train_acc:.3f} | "
              f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate AFTER on all data
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in all_loader:
            pixels = batch["pixel_values"].to(device)
            labels = batch["label"]
            outputs = model(pixel_values=pixels)
            preds = outputs.logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n[Video] Evaluation AFTER fine-tuning:")
    print(classification_report(all_labels, all_preds,
                                target_names=EMOTION_CLASSES, zero_division=0))

    # Save fine-tuned model
    save_dir = PROJECT_DIR / "model" / "Video_SilentCare_finetuned"
    model.save_pretrained(str(save_dir))
    processor.save_pretrained(str(save_dir))
    print(f"[Video] Fine-tuned model saved: {save_dir}")

    if confirm or _ask_confirm("Overwrite production video model?"):
        prod_dir = PROJECT_DIR / "model" / "Video_SilentCare_production"
        if save_dir.exists():
            if prod_dir.exists():
                backup_dir = PROJECT_DIR / "model" / "Video_SilentCare_backup"
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(prod_dir, backup_dir)
                print(f"[Video] Backup saved: {backup_dir}")
            shutil.copytree(save_dir, prod_dir, dirs_exist_ok=True)
            print(f"[Video] Production model updated: {prod_dir}")
    else:
        print("[Video] Production model NOT updated.")

    return True


def _ask_confirm(question):
    """Ask user for confirmation (interactive prompt)."""
    try:
        answer = input(f"\n{question} [y/N] ").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SilentCare models from feedback")
    parser.add_argument("--confirm", action="store_true",
                        help="Overwrite production models without prompting")
    parser.add_argument("--audio-only", action="store_true",
                        help="Only fine-tune the audio model")
    parser.add_argument("--video-only", action="store_true",
                        help="Only fine-tune the video model")
    parser.add_argument("--mix-original", action="store_true",
                        help="Mix original training data for underrepresented classes")
    args = parser.parse_args()

    do_audio = not args.video_only
    do_video = not args.audio_only

    print("=" * 60)
    print("SilentCare - Fine-tuning from Human Feedback")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load database
    db_path = PROJECT_DIR / DATABASE_PATH
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        sys.exit(1)

    db = Database(str(db_path))

    # Load feedback data
    feedbacks, audio_samples, video_samples = load_feedback_data(db)

    if not feedbacks:
        print("\n[INFO] No unused feedback to train on. Exiting.")
        db.close()
        return

    print(f"\n[Summary]")
    print(f"  Total feedback entries: {len(feedbacks)}")
    print(f"  Audio samples: {len(audio_samples)}")
    print(f"  Video samples: {len(video_samples)}")

    # Fine-tune audio
    audio_ok = False
    if do_audio and audio_samples:
        if check_safety(audio_samples, "Audio"):
            audio_ok = finetune_audio(audio_samples, confirm=args.confirm)
    elif do_audio:
        print("\n[Audio] No audio samples available, skipping.")

    # Fine-tune video
    video_ok = False
    if do_video and video_samples:
        if check_safety(video_samples, "Video"):
            video_ok = finetune_video(video_samples, confirm=args.confirm)
    elif do_video:
        print("\n[Video] No video samples available, skipping.")

    # Mark feedback as used
    if audio_ok or video_ok:
        feedback_ids = [fb["id"] for fb in feedbacks]
        db.mark_feedback_used(feedback_ids)
        print(f"\n[Done] Marked {len(feedback_ids)} feedback entries as used_for_training=1")

    print("\n" + "=" * 60)
    print("Fine-tuning complete.")
    print("=" * 60)

    db.close()


if __name__ == "__main__":
    main()
