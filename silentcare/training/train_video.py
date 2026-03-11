"""
SilentCare - Video Model Training Script
==========================================
ResNet50 (pretrained ImageNet, PyTorch) + classification head + LSTM temporal
Fine-tuned on RAF-DB remapped to 4 classes: DISTRESS, ANGRY, ALERT, CALM

Step 1: Train static backbone on RAF-DB (frame-level)
Step 2: Build temporal LSTM head for sequence-level prediction

RAF-DB label mapping (1-indexed):
  1=Surprise -> ALERT
  2=Fear     -> DISTRESS
  3=Disgust  -> DISTRESS
  4=Happy    -> CALM
  5=Sad      -> DISTRESS
  6=Angry    -> ANGRY
  7=Neutral  -> CALM

Output:
  model/Video_SilentCare_model.pth (static backbone + head)
  model/Video_SilentCare_temporal.pth (LSTM temporal head)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

# ============================================
# Configuration
# ============================================
CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4

# RAF-DB 1-indexed label -> SilentCare class index
# 1=Surprise->ALERT(2), 2=Fear->DISTRESS(0), 3=Disgust->DISTRESS(0),
# 4=Happy->CALM(3), 5=Sad->DISTRESS(0), 6=Angry->ANGRY(1), 7=Neutral->CALM(3)
RAFDB_TO_SILENTCARE = {
    1: 2,  # Surprise -> ALERT
    2: 0,  # Fear -> DISTRESS
    3: 0,  # Disgust -> DISTRESS
    4: 3,  # Happy -> CALM
    5: 0,  # Sad -> DISTRESS
    6: 1,  # Angry -> ANGRY
    7: 3,  # Neutral -> CALM
}

# Training params
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 8
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows compatibility
IMAGE_SIZE = 224

# LSTM temporal head params
LSTM_SEQ_LEN = 15  # frames per segment
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
FEATURE_DIM = 512  # output of backbone feature extractor

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RAFDB_IMAGE_DIR = PROJECT_DIR.parent / "data" / "image" / "RAF-DB" / "aligned"
RAFDB_LABEL_FILE = PROJECT_DIR.parent / "data" / "image" / "EmoLabel" / "list_patition_label.txt"
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "Video_SilentCare_model.pth"
TEMPORAL_MODEL_PATH = MODEL_DIR / "Video_SilentCare_temporal.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# Dataset
# ============================================
class RAFDBDataset(Dataset):
    """RAF-DB dataset with 7->4 class remapping."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_rafdb_data():
    """Load RAF-DB labels and remap to 4 SilentCare classes."""
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

            if rafdb_label not in RAFDB_TO_SILENTCARE:
                skipped += 1
                continue

            silentcare_label = RAFDB_TO_SILENTCARE[rafdb_label]

            # Convert filename to aligned version
            # Format: train_00001.jpg -> train_00001_aligned.jpg
            base = filename.replace(".jpg", "_aligned.jpg")
            img_path = RAFDB_IMAGE_DIR / base

            if img_path.exists():
                image_paths.append(str(img_path))
                labels.append(silentcare_label)
            else:
                skipped += 1

    labels = np.array(labels)

    print(f"Loaded {len(image_paths)} images, skipped {skipped}")
    print(f"Class distribution:")
    for i, cls in enumerate(CLASSES):
        count = np.sum(labels == i)
        print(f"  {cls}: {count}")

    return image_paths, labels


# ============================================
# Model
# ============================================
class SilentCareVideoModel(nn.Module):
    """
    ResNet50 backbone + classification head for FER.
    Also provides feature extraction for temporal LSTM.
    """

    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM, freeze_backbone=False):
        super().__init__()

        # Load pretrained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final FC layer, keep everything up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (batch, 2048, 1, 1)

        # Projection to feature_dim
        self.projection = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def extract_features(self, x):
        """Extract feature_dim dimensional features (for LSTM input)."""
        x = self.features(x)
        x = x.flatten(1)  # (batch, 2048)
        x = self.projection(x)  # (batch, feature_dim)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


class TemporalLSTMHead(nn.Module):
    """
    LSTM temporal head for sequence-level emotion prediction.
    Takes sequence of per-frame features and outputs class probabilities.
    """

    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=LSTM_HIDDEN,
                 num_layers=LSTM_LAYERS, num_classes=NUM_CLASSES,
                 dropout=LSTM_DROPOUT):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, num_classes)
        """
        # LSTM output
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        logits = self.classifier(last_hidden)
        return logits


# ============================================
# Training functions
# ============================================
def get_transforms(is_train=True):
    """Get image transforms for training/validation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def get_weighted_sampler(labels):
    """Create weighted sampler for imbalanced dataset."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_static_model():
    """Train the static (frame-level) FER model on RAF-DB."""
    print("=" * 60)
    print("SilentCare - Video Model Training (Static)")
    print("ResNet50 + Classification Head on RAF-DB")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    image_paths, labels = load_rafdb_data()
    if len(image_paths) == 0:
        print("ERROR: No data loaded!")
        return None

    # Split: train / val / test (70/15/15)
    paths_temp, paths_test, labels_temp, labels_test = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_temp, labels_temp, test_size=0.176, random_state=42, stratify=labels_temp
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(paths_train)}")
    print(f"  Val:   {len(paths_val)}")
    print(f"  Test:  {len(paths_test)}")

    # Datasets and DataLoaders
    train_dataset = RAFDBDataset(paths_train, labels_train, transform=get_transforms(True))
    val_dataset = RAFDBDataset(paths_val, labels_val, transform=get_transforms(False))
    test_dataset = RAFDBDataset(paths_test, labels_test, transform=get_transforms(False))

    train_sampler = get_weighted_sampler(labels_train)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model
    model = SilentCareVideoModel(
        num_classes=NUM_CLASSES,
        feature_dim=FEATURE_DIM,
        freeze_backbone=False,
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss with class weights
    class_counts = np.bincount(labels_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(DEVICE)
    )

    # Optimizer with differential learning rates
    backbone_params = list(model.features.parameters())
    head_params = list(model.projection.parameters()) + list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},  # backbone: lower LR
        {"params": head_params, "lr": LEARNING_RATE},             # head: normal LR
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-7
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model for evaluation
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True))

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluation on Test Set")
    print("=" * 60)

    test_loss, test_acc, y_pred, y_true = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\nF1 per class:")
    for cls, f1 in zip(CLASSES, f1_per_class):
        print(f"  {cls}: {f1:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")

    # Save training history
    history_path = MODEL_DIR / "video_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save temporal LSTM head (untrained, architecture only)
    temporal_model = TemporalLSTMHead(
        input_dim=FEATURE_DIM,
        hidden_dim=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        num_classes=NUM_CLASSES,
    )
    torch.save(temporal_model.state_dict(), str(TEMPORAL_MODEL_PATH))
    print(f"\nSaved temporal LSTM head to {TEMPORAL_MODEL_PATH}")
    print(f"(Untrained - will be used with static features at inference)")

    print("\n" + "=" * 60)
    print("Video model training complete!")
    print("=" * 60)

    return {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": {cls: float(f1) for cls, f1 in zip(CLASSES, f1_per_class)},
    }


if __name__ == "__main__":
    results = train_static_model()
    if results:
        print("\n=== SUMMARY ===")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"F1 Macro: {results['f1_macro']:.4f}")
        for cls, f1 in results["f1_per_class"].items():
            print(f"  {cls}: F1={f1:.4f}")
