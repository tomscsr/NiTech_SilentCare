"""
SilentCare - Video Model Wrapper
==================================
Default: ViT from HuggingFace (trpakov/vit-face-expression, FER-2013).
Optional: ResNet50 (use_resnet=True) for comparison (trained on RAF-DB).

Face detection (Haar cascade) + cropping before inference.
ViT outputs 7 FER emotions, remapped to 4 SilentCare classes.
ResNet50 outputs 4-class logits directly.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import cv2
from PIL import Image
from pathlib import Path

CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4

# HuggingFace FER labels -> SilentCare class index
FER_TO_SILENTCARE = {
    "angry": 1,      # ANGRY
    "disgust": 0,     # DISTRESS
    "fear": 0,        # DISTRESS
    "happy": 3,       # CALM
    "neutral": 3,     # CALM
    "sad": 0,         # DISTRESS
    "surprise": 2,    # ALERT
}

# ResNet50 architecture constants (must match train_video.py)
FEATURE_DIM = 512


class VideoModel:
    """
    Video emotion classifier.
    Default: HuggingFace ViT (7-class FER -> 4-class mapping).
    Optional: ResNet50 fine-tuned on RAF-DB (use_resnet=True, 4-class direct output).

    Face detection + CLAHE/LAB enhancement + classification.
    """

    def __init__(self, model_path=None, use_resnet=False,
                 vit_model_name="trpakov/vit-face-expression", **kwargs):
        """
        Args:
            model_path: Path to ResNet50 .pth file (only used if use_resnet=True).
            use_resnet: If True, use ResNet50 instead of ViT.
            vit_model_name: HuggingFace model name for ViT.
        """
        self.classes = CLASSES
        self._use_vit = not use_resnet

        # Face detector (Haar cascade, ships with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # CLAHE for low-light / IR camera enhancement
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        if use_resnet:
            self._load_resnet50(model_path)
        else:
            self._load_vit(vit_model_name)

        self._ready = True

    def _load_resnet50(self, model_path):
        """Load the locally trained ResNet50 model."""
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        # Auto-detect model path if not provided
        if model_path is None:
            project_dir = Path(__file__).resolve().parent.parent.parent
            model_path = str(project_dir / "model" / "Video_SilentCare_model.pth")

        print(f"[VideoModel] Loading ResNet50 from: {model_path}")

        # Build the same architecture as train_video.py
        backbone = models.resnet50(weights=None)
        self._features = nn.Sequential(*list(backbone.children())[:-1])
        self._projection = nn.Sequential(
            nn.Linear(2048, FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self._classifier_head = nn.Linear(FEATURE_DIM, NUM_CLASSES)

        # Load trained weights
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        # Map keys from saved model to our attribute names
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("features."):
                mapped[k] = v
            elif k.startswith("projection."):
                mapped[k] = v
            elif k.startswith("classifier."):
                mapped[k.replace("classifier.", "classifier_head.")] = v
            else:
                mapped[k] = v

        # Load into sub-modules
        features_sd = {k.replace("features.", ""): v for k, v in mapped.items() if k.startswith("features.")}
        projection_sd = {k.replace("projection.", ""): v for k, v in mapped.items() if k.startswith("projection.")}
        classifier_sd = {k.replace("classifier_head.", ""): v for k, v in mapped.items() if k.startswith("classifier_head.")}

        self._features.load_state_dict(features_sd)
        self._projection.load_state_dict(projection_sd)
        self._classifier_head.load_state_dict(classifier_sd)

        self._features.eval()
        self._projection.eval()
        self._classifier_head.eval()

        # ImageNet normalization transform (must match training)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._torch = torch
        print("[VideoModel] ResNet50 loaded successfully.")

    def _load_vit(self, model_name):
        """Load the HuggingFace ViT model (production)."""
        from transformers import pipeline as hf_pipeline
        print(f"[VideoModel] Loading ViT from HuggingFace: {model_name}")
        self._vit_classifier = hf_pipeline(
            "image-classification",
            model=model_name,
            top_k=7,
        )
        print("[VideoModel] ViT loaded successfully.")

    @property
    def ready(self):
        return self._ready

    def _detect_face(self, frame_bgr):
        """
        Detect the largest face in a BGR frame and return cropped face.
        Uses CLAHE enhancement for low-light/IR cameras.

        Args:
            frame_bgr: numpy array (H, W, 3) BGR

        Returns:
            Cropped face as numpy array (BGR), or None if no face found
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to handle dark/IR camera images
        gray_enhanced = self._clahe.apply(gray)

        # Try detection on enhanced image first
        faces = self.face_cascade.detectMultiScale(
            gray_enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        if len(faces) == 0:
            return None

        # Pick largest face by area
        areas = [w * h for (x, y, w, h) in faces]
        idx = int(np.argmax(areas))
        x, y, w, h = faces[idx]

        # Add 20% margin around face for context
        margin = 0.2
        mx, my = int(w * margin), int(h * margin)
        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(fw, x + w + mx)
        y2 = min(fh, y + h + my)

        return frame_bgr[y1:y2, x1:x2]

    def _enhance_face(self, face_bgr):
        """
        Enhance a dark face crop for better classification.
        Uses histogram equalization on L channel of LAB color space.
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        if gray.mean() > 80:
            return face_bgr  # Already bright enough

        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _classify_face_resnet(self, face_bgr):
        """
        Run ResNet50 classifier on a cropped face image.

        Returns:
            np.array of shape (4,) with SilentCare class probabilities
        """
        face_bgr = self._enhance_face(face_bgr)

        # Convert BGR to RGB PIL image
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face_rgb)

        # Apply ImageNet transforms
        tensor = self._transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        # Inference
        with self._torch.no_grad():
            features = self._features(tensor)
            features = features.flatten(1)  # (1, 2048)
            features = self._projection(features)  # (1, 512)
            logits = self._classifier_head(features)  # (1, 4)
            probs = self._torch.nn.functional.softmax(logits, dim=1)

        return probs[0].numpy().astype(np.float64)

    def _classify_face_vit(self, face_bgr):
        """
        Run HuggingFace ViT classifier on a cropped face image.

        Returns:
            np.array of shape (4,) with SilentCare class probabilities
        """
        face_bgr = self._enhance_face(face_bgr)

        # Convert BGR to RGB PIL image
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(face_rgb)

        # Classify with HuggingFace pipeline
        results = self._vit_classifier(image)

        # Map 7 FER labels to 4 SilentCare classes
        probs = np.zeros(NUM_CLASSES, dtype=np.float64)
        for r in results:
            label = r["label"]
            if label in FER_TO_SILENTCARE:
                idx = FER_TO_SILENTCARE[label]
                probs[idx] += r["score"]

        return probs

    def _classify_face(self, face_bgr):
        """Route to the active classifier backend."""
        if self._use_vit:
            return self._classify_face_vit(face_bgr)
        else:
            return self._classify_face_resnet(face_bgr)

    def predict_frame(self, frame):
        """
        Predict emotion from a single frame.
        Detects face first; returns None if no face found.

        Args:
            frame: numpy array (H, W, 3) BGR from OpenCV, or PIL Image

        Returns:
            dict with 'probabilities', 'predicted_class', 'confidence'
            or None if no face detected
        """
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
            # RGB to BGR
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = frame[:, :, ::-1]

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        face_crop = self._detect_face(frame)
        if face_crop is None:
            return None

        probs = self._classify_face(face_crop)
        predicted_idx = int(np.argmax(probs))

        return {
            "probabilities": probs,
            "predicted_class": self.classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
        }

    def predict_sequence(self, frames):
        """
        Predict emotion from a sequence of frames.
        Averages per-frame predictions (recent frames weighted more).

        Args:
            frames: list of numpy arrays (H, W, 3) BGR or PIL Images

        Returns:
            dict or None if no face found in any frame
        """
        if not frames:
            return None

        frame_probs_list = []

        for frame in frames:
            result = self.predict_frame(frame)
            if result is not None:
                frame_probs_list.append(result["probabilities"])

        # No face found in any frame
        if not frame_probs_list:
            return None

        # Weighted average (recent frames weighted more)
        weights = np.linspace(0.5, 1.0, len(frame_probs_list))
        weights /= weights.sum()
        probs = np.zeros(NUM_CLASSES, dtype=np.float64)
        for w, fp in zip(weights, frame_probs_list):
            probs += w * fp

        predicted_idx = int(np.argmax(probs))

        return {
            "probabilities": probs,
            "predicted_class": self.classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "frame_probabilities": frame_probs_list,
        }

    def predict(self, frames):
        """
        Main predict method. Accepts single frame or list of frames.
        Returns None if no face detected.
        """
        if isinstance(frames, list):
            return self.predict_sequence(frames)
        else:
            return self.predict_frame(frames)
