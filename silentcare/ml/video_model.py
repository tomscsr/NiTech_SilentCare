"""
SilentCare - Video Model Wrapper
==================================
Supports 5 backends:
  - vit_trpakov (default, production): trpakov/vit-face-expression
  - vit_dima806: dima806/facial_emotions_image_detection
  - resnet50: locally trained ResNet50 on RAF-DB
  - efficientnet_b2: locally trained EfficientNet-B2 on RAF-DB
  - mobilenet_v3: locally trained MobileNetV3-Large on RAF-DB

Face detection (Haar cascade) + cropping before inference.
ViT models output 7 FER emotions, remapped to 4 SilentCare classes.
Local models output 4-class logits directly.
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
# Works for both trpakov and dima806 ViTs (same label names)
FER_TO_SILENTCARE = {
    "angry": 1,      # ANGRY
    "disgust": 0,     # DISTRESS
    "fear": 0,        # DISTRESS
    "happy": 3,       # CALM
    "neutral": 3,     # CALM
    "sad": 0,         # DISTRESS
    "surprise": 2,    # ALERT
}

# Architecture constants (must match train_video.py)
FEATURE_DIM = 512

# Backend -> model filename mapping
BACKEND_MODEL_FILES = {
    "resnet50": "Video_SilentCare_model.pth",
    "efficientnet_b2": "EfficientNet_B2_SilentCare.pth",
    "mobilenet_v3": "MobileNetV3_SilentCare.pth",
}

# Backend -> raw feature dimension (before projection)
BACKEND_FEATURE_DIMS = {
    "resnet50": 2048,
    "efficientnet_b2": 1408,
    "mobilenet_v3": 960,
}

# HuggingFace ViT model names
VIT_MODELS = {
    "vit_trpakov": "trpakov/vit-face-expression",
    "vit_dima806": "dima806/facial_emotions_image_detection",
}

ALL_BACKENDS = list(VIT_MODELS.keys()) + list(BACKEND_MODEL_FILES.keys())


class VideoModel:
    """
    Video emotion classifier with 5 backend options.
    Default: HuggingFace ViT trpakov (production).

    Face detection + CLAHE/LAB enhancement + classification.
    """

    def __init__(self, model_path=None, use_resnet=False,
                 vit_model_name="trpakov/vit-face-expression",
                 backend=None, **kwargs):
        """
        Args:
            model_path: Path to .pth file (only for local models).
            use_resnet: Legacy flag. If True, equivalent to backend="resnet50".
            vit_model_name: HuggingFace model name (only when backend is None and use_resnet=False).
            backend: One of "vit_trpakov", "vit_dima806", "resnet50",
                     "efficientnet_b2", "mobilenet_v3". Overrides use_resnet/vit_model_name.
        """
        self.classes = CLASSES

        # Resolve backend
        if backend is not None:
            self._backend = backend
        elif use_resnet:
            self._backend = "resnet50"
        else:
            # Map vit_model_name to backend key
            rev = {v: k for k, v in VIT_MODELS.items()}
            self._backend = rev.get(vit_model_name, "vit_trpakov")

        self._use_vit = self._backend.startswith("vit_")

        # Face detector (Haar cascade, ships with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # CLAHE for low-light / IR camera enhancement
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        if self._use_vit:
            hf_name = VIT_MODELS.get(self._backend, vit_model_name)
            self._load_vit(hf_name)
        elif self._backend in BACKEND_MODEL_FILES:
            self._load_local_model(self._backend, model_path)
        else:
            raise ValueError(f"Unknown backend: {self._backend}. Choose from: {ALL_BACKENDS}")

        self._ready = True

    def _load_local_model(self, backend_name, model_path):
        """Load a locally trained model (ResNet50, EfficientNet-B2, or MobileNetV3)."""
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        raw_dim = BACKEND_FEATURE_DIMS[backend_name]
        model_file = BACKEND_MODEL_FILES[backend_name]

        # Auto-detect model path if not provided
        if model_path is None:
            project_dir = Path(__file__).resolve().parent.parent.parent
            model_path = str(project_dir / "model" / model_file)

        print(f"[VideoModel] Loading {backend_name} from: {model_path}")

        # Build backbone architecture (weights=None, will load from state_dict)
        if backend_name == "resnet50":
            backbone = models.resnet50(weights=None)
            self._features = nn.Sequential(*list(backbone.children())[:-1])
        elif backend_name == "efficientnet_b2":
            backbone = models.efficientnet_b2(weights=None)
            self._features = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d(1),
            )
        elif backend_name == "mobilenet_v3":
            backbone = models.mobilenet_v3_large(weights=None)
            self._features = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d(1),
            )

        # Projection and classifier (same for all local models)
        self._projection = nn.Sequential(
            nn.Linear(raw_dim, FEATURE_DIM),
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
        print(f"[VideoModel] {backend_name} loaded successfully.")

    def _load_vit(self, model_name):
        """Load a HuggingFace ViT model."""
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
        """Legacy alias for _classify_face_local (backward compat)."""
        return self._classify_face_local(face_bgr)

    def _classify_face_local(self, face_bgr):
        """
        Run locally trained classifier on a cropped face image.
        Works for ResNet50, EfficientNet-B2, and MobileNetV3.

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
            features = features.flatten(1)
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
            return self._classify_face_local(face_bgr)

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
