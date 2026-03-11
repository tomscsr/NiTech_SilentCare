"""
SilentCare - Audio Model Wrapper
=================================
Loads YAMNet (frozen) + trained classification head (TensorFlow/Keras).
Exposes predict() returning class probabilities for a raw audio segment.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path


class AudioModel:
    """
    Audio emotion classifier: YAMNet embeddings -> classification head.
    Thread-safe for inference (TF session is reentrant for predict).
    """

    def __init__(self, model_path, classes_path=None, target_sr=22050):
        self.target_sr = target_sr
        self.classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]

        if classes_path and Path(classes_path).exists():
            self.classes = list(np.load(str(classes_path), allow_pickle=True))

        # Load YAMNet
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load trained classification head
        self.head = tf.keras.models.load_model(str(model_path))

        self._ready = True

    @property
    def ready(self):
        return self._ready

    def _extract_embeddings(self, audio, sr):
        """Extract YAMNet embeddings and aggregate via mean+max+std pooling -> 3072-dim."""
        # YAMNet expects 16kHz
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        audio_16k = audio_16k.astype(np.float32)

        # YAMNet returns (scores, embeddings, spectrogram)
        _, embeddings, _ = self.yamnet(audio_16k)
        embeddings_np = embeddings.numpy()

        if len(embeddings_np) == 0:
            return np.zeros(3072, dtype=np.float32)

        # Temporal aggregation
        mean_pool = np.mean(embeddings_np, axis=0)   # 1024
        max_pool = np.max(embeddings_np, axis=0)      # 1024
        std_pool = np.std(embeddings_np, axis=0)      # 1024

        return np.concatenate([mean_pool, max_pool, std_pool]).astype(np.float32)

    def predict(self, audio, sr=None):
        """
        Predict emotion probabilities from raw audio waveform.

        Args:
            audio: numpy array, mono waveform
            sr: sample rate (defaults to self.target_sr)

        Returns:
            dict with keys:
                'probabilities': np.array of shape (4,) - class probabilities
                'predicted_class': str - predicted class name
                'confidence': float - confidence of top prediction
        """
        if sr is None:
            sr = self.target_sr

        # Extract features
        features = self._extract_embeddings(audio, sr)
        features = features.reshape(1, -1)  # (1, 3072)

        # Predict
        probs = self.head.predict(features, verbose=0)[0]  # (4,)

        predicted_idx = int(np.argmax(probs))

        return {
            "probabilities": probs.astype(np.float64),
            "predicted_class": self.classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
        }

    def predict_from_file(self, filepath, sr=None):
        """Predict from an audio file path."""
        if sr is None:
            sr = self.target_sr
        audio, loaded_sr = librosa.load(str(filepath), sr=sr, mono=True)
        return self.predict(audio, loaded_sr)
