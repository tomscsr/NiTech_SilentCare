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

from silentcare.ml.audio_preprocessor import AudioPreprocessor


class AudioModel:
    """
    Audio emotion classifier: YAMNet embeddings -> classification head.
    Thread-safe for inference (TF session is reentrant for predict).
    """

    def __init__(self, model_path, classes_path=None, target_sr=22050,
                 enable_noise_reduction=True, enable_vad=True,
                 noise_prop_decrease=0.75, vad_voice_threshold=0.15):
        self.target_sr = target_sr
        self.classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]

        if classes_path and Path(classes_path).exists():
            self.classes = list(np.load(str(classes_path), allow_pickle=True))

        # Load YAMNet
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

        # Load trained classification head
        self.head = tf.keras.models.load_model(str(model_path))

        # Preprocessor (instantiated once, reused per segment)
        self.preprocessor = AudioPreprocessor(
            enable_noise_reduction=enable_noise_reduction,
            enable_vad=enable_vad,
            noise_prop_decrease=noise_prop_decrease,
            vad_voice_threshold=vad_voice_threshold,
        )

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

    def predict(self, audio, sr=None, rms_threshold=0.01):
        """
        Predict emotion probabilities from raw audio waveform.

        Runs preprocessing (noise reduction, VAD, normalisation)
        before YAMNet feature extraction.

        Args:
            audio: numpy array, mono waveform
            sr: sample rate (defaults to self.target_sr)
            rms_threshold: minimum RMS to consider segment non-silent

        Returns:
            dict with keys:
                'probabilities': np.array of shape (4,) - class probabilities
                'predicted_class': str - predicted class name
                'confidence': float - confidence of top prediction
                'low_confidence': bool - True if VAD flagged non-voice
                    but RMS was above threshold
                'preprocessing': dict - metadata from AudioPreprocessor
            or None if the segment is silent and non-voice
        """
        if sr is None:
            sr = self.target_sr

        # Preprocess: noise reduction -> VAD -> normalisation
        processed_audio, meta = self.preprocessor.preprocess(audio, sr)

        # If no voice detected AND below RMS threshold: skip inference
        if not meta["is_voice"] and meta["original_rms"] < rms_threshold:
            return None

        low_confidence = not meta["is_voice"] and meta["original_rms"] >= rms_threshold

        # Extract features from preprocessed audio
        features = self._extract_embeddings(processed_audio, sr)
        features = features.reshape(1, -1)  # (1, 3072)

        # Predict
        probs = self.head.predict(features, verbose=0)[0]  # (4,)

        predicted_idx = int(np.argmax(probs))

        return {
            "probabilities": probs.astype(np.float64),
            "predicted_class": self.classes[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "low_confidence": low_confidence,
            "preprocessing": meta,
        }

    def predict_from_file(self, filepath, sr=None):
        """Predict from an audio file path."""
        if sr is None:
            sr = self.target_sr
        audio, loaded_sr = librosa.load(str(filepath), sr=sr, mono=True)
        return self.predict(audio, loaded_sr)
