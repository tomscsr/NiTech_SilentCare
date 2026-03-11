"""
SilentCare - Audio Preprocessor
================================
Noise reduction, Voice Activity Detection, and normalisation
applied to raw audio before YAMNet feature extraction.
"""

import numpy as np

try:
    import noisereduce as nr
    _HAS_NOISEREDUCE = True
except ImportError:
    _HAS_NOISEREDUCE = False


class AudioPreprocessor:
    """
    Preprocesses raw audio segments before inference.

    Pipeline: noise reduction -> VAD check -> peak normalisation.
    All steps are optional and configurable.
    """

    def __init__(
        self,
        enable_noise_reduction=True,
        enable_vad=True,
        noise_prop_decrease=0.75,
        vad_voice_threshold=0.15,
        vad_frame_ms=20,
        noise_profile_seconds=0.5,
    ):
        self.enable_noise_reduction = enable_noise_reduction and _HAS_NOISEREDUCE
        self.enable_vad = enable_vad
        self.noise_prop_decrease = noise_prop_decrease
        self.vad_voice_threshold = vad_voice_threshold
        self.vad_frame_ms = vad_frame_ms
        self.noise_profile_seconds = noise_profile_seconds

    def reduce_noise(self, waveform, sample_rate):
        """
        Apply spectral gating noise reduction.

        Uses the first noise_profile_seconds of the segment as the
        stationary noise profile.  Skips if the segment is shorter
        than the profile window or if noisereduce is unavailable.
        """
        if not self.enable_noise_reduction:
            return waveform, False

        profile_samples = int(self.noise_profile_seconds * sample_rate)
        if len(waveform) < profile_samples:
            return waveform, False

        noise_clip = waveform[:profile_samples]
        reduced = nr.reduce_noise(
            y=waveform.astype(np.float64),
            sr=sample_rate,
            y_noise=noise_clip.astype(np.float64),
            prop_decrease=self.noise_prop_decrease,
            stationary=True,
        )
        return reduced.astype(np.float32), True

    def detect_voice(self, waveform, sample_rate):
        """
        Energy-based Voice Activity Detection.

        Splits the signal into fixed-length frames, computes RMS per
        frame, and classifies each frame as voiced or unvoiced using
        a dynamic threshold (mean RMS * 1.5).

        Returns:
            is_voice (bool): True if voiced frame ratio >= vad_voice_threshold
            voiced_ratio (float): fraction of frames classified as voiced
        """
        if not self.enable_vad:
            return True, 1.0

        frame_length = int(self.vad_frame_ms / 1000.0 * sample_rate)
        if frame_length < 1 or len(waveform) < frame_length:
            return True, 1.0

        n_frames = len(waveform) // frame_length
        if n_frames == 0:
            return True, 1.0

        frames = waveform[: n_frames * frame_length].reshape(n_frames, frame_length)
        rms_per_frame = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

        mean_rms = np.mean(rms_per_frame)
        threshold = mean_rms * 1.5

        voiced_count = int(np.sum(rms_per_frame > threshold))
        voiced_ratio = voiced_count / n_frames

        is_voice = voiced_ratio >= self.vad_voice_threshold
        return is_voice, float(voiced_ratio)

    def normalise(self, waveform):
        """
        Peak normalisation to 0.95.

        Skips near-silence segments (peak < 0.01) to avoid amplifying
        noise floors.
        """
        peak = np.max(np.abs(waveform))
        if peak <= 0.01:
            return waveform, False

        normalised = waveform / peak * 0.95
        return normalised.astype(np.float32), True

    def preprocess(self, waveform, sample_rate):
        """
        Full preprocessing pipeline.

        Args:
            waveform: numpy array, mono audio signal (float32)
            sample_rate: int, sample rate in Hz

        Returns:
            processed_waveform: numpy array (same length as input)
            metadata: dict with keys:
                noise_reduced (bool), is_voice (bool),
                voiced_ratio (float),
                original_rms (float), processed_rms (float)
        """
        waveform = np.asarray(waveform, dtype=np.float32)
        original_rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))

        # 1. Noise reduction
        processed, noise_reduced = self.reduce_noise(waveform, sample_rate)

        # 2. Voice Activity Detection
        is_voice, voiced_ratio = self.detect_voice(processed, sample_rate)

        # 3. Peak normalisation
        processed, _ = self.normalise(processed)

        processed_rms = float(np.sqrt(np.mean(processed.astype(np.float64) ** 2)))

        metadata = {
            "noise_reduced": noise_reduced,
            "is_voice": is_voice,
            "voiced_ratio": voiced_ratio,
            "original_rms": original_rms,
            "processed_rms": processed_rms,
        }

        return processed, metadata
