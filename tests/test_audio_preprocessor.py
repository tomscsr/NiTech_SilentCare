"""
Tests for AudioPreprocessor: noise reduction, VAD, normalisation.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from silentcare.ml.audio_preprocessor import AudioPreprocessor


# ============================================
# Helpers
# ============================================

def make_sine(freq=440, duration=1.0, sr=22050, amplitude=0.5):
    """Generate a sine wave (simulates a voiced segment)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration=1.0, sr=22050):
    """Generate near-silence (very low amplitude noise)."""
    n = int(sr * duration)
    return (np.random.randn(n) * 0.0001).astype(np.float32)


def make_voiced(duration=1.0, sr=22050):
    """Generate a signal that VAD should detect as voice (loud bursts)."""
    n = int(sr * duration)
    signal = np.zeros(n, dtype=np.float32)
    # Create loud bursts in 40% of the signal
    burst_len = n // 5
    for i in range(0, n, burst_len * 2):
        end = min(i + burst_len, n)
        signal[i:end] = make_sine(300, (end - i) / sr, sr, amplitude=0.8)[:end - i]
    return signal


# ============================================
# Noise reduction tests
# ============================================

class TestNoiseReduction:
    def test_noise_reduction_runs_without_error(self):
        pp = AudioPreprocessor(enable_noise_reduction=True, enable_vad=False)
        waveform = make_sine(duration=2.0)
        result, reduced = pp.reduce_noise(waveform, 22050)
        assert reduced is True
        assert len(result) == len(waveform)

    def test_noise_reduction_skips_short_segment(self):
        pp = AudioPreprocessor(enable_noise_reduction=True, enable_vad=False)
        waveform = make_sine(duration=0.3)  # < 0.5s
        result, reduced = pp.reduce_noise(waveform, 22050)
        assert reduced is False
        np.testing.assert_array_equal(result, waveform)

    def test_noise_reduction_disabled(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=False)
        waveform = make_sine(duration=2.0)
        result, reduced = pp.reduce_noise(waveform, 22050)
        assert reduced is False
        np.testing.assert_array_equal(result, waveform)


# ============================================
# VAD tests
# ============================================

class TestVAD:
    def test_vad_silent_segment(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=True)
        silence = make_silence(duration=1.0)
        is_voice, ratio = pp.detect_voice(silence, 22050)
        assert is_voice is False
        assert ratio < 0.15

    def test_vad_voiced_segment(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=True)
        voiced = make_voiced(duration=1.0)
        is_voice, ratio = pp.detect_voice(voiced, 22050)
        assert is_voice is True
        assert ratio >= 0.15

    def test_vad_disabled_always_returns_voice(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=False)
        silence = make_silence(duration=1.0)
        is_voice, ratio = pp.detect_voice(silence, 22050)
        assert is_voice is True
        assert ratio == 1.0


# ============================================
# Normalisation tests
# ============================================

class TestNormalisation:
    def test_normalisation_keeps_peak_below_095(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=False)
        waveform = make_sine(amplitude=0.7)
        result, applied = pp.normalise(waveform)
        assert applied is True
        assert np.max(np.abs(result)) <= 0.9501  # small float tolerance

    def test_normalisation_skips_near_silence(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=False)
        near_silent = np.ones(1000, dtype=np.float32) * 0.005
        result, applied = pp.normalise(near_silent)
        assert applied is False
        np.testing.assert_array_equal(result, near_silent)


# ============================================
# Full preprocess() tests
# ============================================

class TestPreprocess:
    def test_returns_correct_metadata_keys(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=False)
        waveform = make_sine(duration=1.0)
        processed, meta = pp.preprocess(waveform, 22050)
        assert "noise_reduced" in meta
        assert "is_voice" in meta
        assert "voiced_ratio" in meta
        assert "original_rms" in meta
        assert "processed_rms" in meta

    def test_output_same_length_as_input(self):
        pp = AudioPreprocessor(enable_noise_reduction=True, enable_vad=True)
        waveform = make_sine(duration=2.0)
        processed, _ = pp.preprocess(waveform, 22050)
        assert len(processed) == len(waveform)

    def test_preprocessor_with_noise_reduction_disabled(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=True)
        waveform = make_sine(duration=1.0)
        _, meta = pp.preprocess(waveform, 22050)
        assert meta["noise_reduced"] is False

    def test_original_rms_is_positive_for_nonsilent(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=False)
        waveform = make_sine(duration=1.0, amplitude=0.5)
        _, meta = pp.preprocess(waveform, 22050)
        assert meta["original_rms"] > 0.0

    def test_silent_segment_flagged_no_voice(self):
        pp = AudioPreprocessor(enable_noise_reduction=False, enable_vad=True)
        silence = make_silence(duration=1.0)
        _, meta = pp.preprocess(silence, 22050)
        assert meta["is_voice"] is False


# ============================================
# Integration with AudioModel (mocked)
# ============================================

class TestAudioModelIntegration:
    def _make_mock_model(self, enable_nr=False, enable_vad=True):
        """Build a minimal AudioModel with mocked TF components."""
        with patch("silentcare.ml.audio_model.hub") as mock_hub, \
             patch("silentcare.ml.audio_model.tf") as mock_tf:

            # Mock YAMNet: return fake embeddings
            mock_yamnet = MagicMock()
            fake_embeddings = MagicMock()
            fake_embeddings.numpy.return_value = np.random.randn(10, 1024).astype(np.float32)
            mock_yamnet.return_value = (None, fake_embeddings, None)
            mock_hub.load.return_value = mock_yamnet

            # Mock Keras head: return fake probabilities
            mock_head = MagicMock()
            mock_head.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
            mock_tf.keras.models.load_model.return_value = mock_head

            from silentcare.ml.audio_model import AudioModel
            model = AudioModel(
                "fake_path.h5",
                enable_noise_reduction=enable_nr,
                enable_vad=enable_vad,
            )
        return model

    def test_predict_returns_none_on_silent_segment(self):
        model = self._make_mock_model()
        silence = make_silence(duration=1.0)
        result = model.predict(silence, sr=22050, rms_threshold=0.01)
        assert result is None

    def test_predict_returns_probabilities_on_voiced_segment(self):
        model = self._make_mock_model()
        voiced = make_voiced(duration=1.0)
        result = model.predict(voiced, sr=22050)
        assert result is not None
        assert "probabilities" in result
        assert len(result["probabilities"]) == 4
        assert "low_confidence" in result
        assert "preprocessing" in result
