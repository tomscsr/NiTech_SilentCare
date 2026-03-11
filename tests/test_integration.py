"""
SilentCare - Step 6: Integration Tests
=========================================
End-to-end pipeline tests with simulated data.
No real audio/video capture needed - uses mock segments.
"""

import sys
import os
import time
import tempfile
import threading
import queue
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from silentcare.core.database import Database
from silentcare.core.alert_manager import AlertManager
from silentcare.core.analysis_pipeline import AnalysisPipeline


# ============================================
# Mock Capture Service
# ============================================
class MockCaptureService:
    """Simulates CaptureService by feeding pre-defined segments."""

    def __init__(self):
        self.segment_queue = queue.Queue(maxsize=50)
        self.enable_audio = True
        self.enable_video = False  # Audio-only for these tests
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def inject_segment(self, audio=None, video_frames=None):
        """Inject a test segment into the queue."""
        segment = {
            "timestamp": time.time(),
            "audio": audio,
            "audio_sr": 22050 if audio is not None else None,
            "video_frames": video_frames or [],
            "has_audio": audio is not None,
            "has_video": len(video_frames) > 0 if video_frames else False,
        }
        self.segment_queue.put(segment, timeout=2)


# ============================================
# Mock Audio Model
# ============================================
class MockAudioModel:
    """Returns pre-configured probabilities."""

    def __init__(self, default_probs=None):
        self.classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
        self._ready = True
        self._probs_queue = []
        self._default = default_probs if default_probs is not None else np.array([0.1, 0.1, 0.1, 0.7])

    def set_next_probs(self, probs_list):
        """Set a sequence of probabilities to return."""
        self._probs_queue = list(probs_list)

    def predict(self, audio, sr=None):
        if self._probs_queue:
            probs = np.array(self._probs_queue.pop(0))
        else:
            probs = self._default.copy()
        idx = int(np.argmax(probs))
        return {
            "probabilities": probs,
            "predicted_class": self.classes[idx],
            "confidence": float(probs[idx]),
        }

    @property
    def ready(self):
        return self._ready


# ============================================
# Helper: create pipeline with mocks
# ============================================
def create_test_pipeline(db_path, audio_probs=None):
    """Create a pipeline with mock capture and audio model."""
    capture = MockCaptureService()
    db = Database(db_path)

    pipeline = AnalysisPipeline(capture_service=capture, db=db)
    pipeline._audio_model = MockAudioModel(audio_probs)
    pipeline._video_model = None
    pipeline._models_loaded = True

    return pipeline, capture, db


# ============================================
# Integration Tests
# ============================================
class TestPipelineIntegration:
    """End-to-end pipeline tests with simulated data."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.alerts_received = []

    def teardown_method(self):
        try:
            os.unlink(self.tmp.name)
        except PermissionError:
            pass  # Windows: DB file still locked by another thread

    def test_calm_segment_no_alert(self):
        """A calm segment should be processed and stored, but trigger no alert."""
        pipeline, capture, db = create_test_pipeline(
            self.tmp.name,
            audio_probs=np.array([0.05, 0.05, 0.1, 0.8])
        )

        # Start pipeline
        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        # Create a fake audio segment (1s of silence)
        audio = np.zeros(22050, dtype=np.float32)
        capture.inject_segment(audio=audio)

        # Process manually (don't start the thread)
        segment = capture.segment_queue.get(timeout=2)
        result = pipeline._process_segment(segment)

        assert result["predicted_class"] == "CALM"
        assert result["alert"] is None

        # Verify DB storage
        segments = db.get_recent_segments(pipeline._session_id, limit=10)
        assert len(segments) == 1
        assert segments[0]["predicted_class"] == "CALM"

        alerts = db.get_recent_alerts(session_id=pipeline._session_id)
        assert len(alerts) == 0

        db.close()

    def test_distress_triggers_alert_after_consecutive(self):
        """Three consecutive DISTRESS segments above threshold should trigger an alert."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            np.array([0.85, 0.05, 0.05, 0.05]),
            np.array([0.82, 0.08, 0.05, 0.05]),
            np.array([0.83, 0.07, 0.05, 0.05]),
        ])

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()
        pipeline.on_alert = lambda a: self.alerts_received.append(a)

        # Non-zero audio so RMS gate doesn't skip it
        audio = np.random.randn(22050).astype(np.float32) * 0.1

        # Segment 1: DISTRESS but no alert yet
        capture.inject_segment(audio=audio)
        seg = capture.segment_queue.get(timeout=2)
        r1 = pipeline._process_segment(seg)
        assert r1["predicted_class"] == "DISTRESS"
        assert r1["alert"] is None

        # Segment 2: DISTRESS again but still not enough (need 3)
        capture.inject_segment(audio=audio)
        seg = capture.segment_queue.get(timeout=2)
        r2 = pipeline._process_segment(seg)
        assert r2["predicted_class"] == "DISTRESS"
        assert r2["alert"] is None

        # Segment 3: DISTRESS -> alert fires
        capture.inject_segment(audio=audio)
        seg = capture.segment_queue.get(timeout=2)
        r3 = pipeline._process_segment(seg)
        assert r3["predicted_class"] == "DISTRESS"
        assert r3["alert"] is not None
        assert r3["alert"]["emotion"] == "DISTRESS"
        assert r3["alert"]["severity"] == "HIGH"

        # Verify alert in DB
        alerts = db.get_recent_alerts(session_id=pipeline._session_id)
        assert len(alerts) == 1
        assert alerts[0]["emotion"] == "DISTRESS"
        assert alerts[0]["severity"] == "HIGH"

        # Verify callback fired
        assert len(self.alerts_received) == 1

        db.close()

    def test_mixed_emotions_break_streak(self):
        """Alternating emotions should not trigger alerts."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            np.array([0.85, 0.05, 0.05, 0.05]),  # DISTRESS
            np.array([0.05, 0.05, 0.05, 0.85]),  # CALM
            np.array([0.85, 0.05, 0.05, 0.05]),  # DISTRESS
            np.array([0.05, 0.85, 0.05, 0.05]),  # ANGRY
        ])

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        # Non-zero audio so RMS gate doesn't skip it
        audio = np.random.randn(22050).astype(np.float32) * 0.1

        for _ in range(4):
            capture.inject_segment(audio=audio)
            seg = capture.segment_queue.get(timeout=2)
            result = pipeline._process_segment(seg)

        alerts = db.get_recent_alerts(session_id=pipeline._session_id)
        assert len(alerts) == 0

        db.close()

    def test_cooldown_prevents_rapid_alerts(self):
        """After an alert, the same emotion shouldn't re-trigger within cooldown."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)

        # 5 consecutive DISTRESS segments
        probs = [np.array([0.85, 0.05, 0.05, 0.05])] * 5
        pipeline._audio_model.set_next_probs(probs)

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        # Non-zero audio so RMS gate doesn't skip it
        audio = np.random.randn(22050).astype(np.float32) * 0.1

        for _ in range(5):
            capture.inject_segment(audio=audio)
            seg = capture.segment_queue.get(timeout=2)
            pipeline._process_segment(seg)

        # Only 1 alert should fire (cooldown blocks the rest)
        alerts = db.get_recent_alerts(session_id=pipeline._session_id)
        assert len(alerts) == 1

        db.close()

    def test_audio_only_fusion_passthrough(self):
        """With video model absent, fusion should pass audio probs through."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            np.array([0.1, 0.1, 0.2, 0.6]),
        ])

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        # Non-zero audio so RMS gate doesn't skip it
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        capture.inject_segment(audio=audio)
        seg = capture.segment_queue.get(timeout=2)
        result = pipeline._process_segment(seg)

        # With no video, fused should equal audio probs
        np.testing.assert_array_almost_equal(
            result["fused_probs"],
            [0.1, 0.1, 0.2, 0.6],
            decimal=4
        )

        db.close()

    def test_session_lifecycle(self):
        """Start session -> process segments -> stop session."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            np.array([0.1, 0.1, 0.1, 0.7]),
            np.array([0.1, 0.1, 0.1, 0.7]),
        ])

        # Start
        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        session = db.get_active_session()
        assert session is not None
        assert session["status"] == "active"

        # Process 2 segments (non-zero audio so RMS gate doesn't skip)
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        for _ in range(2):
            capture.inject_segment(audio=audio)
            seg = capture.segment_queue.get(timeout=2)
            pipeline._process_segment(seg)

        # Stop
        db.stop_session(pipeline._session_id)

        # Verify stats
        stats = db.get_session_stats(pipeline._session_id)
        assert stats["total_segments"] == 2
        assert stats["total_alerts"] == 0

        # No active session
        assert db.get_active_session() is None

        db.close()

    def test_alert_acknowledge_flow(self):
        """Alert -> stored in DB -> acknowledge -> verified."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            np.array([0.85, 0.05, 0.05, 0.05]),
            np.array([0.85, 0.05, 0.05, 0.05]),
            np.array([0.85, 0.05, 0.05, 0.05]),
        ])

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        # Non-zero audio so RMS gate doesn't skip it
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        for _ in range(3):
            capture.inject_segment(audio=audio)
            seg = capture.segment_queue.get(timeout=2)
            pipeline._process_segment(seg)

        # Unacknowledged alerts
        unack = db.get_unacknowledged_alerts(session_id=pipeline._session_id)
        assert len(unack) == 1

        # Acknowledge
        db.acknowledge_alert(unack[0]["id"])

        # Verified
        unack = db.get_unacknowledged_alerts(session_id=pipeline._session_id)
        assert len(unack) == 0

        db.close()

    def test_multiple_emotion_alerts(self):
        """Different emotions can each trigger their own alerts independently."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            # 3x DISTRESS -> alert
            np.array([0.85, 0.05, 0.05, 0.05]),
            np.array([0.85, 0.05, 0.05, 0.05]),
            np.array([0.85, 0.05, 0.05, 0.05]),
            # break
            np.array([0.05, 0.05, 0.05, 0.85]),
            # 3x ANGRY -> alert
            np.array([0.05, 0.80, 0.10, 0.05]),
            np.array([0.05, 0.78, 0.12, 0.05]),
            np.array([0.05, 0.82, 0.08, 0.05]),
        ])

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        # Non-zero audio so RMS gate doesn't skip it
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        for _ in range(7):
            capture.inject_segment(audio=audio)
            seg = capture.segment_queue.get(timeout=2)
            pipeline._process_segment(seg)

        alerts = db.get_recent_alerts(session_id=pipeline._session_id)
        assert len(alerts) == 2
        emotions = [a["emotion"] for a in alerts]
        assert "DISTRESS" in emotions
        assert "ANGRY" in emotions

        db.close()

    def test_segment_data_integrity(self):
        """Verify segment data is stored correctly with all fields."""
        pipeline, capture, db = create_test_pipeline(self.tmp.name)
        pipeline._audio_model.set_next_probs([
            np.array([0.3, 0.2, 0.15, 0.35]),
        ])

        pipeline.alert_manager.reset()
        pipeline._session_id = db.start_session()

        audio = np.random.randn(22050).astype(np.float32) * 0.01
        capture.inject_segment(audio=audio)
        seg = capture.segment_queue.get(timeout=2)
        pipeline._process_segment(seg)

        segments = db.get_recent_segments(pipeline._session_id, limit=1)
        assert len(segments) == 1

        s = segments[0]
        assert s["predicted_class"] == "CALM"  # 0.35 is highest
        assert s["confidence"] is not None
        assert isinstance(s["audio_probs"], list)
        assert len(s["audio_probs"]) == 4
        assert isinstance(s["fused_probs"], list)
        assert len(s["fused_probs"]) == 4
        assert s["timestamp"] is not None

        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
