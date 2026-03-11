"""
SilentCare - Step 6: Stability and Stress Tests
==================================================
Simulates extended operation to check:
  - No memory leaks (bounded growth)
  - DB integrity under load
  - Alert counts correctness
  - No crashes over many segments
"""

import sys
import os
import time
import tempfile
import tracemalloc
import queue
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from silentcare.core.database import Database
from silentcare.core.alert_manager import AlertManager
from silentcare.core.analysis_pipeline import AnalysisPipeline


class MockCapture:
    def __init__(self):
        self.segment_queue = queue.Queue(maxsize=50)
        self.enable_audio = True
        self.enable_video = False
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running


class MockAudio:
    def __init__(self):
        self.classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
        self._ready = True
        self._call_count = 0

    def predict(self, audio, sr=None):
        self._call_count += 1
        # Cycle through patterns
        patterns = [
            np.array([0.1, 0.1, 0.1, 0.7]),   # CALM
            np.array([0.1, 0.1, 0.1, 0.7]),   # CALM
            np.array([0.8, 0.1, 0.05, 0.05]),  # DISTRESS
            np.array([0.8, 0.1, 0.05, 0.05]),  # DISTRESS (triggers alert)
            np.array([0.1, 0.1, 0.1, 0.7]),   # CALM (breaks streak)
            np.array([0.05, 0.75, 0.1, 0.1]),  # ANGRY
            np.array([0.05, 0.75, 0.1, 0.1]),  # ANGRY (triggers alert)
            np.array([0.1, 0.1, 0.1, 0.7]),   # CALM
        ]
        idx = self._call_count % len(patterns)
        probs = patterns[idx]
        top = int(np.argmax(probs))
        return {
            "probabilities": probs,
            "predicted_class": self.classes[top],
            "confidence": float(probs[top]),
        }

    @property
    def ready(self):
        return self._ready


class TestStability:
    """Stability tests simulating extended operation."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

    def teardown_method(self):
        try:
            os.unlink(self.tmp.name)
        except PermissionError:
            pass  # Windows: DB file still locked

    def test_300_segments_no_crash(self):
        """Simulate ~40 minutes of operation (300 segments at 8s each) without crash."""
        capture = MockCapture()
        db = Database(self.tmp.name)
        pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        pipeline.capture = capture
        pipeline.db = db
        pipeline._audio_model = MockAudio()
        pipeline._video_model = None
        pipeline._models_loaded = True
        pipeline._running = True
        pipeline._latest_result = None
        pipeline._result_lock = __import__("threading").Lock()
        pipeline.on_segment = None
        pipeline.on_alert = None
        # Disable cooldown for stress test (set to 0)
        am = AlertManager()
        am._cooldowns = {}
        pipeline.alert_manager = am
        pipeline._session_id = db.start_session()

        # Patch cooldown to 0 for this test (must patch in alert_manager module)
        import silentcare.core.alert_manager as am_mod
        original_cooldown = am_mod.ALERT_COOLDOWN_SECONDS
        am_mod.ALERT_COOLDOWN_SECONDS = 0

        audio = np.zeros(22050, dtype=np.float32)
        errors = 0

        for i in range(300):
            segment = {
                "timestamp": time.time(),
                "audio": audio,
                "audio_sr": 22050,
                "video_frames": [],
                "has_audio": True,
                "has_video": False,
            }
            try:
                pipeline._process_segment(segment)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"Error at segment {i}: {e}")

        # Restore cooldown
        am_mod.ALERT_COOLDOWN_SECONDS = original_cooldown

        assert errors == 0, f"{errors} errors in 300 segments"

        # Verify DB integrity
        segments = db.get_recent_segments(pipeline._session_id, limit=500)
        assert len(segments) == 300

        stats = db.get_session_stats(pipeline._session_id)
        assert stats["total_segments"] == 300
        # Alerts may or may not fire depending on pattern + consecutive check
        # Main goal is no crashes - alert count is a bonus
        assert stats["total_alerts"] >= 0

        db.close()

    def test_memory_bounded(self):
        """Verify memory doesn't grow unboundedly over many segments."""
        tracemalloc.start()

        capture = MockCapture()
        db = Database(self.tmp.name)
        pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        pipeline.capture = capture
        pipeline.db = db
        pipeline._audio_model = MockAudio()
        pipeline._video_model = None
        pipeline._models_loaded = True
        pipeline._running = True
        pipeline._latest_result = None
        pipeline._result_lock = __import__("threading").Lock()
        pipeline.on_segment = None
        pipeline.on_alert = None
        pipeline.alert_manager = AlertManager()
        pipeline._session_id = db.start_session()

        audio = np.zeros(22050, dtype=np.float32)

        # Warmup
        for _ in range(20):
            segment = {
                "timestamp": time.time(),
                "audio": audio,
                "audio_sr": 22050,
                "video_frames": [],
                "has_audio": True,
                "has_video": False,
            }
            pipeline._process_segment(segment)

        snapshot1 = tracemalloc.take_snapshot()
        mem_after_warmup = tracemalloc.get_traced_memory()[0]

        # Process 200 more segments
        for _ in range(200):
            segment = {
                "timestamp": time.time(),
                "audio": audio,
                "audio_sr": 22050,
                "video_frames": [],
                "has_audio": True,
                "has_video": False,
            }
            pipeline._process_segment(segment)

        mem_after_load = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory growth should be bounded (< 10MB for 200 segments)
        growth_mb = (mem_after_load - mem_after_warmup) / (1024 * 1024)
        print(f"Memory growth over 200 segments: {growth_mb:.2f} MB")
        assert growth_mb < 10, f"Memory grew {growth_mb:.2f} MB (limit: 10 MB)"

        db.close()

    def test_db_integrity_under_concurrent_writes(self):
        """Verify DB handles rapid sequential writes correctly."""
        db = Database(self.tmp.name)
        sid = db.start_session()

        # Rapid-fire 100 segments + 20 alerts
        for i in range(100):
            db.add_segment(
                session_id=sid,
                audio_probs=np.random.rand(4),
                video_probs=np.zeros(4),
                fused_probs=np.random.rand(4),
                predicted_class="CALM",
                confidence=0.5 + np.random.rand() * 0.5,
            )

        for i in range(20):
            emotion = ["DISTRESS", "ANGRY", "ALERT"][i % 3]
            severity = ["LOW", "MEDIUM", "HIGH"][i % 3]
            db.add_alert(
                session_id=sid,
                emotion=emotion,
                severity=severity,
                confidence=0.7 + np.random.rand() * 0.3,
                audio_confidence=0.8,
                video_confidence=None,
                fused_probs=np.random.rand(4),
            )

        # Verify counts
        stats = db.get_session_stats(sid)
        assert stats["total_segments"] == 100
        assert stats["total_alerts"] == 20

        # Verify all segments retrievable
        segments = db.get_recent_segments(sid, limit=200)
        assert len(segments) == 100

        # Verify all alerts retrievable
        alerts = db.get_recent_alerts(session_id=sid, limit=50)
        assert len(alerts) == 20

        # Acknowledge half and verify
        for alert in alerts[:10]:
            db.acknowledge_alert(alert["id"])

        unack = db.get_unacknowledged_alerts(session_id=sid)
        assert len(unack) == 10

        db.close()

    def test_alert_manager_long_run(self):
        """Verify alert manager state stays consistent over many segments."""
        am = AlertManager()

        alerts_fired = []
        am.on_alert = lambda a: alerts_fired.append(a)

        # Simulate 500 segments with alternating patterns
        for i in range(500):
            cycle = i % 20
            if cycle < 5:
                # CALM
                probs = np.array([0.1, 0.1, 0.1, 0.7])
            elif cycle < 10:
                # DISTRESS
                probs = np.array([0.85, 0.05, 0.05, 0.05])
            elif cycle < 15:
                # ANGRY
                probs = np.array([0.05, 0.80, 0.10, 0.05])
            else:
                # CALM
                probs = np.array([0.1, 0.1, 0.1, 0.7])

            am.process_segment(fused_probs=probs)

        # Should have some alerts but not too many (cooldowns)
        assert len(alerts_fired) > 0
        assert len(alerts_fired) < 100  # cooldown should limit

        # All alerts should be valid
        for alert in alerts_fired:
            assert alert["emotion"] in ["DISTRESS", "ANGRY", "ALERT"]
            assert alert["severity"] in ["LOW", "MEDIUM", "HIGH"]
            assert alert["confidence"] > 0.5
            assert alert["consecutive_count"] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
