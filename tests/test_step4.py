"""
SilentCare - Step 4 Unit Tests
================================
Tests for: database, alert_manager, analysis_pipeline (fusion logic).
ML model wrappers are tested with mocks (no model files needed).
"""

import sys
import os
import time
import tempfile
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================
# Database tests
# ============================================
class TestDatabase:

    def setup_method(self):
        from silentcare.core.database import Database
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_session_lifecycle(self):
        sid = self.db.start_session()
        assert sid is not None
        assert sid > 0

        session = self.db.get_active_session()
        assert session is not None
        assert session["id"] == sid
        assert session["status"] == "active"

        self.db.stop_session(sid)
        session = self.db.get_active_session()
        assert session is None  # no active session

    def test_add_segment(self):
        sid = self.db.start_session()
        seg_id = self.db.add_segment(
            session_id=sid,
            audio_probs=np.array([0.8, 0.1, 0.05, 0.05]),
            video_probs=np.array([0.6, 0.2, 0.1, 0.1]),
            fused_probs=np.array([0.7, 0.15, 0.08, 0.07]),
            predicted_class="DISTRESS",
            confidence=0.7,
        )
        assert seg_id > 0

        segments = self.db.get_recent_segments(sid, limit=10)
        assert len(segments) == 1
        assert segments[0]["predicted_class"] == "DISTRESS"
        assert isinstance(segments[0]["fused_probs"], list)
        assert len(segments[0]["fused_probs"]) == 4

    def test_add_alert(self):
        sid = self.db.start_session()
        alert_id = self.db.add_alert(
            session_id=sid,
            emotion="DISTRESS",
            severity="HIGH",
            confidence=0.85,
            audio_confidence=0.9,
            video_confidence=0.7,
            fused_probs=np.array([0.85, 0.05, 0.05, 0.05]),
            consecutive_count=3,
        )
        assert alert_id > 0

        alerts = self.db.get_recent_alerts(session_id=sid)
        assert len(alerts) == 1
        assert alerts[0]["emotion"] == "DISTRESS"
        assert alerts[0]["severity"] == "HIGH"

    def test_acknowledge_alert(self):
        sid = self.db.start_session()
        alert_id = self.db.add_alert(
            session_id=sid,
            emotion="ANGRY",
            severity="MEDIUM",
            confidence=0.75,
            audio_confidence=0.8,
            video_confidence=0.6,
            fused_probs=np.array([0.1, 0.75, 0.1, 0.05]),
        )

        unack = self.db.get_unacknowledged_alerts(session_id=sid)
        assert len(unack) == 1

        self.db.acknowledge_alert(alert_id)

        unack = self.db.get_unacknowledged_alerts(session_id=sid)
        assert len(unack) == 0

    def test_session_stats(self):
        sid = self.db.start_session()

        for _ in range(5):
            self.db.add_segment(
                session_id=sid,
                audio_probs=np.array([0.5, 0.2, 0.2, 0.1]),
                video_probs=np.array([0.4, 0.3, 0.2, 0.1]),
                fused_probs=np.array([0.45, 0.25, 0.2, 0.1]),
                predicted_class="DISTRESS",
                confidence=0.45,
            )

        self.db.add_alert(sid, "DISTRESS", "HIGH", 0.8, 0.85, 0.7,
                          np.array([0.8, 0.1, 0.05, 0.05]), 3)
        self.db.add_alert(sid, "ANGRY", "LOW", 0.72, 0.75, 0.65,
                          np.array([0.1, 0.72, 0.1, 0.08]))

        stats = self.db.get_session_stats(sid)
        assert stats["total_segments"] == 5
        assert stats["total_alerts"] == 2
        assert stats["alerts_by_emotion"]["DISTRESS"] == 1
        assert stats["alerts_by_emotion"]["ANGRY"] == 1


# ============================================
# AlertManager tests
# ============================================
class TestAlertManager:

    def setup_method(self):
        from silentcare.core.alert_manager import AlertManager
        self.fired_alerts = []
        self.am = AlertManager(on_alert=lambda a: self.fired_alerts.append(a))

    def test_calm_never_triggers(self):
        for _ in range(10):
            result = self.am.process_segment(
                fused_probs=np.array([0.0, 0.0, 0.0, 1.0])
            )
            assert result is None
        assert len(self.fired_alerts) == 0

    def test_single_segment_not_enough(self):
        # One DISTRESS detection should not trigger (need 3 consecutive)
        result = self.am.process_segment(
            fused_probs=np.array([0.85, 0.05, 0.05, 0.05])
        )
        assert result is None

    def test_two_segments_not_enough(self):
        # Two consecutive DISTRESS still not enough (need 3)
        self.am.process_segment(
            fused_probs=np.array([0.85, 0.05, 0.05, 0.05])
        )
        result = self.am.process_segment(
            fused_probs=np.array([0.80, 0.10, 0.05, 0.05])
        )
        assert result is None

    def test_consecutive_triggers_alert(self):
        # Three consecutive DISTRESS above threshold should trigger
        self.am.process_segment(
            fused_probs=np.array([0.85, 0.05, 0.05, 0.05])
        )
        self.am.process_segment(
            fused_probs=np.array([0.80, 0.10, 0.05, 0.05])
        )
        result = self.am.process_segment(
            fused_probs=np.array([0.82, 0.08, 0.05, 0.05])
        )
        assert result is not None
        assert result["emotion"] == "DISTRESS"
        assert result["severity"] == "HIGH"
        assert result["consecutive_count"] == 3

    def test_interrupted_streak_no_alert(self):
        # DISTRESS x2, then CALM, then DISTRESS -> streak broken
        self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        self.am.process_segment(fused_probs=np.array([0.05, 0.05, 0.05, 0.85]))
        result = self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        assert result is None  # streak was reset

    def test_severity_escalation(self):
        # 4 consecutive -> alert at count=3 with HIGH severity
        for _ in range(4):
            self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))

        # Should have fired once at count=3 with HIGH (cooldown prevents 4th)
        assert len(self.fired_alerts) == 1
        assert self.fired_alerts[0]["severity"] == "HIGH"
        assert self.fired_alerts[0]["consecutive_count"] == 3

    def test_cooldown_prevents_repeat(self):
        # Trigger alert at 3rd segment, then immediately try again
        self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        assert len(self.fired_alerts) == 1

        # Continue DISTRESS - should be in cooldown
        result = self.am.process_segment(fused_probs=np.array([0.85, 0.05, 0.05, 0.05]))
        assert result is None  # cooldown active

    def test_below_threshold_no_alert(self):
        # DISTRESS at 0.40 (below 0.60 threshold)
        for _ in range(5):
            result = self.am.process_segment(
                fused_probs=np.array([0.40, 0.25, 0.20, 0.15])
            )
        assert result is None
        assert len(self.fired_alerts) == 0

    def test_distress_too_close_to_calm_no_alert(self):
        # DISTRESS at 0.61, CALM at 0.37 -> margin = 0.24 < 0.25 required
        # Above threshold (0.60) but too close to CALM -> no alert
        for _ in range(5):
            result = self.am.process_segment(
                fused_probs=np.array([0.61, 0.01, 0.01, 0.37])
            )
        assert result is None
        assert len(self.fired_alerts) == 0

    def test_different_class_alert(self):
        # ANGRY needs 3 consecutive above 0.50
        self.am.process_segment(fused_probs=np.array([0.05, 0.80, 0.10, 0.05]))
        self.am.process_segment(fused_probs=np.array([0.05, 0.75, 0.10, 0.10]))
        result = self.am.process_segment(fused_probs=np.array([0.05, 0.78, 0.10, 0.07]))
        assert result is not None
        assert result["emotion"] == "ANGRY"


# ============================================
# Fusion logic tests
# ============================================
class TestFusion:

    def setup_method(self):
        from silentcare.core.analysis_pipeline import AnalysisPipeline
        # Create a minimal pipeline just to test fusion
        # We pass None for capture_service since we won't start it
        self.pipeline = AnalysisPipeline.__new__(AnalysisPipeline)

    def test_both_modalities(self):
        audio = np.array([0.8, 0.1, 0.05, 0.05])
        video = np.array([0.7, 0.15, 0.1, 0.05])
        fused = self.pipeline._fuse_predictions(audio, video)

        assert fused.shape == (4,)
        assert abs(fused.sum() - 1.0) < 1e-6
        assert np.argmax(fused) == 0  # DISTRESS should be top

    def test_audio_only(self):
        audio = np.array([0.1, 0.1, 0.1, 0.7])
        fused = self.pipeline._fuse_predictions(audio, None)
        np.testing.assert_array_almost_equal(fused, audio)

    def test_video_only(self):
        video = np.array([0.1, 0.8, 0.05, 0.05])
        fused = self.pipeline._fuse_predictions(None, video)
        np.testing.assert_array_almost_equal(fused, video)

    def test_both_none_defaults_calm(self):
        fused = self.pipeline._fuse_predictions(None, None)
        assert np.argmax(fused) == 3  # CALM

    def test_agreement_boost(self):
        # When both agree on top class, confidence should be boosted
        audio = np.array([0.5, 0.2, 0.2, 0.1])
        video = np.array([0.6, 0.2, 0.1, 0.1])

        fused = self.pipeline._fuse_predictions(audio, video)
        # DISTRESS should be boosted by agreement
        assert fused[0] > 0.5  # should be higher than raw weighted average

    def test_disagreement_no_boost(self):
        audio = np.array([0.5, 0.2, 0.2, 0.1])  # top: DISTRESS
        video = np.array([0.1, 0.6, 0.2, 0.1])   # top: ANGRY

        fused = self.pipeline._fuse_predictions(audio, video)
        # No agreement boost
        assert fused.shape == (4,)
        assert abs(fused.sum() - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
