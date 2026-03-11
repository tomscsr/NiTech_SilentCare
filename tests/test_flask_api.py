"""
SilentCare - Step 6: Flask API Integration Tests
===================================================
Tests all API endpoints using Flask test client.
"""

import sys
import os
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from silentcare.core.database import Database
from silentcare.core.capture_service import CaptureService
from silentcare.core.analysis_pipeline import AnalysisPipeline
from silentcare.app.routes import create_app


class MockCaptureForFlask:
    """Minimal mock for Flask tests."""
    def __init__(self):
        self.enable_audio = True
        self.enable_video = False
        self.audio_device = None
        self._running = False
        import queue
        self.segment_queue = queue.Queue()

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def get_current_frame(self):
        return None  # No camera in tests

    def get_audio_buffer_copy(self):
        return None, None  # No mic in tests

    def set_audio_device(self, device_index):
        self.audio_device = device_index

    @staticmethod
    def list_audio_devices():
        return [{"id": 0, "name": "Mock Microphone"}]


class TestFlaskAPI:
    """Test all Flask API endpoints."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

        self.db = Database(self.tmp.name)
        self.capture = MockCaptureForFlask()

        self.pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        self.pipeline.capture = self.capture
        self.pipeline.db = self.db
        self.pipeline._audio_model = None
        self.pipeline._video_model = None
        self.pipeline._models_loaded = False
        self.pipeline._running = False
        self.pipeline._session_id = None
        self.pipeline._latest_result = None
        self.pipeline._result_lock = __import__("threading").Lock()
        self.pipeline.on_segment = None
        self.pipeline.on_alert = None
        self.pipeline.alert_manager = __import__(
            "silentcare.core.alert_manager", fromlist=["AlertManager"]
        ).AlertManager()

        self.app = create_app(self.capture, self.pipeline, self.db)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    # =============================================
    # GET /
    # =============================================
    def test_dashboard_page(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"SilentCare" in resp.data

    # =============================================
    # GET /api/status
    # =============================================
    def test_status_inactive(self):
        resp = self.client.get("/api/status")
        data = json.loads(resp.data)
        assert resp.status_code == 200
        assert data["running"] is False
        assert data["capture_running"] is False

    def test_status_active(self):
        self.pipeline._running = True
        self.pipeline._session_id = 1
        self.pipeline._models_loaded = True
        self.capture._running = True

        resp = self.client.get("/api/status")
        data = json.loads(resp.data)
        assert data["running"] is True
        assert data["capture_running"] is True
        assert data["models_loaded"] is True

    # =============================================
    # GET /api/segments
    # =============================================
    def test_segments_empty(self):
        resp = self.client.get("/api/segments")
        data = json.loads(resp.data)
        assert resp.status_code == 200
        assert data == []

    def test_segments_with_data(self):
        sid = self.db.start_session()
        self.pipeline._running = True
        self.pipeline._session_id = sid

        self.db.add_segment(
            session_id=sid,
            audio_probs=np.array([0.8, 0.1, 0.05, 0.05]),
            video_probs=np.zeros(4),
            fused_probs=np.array([0.8, 0.1, 0.05, 0.05]),
            predicted_class="DISTRESS",
            confidence=0.8,
        )

        resp = self.client.get("/api/segments?limit=10")
        data = json.loads(resp.data)
        assert len(data) == 1
        assert data[0]["predicted_class"] == "DISTRESS"

    # =============================================
    # GET /api/alerts
    # =============================================
    def test_alerts_empty(self):
        resp = self.client.get("/api/alerts")
        data = json.loads(resp.data)
        assert resp.status_code == 200
        assert data == []

    def test_alerts_with_data(self):
        sid = self.db.start_session()
        self.pipeline._running = True
        self.pipeline._session_id = sid

        self.db.add_alert(
            session_id=sid,
            emotion="DISTRESS",
            severity="HIGH",
            confidence=0.9,
            audio_confidence=0.92,
            video_confidence=None,
            fused_probs=np.array([0.9, 0.05, 0.03, 0.02]),
            consecutive_count=3,
        )

        resp = self.client.get("/api/alerts?limit=10")
        data = json.loads(resp.data)
        assert len(data) == 1
        assert data[0]["emotion"] == "DISTRESS"
        assert data[0]["severity"] == "HIGH"
        assert data[0]["consecutive_count"] == 3

    # =============================================
    # POST /api/alerts/<id>/ack
    # =============================================
    def test_acknowledge_alert(self):
        sid = self.db.start_session()
        alert_id = self.db.add_alert(
            session_id=sid,
            emotion="ANGRY",
            severity="MEDIUM",
            confidence=0.75,
            audio_confidence=0.8,
            video_confidence=None,
            fused_probs=np.array([0.1, 0.75, 0.1, 0.05]),
        )

        resp = self.client.post(f"/api/alerts/{alert_id}/ack")
        data = json.loads(resp.data)
        assert resp.status_code == 200
        assert data["status"] == "ok"

        unack = self.db.get_unacknowledged_alerts(session_id=sid)
        assert len(unack) == 0

    # =============================================
    # GET /api/stats
    # =============================================
    def test_stats_no_session(self):
        resp = self.client.get("/api/stats")
        data = json.loads(resp.data)
        assert data["total_segments"] == 0
        assert data["total_alerts"] == 0

    def test_stats_with_session(self):
        sid = self.db.start_session()
        self.pipeline._running = True
        self.pipeline._session_id = sid

        for _ in range(3):
            self.db.add_segment(
                session_id=sid,
                audio_probs=np.zeros(4),
                video_probs=np.zeros(4),
                fused_probs=np.zeros(4),
                predicted_class="CALM",
                confidence=0.5,
            )

        self.db.add_alert(sid, "DISTRESS", "HIGH", 0.85, 0.9, None,
                          np.array([0.85, 0.05, 0.05, 0.05]))

        resp = self.client.get("/api/stats")
        data = json.loads(resp.data)
        assert data["total_segments"] == 3
        assert data["total_alerts"] == 1
        assert data["alerts_by_emotion"]["DISTRESS"] == 1

    # =============================================
    # POST /api/start and /api/stop
    # =============================================
    def test_start_stop_cycle(self):
        # Manually make pipeline startable (mock load_models and thread)
        self.pipeline._models_loaded = True
        self.pipeline.load_models = lambda: None

        # Override start/stop to avoid real threads
        original_start = self.pipeline.start
        original_stop = self.pipeline.stop

        def mock_start():
            self.pipeline._running = True
            self.pipeline._session_id = self.db.start_session()

        def mock_stop():
            self.pipeline._running = False
            if self.pipeline._session_id:
                self.db.stop_session(self.pipeline._session_id)

        self.pipeline.start = mock_start
        self.pipeline.stop = mock_stop

        # Start
        resp = self.client.post("/api/start")
        data = json.loads(resp.data)
        assert data["status"] == "started"
        assert data["session_id"] is not None

        # Start again -> already running
        resp = self.client.post("/api/start")
        data = json.loads(resp.data)
        assert data["status"] == "already_running"

        # Stop
        resp = self.client.post("/api/stop")
        data = json.loads(resp.data)
        assert data["status"] == "stopped"

        # Stop again -> not running
        resp = self.client.post("/api/stop")
        data = json.loads(resp.data)
        assert data["status"] == "not_running"

    # =============================================
    # Content type checks
    # =============================================
    def test_api_returns_json(self):
        for endpoint in ["/api/status", "/api/segments", "/api/alerts", "/api/stats"]:
            resp = self.client.get(endpoint)
            assert resp.content_type == "application/json"

    def test_dashboard_returns_html(self):
        resp = self.client.get("/")
        assert "text/html" in resp.content_type

    # =============================================
    # GET /api/audio_data
    # =============================================
    def test_audio_data_empty(self):
        """Audio data returns empty waveform when capture is not running."""
        resp = self.client.get("/api/audio_data")
        data = json.loads(resp.data)
        assert resp.status_code == 200
        assert data["waveform"] == []

    # =============================================
    # GET /api/video_feed
    # =============================================
    def test_video_feed_returns_mjpeg(self):
        """Video feed returns multipart MJPEG content type."""
        resp = self.client.get("/api/video_feed")
        assert resp.status_code == 200
        assert "multipart/x-mixed-replace" in resp.content_type

    # =============================================
    # Dashboard contains live feed sections
    # =============================================
    def test_dashboard_has_live_feeds(self):
        resp = self.client.get("/")
        assert b"video-feed" in resp.data
        assert b"waveform-canvas" in resp.data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
