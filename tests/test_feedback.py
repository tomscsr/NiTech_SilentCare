"""
SilentCare - Feedback System Tests
====================================
Tests for:
  - Database feedback table and CRUD operations
  - CaptureService circular buffer
  - FeedbackService save logic
  - Feedback API endpoints
  - WAV file writing
"""

import sys
import os
import time
import tempfile
import shutil
import wave
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from silentcare.core.database import Database
from silentcare.core.capture_service import CaptureService
from silentcare.core.feedback_service import FeedbackService


# ============================================
# Database feedback tests
# ============================================
class TestDatabaseFeedback:
    """Tests for feedback table operations."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))
        self.session_id = self.db.start_session()
        self.segment_id = self.db.add_segment(
            session_id=self.session_id,
            audio_probs=np.array([0.1, 0.2, 0.3, 0.4]),
            video_probs=np.array([0.2, 0.1, 0.3, 0.4]),
            fused_probs=np.array([0.15, 0.15, 0.3, 0.4]),
            predicted_class="CALM",
            confidence=0.4,
        )
        self.alert_id = self.db.add_alert(
            session_id=self.session_id,
            emotion="DISTRESS",
            severity="HIGH",
            confidence=0.85,
            audio_confidence=0.7,
            video_confidence=0.9,
            fused_probs=np.array([0.85, 0.05, 0.05, 0.05]),
            consecutive_count=3,
            segment_id=self.segment_id,
        )

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_feedback(self):
        fb_id = self.db.add_feedback(
            session_id=self.session_id,
            segment_id=self.segment_id,
            alert_id=self.alert_id,
            report_type="FALSE_ALERT",
            predicted_class="DISTRESS",
            correct_class="CALM",
            notes="Test note",
        )
        assert fb_id is not None
        assert fb_id > 0

    def test_get_feedback(self):
        self.db.add_feedback(
            session_id=self.session_id,
            segment_id=self.segment_id,
            alert_id=self.alert_id,
            report_type="FALSE_ALERT",
            predicted_class="DISTRESS",
            correct_class="CALM",
        )
        feedbacks = self.db.get_feedback(limit=10)
        assert len(feedbacks) == 1
        assert feedbacks[0]["report_type"] == "FALSE_ALERT"
        assert feedbacks[0]["predicted_class"] == "DISTRESS"
        assert feedbacks[0]["correct_class"] == "CALM"

    def test_get_feedback_filtered(self):
        self.db.add_feedback(
            session_id=self.session_id,
            segment_id=self.segment_id,
            alert_id=None,
            report_type="MISSED_DETECTION",
            predicted_class="CALM",
            correct_class="DISTRESS",
        )
        feedbacks = self.db.get_feedback(used_for_training=False)
        assert len(feedbacks) == 1
        feedbacks = self.db.get_feedback(used_for_training=True)
        assert len(feedbacks) == 0

    def test_feedback_stats(self):
        self.db.add_feedback(
            session_id=self.session_id, segment_id=self.segment_id,
            alert_id=self.alert_id, report_type="FALSE_ALERT",
            predicted_class="DISTRESS", correct_class="CALM",
        )
        self.db.add_feedback(
            session_id=self.session_id, segment_id=self.segment_id,
            alert_id=None, report_type="MISSED_DETECTION",
            predicted_class="CALM", correct_class="DISTRESS",
        )
        stats = self.db.get_feedback_stats()
        assert stats["total"] == 2
        assert stats["by_type"]["FALSE_ALERT"] == 1
        assert stats["by_type"]["MISSED_DETECTION"] == 1
        assert stats["ready_for_training"] == 2
        assert len(stats["confusions"]) == 2

    def test_mark_feedback_used(self):
        fb_id = self.db.add_feedback(
            session_id=self.session_id, segment_id=self.segment_id,
            alert_id=None, report_type="MISSED_DETECTION",
            predicted_class="CALM", correct_class="DISTRESS",
        )
        self.db.mark_feedback_used([fb_id])
        feedbacks = self.db.get_feedback(used_for_training=True)
        assert len(feedbacks) == 1
        assert feedbacks[0]["used_for_training"] == 1

    def test_get_segment_by_id(self):
        seg = self.db.get_segment_by_id(self.segment_id)
        assert seg is not None
        assert seg["predicted_class"] == "CALM"

    def test_get_alert_by_id(self):
        alert = self.db.get_alert_by_id(self.alert_id)
        assert alert is not None
        assert alert["emotion"] == "DISTRESS"
        assert alert["segment_id"] == self.segment_id

    def test_get_segment_near_timestamp(self):
        seg = self.db.get_segment_by_id(self.segment_id)
        found = self.db.get_segment_near_timestamp(seg["timestamp"])
        assert found is not None
        assert found["id"] == self.segment_id

    def test_update_feedback_files(self):
        fb_id = self.db.add_feedback(
            session_id=self.session_id, segment_id=self.segment_id,
            alert_id=None, report_type="MISSED_DETECTION",
            predicted_class="CALM", correct_class="DISTRESS",
        )
        self.db.update_feedback_files(
            fb_id, audio_saved=True, video_saved=True,
            audio_path="data/feedback/audio/1.wav",
            video_path="data/feedback/video/1",
        )
        feedbacks = self.db.get_feedback()
        assert feedbacks[0]["audio_saved"] == 1
        assert feedbacks[0]["video_saved"] == 1
        assert feedbacks[0]["audio_path"] == "data/feedback/audio/1.wav"


# ============================================
# CaptureService buffer tests
# ============================================
class TestCaptureBuffer:
    """Tests for the circular segment buffer."""

    def test_buffer_segment_audio(self):
        cs = CaptureService(enable_audio=False, enable_video=False)
        segment = {
            "timestamp": time.time(),
            "audio": np.random.randn(22050).astype(np.float32),
            "audio_sr": 22050,
            "video_frames": [],
            "has_audio": True,
            "has_video": False,
        }
        cs._buffer_segment(segment)
        assert len(cs._segment_buffer) == 1
        entry = cs._segment_buffer[0]
        assert entry["audio"] is not None
        assert len(entry["video_jpegs"]) == 0

    def test_buffer_segment_with_video(self):
        cs = CaptureService(enable_audio=False, enable_video=False)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        segment = {
            "timestamp": time.time(),
            "audio": None,
            "audio_sr": None,
            "video_frames": [frame, frame, frame],
            "has_audio": False,
            "has_video": True,
        }
        cs._buffer_segment(segment)
        entry = cs._segment_buffer[0]
        assert len(entry["video_jpegs"]) == 3
        # Each jpeg should be bytes
        assert isinstance(entry["video_jpegs"][0], bytes)

    def test_get_buffered_segment_found(self):
        cs = CaptureService(enable_audio=False, enable_video=False)
        ts = time.time()
        segment = {
            "timestamp": ts,
            "audio": np.random.randn(11025).astype(np.float32),
            "audio_sr": 22050,
            "video_frames": [],
            "has_audio": True,
            "has_video": False,
        }
        cs._buffer_segment(segment)
        found = cs.get_buffered_segment(ts)
        assert found is not None
        assert found["audio"] is not None
        assert abs(found["timestamp"] - ts) < 0.01

    def test_get_buffered_segment_not_found(self):
        cs = CaptureService(enable_audio=False, enable_video=False)
        ts = time.time()
        cs._buffer_segment({
            "timestamp": ts,
            "audio": None, "audio_sr": None,
            "video_frames": [], "has_audio": False, "has_video": False,
        })
        not_found = cs.get_buffered_segment(ts + 100)
        assert not_found is None

    def test_buffer_max_size(self):
        cs = CaptureService(enable_audio=False, enable_video=False)
        for i in range(35):
            cs._buffer_segment({
                "timestamp": time.time() + i,
                "audio": None, "audio_sr": None,
                "video_frames": [], "has_audio": False, "has_video": False,
            })
        assert len(cs._segment_buffer) == 30

    def test_buffer_returns_copy(self):
        """Returned data should be independent from the buffer."""
        cs = CaptureService(enable_audio=False, enable_video=False)
        ts = time.time()
        audio = np.ones(100, dtype=np.float32)
        cs._buffer_segment({
            "timestamp": ts,
            "audio": audio, "audio_sr": 22050,
            "video_frames": [], "has_audio": True, "has_video": False,
        })
        result = cs.get_buffered_segment(ts)
        result["audio"][0] = 999.0
        # Buffer should not be modified
        original = cs._segment_buffer[0]
        assert original["audio"][0] != 999.0


# ============================================
# FeedbackService tests
# ============================================
class TestFeedbackService:
    """Tests for FeedbackService save logic."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))
        self.session_id = self.db.start_session()

        self.capture = CaptureService(enable_audio=False, enable_video=False)

        self.segment_id = self.db.add_segment(
            session_id=self.session_id,
            audio_probs=np.array([0.8, 0.1, 0.05, 0.05]),
            video_probs=np.array([0.7, 0.1, 0.1, 0.1]),
            fused_probs=np.array([0.75, 0.1, 0.075, 0.075]),
            predicted_class="DISTRESS",
            confidence=0.75,
        )

        # Get segment timestamp and buffer a matching segment
        seg = self.db.get_segment_by_id(self.segment_id)
        from datetime import datetime
        dt = datetime.fromisoformat(seg["timestamp"])
        seg_unix_ts = dt.timestamp()

        self.capture._buffer_segment({
            "timestamp": seg_unix_ts,
            "audio": np.random.randn(22050).astype(np.float32) * 0.1,
            "audio_sr": 22050,
            "video_frames": [
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                for _ in range(3)
            ],
            "has_audio": True,
            "has_video": True,
        })

        self.alert_id = self.db.add_alert(
            session_id=self.session_id,
            emotion="DISTRESS",
            severity="HIGH",
            confidence=0.75,
            audio_confidence=0.8,
            video_confidence=0.7,
            fused_probs=np.array([0.75, 0.1, 0.075, 0.075]),
            segment_id=self.segment_id,
        )

        self.feedback_dir = os.path.join(self.tmpdir, "feedback")
        self.service = FeedbackService(
            db=self.db,
            capture_service=self.capture,
            data_dir=self.feedback_dir,
        )

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_report_false_alert(self):
        fb_id = self.service.report_false_alert(
            alert_id=self.alert_id,
            correct_class="CALM",
            notes="Not distressed",
        )
        assert fb_id > 0
        feedbacks = self.db.get_feedback()
        assert len(feedbacks) == 1
        assert feedbacks[0]["report_type"] == "FALSE_ALERT"
        assert feedbacks[0]["predicted_class"] == "DISTRESS"
        assert feedbacks[0]["correct_class"] == "CALM"

    def test_report_missed_detection(self):
        fb_id = self.service.report_missed_detection(
            segment_id=self.segment_id,
            correct_class="ANGRY",
        )
        assert fb_id > 0
        feedbacks = self.db.get_feedback()
        assert feedbacks[0]["report_type"] == "MISSED_DETECTION"

    def test_report_wrong_classification(self):
        fb_id = self.service.report_wrong_classification(
            alert_id=self.alert_id,
            correct_class="ANGRY",
        )
        assert fb_id > 0
        feedbacks = self.db.get_feedback()
        assert feedbacks[0]["report_type"] == "WRONG_CLASSIFICATION"

    def test_invalid_class_raises(self):
        with pytest.raises(ValueError, match="Invalid class"):
            self.service.report_false_alert(
                alert_id=self.alert_id,
                correct_class="INVALID",
            )

    def test_missing_alert_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.service.report_false_alert(
                alert_id=9999,
                correct_class="CALM",
            )

    def test_missing_segment_raises(self):
        with pytest.raises(ValueError, match="not found"):
            self.service.report_missed_detection(
                segment_id=9999,
                correct_class="DISTRESS",
            )

    def test_audio_saved_to_disk(self):
        fb_id = self.service.report_false_alert(
            alert_id=self.alert_id,
            correct_class="CALM",
        )
        audio_path = os.path.join(self.feedback_dir, "audio", f"{fb_id}.wav")
        assert os.path.exists(audio_path)
        with open(audio_path, "rb") as f:
            assert f.read(4) == b"RIFF"

    def test_video_saved_to_disk(self):
        fb_id = self.service.report_false_alert(
            alert_id=self.alert_id,
            correct_class="CALM",
        )
        video_dir = os.path.join(self.feedback_dir, "video", str(fb_id))
        assert os.path.isdir(video_dir)
        frames = [f for f in os.listdir(video_dir) if f.endswith(".jpg")]
        assert len(frames) == 3

    def test_wrong_classification_needs_reference(self):
        with pytest.raises(ValueError, match="Either alert_id or segment_id"):
            self.service.report_wrong_classification(
                correct_class="CALM",
            )


# ============================================
# WAV writing test
# ============================================
class TestWavWriting:

    def test_write_valid_wav(self):
        tmpdir = tempfile.mkdtemp()
        wav_path = os.path.join(tmpdir, "test.wav")

        audio = np.random.randn(22050).astype(np.float32) * 0.5
        FeedbackService._write_wav(wav_path, audio, 22050)

        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 22050

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_wav_clipping(self):
        """Audio values outside [-1, 1] should be clipped."""
        tmpdir = tempfile.mkdtemp()
        wav_path = os.path.join(tmpdir, "test_clip.wav")

        audio = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        FeedbackService._write_wav(wav_path, audio, 22050)

        with wave.open(wav_path, "rb") as wf:
            frames = wf.readframes(4)
            samples = np.frombuffer(frames, dtype=np.int16)
            # Clipped values should be at int16 limits
            assert samples[0] == 32767
            assert samples[1] == -32768

        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================
# API endpoint tests
# ============================================
class TestFeedbackAPI:
    """Tests for feedback-related Flask API endpoints."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmpdir, "test.db"))
        self.session_id = self.db.start_session()

        self.segment_id = self.db.add_segment(
            session_id=self.session_id,
            audio_probs=np.array([0.7, 0.1, 0.1, 0.1]),
            video_probs=np.array([0.6, 0.2, 0.1, 0.1]),
            fused_probs=np.array([0.65, 0.15, 0.1, 0.1]),
            predicted_class="DISTRESS",
            confidence=0.65,
        )

        self.alert_id = self.db.add_alert(
            session_id=self.session_id,
            emotion="DISTRESS",
            severity="HIGH",
            confidence=0.65,
            audio_confidence=0.7,
            video_confidence=0.6,
            fused_probs=np.array([0.65, 0.15, 0.1, 0.1]),
            segment_id=self.segment_id,
        )

        self.capture = CaptureService(enable_audio=False, enable_video=False)

        _session_id = self.session_id

        class MockPipeline:
            is_running = True
            session_id = _session_id
            _models_loaded = True
            latest_result = None
            on_segment = None
            on_alert = None

        from silentcare.app.routes import create_app
        app = create_app(self.capture, MockPipeline(), self.db)
        app.config["TESTING"] = True
        self.client = app.test_client()

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_false_alert_endpoint(self):
        resp = self.client.post(
            "/api/feedback/false_alert",
            json={"alert_id": self.alert_id, "correct_class": "CALM"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "feedback_id" in data

    def test_missed_detection_endpoint(self):
        resp = self.client.post(
            "/api/feedback/missed_detection",
            json={"segment_id": self.segment_id, "correct_class": "ANGRY"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_wrong_classification_endpoint(self):
        resp = self.client.post(
            "/api/feedback/wrong_classification",
            json={"alert_id": self.alert_id, "correct_class": "ANGRY"},
        )
        assert resp.status_code == 200

    def test_feedback_list_endpoint(self):
        self.client.post(
            "/api/feedback/false_alert",
            json={"alert_id": self.alert_id, "correct_class": "CALM"},
        )
        resp = self.client.get("/api/feedback")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1

    def test_feedback_stats_endpoint(self):
        self.client.post(
            "/api/feedback/false_alert",
            json={"alert_id": self.alert_id, "correct_class": "CALM"},
        )
        resp = self.client.get("/api/feedback/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 1
        assert data["by_type"]["FALSE_ALERT"] == 1

    def test_feedback_export_csv(self):
        self.client.post(
            "/api/feedback/false_alert",
            json={"alert_id": self.alert_id, "correct_class": "CALM"},
        )
        resp = self.client.get("/api/feedback/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.content_type
        csv_text = resp.data.decode()
        assert "FALSE_ALERT" in csv_text

    def test_validation_error_invalid_class(self):
        resp = self.client.post(
            "/api/feedback/false_alert",
            json={"alert_id": self.alert_id, "correct_class": "INVALID"},
        )
        assert resp.status_code == 400

    def test_validation_error_missing_params(self):
        resp = self.client.post(
            "/api/feedback/false_alert",
            json={"correct_class": "CALM"},
        )
        assert resp.status_code == 400
