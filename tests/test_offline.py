"""
SilentCare - Offline Analysis Tests
======================================
Tests:
  - OfflineExtractor (video info, segment extraction)
  - OfflinePipeline (analysis with mock models)
  - Offline API endpoints (upload, info, analyze, status, results, control)
  - DB migration (is_offline, video_filename)
"""

import sys
import os
import json
import time
import tempfile
import threading
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from silentcare.core.database import Database
from silentcare.core.offline_extractor import OfflineExtractor
from silentcare.core.offline_pipeline import OfflinePipeline
from silentcare.core.alert_manager import AlertManager


# =============================================
# Test video creation helper
# =============================================
def create_test_video(path, duration_s=6, fps=15, width=320, height=240, with_audio=True):
    """Create a minimal test video file using OpenCV + ffmpeg for audio."""
    # Create video with OpenCV
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    total_frames = int(duration_s * fps)
    for i in range(total_frames):
        # Create a simple colored frame (changes over time)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_val = int(255 * i / total_frames)
        frame[:, :, 1] = color_val  # Green channel varies
        frame[:, :, 2] = 128        # Red constant
        writer.write(frame)

    writer.release()

    if with_audio:
        # Add a silent audio track via ffmpeg
        video_tmp = str(path) + ".tmp.mp4"
        os.rename(str(path), video_tmp)
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_tmp,
                    "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono",
                    "-t", str(duration_s),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    str(path),
                ],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                # Fallback: just use the video without audio
                os.rename(video_tmp, str(path))
        except Exception:
            if os.path.exists(video_tmp):
                os.rename(video_tmp, str(path))
        finally:
            if os.path.exists(video_tmp):
                try:
                    os.unlink(video_tmp)
                except OSError:
                    pass


# =============================================
# Mock classes
# =============================================
class MockAudioModel:
    """Mock audio model that returns predictable results."""
    def predict(self, audio, sr=None):
        # Return CALM for silence, DISTRESS for loud audio
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 0.1:
            probs = np.array([0.7, 0.1, 0.1, 0.1])
        else:
            probs = np.array([0.05, 0.05, 0.1, 0.8])
        idx = int(np.argmax(probs))
        classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
        return {
            "probabilities": probs,
            "predicted_class": classes[idx],
            "confidence": float(probs[idx]),
        }


class MockVideoModel:
    """Mock video model that returns predictable results."""
    def predict(self, frames):
        if not frames:
            return None
        probs = np.array([0.1, 0.1, 0.2, 0.6])
        idx = int(np.argmax(probs))
        classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
        return {
            "probabilities": probs,
            "predicted_class": classes[idx],
            "confidence": float(probs[idx]),
        }


class MockPipeline:
    """Mock AnalysisPipeline for reuse by OfflinePipeline."""
    def __init__(self):
        self._audio_model = MockAudioModel()
        self._video_model = MockVideoModel()
        self._models_loaded = True


class MockCaptureForFlask:
    """Minimal mock for Flask tests."""
    def __init__(self):
        self.enable_audio = True
        self.enable_video = False
        self.audio_device = None
        self._running = False
        import queue
        self.segment_queue = queue.Queue()
        self._segment_buffer = []

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    @property
    def is_running(self):
        return self._running

    def get_current_frame(self):
        return None

    def get_audio_buffer_copy(self):
        return None, None

    def set_audio_device(self, device_index):
        self.audio_device = device_index

    def get_buffered_segment(self, timestamp, tolerance=10.0):
        return None

    @staticmethod
    def list_audio_devices():
        return [{"id": 0, "name": "Mock Microphone"}]


# =============================================
# DB Migration Tests
# =============================================
class TestDBOfflineMigration:
    """Test that offline columns exist in sessions table."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(self.tmp.name)

    def teardown_method(self):
        self.db.close()
        os.unlink(self.tmp.name)

    def test_start_offline_session(self):
        sid = self.db.start_offline_session("test_video.mp4")
        assert sid is not None
        assert sid > 0

    def test_offline_session_has_columns(self):
        sid = self.db.start_offline_session("my_video.mp4")

        row = self.db._conn.execute(
            "SELECT is_offline, video_filename FROM sessions WHERE id = ?",
            (sid,)
        ).fetchone()

        assert row["is_offline"] == 1
        assert row["video_filename"] == "my_video.mp4"

    def test_regular_session_defaults(self):
        sid = self.db.start_session()

        row = self.db._conn.execute(
            "SELECT is_offline, video_filename FROM sessions WHERE id = ?",
            (sid,)
        ).fetchone()

        assert row["is_offline"] == 0
        assert row["video_filename"] is None

    def test_stop_offline_session(self):
        sid = self.db.start_offline_session("video.mp4")
        self.db.stop_session(sid)

        row = self.db._conn.execute(
            "SELECT status, stopped_at FROM sessions WHERE id = ?",
            (sid,)
        ).fetchone()

        assert row["status"] == "stopped"
        assert row["stopped_at"] is not None


# =============================================
# OfflineExtractor Tests
# =============================================
class TestOfflineExtractor:
    """Test video extraction."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmpdir, "test.mp4")
        create_test_video(self.video_path, duration_s=8, fps=15)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_info(self):
        ext = OfflineExtractor(self.video_path)
        info = ext.get_info()

        assert info["duration"] > 0
        assert info["fps"] > 0
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["total_segments"] > 0
        assert isinstance(info["has_audio"], bool)

    def test_get_info_cached(self):
        ext = OfflineExtractor(self.video_path)
        info1 = ext.get_info()
        info2 = ext.get_info()
        assert info1 is info2  # Same object (cached)

    def test_extract_segment(self):
        ext = OfflineExtractor(self.video_path)
        segment = ext.extract_segment(0.0, 5.0)

        assert segment["timestamp"] == 0.0
        assert isinstance(segment["video_frames"], list)
        assert len(segment["video_frames"]) > 0
        assert segment["has_video"] is True
        # Each frame should be a numpy array
        assert isinstance(segment["video_frames"][0], np.ndarray)

    def test_extract_segment_frames_are_bgr(self):
        ext = OfflineExtractor(self.video_path)
        segment = ext.extract_segment(0.0, 5.0)

        frame = segment["video_frames"][0]
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # BGR channels

    def test_iter_segments(self):
        ext = OfflineExtractor(self.video_path)
        segments = list(ext.iter_segments())

        assert len(segments) > 0
        for idx, total, seg in segments:
            assert isinstance(idx, int)
            assert isinstance(total, int)
            assert isinstance(seg, dict)
            assert "timestamp" in seg
            assert "audio" in seg
            assert "video_frames" in seg

    def test_iter_segments_count_matches_info(self):
        ext = OfflineExtractor(self.video_path)
        info = ext.get_info()
        segments = list(ext.iter_segments())

        assert len(segments) == info["total_segments"]

    def test_invalid_video_path(self):
        ext = OfflineExtractor("/nonexistent/video.mp4")
        with pytest.raises(ValueError, match="Cannot open video"):
            ext.get_info()


# =============================================
# OfflinePipeline Tests
# =============================================
class TestOfflinePipeline:
    """Test offline analysis pipeline."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmpdir, "test.mp4")
        create_test_video(self.video_path, duration_s=8, fps=15, with_audio=False)

        self.db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_tmp.close()
        self.db = Database(self.db_tmp.name)
        self.mock_pipeline = MockPipeline()

    def teardown_method(self):
        self.db.close()
        os.unlink(self.db_tmp.name)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_analyze_complete(self):
        received = []
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
            on_segment=lambda r: received.append(r),
        )

        results = pipe.analyze_complete(self.video_path)

        assert results["session_id"] is not None
        assert len(results["segments"]) > 0
        assert results["stats"]["total_segments"] > 0
        assert pipe.status == OfflinePipeline.STATUS_COMPLETE
        assert len(received) == len(results["segments"])

    def test_analyze_complete_segments_have_expected_keys(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        results = pipe.analyze_complete(self.video_path)
        seg = results["segments"][0]

        assert "segment_id" in seg
        assert "predicted_class" in seg
        assert "confidence" in seg
        assert "fused_probs" in seg
        assert "audio_probs" in seg
        assert "video_probs" in seg
        assert "streak" in seg

    def test_analyze_complete_stores_in_db(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        results = pipe.analyze_complete(self.video_path)
        sid = results["session_id"]

        # Check DB has segments
        db_segments = self.db.get_recent_segments(sid, limit=100)
        assert len(db_segments) == len(results["segments"])

    def test_progress_tracking(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        results = pipe.analyze_complete(self.video_path)

        assert pipe.progress["percent"] == 100.0
        assert pipe.progress["current_segment"] == pipe.progress["total_segments"]

    def test_analyze_realtime(self):
        received = []
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
            on_segment=lambda r: received.append(r),
        )

        pipe.analyze_realtime(self.video_path)

        # Wait for completion (with timeout)
        timeout = 60
        start = time.time()
        while pipe.status not in ("complete", "error") and time.time() - start < timeout:
            time.sleep(0.5)

        assert pipe.status == OfflinePipeline.STATUS_COMPLETE
        assert len(received) > 0

    def test_stop_during_analysis(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        pipe.analyze_realtime(self.video_path)
        time.sleep(0.5)  # Let it start
        pipe.stop()

        assert pipe.status == OfflinePipeline.STATUS_IDLE

    def test_pause_resume(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        pipe.analyze_realtime(self.video_path)
        time.sleep(0.3)

        pipe.pause()
        assert pipe.status == OfflinePipeline.STATUS_PAUSED

        pipe.resume()
        assert pipe.status == OfflinePipeline.STATUS_RUNNING

        pipe.stop()

    def test_get_results(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        results = pipe.analyze_complete(self.video_path)
        get_results = pipe.get_results()

        assert get_results["session_id"] == results["session_id"]
        assert get_results["status"] == "complete"
        assert len(get_results["segments"]) > 0

    def test_offline_session_marked(self):
        pipe = OfflinePipeline(
            pipeline=self.mock_pipeline,
            db=self.db,
        )

        results = pipe.analyze_complete(self.video_path)
        sid = results["session_id"]

        row = self.db._conn.execute(
            "SELECT is_offline, video_filename FROM sessions WHERE id = ?",
            (sid,)
        ).fetchone()

        assert row["is_offline"] == 1
        assert row["video_filename"] is not None


# =============================================
# Flask Offline API Tests
# =============================================
class TestOfflineAPI:
    """Test offline API endpoints."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmpdir, "test.mp4")
        create_test_video(self.video_path, duration_s=6, fps=15, with_audio=False)

        self.db_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_tmp.close()
        self.db = Database(self.db_tmp.name)
        self.capture = MockCaptureForFlask()

        # Create a mock pipeline
        from silentcare.core.analysis_pipeline import AnalysisPipeline
        self.pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        self.pipeline.capture = self.capture
        self.pipeline.db = self.db
        self.pipeline._audio_model = MockAudioModel()
        self.pipeline._video_model = MockVideoModel()
        self.pipeline._models_loaded = True
        self.pipeline._running = False
        self.pipeline._session_id = None
        self.pipeline._latest_result = None
        self.pipeline._result_lock = threading.Lock()
        self.pipeline.on_segment = None
        self.pipeline.on_alert = None
        self.pipeline.alert_manager = AlertManager()

        from silentcare.app.routes import create_app
        self.app = create_app(self.capture, self.pipeline, self.db)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def teardown_method(self):
        self.db.close()
        os.unlink(self.db_tmp.name)
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_offline_page(self):
        resp = self.client.get("/offline")
        assert resp.status_code == 200
        assert b"SilentCare" in resp.data
        assert b"offline" in resp.data.lower()

    def test_upload_no_file(self):
        resp = self.client.post("/api/offline/upload")
        assert resp.status_code == 400

    def test_upload_success(self):
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"
        assert "job_id" in data
        assert "info" in data
        assert data["info"]["total_segments"] > 0

    def test_upload_invalid_extension(self):
        # Create a text file with wrong extension
        txt_path = os.path.join(self.tmpdir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("not a video")

        with open(txt_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.txt")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 400
        assert b"Unsupported format" in resp.data

    def test_info_endpoint(self):
        # Upload first
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        # Get info
        resp = self.client.get(f"/api/offline/info/{job_id}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["job_id"] == job_id
        assert "info" in data

    def test_info_not_found(self):
        resp = self.client.get("/api/offline/info/nonexistent")
        assert resp.status_code == 404

    def test_analyze_complete(self):
        # Upload
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        # Start analysis in complete mode
        resp = self.client.post(
            f"/api/offline/analyze/{job_id}",
            data=json.dumps({"mode": "complete"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "started"

        # Wait for completion
        for _ in range(30):
            time.sleep(0.5)
            resp = self.client.get(f"/api/offline/status/{job_id}")
            status_data = json.loads(resp.data)
            if status_data["status"] in ("complete", "error"):
                break

        assert status_data["status"] == "complete"

    def test_results_endpoint(self):
        # Upload + analyze
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        self.client.post(
            f"/api/offline/analyze/{job_id}",
            data=json.dumps({"mode": "complete"}),
            content_type="application/json",
        )

        # Wait for completion
        for _ in range(30):
            time.sleep(0.5)
            resp = self.client.get(f"/api/offline/status/{job_id}")
            if json.loads(resp.data)["status"] == "complete":
                break

        # Get results
        resp = self.client.get(f"/api/offline/results/{job_id}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "complete"
        assert len(data["segments"]) > 0

    def test_status_idle(self):
        # Upload but don't analyze
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        resp = self.client.get(f"/api/offline/status/{job_id}")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "idle"

    def test_control_not_started(self):
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        resp = self.client.post(
            f"/api/offline/control/{job_id}",
            data=json.dumps({"action": "pause"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_control_invalid_action(self):
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        # Start first
        self.client.post(
            f"/api/offline/analyze/{job_id}",
            data=json.dumps({"mode": "realtime"}),
            content_type="application/json",
        )
        time.sleep(0.3)

        # Invalid action
        resp = self.client.post(
            f"/api/offline/control/{job_id}",
            data=json.dumps({"action": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

        # Cleanup
        self.client.post(
            f"/api/offline/control/{job_id}",
            data=json.dumps({"action": "stop"}),
            content_type="application/json",
        )

    def test_analyze_invalid_mode(self):
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "test.mp4")},
                content_type="multipart/form-data",
            )
        job_id = json.loads(resp.data)["job_id"]

        resp = self.client.post(
            f"/api/offline/analyze/{job_id}",
            data=json.dumps({"mode": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_upload_cleans_old_files(self):
        """New upload should clean up old files in the upload directory."""
        # First upload
        with open(self.video_path, "rb") as f:
            resp1 = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "first.mp4")},
                content_type="multipart/form-data",
            )
        assert resp1.status_code == 200

        # Second upload (should clean up first)
        with open(self.video_path, "rb") as f:
            resp2 = self.client.post(
                "/api/offline/upload",
                data={"file": (f, "second.mp4")},
                content_type="multipart/form-data",
            )
        assert resp2.status_code == 200

        # Check that the upload directory only has the second file
        from pathlib import Path
        from silentcare.app.config import OFFLINE_UPLOAD_DIR
        project_dir = Path(__file__).resolve().parent.parent
        upload_dir = project_dir / OFFLINE_UPLOAD_DIR

        if upload_dir.exists():
            files = list(upload_dir.iterdir())
            # Should only have the second file
            assert len([f for f in files if f.is_file()]) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
