"""
SilentCare - Offline Analysis Pipeline
=========================================
Analyzes uploaded video files using the same ML models as live mode.
Two modes: complete (batch) and realtime (simulated with SSE).
Reuses models from the main AnalysisPipeline (no duplication).
"""

import time
import threading
import numpy as np

from silentcare.app.config import (
    AUDIO_WEIGHT,
    VIDEO_WEIGHT,
    AGREEMENT_BOOST,
    UNCERTAINTY_THRESHOLD,
    VIDEO_MIN_CONFIDENCE,
    AUDIO_MIN_RMS,
    EMOTION_CLASSES,
    NUM_CLASSES,
    SEGMENT_STEP_S,
)
from silentcare.core.alert_manager import AlertManager
from silentcare.core.offline_extractor import OfflineExtractor


class OfflinePipeline:
    """
    Offline analysis pipeline. Reuses loaded models from the main pipeline.
    Supports complete (batch) and realtime (simulated) analysis modes.
    """

    # Status constants
    STATUS_IDLE = "idle"
    STATUS_RUNNING = "running"
    STATUS_PAUSED = "paused"
    STATUS_COMPLETE = "complete"
    STATUS_ERROR = "error"

    def __init__(self, pipeline, db, on_segment=None, on_alert=None):
        """
        Args:
            pipeline: AnalysisPipeline instance (for model reuse)
            db: Database instance
            on_segment: callback(segment_result) for each analyzed segment
            on_alert: callback(alert_dict) when an alert fires
        """
        # Reuse loaded models from main pipeline
        self._audio_model = pipeline._audio_model
        self._video_model = pipeline._video_model

        self.db = db
        self.on_segment = on_segment
        self.on_alert = on_alert

        self.alert_manager = AlertManager(on_alert=self._handle_alert)

        # State
        self._status = self.STATUS_IDLE
        self._session_id = None
        self._thread = None
        self._current_segment_id = None

        # Control
        self._running = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # Progress
        self._current_index = 0
        self._total_segments = 0

        # Results accumulator (for complete mode)
        self._results = []
        self._alerts = []
        self._error = None

    @property
    def status(self):
        return self._status

    @property
    def progress(self):
        total = max(1, self._total_segments)
        pct = min(100.0, round(100 * self._current_index / total, 1))
        return {
            "current_segment": self._current_index,
            "total_segments": self._total_segments,
            "percent": pct,
        }

    @property
    def session_id(self):
        return self._session_id

    def analyze_complete(self, video_path):
        """
        Run complete analysis synchronously. Returns all results.

        Args:
            video_path: Path to the video file

        Returns:
            dict with session_id, segments, alerts, stats
        """
        self._reset()
        self._status = self.STATUS_RUNNING

        try:
            extractor = OfflineExtractor(video_path)
            info = extractor.get_info()
            filename = str(video_path).split("\\")[-1].split("/")[-1]

            self._session_id = self.db.start_offline_session(filename)
            self._total_segments = info["total_segments"]

            for idx, total, segment in extractor.iter_segments():
                if not self._running:
                    break

                # Respect pause
                self._pause_event.wait()

                self._current_index = idx + 1
                result = self._process_segment(segment)
                self._results.append(result)

                if self.on_segment:
                    self.on_segment(result)

            self.db.stop_session(self._session_id)
            self._status = self.STATUS_COMPLETE

            return {
                "session_id": self._session_id,
                "segments": self._results,
                "alerts": self._alerts,
                "stats": self.db.get_session_stats(self._session_id),
                "video_info": info,
            }

        except Exception as e:
            self._status = self.STATUS_ERROR
            self._error = str(e)
            if self._session_id:
                self.db.stop_session(self._session_id)
            raise

    def analyze_realtime(self, video_path):
        """
        Run analysis in a background thread with simulated real-time pacing.
        Fires on_segment/on_alert callbacks for SSE streaming.

        Args:
            video_path: Path to the video file
        """
        self._reset()
        self._status = self.STATUS_RUNNING

        self._thread = threading.Thread(
            target=self._realtime_loop,
            args=(video_path,),
            daemon=True,
        )
        self._thread.start()

    def _realtime_loop(self, video_path):
        """Background thread for realtime analysis."""
        try:
            extractor = OfflineExtractor(video_path)
            info = extractor.get_info()
            filename = str(video_path).split("\\")[-1].split("/")[-1]

            self._session_id = self.db.start_offline_session(filename)
            self._total_segments = info["total_segments"]

            for idx, total, segment in extractor.iter_segments():
                if not self._running:
                    break

                # Respect pause
                self._pause_event.wait()
                if not self._running:
                    break

                self._current_index = idx + 1
                result = self._process_segment(segment)
                self._results.append(result)

                if self.on_segment:
                    self.on_segment(result)

                # Simulate real-time pacing (wait between segments)
                if self._running and idx + 1 < total:
                    # Sleep in small increments to allow stop/pause
                    for _ in range(int(SEGMENT_STEP_S * 10)):
                        if not self._running:
                            break
                        self._pause_event.wait()
                        time.sleep(0.1)

            if self._session_id:
                self.db.stop_session(self._session_id)

            if self._running:
                self._status = self.STATUS_COMPLETE
            else:
                self._status = self.STATUS_IDLE

        except Exception as e:
            self._status = self.STATUS_ERROR
            self._error = str(e)
            if self._session_id:
                try:
                    self.db.stop_session(self._session_id)
                except Exception:
                    pass

    def pause(self):
        """Pause the analysis."""
        if self._status == self.STATUS_RUNNING:
            self._pause_event.clear()
            self._status = self.STATUS_PAUSED

    def resume(self):
        """Resume paused analysis."""
        if self._status == self.STATUS_PAUSED:
            self._status = self.STATUS_RUNNING
            self._pause_event.set()

    def stop(self):
        """Stop the analysis."""
        self._running = False
        self._pause_event.set()  # Unblock if paused

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        if self._session_id and self._status != self.STATUS_COMPLETE:
            try:
                self.db.stop_session(self._session_id)
            except Exception:
                pass

        self._status = self.STATUS_IDLE

    def get_results(self):
        """Get accumulated results."""
        return {
            "session_id": self._session_id,
            "status": self._status,
            "progress": self.progress,
            "segments": self._results,
            "alerts": self._alerts,
            "error": self._error,
        }

    def _reset(self):
        """Reset state for a new analysis."""
        if self._thread and self._thread.is_alive():
            self._running = False
            self._pause_event.set()
            self._thread.join(timeout=5)

        self._status = self.STATUS_IDLE
        self._session_id = None
        self._running = True
        self._pause_event.set()
        self._current_index = 0
        self._total_segments = 0
        self._results = []
        self._alerts = []
        self._error = None
        self._current_segment_id = None
        self.alert_manager.reset()

    # =============================================
    # Segment processing (reused from AnalysisPipeline)
    # =============================================
    def _process_segment(self, segment):
        """Process a single segment through models and fusion."""
        audio_probs = None
        video_probs = None

        # Audio inference
        if segment["has_audio"] and self._audio_model is not None:
            try:
                audio = segment["audio"]
                rms = float(np.sqrt(np.mean(audio ** 2)))

                if rms >= AUDIO_MIN_RMS:
                    audio_result = self._audio_model.predict(
                        audio, sr=segment["audio_sr"]
                    )
                    audio_probs = audio_result["probabilities"]
            except Exception:
                pass

        # Video inference
        if segment["has_video"] and self._video_model is not None:
            try:
                video_result = self._video_model.predict(segment["video_frames"])
                if video_result is not None:
                    video_probs = video_result["probabilities"]
            except Exception:
                pass

        # Fusion
        fused_probs = self._fuse_predictions(audio_probs, video_probs)

        predicted_idx = int(np.argmax(fused_probs))
        predicted_class = EMOTION_CLASSES[predicted_idx]
        confidence = float(fused_probs[predicted_idx])

        # Store segment in DB
        segment_id = self.db.add_segment(
            session_id=self._session_id,
            audio_probs=audio_probs if audio_probs is not None else np.zeros(NUM_CLASSES),
            video_probs=video_probs if video_probs is not None else np.zeros(NUM_CLASSES),
            fused_probs=fused_probs,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        self._current_segment_id = segment_id

        # Run through alert manager
        alert = self.alert_manager.process_segment(
            fused_probs=fused_probs,
            audio_probs=audio_probs,
            video_probs=video_probs,
            session_id=self._session_id,
        )

        return {
            "segment_id": segment_id,
            "timestamp": segment["timestamp"],
            "predicted_class": predicted_class,
            "confidence": confidence,
            "fused_probs": fused_probs.tolist(),
            "audio_probs": audio_probs.tolist() if audio_probs is not None else None,
            "video_probs": video_probs.tolist() if video_probs is not None else None,
            "alert": alert,
            "streak": self.alert_manager.current_streak,
        }

    def _fuse_predictions(self, audio_probs, video_probs):
        """Fuse audio and video predictions (same logic as AnalysisPipeline)."""
        if audio_probs is None and video_probs is None:
            return np.array([0.0, 0.0, 0.0, 1.0])

        if audio_probs is None:
            return np.array(video_probs, dtype=np.float64)

        if video_probs is None:
            return np.array(audio_probs, dtype=np.float64)

        audio_probs = np.array(audio_probs, dtype=np.float64)
        video_probs = np.array(video_probs, dtype=np.float64)

        video_conf = float(np.max(video_probs))
        audio_conf = float(np.max(audio_probs))

        if video_conf < VIDEO_MIN_CONFIDENCE:
            video_scale = video_conf / VIDEO_MIN_CONFIDENCE
            adj_video_w = VIDEO_WEIGHT * video_scale
            adj_audio_w = 1.0 - adj_video_w
        elif audio_conf < VIDEO_MIN_CONFIDENCE:
            audio_scale = audio_conf / VIDEO_MIN_CONFIDENCE
            adj_audio_w = AUDIO_WEIGHT * audio_scale
            adj_video_w = 1.0 - adj_audio_w
        else:
            adj_audio_w = AUDIO_WEIGHT
            adj_video_w = VIDEO_WEIGHT

        fused = adj_audio_w * audio_probs + adj_video_w * video_probs

        audio_top = int(np.argmax(audio_probs))
        video_top = int(np.argmax(video_probs))

        if audio_top == video_top:
            fused[audio_top] *= AGREEMENT_BOOST

        total = fused.sum()
        if total > 0:
            fused /= total

        top_conf = float(np.max(fused))
        if top_conf < UNCERTAINTY_THRESHOLD:
            fused = np.array([0.0, 0.0, 0.0, 1.0])

        return fused

    def _handle_alert(self, alert):
        """Called by AlertManager when an alert fires."""
        segment_id = self._current_segment_id
        alert_id = self.db.add_alert(
            session_id=alert["session_id"],
            emotion=alert["emotion"],
            severity=alert["severity"],
            confidence=alert["confidence"],
            audio_confidence=alert["audio_confidence"],
            video_confidence=alert["video_confidence"],
            fused_probs=alert["fused_probs"],
            consecutive_count=alert["consecutive_count"],
            segment_id=segment_id,
        )

        alert["id"] = alert_id
        alert["segment_id"] = segment_id
        self._alerts.append(dict(alert))

        if self.on_alert:
            self.on_alert(alert)
