"""
SilentCare - Analysis Pipeline
=================================
Orchestrates: capture -> ML inference -> fusion -> alert manager -> DB.
Runs in its own thread consuming segments from CaptureService.
"""

import time
import threading
import numpy as np
import queue

from silentcare.app.config import (
    AUDIO_WEIGHT,
    VIDEO_WEIGHT,
    AGREEMENT_BOOST,
    UNCERTAINTY_THRESHOLD,
    VIDEO_MIN_CONFIDENCE,
    AUDIO_MIN_RMS,
    EMOTION_CLASSES,
    NUM_CLASSES,
    AUDIO_MODEL_PATH,
    AUDIO_CLASSES_PATH,
    VIDEO_MODEL_PATH,
)
from silentcare.core.database import Database
from silentcare.core.alert_manager import AlertManager


class AnalysisPipeline:
    """
    Main analysis pipeline. Consumes segments, runs inference, fuses results,
    manages alerts, stores to DB.
    """

    def __init__(self, capture_service, db=None, on_segment=None, on_alert=None):
        """
        Args:
            capture_service: CaptureService instance providing segment_queue
            db: Database instance (created internally if None)
            on_segment: callback(segment_result) for each analyzed segment
            on_alert: callback(alert_dict) when an alert fires
        """
        self.capture = capture_service
        self.db = db or Database()
        self.on_segment = on_segment
        self.on_alert = on_alert

        self.alert_manager = AlertManager(on_alert=self._handle_alert)

        # Models (lazy-loaded)
        self._audio_model = None
        self._video_model = None
        self._models_loaded = False

        # State
        self._running = False
        self._thread = None
        self._session_id = None

        # Latest result for dashboard polling
        self._latest_result = None
        self._result_lock = threading.Lock()

    def load_models(self):
        """Load ML models. Call before start() or models load on first segment."""
        from pathlib import Path

        project_dir = Path(__file__).resolve().parent.parent.parent

        # Audio model
        audio_path = project_dir / AUDIO_MODEL_PATH
        classes_path = project_dir / AUDIO_CLASSES_PATH
        if audio_path.exists():
            from silentcare.ml.audio_model import AudioModel
            self._audio_model = AudioModel(
                model_path=str(audio_path),
                classes_path=str(classes_path) if classes_path.exists() else None,
            )
            print("[Pipeline] Audio model loaded.")
        else:
            print(f"[Pipeline] WARNING: Audio model not found at {audio_path}")

        # Video model (ViT HuggingFace, trained on FER-2013)
        try:
            from silentcare.ml.video_model import VideoModel
            self._video_model = VideoModel()
            print("[Pipeline] Video model loaded (ViT HuggingFace).")
        except Exception as e:
            print(f"[Pipeline] WARNING: Video model failed to load: {e}")

        self._models_loaded = True

    def start(self):
        """Start the analysis pipeline thread."""
        if self._running:
            return

        if not self._models_loaded:
            self.load_models()

        self._running = True
        self.alert_manager.reset()

        # Start DB session
        self._session_id = self.db.start_session()
        print(f"[Pipeline] Session {self._session_id} started.")

        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the pipeline."""
        self._running = False

        if self._session_id:
            self.db.stop_session(self._session_id)
            print(f"[Pipeline] Session {self._session_id} stopped.")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def is_running(self):
        return self._running

    @property
    def session_id(self):
        return self._session_id

    @property
    def latest_result(self):
        with self._result_lock:
            return self._latest_result

    # =============================================
    # Fusion logic
    # =============================================
    def _fuse_predictions(self, audio_probs, video_probs):
        """
        Fuse audio and video predictions.

        Strategy:
          1. Weighted average: audio * 0.30 + video * 0.70
          2. Agreement boost: if both agree on top class, boost by 1.3x
          3. Uncertainty check: if top confidence < threshold, default to CALM

        Args:
            audio_probs: np.array (4,) or None
            video_probs: np.array (4,) or None

        Returns:
            np.array (4,) fused probabilities (normalized)
        """
        # Handle missing modalities
        if audio_probs is None and video_probs is None:
            return np.array([0.0, 0.0, 0.0, 1.0])  # default CALM

        if audio_probs is None:
            return np.array(video_probs, dtype=np.float64)

        if video_probs is None:
            return np.array(audio_probs, dtype=np.float64)

        audio_probs = np.array(audio_probs, dtype=np.float64)
        video_probs = np.array(video_probs, dtype=np.float64)

        # Step 1: Adaptive weighted average
        # If video confidence is low (domain shift), reduce its weight
        video_conf = float(np.max(video_probs))
        audio_conf = float(np.max(audio_probs))

        if video_conf < VIDEO_MIN_CONFIDENCE:
            # Video uncertain -> scale down its weight proportionally
            video_scale = video_conf / VIDEO_MIN_CONFIDENCE  # 0..1
            adj_video_w = VIDEO_WEIGHT * video_scale
            adj_audio_w = 1.0 - adj_video_w
        elif audio_conf < VIDEO_MIN_CONFIDENCE:
            # Audio uncertain -> scale down its weight proportionally
            audio_scale = audio_conf / VIDEO_MIN_CONFIDENCE
            adj_audio_w = AUDIO_WEIGHT * audio_scale
            adj_video_w = 1.0 - adj_audio_w
        else:
            adj_audio_w = AUDIO_WEIGHT
            adj_video_w = VIDEO_WEIGHT

        fused = adj_audio_w * audio_probs + adj_video_w * video_probs

        # Step 2: Agreement boost
        audio_top = int(np.argmax(audio_probs))
        video_top = int(np.argmax(video_probs))

        if audio_top == video_top:
            fused[audio_top] *= AGREEMENT_BOOST

        # Re-normalize
        total = fused.sum()
        if total > 0:
            fused /= total

        # Step 3: Uncertainty check
        top_conf = float(np.max(fused))
        if top_conf < UNCERTAINTY_THRESHOLD:
            # Too uncertain, default to CALM
            fused = np.array([0.0, 0.0, 0.0, 1.0])

        return fused

    # =============================================
    # Analysis loop
    # =============================================
    def _analysis_loop(self):
        """Main loop: consume segments, analyze, fuse, alert."""
        while self._running:
            try:
                segment = self.capture.segment_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                result = self._process_segment(segment)

                # Store latest for dashboard
                with self._result_lock:
                    self._latest_result = result

                # Callback
                if self.on_segment:
                    self.on_segment(result)

            except Exception as e:
                print(f"[Pipeline] Error processing segment: {e}")

    def _process_segment(self, segment):
        """Process a single segment through models and fusion."""
        audio_probs = None
        video_probs = None

        # Audio inference
        if segment["has_audio"] and self._audio_model is not None:
            try:
                audio = segment["audio"]
                rms = float(np.sqrt(np.mean(audio ** 2)))

                if rms < AUDIO_MIN_RMS:
                    # Silence / ambient noise -> skip audio (unreliable on silence)
                    print(f"[Pipeline] Audio RMS={rms:.4f} -> silence, skipping")
                else:
                    audio_result = self._audio_model.predict(
                        audio, sr=segment["audio_sr"]
                    )
                    audio_probs = audio_result["probabilities"]
                    print(f"[Pipeline] Audio RMS={rms:.4f} | "
                          f"D={audio_probs[0]:.2f} A={audio_probs[1]:.2f} "
                          f"L={audio_probs[2]:.2f} C={audio_probs[3]:.2f} "
                          f"-> {audio_result['predicted_class']} ({audio_result['confidence']:.2f})")
            except Exception as e:
                print(f"[Pipeline] Audio inference error: {e}")

        # Video inference
        if segment["has_video"] and self._video_model is not None:
            try:
                video_result = self._video_model.predict(segment["video_frames"])
                if video_result is not None:
                    video_probs = video_result["probabilities"]
                    print(f"[Pipeline] Video | "
                          f"D={video_probs[0]:.2f} A={video_probs[1]:.2f} "
                          f"L={video_probs[2]:.2f} C={video_probs[3]:.2f} "
                          f"-> {video_result['predicted_class']} ({video_result['confidence']:.2f})")
                else:
                    print("[Pipeline] Video | no face detected -> audio only")
            except Exception as e:
                print(f"[Pipeline] Video inference error: {e}")

        # Fusion
        fused_probs = self._fuse_predictions(audio_probs, video_probs)

        predicted_idx = int(np.argmax(fused_probs))
        predicted_class = EMOTION_CLASSES[predicted_idx]
        confidence = float(fused_probs[predicted_idx])
        print(f"[Pipeline] Fused -> {predicted_class} ({confidence:.2f}) | "
              f"streak: {self.alert_manager._consecutive_count}x {self.alert_manager._consecutive_class}")

        # Store segment in DB
        segment_id = self.db.add_segment(
            session_id=self._session_id,
            audio_probs=audio_probs if audio_probs is not None else np.zeros(NUM_CLASSES),
            video_probs=video_probs if video_probs is not None else np.zeros(NUM_CLASSES),
            fused_probs=fused_probs,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        # Store current segment_id for alert handler
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

    def _handle_alert(self, alert):
        """Called by AlertManager when an alert fires."""
        # Store in DB with segment linkage
        segment_id = getattr(self, '_current_segment_id', None)
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

        # Add IDs for SSE broadcast
        alert["id"] = alert_id
        alert["segment_id"] = segment_id

        print(f"[ALERT] {alert['severity']} - {alert['emotion']} "
              f"(confidence: {alert['confidence']:.2f}, "
              f"consecutive: {alert['consecutive_count']})")

        # Forward to external callback
        if self.on_alert:
            self.on_alert(alert)
