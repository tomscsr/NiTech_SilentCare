"""
SilentCare - Audio + Video Capture Service
=============================================
Threaded capture:
  - Audio: sounddevice ring buffer at 22050Hz mono
  - Video: OpenCV webcam at ~15fps
  - Produces 10s segments with 2s overlap every 8s
  - Thread-safe segment queue for consumer (analysis pipeline)
"""

import time
import threading
import queue
from collections import deque
import numpy as np
import cv2

try:
    import sounddevice as sd
except OSError:
    sd = None

from silentcare.app.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    SEGMENT_DURATION_S,
    SEGMENT_STEP_S,
    VIDEO_FPS_TARGET,
    VIDEO_FRAMES_PER_SEGMENT,
)


class CaptureService:
    """
    Manages concurrent audio + video capture.
    Produces Segment objects on a queue every SEGMENT_STEP_S seconds.
    """

    def __init__(self, audio_device=None, video_device=0,
                 enable_audio=True, enable_video=True):
        """
        Args:
            audio_device: sounddevice device index (None = default)
            video_device: OpenCV camera index (0 = default webcam)
            enable_audio: whether to capture audio
            enable_video: whether to capture video
        """
        self.audio_device = audio_device
        self.video_device = video_device
        self.enable_audio = enable_audio and (sd is not None)
        self.enable_video = enable_video

        # Output queue of segments
        self.segment_queue = queue.Queue(maxsize=10)

        # Internal state
        self._running = False
        self._audio_thread = None
        self._video_thread = None
        self._segment_thread = None

        # Audio ring buffer: stores SEGMENT_DURATION_S of audio
        self._audio_buffer_size = AUDIO_SAMPLE_RATE * SEGMENT_DURATION_S
        self._audio_buffer = np.zeros(self._audio_buffer_size, dtype=np.float32)
        self._audio_write_pos = 0
        self._audio_lock = threading.Lock()
        self._audio_stream = None

        # Video frame buffer: stores recent frames
        self._video_frames = []
        self._video_lock = threading.Lock()
        self._video_cap = None

        # Segment buffer for feedback system (last 30 segments, JPEG-compressed)
        self._segment_buffer = deque(maxlen=30)
        self._buffer_lock = threading.Lock()

    def start(self):
        """Start capture threads."""
        if self._running:
            return

        self._running = True

        if self.enable_audio:
            self._start_audio()

        if self.enable_video:
            self._video_thread = threading.Thread(
                target=self._video_capture_loop, daemon=True
            )
            self._video_thread.start()

        # Segment production thread
        self._segment_thread = threading.Thread(
            target=self._segment_loop, daemon=True
        )
        self._segment_thread.start()

    def stop(self):
        """Stop all capture threads."""
        self._running = False

        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None

        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

        # Wait for threads to finish
        for t in [self._audio_thread, self._video_thread, self._segment_thread]:
            if t is not None and t.is_alive():
                t.join(timeout=3)

    @property
    def is_running(self):
        return self._running

    # =============================================
    # Audio capture
    # =============================================
    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if not self._running:
            return

        audio = indata[:, 0].astype(np.float32)  # mono

        with self._audio_lock:
            space = self._audio_buffer_size - self._audio_write_pos
            if len(audio) <= space:
                self._audio_buffer[self._audio_write_pos:self._audio_write_pos + len(audio)] = audio
                self._audio_write_pos += len(audio)
            else:
                # Shift buffer left and append
                shift = len(audio) - space
                self._audio_buffer[:-shift] = self._audio_buffer[shift:]
                self._audio_buffer[-len(audio):] = audio
                self._audio_write_pos = self._audio_buffer_size

    def _start_audio(self):
        """Start audio stream."""
        try:
            self._audio_stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype="float32",
                device=self.audio_device,
                callback=self._audio_callback,
                blocksize=int(AUDIO_SAMPLE_RATE * 0.1),  # 100ms blocks
            )
            self._audio_stream.start()
        except Exception as e:
            print(f"[CaptureService] Audio start failed: {e}")
            self.enable_audio = False

    def set_audio_device(self, device_index):
        """Change audio input device. Restarts audio stream if running."""
        self.audio_device = device_index
        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        # Reset buffer
        with self._audio_lock:
            self._audio_buffer[:] = 0
            self._audio_write_pos = 0
        if self._running and self.enable_audio:
            self._start_audio()

    @staticmethod
    def list_audio_devices():
        """List available audio input devices. Returns list of {id, name}."""
        if sd is None:
            return []
        devices = sd.query_devices()
        inputs = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                inputs.append({"id": i, "name": d["name"]})
        return inputs

    def _get_audio_segment(self):
        """Get the current audio buffer as a segment."""
        with self._audio_lock:
            if self._audio_write_pos < AUDIO_SAMPLE_RATE:
                return None  # not enough audio yet
            # Return a copy of the filled portion
            end = self._audio_write_pos
            start = max(0, end - self._audio_buffer_size)
            return self._audio_buffer[start:end].copy()

    # =============================================
    # Video capture
    # =============================================
    def _video_capture_loop(self):
        """Continuously capture video frames."""
        self._video_cap = cv2.VideoCapture(self.video_device)

        if not self._video_cap.isOpened():
            print("[CaptureService] Video capture failed to open.")
            self.enable_video = False
            return

        frame_interval = 1.0 / VIDEO_FPS_TARGET

        while self._running:
            ret, frame = self._video_cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            timestamp = time.time()

            with self._video_lock:
                self._video_frames.append((timestamp, frame))

                # Keep only frames from the last SEGMENT_DURATION_S
                cutoff = timestamp - SEGMENT_DURATION_S - 1
                self._video_frames = [
                    (t, f) for t, f in self._video_frames if t >= cutoff
                ]

            # Throttle to target FPS
            time.sleep(frame_interval)

        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None

    def _get_video_frames(self):
        """Get evenly-spaced frames from the last SEGMENT_DURATION_S."""
        with self._video_lock:
            if not self._video_frames:
                return []

            now = time.time()
            cutoff = now - SEGMENT_DURATION_S

            # Filter to segment window
            segment_frames = [
                (t, f) for t, f in self._video_frames if t >= cutoff
            ]

            if len(segment_frames) < 3:
                return []

            # Sample VIDEO_FRAMES_PER_SEGMENT evenly
            n = len(segment_frames)
            target = VIDEO_FRAMES_PER_SEGMENT
            if n <= target:
                return [f for _, f in segment_frames]

            indices = np.linspace(0, n - 1, target, dtype=int)
            return [segment_frames[i][1] for i in indices]

    # =============================================
    # Live access (for dashboard streaming)
    # =============================================
    def get_current_frame(self):
        """Get the latest video frame (for MJPEG streaming). Returns JPEG bytes or None."""
        with self._video_lock:
            if not self._video_frames:
                return None
            _, frame = self._video_frames[-1]
            # Flip horizontally to remove mirror effect
            frame = cv2.flip(frame, 1)
            # Enhance dark/IR camera images for display
            gray_check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_check.mean() < 80:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                return jpeg.tobytes()
            return None

    def get_audio_buffer_copy(self):
        """Get a copy of the current audio ring buffer (for spectrogram). Returns (audio, sr) or (None, None)."""
        with self._audio_lock:
            if self._audio_write_pos < AUDIO_SAMPLE_RATE:
                return None, None
            end = self._audio_write_pos
            # Return last ~0.5s for spectrogram display
            samples = min(end, AUDIO_SAMPLE_RATE // 2)
            return self._audio_buffer[end - samples:end].copy(), AUDIO_SAMPLE_RATE

    # =============================================
    # Segment production
    # =============================================
    def _segment_loop(self):
        """Produce segments at regular intervals."""
        # Wait for buffers to fill
        time.sleep(SEGMENT_DURATION_S + 1)

        while self._running:
            audio = self._get_audio_segment() if self.enable_audio else None
            frames = self._get_video_frames() if self.enable_video else []

            segment = {
                "timestamp": time.time(),
                "audio": audio,
                "audio_sr": AUDIO_SAMPLE_RATE if audio is not None else None,
                "video_frames": frames,
                "has_audio": audio is not None,
                "has_video": len(frames) > 0,
            }

            # Store compressed copy in circular buffer for feedback
            self._buffer_segment(segment)

            try:
                self.segment_queue.put(segment, timeout=2)
            except queue.Full:
                # Drop oldest segment to prevent backlog
                try:
                    self.segment_queue.get_nowait()
                except queue.Empty:
                    pass
                self.segment_queue.put(segment, timeout=2)

            time.sleep(SEGMENT_STEP_S)

    # =============================================
    # Segment buffer (for feedback system)
    # =============================================
    def _buffer_segment(self, segment):
        """Store a compressed copy of the segment in the circular buffer."""
        video_jpegs = []
        for frame in segment.get("video_frames", []):
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                video_jpegs.append(jpeg.tobytes())

        entry = {
            "timestamp": segment["timestamp"],
            "audio": segment["audio"].copy() if segment["audio"] is not None else None,
            "audio_sr": segment.get("audio_sr"),
            "video_jpegs": video_jpegs,
        }

        with self._buffer_lock:
            self._segment_buffer.append(entry)

    def get_buffered_segment(self, timestamp, tolerance=10.0):
        """Find buffered segment closest to given unix timestamp.

        Args:
            timestamp: Unix timestamp (float) to match
            tolerance: Maximum allowed difference in seconds

        Returns:
            Buffer entry dict with audio + video_jpegs, or None
        """
        with self._buffer_lock:
            best = None
            best_diff = float('inf')
            for entry in self._segment_buffer:
                diff = abs(entry["timestamp"] - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best = entry

            if best is not None and best_diff <= tolerance:
                return {
                    "timestamp": best["timestamp"],
                    "audio": best["audio"].copy() if best["audio"] is not None else None,
                    "audio_sr": best["audio_sr"],
                    "video_jpegs": list(best["video_jpegs"]),
                }
            return None
