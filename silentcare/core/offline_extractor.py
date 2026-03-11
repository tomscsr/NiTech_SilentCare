"""
SilentCare - Offline Video Extractor
=======================================
Extracts audio waveform + video frames from an MP4 file,
producing segments in the same format as CaptureService.

Uses:
  - OpenCV for video frames (already a dependency)
  - ffmpeg subprocess for audio extraction (ffmpeg must be in PATH)
"""

import math
import subprocess
import tempfile
import struct
import numpy as np
import cv2
from pathlib import Path

from silentcare.app.config import (
    AUDIO_SAMPLE_RATE,
    SEGMENT_DURATION_S,
    SEGMENT_STEP_S,
    VIDEO_FRAMES_PER_SEGMENT,
)


class OfflineExtractor:
    """Extracts audio + video from a video file, segment by segment."""

    def __init__(self, video_path):
        self.video_path = str(Path(video_path).resolve())
        self._info = None

    def get_info(self):
        """Return video metadata: duration, fps, width, height, has_audio, total_segments."""
        if self._info is not None:
            return self._info

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        # Check for audio stream via ffprobe
        has_audio = self._check_audio()

        # Calculate total segments: how many start positions 0, step, 2*step, ...
        # are strictly less than duration
        if duration <= 0:
            total_segments = 0
        else:
            total_segments = math.ceil(duration / SEGMENT_STEP_S)

        self._info = {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "has_audio": has_audio,
            "total_segments": total_segments,
        }
        return self._info

    def _check_audio(self):
        """Check if the video file has an audio stream."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "a",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    self.video_path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            return "audio" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _extract_audio_full(self):
        """Extract the full audio track as a float32 numpy array at AUDIO_SAMPLE_RATE."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-i", self.video_path,
                    "-vn",                          # no video
                    "-acodec", "pcm_s16le",         # 16-bit PCM
                    "-ar", str(AUDIO_SAMPLE_RATE),  # resample
                    "-ac", "1",                     # mono
                    "-f", "wav",                    # WAV format
                    "pipe:1",                       # stdout
                ],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                return None

            wav_data = result.stdout
            # Skip WAV header (44 bytes)
            if len(wav_data) <= 44:
                return None

            audio_bytes = wav_data[44:]
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int16.astype(np.float32) / 32768.0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def extract_segment(self, start_s, duration_s=None, full_audio=None):
        """
        Extract a single segment from the video.

        Args:
            start_s: Start time in seconds
            duration_s: Segment duration (default SEGMENT_DURATION_S)
            full_audio: Pre-extracted full audio array (optimization)

        Returns:
            Segment dict compatible with AnalysisPipeline._process_segment
        """
        if duration_s is None:
            duration_s = SEGMENT_DURATION_S

        info = self.get_info()

        # Clamp to video duration
        end_s = min(start_s + duration_s, info["duration"])
        actual_duration = end_s - start_s

        # --- Video frames ---
        frames = self._extract_video_frames(start_s, actual_duration, info["fps"])

        # --- Audio ---
        audio = None
        if full_audio is not None:
            start_sample = int(start_s * AUDIO_SAMPLE_RATE)
            end_sample = int(end_s * AUDIO_SAMPLE_RATE)
            end_sample = min(end_sample, len(full_audio))
            if start_sample < end_sample:
                audio = full_audio[start_sample:end_sample]
        elif info["has_audio"]:
            audio = self._extract_audio_segment(start_s, actual_duration)

        return {
            "timestamp": start_s,
            "audio": audio,
            "audio_sr": AUDIO_SAMPLE_RATE if audio is not None else None,
            "video_frames": frames,
            "has_audio": audio is not None and len(audio) > AUDIO_SAMPLE_RATE * 0.5,
            "has_video": len(frames) >= 3,
        }

    def _extract_video_frames(self, start_s, duration_s, fps):
        """Extract evenly-spaced video frames from a time range."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return []

        start_frame = int(start_s * fps)
        end_frame = int((start_s + duration_s) * fps)
        total_frames_in_range = max(1, end_frame - start_frame)

        target = min(VIDEO_FRAMES_PER_SEGMENT, total_frames_in_range)
        if total_frames_in_range <= target:
            frame_indices = list(range(start_frame, end_frame))
        else:
            frame_indices = np.linspace(
                start_frame, end_frame - 1, target, dtype=int
            ).tolist()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def _extract_audio_segment(self, start_s, duration_s):
        """Extract a single audio segment via ffmpeg."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-ss", str(start_s),
                    "-i", self.video_path,
                    "-t", str(duration_s),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(AUDIO_SAMPLE_RATE),
                    "-ac", "1",
                    "-f", "wav",
                    "pipe:1",
                ],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                return None

            wav_data = result.stdout
            if len(wav_data) <= 44:
                return None

            audio_bytes = wav_data[44:]
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int16.astype(np.float32) / 32768.0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def iter_segments(self, segment_duration=None, step=None):
        """
        Yield segments from start to end of video.

        Args:
            segment_duration: Duration of each segment (default SEGMENT_DURATION_S)
            step: Step between segments (default SEGMENT_STEP_S)

        Yields:
            (segment_index, total_segments, segment_dict)
        """
        if segment_duration is None:
            segment_duration = SEGMENT_DURATION_S
        if step is None:
            step = SEGMENT_STEP_S

        info = self.get_info()
        duration = info["duration"]

        if duration <= 0:
            return

        # Pre-extract full audio for efficiency (one ffmpeg call)
        full_audio = None
        if info["has_audio"]:
            full_audio = self._extract_audio_full()

        total = info["total_segments"]
        start_s = 0.0
        idx = 0

        while start_s < duration:
            segment = self.extract_segment(
                start_s, segment_duration, full_audio=full_audio
            )
            yield idx, total, segment
            idx += 1
            start_s += step
