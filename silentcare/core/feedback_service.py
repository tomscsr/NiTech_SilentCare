"""
SilentCare - Feedback Service
================================
Human-in-the-loop feedback: saves audio/video segments from the
circular buffer, creates feedback entries in the database.
"""

import struct
import numpy as np
from pathlib import Path
from datetime import datetime


class FeedbackService:
    """Manages feedback reports: saves segment data and creates DB entries."""

    VALID_CLASSES = {"DISTRESS", "ANGRY", "ALERT", "CALM"}

    def __init__(self, db, capture_service, data_dir=None):
        self.db = db
        self.capture = capture_service

        if data_dir is None:
            project_dir = Path(__file__).resolve().parent.parent.parent
            self.data_dir = project_dir / "data" / "feedback"
        else:
            self.data_dir = Path(data_dir)

        # Ensure directories exist
        (self.data_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "video").mkdir(parents=True, exist_ok=True)

    def report_false_alert(self, alert_id, correct_class, notes=None):
        """Report a false alert.

        Args:
            alert_id: ID of the alert that was incorrect
            correct_class: What the actual emotion was
            notes: Optional user comment

        Returns:
            feedback_id
        """
        if correct_class not in self.VALID_CLASSES:
            raise ValueError(f"Invalid class: {correct_class}")

        alert = self.db.get_alert_by_id(alert_id)
        if alert is None:
            raise ValueError(f"Alert {alert_id} not found")

        # Find associated segment
        segment_id = alert.get("segment_id")
        segment = None
        if segment_id:
            segment = self.db.get_segment_by_id(segment_id)
        if segment is None:
            segment = self.db.get_segment_near_timestamp(alert["timestamp"])
            segment_id = segment["id"] if segment else None

        feedback_id = self.db.add_feedback(
            session_id=alert.get("session_id"),
            segment_id=segment_id,
            alert_id=alert_id,
            report_type="FALSE_ALERT",
            predicted_class=alert["emotion"],
            correct_class=correct_class,
            notes=notes,
        )

        audio_saved, video_saved = self._save_segment_data(
            feedback_id, segment
        )

        if audio_saved or video_saved:
            audio_path = f"data/feedback/audio/{feedback_id}.wav" if audio_saved else None
            video_path = f"data/feedback/video/{feedback_id}" if video_saved else None
            self.db.update_feedback_files(
                feedback_id, audio_saved, video_saved, audio_path, video_path
            )

        return feedback_id

    def report_missed_detection(self, segment_id, correct_class, notes=None):
        """Report a missed detection.

        Args:
            segment_id: ID of the segment where detection was missed
            correct_class: What emotion should have been detected
            notes: Optional user comment

        Returns:
            feedback_id
        """
        if correct_class not in self.VALID_CLASSES:
            raise ValueError(f"Invalid class: {correct_class}")

        segment = self.db.get_segment_by_id(segment_id)
        if segment is None:
            raise ValueError(f"Segment {segment_id} not found")

        feedback_id = self.db.add_feedback(
            session_id=segment.get("session_id"),
            segment_id=segment_id,
            alert_id=None,
            report_type="MISSED_DETECTION",
            predicted_class=segment.get("predicted_class", "CALM"),
            correct_class=correct_class,
            notes=notes,
        )

        audio_saved, video_saved = self._save_segment_data(
            feedback_id, segment
        )

        if audio_saved or video_saved:
            audio_path = f"data/feedback/audio/{feedback_id}.wav" if audio_saved else None
            video_path = f"data/feedback/video/{feedback_id}" if video_saved else None
            self.db.update_feedback_files(
                feedback_id, audio_saved, video_saved, audio_path, video_path
            )

        return feedback_id

    def report_wrong_classification(self, correct_class, notes=None,
                                    alert_id=None, segment_id=None):
        """Report a wrong classification.

        Args:
            correct_class: The correct emotion class
            notes: Optional user comment
            alert_id: ID of the alert (optional)
            segment_id: ID of the segment (optional)

        Returns:
            feedback_id
        """
        if correct_class not in self.VALID_CLASSES:
            raise ValueError(f"Invalid class: {correct_class}")

        if alert_id is None and segment_id is None:
            raise ValueError("Either alert_id or segment_id is required")

        alert = None
        segment = None
        session_id = None
        predicted_class = None

        if alert_id is not None:
            alert = self.db.get_alert_by_id(alert_id)
            if alert is None:
                raise ValueError(f"Alert {alert_id} not found")
            predicted_class = alert["emotion"]
            session_id = alert.get("session_id")

            seg_id = alert.get("segment_id")
            if seg_id:
                segment = self.db.get_segment_by_id(seg_id)
                segment_id = seg_id
            elif segment_id is None:
                segment = self.db.get_segment_near_timestamp(alert["timestamp"])
                segment_id = segment["id"] if segment else None

        if segment_id is not None and segment is None:
            segment = self.db.get_segment_by_id(segment_id)
            if segment is None:
                raise ValueError(f"Segment {segment_id} not found")
            predicted_class = predicted_class or segment.get("predicted_class")
            session_id = session_id or segment.get("session_id")

        feedback_id = self.db.add_feedback(
            session_id=session_id,
            segment_id=segment_id,
            alert_id=alert_id,
            report_type="WRONG_CLASSIFICATION",
            predicted_class=predicted_class,
            correct_class=correct_class,
            notes=notes,
        )

        audio_saved, video_saved = self._save_segment_data(
            feedback_id, segment
        )

        if audio_saved or video_saved:
            audio_path = f"data/feedback/audio/{feedback_id}.wav" if audio_saved else None
            video_path = f"data/feedback/video/{feedback_id}" if video_saved else None
            self.db.update_feedback_files(
                feedback_id, audio_saved, video_saved, audio_path, video_path
            )

        return feedback_id

    def _save_segment_data(self, feedback_id, segment):
        """Save audio and video data for a feedback entry.

        Returns:
            (audio_saved: bool, video_saved: bool)
        """
        if segment is None:
            return False, False

        timestamp = segment.get("timestamp")
        if timestamp is None:
            return False, False

        # Convert ISO timestamp to unix timestamp for buffer lookup
        try:
            dt = datetime.fromisoformat(timestamp)
            unix_ts = dt.timestamp()
        except (ValueError, TypeError):
            try:
                unix_ts = float(timestamp)
            except (ValueError, TypeError):
                return False, False

        buffer_entry = self.capture.get_buffered_segment(unix_ts)
        if buffer_entry is None:
            return False, False

        audio_saved = False
        video_saved = False

        # Save audio as WAV
        if buffer_entry.get("audio") is not None:
            audio_path = self.data_dir / "audio" / f"{feedback_id}.wav"
            sr = buffer_entry.get("audio_sr", 22050)
            self._write_wav(audio_path, buffer_entry["audio"], sr)
            audio_saved = True

        # Save video frames as JPEGs
        if buffer_entry.get("video_jpegs"):
            video_dir = self.data_dir / "video" / str(feedback_id)
            video_dir.mkdir(parents=True, exist_ok=True)
            for i, jpeg_bytes in enumerate(buffer_entry["video_jpegs"]):
                frame_path = video_dir / f"frame_{i:02d}.jpg"
                frame_path.write_bytes(jpeg_bytes)
            video_saved = True

        return audio_saved, video_saved

    @staticmethod
    def _write_wav(path, audio_data, sample_rate):
        """Write audio data as a WAV file (16-bit PCM).

        Uses struct module to avoid scipy dependency.
        """
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

        num_samples = len(audio_int16)
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align

        with open(path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))
            f.write(struct.pack('<H', 1))   # PCM format
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(audio_int16.tobytes())
