"""
SilentCare - Alert Manager
============================
Anti-false-positive logic:
  - Consecutive segment requirement: N segments must agree before triggering
  - Cooldown timer: minimum time between same-type alerts
  - Severity levels: LOW (1 seg), MEDIUM (2 seg), HIGH (3+ seg)
  - Threshold checks per emotion class
"""

import time
import numpy as np
from collections import deque

from silentcare.app.config import (
    EMOTION_CLASSES,
    ALERT_THRESHOLDS,
    CONSECUTIVE_SEGMENTS_FOR_ALERT,
    ALERT_COOLDOWN_SECONDS,
    SEVERITY_WINDOW,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    CLASS_CALM,
    CLASS_DISTRESS,
    DISTRESS_CALM_MARGIN,
)


class AlertManager:
    """
    Manages alert state: tracks consecutive detections, cooldowns, severity.
    Feed it fused predictions segment-by-segment; it decides when to fire alerts.
    """

    def __init__(self, on_alert=None):
        """
        Args:
            on_alert: optional callback(alert_dict) called when an alert fires
        """
        self.on_alert = on_alert

        # Track recent predictions (sliding window)
        self._history = deque(maxlen=SEVERITY_WINDOW + 2)

        # Cooldown tracking: emotion -> last_alert_timestamp
        self._cooldowns = {}

        # Consecutive same-class counter
        self._consecutive_class = None
        self._consecutive_count = 0

    def reset(self):
        """Reset all state (e.g., at session start)."""
        self._history.clear()
        self._cooldowns.clear()
        self._consecutive_class = None
        self._consecutive_count = 0

    def _compute_severity(self, consecutive_count):
        """Map consecutive count to severity level."""
        if consecutive_count >= 3:
            return SEVERITY_HIGH
        elif consecutive_count >= 2:
            return SEVERITY_MEDIUM
        else:
            return SEVERITY_LOW

    def _is_in_cooldown(self, emotion):
        """Check if this emotion type is still in cooldown."""
        if emotion not in self._cooldowns:
            return False
        elapsed = time.time() - self._cooldowns[emotion]
        return elapsed < ALERT_COOLDOWN_SECONDS

    def _update_cooldown(self, emotion):
        """Record that we just fired an alert for this emotion."""
        self._cooldowns[emotion] = time.time()

    def process_segment(self, fused_probs, audio_probs=None, video_probs=None,
                        session_id=None):
        """
        Process a single segment's fused predictions.
        Returns an alert dict if alert should fire, else None.

        Args:
            fused_probs: np.array of shape (4,) - fused class probabilities
            audio_probs: np.array of shape (4,) - raw audio probs (for logging)
            video_probs: np.array of shape (4,) - raw video probs (for logging)
            session_id: current session id (for DB storage)

        Returns:
            alert dict or None
        """
        fused_probs = np.array(fused_probs, dtype=np.float64)
        predicted_idx = int(np.argmax(fused_probs))
        predicted_class = EMOTION_CLASSES[predicted_idx]
        confidence = float(fused_probs[predicted_idx])

        # Store in history
        self._history.append({
            "class": predicted_class,
            "class_idx": predicted_idx,
            "confidence": confidence,
            "fused_probs": fused_probs,
            "audio_probs": audio_probs,
            "video_probs": video_probs,
            "timestamp": time.time(),
        })

        # Update consecutive counter
        if predicted_class == self._consecutive_class:
            self._consecutive_count += 1
        else:
            self._consecutive_class = predicted_class
            self._consecutive_count = 1

        # CALM never triggers alerts
        if predicted_idx == CLASS_CALM:
            return None

        # Check if this class has a threshold defined
        threshold = ALERT_THRESHOLDS.get(predicted_class)
        if threshold is None:
            return None

        # Check confidence against threshold
        if confidence < threshold:
            return None

        # DISTRESS margin check: must clearly exceed CALM to avoid
        # false positives from neutral faces (RAF-DB domain shift)
        if predicted_idx == CLASS_DISTRESS:
            calm_prob = float(fused_probs[CLASS_CALM])
            if confidence - calm_prob < DISTRESS_CALM_MARGIN:
                return None

        # Check consecutive requirement
        if self._consecutive_count < CONSECUTIVE_SEGMENTS_FOR_ALERT:
            return None

        # Check cooldown
        if self._is_in_cooldown(predicted_class):
            return None

        # All checks passed -> fire alert
        severity = self._compute_severity(self._consecutive_count)
        self._update_cooldown(predicted_class)

        alert = {
            "emotion": predicted_class,
            "severity": severity,
            "confidence": confidence,
            "consecutive_count": self._consecutive_count,
            "fused_probs": fused_probs,
            "audio_confidence": float(audio_probs[predicted_idx]) if audio_probs is not None else None,
            "video_confidence": float(video_probs[predicted_idx]) if video_probs is not None else None,
            "session_id": session_id,
        }

        # Fire callback if registered
        if self.on_alert:
            self.on_alert(alert)

        return alert

    @property
    def current_streak(self):
        """Get current consecutive class and count."""
        return {
            "class": self._consecutive_class,
            "count": self._consecutive_count,
        }

    @property
    def recent_history(self):
        """Get recent prediction history."""
        return list(self._history)
