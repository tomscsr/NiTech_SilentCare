"""
SilentCare - SQLite Database Layer
====================================
Tables:
  - alerts: triggered alerts with severity, confidence, acknowledgment
  - segments: raw analysis results per 10s segment
  - sessions: monitoring session start/stop tracking
"""

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path


class Database:
    """Thread-safe SQLite database for SilentCare."""

    def __init__(self, db_path="silentcare.db"):
        self.db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self):
        """Thread-local connection with autocommit for cross-thread visibility."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path, isolation_level=None
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT NOT NULL,
                audio_probs TEXT,
                video_probs TEXT,
                fused_probs TEXT,
                predicted_class TEXT,
                confidence REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                audio_confidence REAL,
                video_confidence REAL,
                fused_probs TEXT,
                consecutive_count INTEGER DEFAULT 1,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id);
            CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id);
            CREATE INDEX IF NOT EXISTS idx_segments_timestamp ON segments(timestamp);
        """)
        conn.commit()

        # Migration: add segment_id to alerts table if not present
        try:
            conn.execute(
                "ALTER TABLE alerts ADD COLUMN segment_id INTEGER REFERENCES segments(id)"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migration: add offline columns to sessions table
        for col_sql in [
            "ALTER TABLE sessions ADD COLUMN is_offline INTEGER DEFAULT 0",
            "ALTER TABLE sessions ADD COLUMN video_filename TEXT",
        ]:
            try:
                conn.execute(col_sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Feedback table
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id INTEGER,
                segment_id INTEGER,
                alert_id INTEGER,
                report_type TEXT NOT NULL,
                predicted_class TEXT,
                correct_class TEXT NOT NULL,
                audio_saved INTEGER DEFAULT 0,
                video_saved INTEGER DEFAULT 0,
                audio_path TEXT,
                video_path TEXT,
                notes TEXT,
                used_for_training INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (segment_id) REFERENCES segments(id),
                FOREIGN KEY (alert_id) REFERENCES alerts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(report_type);
        """)
        conn.commit()

    # =============================================
    # Session management
    # =============================================
    def start_session(self):
        """Start a new monitoring session. Returns session_id."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO sessions (started_at, status) VALUES (?, 'active')",
            (now,)
        )
        self._conn.commit()
        return cursor.lastrowid

    def start_offline_session(self, video_filename):
        """Start an offline analysis session. Returns session_id."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO sessions (started_at, status, is_offline, video_filename) VALUES (?, 'active', 1, ?)",
            (now, video_filename)
        )
        self._conn.commit()
        return cursor.lastrowid

    def stop_session(self, session_id):
        """Stop a monitoring session."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE sessions SET stopped_at = ?, status = 'stopped' WHERE id = ?",
            (now, session_id)
        )
        self._conn.commit()

    def get_active_session(self):
        """Get the current active session, or None."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # =============================================
    # Segment storage
    # =============================================
    def add_segment(self, session_id, audio_probs, video_probs, fused_probs,
                    predicted_class, confidence):
        """Store a segment analysis result."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO segments
               (session_id, timestamp, audio_probs, video_probs, fused_probs,
                predicted_class, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                json.dumps(audio_probs.tolist() if hasattr(audio_probs, 'tolist') else list(audio_probs)),
                json.dumps(video_probs.tolist() if hasattr(video_probs, 'tolist') else list(video_probs)),
                json.dumps(fused_probs.tolist() if hasattr(fused_probs, 'tolist') else list(fused_probs)),
                predicted_class,
                float(confidence),
            )
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_recent_segments(self, session_id, limit=20):
        """Get the most recent segments for a session."""
        rows = self._conn.execute(
            """SELECT * FROM segments
               WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for key in ("audio_probs", "video_probs", "fused_probs"):
                if d[key]:
                    d[key] = json.loads(d[key])
            results.append(d)
        return list(reversed(results))  # chronological order

    # =============================================
    # Alert storage
    # =============================================
    def add_alert(self, session_id, emotion, severity, confidence,
                  audio_confidence, video_confidence, fused_probs,
                  consecutive_count=1, segment_id=None):
        """Store a triggered alert."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO alerts
               (session_id, timestamp, emotion, severity, confidence,
                audio_confidence, video_confidence, fused_probs, consecutive_count,
                segment_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                now,
                emotion,
                severity,
                float(confidence),
                float(audio_confidence) if audio_confidence is not None else None,
                float(video_confidence) if video_confidence is not None else None,
                json.dumps(fused_probs.tolist() if hasattr(fused_probs, 'tolist') else list(fused_probs)),
                consecutive_count,
                segment_id,
            )
        )
        self._conn.commit()
        return cursor.lastrowid

    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert."""
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE alerts SET acknowledged = 1, acknowledged_at = ? WHERE id = ?",
            (now, alert_id)
        )
        self._conn.commit()

    def get_recent_alerts(self, session_id=None, limit=50):
        """Get recent alerts, optionally filtered by session."""
        if session_id:
            rows = self._conn.execute(
                """SELECT * FROM alerts
                   WHERE session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (session_id, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d["fused_probs"]:
                d["fused_probs"] = json.loads(d["fused_probs"])
            results.append(d)
        return list(reversed(results))

    def get_unacknowledged_alerts(self, session_id=None):
        """Get alerts that haven't been acknowledged."""
        if session_id:
            rows = self._conn.execute(
                """SELECT * FROM alerts
                   WHERE session_id = ? AND acknowledged = 0
                   ORDER BY id DESC""",
                (session_id,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY id DESC"
            ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d["fused_probs"]:
                d["fused_probs"] = json.loads(d["fused_probs"])
            results.append(d)
        return results

    # =============================================
    # Statistics
    # =============================================
    def get_session_stats(self, session_id):
        """Get statistics for a session."""
        total_segments = self._conn.execute(
            "SELECT COUNT(*) FROM segments WHERE session_id = ?",
            (session_id,)
        ).fetchone()[0]

        total_alerts = self._conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE session_id = ?",
            (session_id,)
        ).fetchone()[0]

        alert_counts = {}
        rows = self._conn.execute(
            """SELECT emotion, COUNT(*) as cnt FROM alerts
               WHERE session_id = ?
               GROUP BY emotion""",
            (session_id,)
        ).fetchall()
        for row in rows:
            alert_counts[row["emotion"]] = row["cnt"]

        severity_counts = {}
        rows = self._conn.execute(
            """SELECT severity, COUNT(*) as cnt FROM alerts
               WHERE session_id = ?
               GROUP BY severity""",
            (session_id,)
        ).fetchall()
        for row in rows:
            severity_counts[row["severity"]] = row["cnt"]

        return {
            "total_segments": total_segments,
            "total_alerts": total_alerts,
            "alerts_by_emotion": alert_counts,
            "alerts_by_severity": severity_counts,
        }

    # =============================================
    # Feedback
    # =============================================
    def add_feedback(self, session_id, segment_id, alert_id, report_type,
                     predicted_class, correct_class, audio_saved=False,
                     video_saved=False, audio_path=None, video_path=None,
                     notes=None):
        """Store a feedback report."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO feedback
               (timestamp, session_id, segment_id, alert_id, report_type,
                predicted_class, correct_class, audio_saved, video_saved,
                audio_path, video_path, notes, used_for_training)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                now, session_id, segment_id, alert_id, report_type,
                predicted_class, correct_class,
                1 if audio_saved else 0,
                1 if video_saved else 0,
                audio_path, video_path, notes,
            )
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_feedback(self, limit=50, used_for_training=None):
        """Get feedback reports."""
        if used_for_training is not None:
            rows = self._conn.execute(
                """SELECT * FROM feedback
                   WHERE used_for_training = ?
                   ORDER BY id DESC LIMIT ?""",
                (1 if used_for_training else 0, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM feedback ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_feedback_stats(self):
        """Get feedback statistics: counts by type, confusion matrix, training readiness."""
        # Count by type
        type_counts = {}
        rows = self._conn.execute(
            "SELECT report_type, COUNT(*) as cnt FROM feedback GROUP BY report_type"
        ).fetchall()
        for row in rows:
            type_counts[row["report_type"]] = row["cnt"]

        # Confusion matrix: predicted -> correct
        confusions = []
        rows = self._conn.execute(
            """SELECT predicted_class, correct_class, COUNT(*) as cnt
               FROM feedback
               GROUP BY predicted_class, correct_class
               ORDER BY cnt DESC"""
        ).fetchall()
        for row in rows:
            confusions.append({
                "predicted": row["predicted_class"],
                "correct": row["correct_class"],
                "count": row["cnt"],
            })

        # Ready for training
        ready = self._conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE used_for_training = 0"
        ).fetchone()[0]

        total = self._conn.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

        return {
            "total": total,
            "by_type": type_counts,
            "confusions": confusions[:10],
            "ready_for_training": ready,
        }

    def mark_feedback_used(self, feedback_ids):
        """Mark feedback entries as used for training."""
        if not feedback_ids:
            return
        placeholders = ",".join("?" * len(feedback_ids))
        self._conn.execute(
            f"UPDATE feedback SET used_for_training = 1 WHERE id IN ({placeholders})",
            feedback_ids
        )
        self._conn.commit()

    def update_feedback_files(self, feedback_id, audio_saved, video_saved,
                              audio_path, video_path):
        """Update feedback entry with saved file information."""
        self._conn.execute(
            """UPDATE feedback
               SET audio_saved = ?, video_saved = ?,
                   audio_path = ?, video_path = ?
               WHERE id = ?""",
            (
                1 if audio_saved else 0,
                1 if video_saved else 0,
                audio_path, video_path, feedback_id,
            )
        )
        self._conn.commit()

    def get_segment_by_id(self, segment_id):
        """Get a segment by its ID."""
        row = self._conn.execute(
            "SELECT * FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for key in ("audio_probs", "video_probs", "fused_probs"):
            if d[key]:
                d[key] = json.loads(d[key])
        return d

    def get_alert_by_id(self, alert_id):
        """Get an alert by its ID."""
        row = self._conn.execute(
            "SELECT * FROM alerts WHERE id = ?", (alert_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("fused_probs"):
            d["fused_probs"] = json.loads(d["fused_probs"])
        return d

    def get_segment_near_timestamp(self, timestamp_iso, tolerance_s=10):
        """Find the segment closest to a given ISO timestamp."""
        rows = self._conn.execute(
            """SELECT * FROM segments
               WHERE abs(julianday(timestamp) - julianday(?)) * 86400 < ?
               ORDER BY abs(julianday(timestamp) - julianday(?))
               LIMIT 1""",
            (timestamp_iso, tolerance_s, timestamp_iso)
        ).fetchall()
        if not rows:
            return None
        d = dict(rows[0])
        for key in ("audio_probs", "video_probs", "fused_probs"):
            if d[key]:
                d[key] = json.loads(d[key])
        return d

    def close(self):
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
