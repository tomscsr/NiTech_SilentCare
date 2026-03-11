"""
SilentCare - Flask Routes + SSE
==================================
API endpoints:
  GET  /                         - Dashboard page
  GET  /offline                  - Offline test page
  GET  /api/status               - System status (running, models loaded, etc.)
  GET  /api/segments?limit=20    - Recent segment analysis results
  GET  /api/alerts?limit=50      - Recent alerts
  POST /api/alerts/<id>/ack      - Acknowledge an alert
  GET  /api/stats                - Session statistics
  GET  /api/stream               - SSE stream for real-time updates
  POST /api/start                - Start monitoring
  POST /api/stop                 - Stop monitoring
  GET  /api/audio_devices        - List audio input devices
  POST /api/audio_devices        - Change audio input device
  GET  /api/video_feed           - MJPEG live video stream
  GET  /api/audio_data           - Audio waveform samples (JSON)
  POST /api/offline/upload       - Upload MP4 for offline analysis
  GET  /api/offline/info/<id>    - Video metadata
  POST /api/offline/analyze/<id> - Start analysis (realtime or complete)
  GET  /api/offline/status/<id>  - Analysis progress
  GET  /api/offline/results/<id> - Full analysis results
  POST /api/offline/control/<id> - Pause/Resume/Stop
  GET  /api/offline/stream/<id>  - SSE stream for offline realtime mode
"""

import json
import time
import uuid
import shutil
import threading
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

from silentcare.app.config import OFFLINE_UPLOAD_DIR, OFFLINE_MAX_FILE_SIZE_MB

# These will be set by run.py before the app starts
_capture_service = None
_pipeline = None
_db = None
_feedback_service = None

# SSE subscribers
_sse_clients = []
_sse_lock = threading.Lock()

# Offline job tracking: job_id -> {path, extractor, pipeline, ...}
_offline_jobs = {}
_offline_lock = threading.Lock()

# Offline SSE subscribers: job_id -> [queue, ...]
_offline_sse_clients = {}
_offline_sse_lock = threading.Lock()


def create_app(capture_service, pipeline, db):
    """Create and configure the Flask app."""
    global _capture_service, _pipeline, _db, _feedback_service
    _capture_service = capture_service
    _pipeline = pipeline
    _db = db

    from silentcare.core.feedback_service import FeedbackService
    _feedback_service = FeedbackService(db=db, capture_service=capture_service)

    app = Flask(
        __name__,
        template_folder="../../templates",
        static_folder="../../static",
    )

    # Register SSE broadcast callback on the pipeline
    _pipeline.on_segment = _broadcast_segment
    _pipeline.on_alert = _broadcast_alert

    # =============================================
    # Pages
    # =============================================
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/offline")
    def offline_page():
        return render_template("offline.html")

    # Serve uploaded videos for the <video> player
    @app.route("/data/offline/<path:filename>")
    def serve_offline_file(filename):
        project_dir = Path(__file__).resolve().parent.parent.parent
        upload_dir = project_dir / OFFLINE_UPLOAD_DIR
        return send_from_directory(str(upload_dir), filename)

    # =============================================
    # API endpoints
    # =============================================
    @app.route("/api/status")
    def api_status():
        session = _db.get_active_session()
        return jsonify({
            "running": _pipeline.is_running if _pipeline else False,
            "capture_running": _capture_service.is_running if _capture_service else False,
            "session_id": _pipeline.session_id if _pipeline else None,
            "audio_enabled": _capture_service.enable_audio if _capture_service else False,
            "video_enabled": _capture_service.enable_video if _capture_service else False,
            "models_loaded": _pipeline._models_loaded if _pipeline else False,
            "latest_result": _pipeline.latest_result if _pipeline else None,
        })

    @app.route("/api/segments")
    def api_segments():
        limit = request.args.get("limit", 20, type=int)
        session_id = _pipeline.session_id if _pipeline and _pipeline.is_running else None
        if session_id is None:
            return jsonify([])
        segments = _db.get_recent_segments(session_id, limit=limit)
        return jsonify(segments)

    @app.route("/api/alerts")
    def api_alerts():
        limit = request.args.get("limit", 50, type=int)
        session_id = _pipeline.session_id if _pipeline and _pipeline.is_running else None
        alerts = _db.get_recent_alerts(session_id=session_id, limit=limit)
        return jsonify(alerts)

    @app.route("/api/alerts/<int:alert_id>/ack", methods=["POST"])
    def api_acknowledge_alert(alert_id):
        _db.acknowledge_alert(alert_id)
        return jsonify({"status": "ok", "alert_id": alert_id})

    @app.route("/api/stats")
    def api_stats():
        session_id = _pipeline.session_id if _pipeline and _pipeline.is_running else None
        if session_id is None:
            return jsonify({
                "total_segments": 0,
                "total_alerts": 0,
                "alerts_by_emotion": {},
                "alerts_by_severity": {},
            })
        return jsonify(_db.get_session_stats(session_id))

    @app.route("/api/start", methods=["POST"])
    def api_start():
        if _pipeline.is_running:
            return jsonify({"status": "already_running"})

        _capture_service.start()
        _pipeline.start()
        return jsonify({
            "status": "started",
            "session_id": _pipeline.session_id,
        })

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        if not _pipeline.is_running:
            return jsonify({"status": "not_running"})

        _pipeline.stop()
        _capture_service.stop()
        return jsonify({"status": "stopped"})

    # =============================================
    # Audio device selection
    # =============================================
    @app.route("/api/audio_devices")
    def api_audio_devices():
        from silentcare.core.capture_service import CaptureService
        devices = CaptureService.list_audio_devices()
        current = _capture_service.audio_device if _capture_service else None
        return jsonify({"devices": devices, "current": current})

    @app.route("/api/audio_devices", methods=["POST"])
    def api_set_audio_device():
        data = request.get_json()
        device_id = data.get("device_id")
        if device_id is not None and _capture_service:
            _capture_service.set_audio_device(int(device_id))
            return jsonify({"status": "ok", "device_id": device_id})
        return jsonify({"status": "error", "message": "invalid device_id"}), 400

    # =============================================
    # Live video feed (MJPEG stream)
    # =============================================
    @app.route("/api/video_feed")
    def api_video_feed():
        def generate():
            while _capture_service and _capture_service.is_running:
                frame = _capture_service.get_current_frame()
                if frame is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.05)  # ~20fps

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # =============================================
    # Live audio waveform
    # =============================================
    @app.route("/api/audio_data")
    def api_audio_data():
        if not _capture_service:
            return jsonify({"waveform": []})
        audio, sr = _capture_service.get_audio_buffer_copy()
        if audio is None:
            return jsonify({"waveform": []})

        # Downsample to 200 points for display
        n = len(audio)
        step = max(1, n // 200)
        waveform = audio[::step][:200].tolist()

        return jsonify({"waveform": waveform})

    # =============================================
    # Feedback endpoints
    # =============================================
    @app.route("/api/feedback/false_alert", methods=["POST"])
    def api_feedback_false_alert():
        data = request.get_json()
        alert_id = data.get("alert_id")
        correct_class = data.get("correct_class")
        notes = data.get("notes")

        if not alert_id or not correct_class:
            return jsonify({"error": "alert_id and correct_class required"}), 400

        try:
            feedback_id = _feedback_service.report_false_alert(
                alert_id=int(alert_id),
                correct_class=correct_class,
                notes=notes,
            )
            return jsonify({"status": "ok", "feedback_id": feedback_id})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/feedback/missed_detection", methods=["POST"])
    def api_feedback_missed_detection():
        data = request.get_json()
        segment_id = data.get("segment_id")
        correct_class = data.get("correct_class")
        notes = data.get("notes")

        if not segment_id or not correct_class:
            return jsonify({"error": "segment_id and correct_class required"}), 400

        try:
            feedback_id = _feedback_service.report_missed_detection(
                segment_id=int(segment_id),
                correct_class=correct_class,
                notes=notes,
            )
            return jsonify({"status": "ok", "feedback_id": feedback_id})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/feedback/wrong_classification", methods=["POST"])
    def api_feedback_wrong_classification():
        data = request.get_json()
        alert_id = data.get("alert_id")
        segment_id = data.get("segment_id")
        correct_class = data.get("correct_class")
        notes = data.get("notes")

        if not correct_class:
            return jsonify({"error": "correct_class required"}), 400
        if not alert_id and not segment_id:
            return jsonify({"error": "alert_id or segment_id required"}), 400

        try:
            feedback_id = _feedback_service.report_wrong_classification(
                correct_class=correct_class,
                notes=notes,
                alert_id=int(alert_id) if alert_id else None,
                segment_id=int(segment_id) if segment_id else None,
            )
            return jsonify({"status": "ok", "feedback_id": feedback_id})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/feedback")
    def api_feedback_list():
        limit = request.args.get("limit", 50, type=int)
        used = request.args.get("used_for_training", None)
        if used is not None:
            used = int(used) == 1
        feedbacks = _db.get_feedback(limit=limit, used_for_training=used)
        return jsonify(feedbacks)

    @app.route("/api/feedback/stats")
    def api_feedback_stats():
        stats = _db.get_feedback_stats()
        return jsonify(stats)

    @app.route("/api/feedback/export")
    def api_feedback_export():
        import csv
        import io

        feedbacks = _db.get_feedback(limit=10000)

        output = io.StringIO()
        if feedbacks:
            writer = csv.DictWriter(output, fieldnames=feedbacks[0].keys())
            writer.writeheader()
            for fb in feedbacks:
                writer.writerow(fb)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=feedback_export.csv"},
        )

    # =============================================
    # Offline analysis endpoints
    # =============================================
    @app.route("/api/offline/upload", methods=["POST"])
    def api_offline_upload():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Validate extension
        filename = secure_filename(file.filename)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("mp4", "avi", "mov", "mkv", "webm"):
            return jsonify({"error": "Unsupported format. Use MP4, AVI, MOV, MKV or WEBM."}), 400

        # Setup upload directory
        project_dir = Path(__file__).resolve().parent.parent.parent
        upload_dir = project_dir / OFFLINE_UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Clean up old uploaded files
        for old_file in upload_dir.iterdir():
            if old_file.is_file():
                try:
                    old_file.unlink()
                except OSError:
                    pass

        # Generate job ID and save file
        job_id = str(uuid.uuid4())[:8]
        saved_name = f"{job_id}_{filename}"
        save_path = upload_dir / saved_name
        file.save(str(save_path))

        # Check file size
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        if file_size_mb > OFFLINE_MAX_FILE_SIZE_MB:
            save_path.unlink()
            return jsonify({"error": f"File too large ({file_size_mb:.0f}MB > {OFFLINE_MAX_FILE_SIZE_MB}MB)"}), 400

        # Get video info
        from silentcare.core.offline_extractor import OfflineExtractor
        try:
            extractor = OfflineExtractor(str(save_path))
            info = extractor.get_info()
        except Exception as e:
            save_path.unlink()
            return jsonify({"error": f"Cannot read video: {e}"}), 400

        # Store job
        with _offline_lock:
            _offline_jobs[job_id] = {
                "path": str(save_path),
                "filename": saved_name,
                "original_name": filename,
                "extractor": extractor,
                "pipeline": None,
                "info": info,
            }

        return jsonify({
            "status": "ok",
            "job_id": job_id,
            "filename": filename,
            "video_url": f"/data/offline/{saved_name}",
            "info": info,
        })

    @app.route("/api/offline/info/<job_id>")
    def api_offline_info(job_id):
        with _offline_lock:
            job = _offline_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify({
            "job_id": job_id,
            "filename": job["original_name"],
            "info": job["info"],
        })

    @app.route("/api/offline/analyze/<job_id>", methods=["POST"])
    def api_offline_analyze(job_id):
        with _offline_lock:
            job = _offline_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        data = request.get_json() or {}
        mode = data.get("mode", "complete")
        if mode not in ("realtime", "complete"):
            return jsonify({"error": "mode must be 'realtime' or 'complete'"}), 400

        # Check if models are loaded
        if not _pipeline or not _pipeline._models_loaded:
            return jsonify({"error": "ML models not loaded. Start the server without --no-capture or load models first."}), 503

        # Create offline pipeline
        from silentcare.core.offline_pipeline import OfflinePipeline
        offline_pipe = OfflinePipeline(
            pipeline=_pipeline,
            db=_db,
            on_segment=lambda r: _offline_broadcast_segment(job_id, r),
            on_alert=lambda a: _offline_broadcast_alert(job_id, a),
        )

        with _offline_lock:
            # Stop previous pipeline if any
            if job.get("pipeline") and job["pipeline"].status in ("running", "paused"):
                job["pipeline"].stop()
            job["pipeline"] = offline_pipe

        if mode == "complete":
            # Run in background thread so we don't block
            def run_complete():
                try:
                    offline_pipe.analyze_complete(job["path"])
                except Exception as e:
                    print(f"[Offline] Complete analysis error: {e}")

            t = threading.Thread(target=run_complete, daemon=True)
            t.start()
        else:
            offline_pipe.analyze_realtime(job["path"])

        return jsonify({
            "status": "started",
            "mode": mode,
            "job_id": job_id,
            "total_segments": job["info"]["total_segments"],
        })

    @app.route("/api/offline/status/<job_id>")
    def api_offline_status(job_id):
        with _offline_lock:
            job = _offline_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        pipe = job.get("pipeline")
        if not pipe:
            return jsonify({
                "status": "idle",
                "progress": {"current_segment": 0, "total_segments": 0, "percent": 0},
            })

        return jsonify({
            "status": pipe.status,
            "progress": pipe.progress,
            "session_id": pipe.session_id,
        })

    @app.route("/api/offline/results/<job_id>")
    def api_offline_results(job_id):
        with _offline_lock:
            job = _offline_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        pipe = job.get("pipeline")
        if not pipe:
            return jsonify({"error": "Analysis not started"}), 400

        results = pipe.get_results()

        # Convert any numpy arrays for JSON serialization
        clean_segments = []
        for seg in results.get("segments", []):
            clean = {}
            for k, v in seg.items():
                if hasattr(v, "tolist"):
                    clean[k] = v.tolist()
                elif isinstance(v, dict):
                    clean[k] = {
                        dk: dv.tolist() if hasattr(dv, "tolist") else dv
                        for dk, dv in v.items()
                    }
                else:
                    clean[k] = v
            clean_segments.append(clean)

        clean_alerts = []
        for alert in results.get("alerts", []):
            clean = {}
            for k, v in alert.items():
                if hasattr(v, "tolist"):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            clean_alerts.append(clean)

        # Get DB stats if session exists
        stats = {}
        if results.get("session_id"):
            stats = _db.get_session_stats(results["session_id"])

        return jsonify({
            "status": results["status"],
            "progress": results["progress"],
            "session_id": results["session_id"],
            "segments": clean_segments,
            "alerts": clean_alerts,
            "stats": stats,
            "error": results.get("error"),
        })

    @app.route("/api/offline/control/<job_id>", methods=["POST"])
    def api_offline_control(job_id):
        with _offline_lock:
            job = _offline_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404

        pipe = job.get("pipeline")
        if not pipe:
            return jsonify({"error": "Analysis not started"}), 400

        data = request.get_json() or {}
        action = data.get("action")

        if action == "pause":
            pipe.pause()
        elif action == "resume":
            pipe.resume()
        elif action == "stop":
            pipe.stop()
        else:
            return jsonify({"error": "action must be 'pause', 'resume' or 'stop'"}), 400

        return jsonify({"status": "ok", "action": action, "pipeline_status": pipe.status})

    @app.route("/api/offline/stream/<job_id>")
    def api_offline_stream(job_id):
        with _offline_lock:
            job = _offline_jobs.get(job_id)
        if not job:
            return Response("Job not found", status=404)

        def event_stream():
            q = _offline_sse_subscribe(job_id)
            try:
                while True:
                    try:
                        event = q.get(timeout=30)
                        yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                    except Exception:
                        yield f": keepalive\n\n"
                        # Check if analysis is done
                        pipe = job.get("pipeline")
                        if pipe and pipe.status in ("complete", "error", "idle"):
                            yield f"event: done\ndata: {json.dumps({'status': pipe.status})}\n\n"
                            break
            finally:
                _offline_sse_unsubscribe(job_id, q)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # =============================================
    # SSE stream (live)
    # =============================================
    @app.route("/api/stream")
    def api_stream():
        def event_stream():
            q = _sse_subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=30)
                        yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                    except Exception:
                        # Send keepalive
                        yield f": keepalive\n\n"
            finally:
                _sse_unsubscribe(q)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# =============================================
# SSE helpers
# =============================================
def _sse_subscribe():
    import queue
    q = queue.Queue(maxsize=50)
    with _sse_lock:
        _sse_clients.append(q)
    return q


def _sse_unsubscribe(q):
    with _sse_lock:
        if q in _sse_clients:
            _sse_clients.remove(q)


def _sse_broadcast(event_type, data):
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait({"type": event_type, "data": data})
            except Exception:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def _broadcast_segment(result):
    """Called by pipeline for each analyzed segment."""
    # Convert numpy arrays to lists for JSON
    data = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            data[k] = v.tolist()
        elif isinstance(v, dict):
            data[k] = {
                dk: dv.tolist() if hasattr(dv, "tolist") else dv
                for dk, dv in v.items()
            }
        else:
            data[k] = v
    _sse_broadcast("segment", data)


def _broadcast_alert(alert):
    """Called by pipeline when an alert fires."""
    data = {}
    for k, v in alert.items():
        if hasattr(v, "tolist"):
            data[k] = v.tolist()
        else:
            data[k] = v
    _sse_broadcast("alert", data)


# =============================================
# Offline SSE helpers
# =============================================
def _offline_sse_subscribe(job_id):
    import queue
    q = queue.Queue(maxsize=100)
    with _offline_sse_lock:
        if job_id not in _offline_sse_clients:
            _offline_sse_clients[job_id] = []
        _offline_sse_clients[job_id].append(q)
    return q


def _offline_sse_unsubscribe(job_id, q):
    with _offline_sse_lock:
        if job_id in _offline_sse_clients:
            if q in _offline_sse_clients[job_id]:
                _offline_sse_clients[job_id].remove(q)


def _offline_sse_broadcast(job_id, event_type, data):
    with _offline_sse_lock:
        clients = _offline_sse_clients.get(job_id, [])
        dead = []
        for q in clients:
            try:
                q.put_nowait({"type": event_type, "data": data})
            except Exception:
                dead.append(q)
        for q in dead:
            clients.remove(q)


def _offline_broadcast_segment(job_id, result):
    """Called by offline pipeline for each analyzed segment."""
    data = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            data[k] = v.tolist()
        elif isinstance(v, dict):
            data[k] = {
                dk: dv.tolist() if hasattr(dv, "tolist") else dv
                for dk, dv in v.items()
            }
        else:
            data[k] = v
    _offline_sse_broadcast(job_id, "segment", data)


def _offline_broadcast_alert(job_id, alert):
    """Called by offline pipeline when an alert fires."""
    data = {}
    for k, v in alert.items():
        if hasattr(v, "tolist"):
            data[k] = v.tolist()
        else:
            data[k] = v
    _offline_sse_broadcast(job_id, "alert", data)
