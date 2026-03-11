"""
SilentCare - Main Entry Point
===============================
Initializes all services and starts the Flask dashboard.

Usage:
  python run.py                    # Default: audio + video, port 5000
  python run.py --no-audio         # Video only
  python run.py --no-video         # Audio only
  python run.py --port 8080        # Custom port
  python run.py --debug            # Flask debug mode
  python run.py --no-capture       # Dashboard only (no capture, for testing)
"""

import sys
import argparse
import signal
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from silentcare.app.config import FLASK_HOST, FLASK_PORT
from silentcare.core.database import Database
from silentcare.core.capture_service import CaptureService
from silentcare.core.analysis_pipeline import AnalysisPipeline
from silentcare.app.routes import create_app


def main():
    parser = argparse.ArgumentParser(description="SilentCare - Emotional Monitoring System")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help="Flask server port")
    parser.add_argument("--host", type=str, default=FLASK_HOST, help="Flask server host")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio capture")
    parser.add_argument("--no-video", action="store_true", help="Disable video capture")
    parser.add_argument("--no-capture", action="store_true", help="Dashboard only, no capture")
    parser.add_argument("--audio-device", type=int, default=None, help="Audio device index")
    parser.add_argument("--video-device", type=int, default=0, help="Video device index")
    args = parser.parse_args()

    print("=" * 60)
    print("SilentCare - Emotional Monitoring System")
    print("=" * 60)

    # Database
    db_path = PROJECT_DIR / "silentcare.db"
    db = Database(str(db_path))
    print(f"[Init] Database: {db_path}")

    # Capture service
    capture = CaptureService(
        audio_device=args.audio_device,
        video_device=args.video_device,
        enable_audio=not args.no_audio and not args.no_capture,
        enable_video=not args.no_video and not args.no_capture,
    )
    print(f"[Init] Capture: audio={'ON' if capture.enable_audio else 'OFF'}, "
          f"video={'ON' if capture.enable_video else 'OFF'}")

    # Analysis pipeline
    pipeline = AnalysisPipeline(capture_service=capture, db=db)

    # Pre-load models (can take a moment)
    if not args.no_capture:
        print("[Init] Loading ML models...")
        pipeline.load_models()

    # Flask app
    app = create_app(capture_service=capture, pipeline=pipeline, db=db)

    # Graceful shutdown
    def shutdown(signum, frame):
        print("\n[Shutdown] Stopping services...")
        if pipeline.is_running:
            pipeline.stop()
        if capture.is_running:
            capture.stop()
        db.close()
        print("[Shutdown] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start Flask
    display_host = "localhost" if args.host == "0.0.0.0" else args.host
    print(f"\n[Ready] Dashboard: http://{display_host}:{args.port}")
    print("[Ready] Press Ctrl+C to stop.\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
        use_reloader=False,  # Don't reload (would break threads)
    )


if __name__ == "__main__":
    main()
