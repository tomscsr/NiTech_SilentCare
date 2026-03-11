"""
SilentCare - Video Pipeline Diagnostic Tool
=============================================
Captures 30 seconds of webcam feed, runs face detection + ResNet50
inference on sampled frames, and saves diagnostic images + a summary
report.

Usage:
    python scripts/diagnose_video.py
    python scripts/diagnose_video.py --duration 60 --device 1
    python scripts/diagnose_video.py --output diagnostic_run2
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="SilentCare video pipeline diagnostics")
    parser.add_argument("--duration", type=int, default=30, help="Capture duration in seconds (default: 30)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--output", type=str, default="diagnostic", help="Output directory name (default: diagnostic)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sampled frames (default: 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SilentCare - Video Pipeline Diagnostic")
    print("=" * 60)
    print(f"Duration:    {args.duration}s")
    print(f"Camera:      device {args.device}")
    print(f"Interval:    {args.interval}s")
    print(f"Output:      {output_dir}")
    print()

    # --- Load model ---
    print("[1/3] Loading ResNet50 model...")
    from silentcare.ml.video_model import VideoModel
    model = VideoModel()
    print()

    # --- Open camera ---
    print("[2/3] Opening camera...")
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera device {args.device}")
        sys.exit(1)

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera opened: {cap_w}x{cap_h} @ {cap_fps:.0f} fps")
    print()

    # --- Capture loop ---
    print(f"[3/3] Capturing for {args.duration} seconds (sampling every {args.interval}s)...")
    print()

    BRIGHTNESS_THRESHOLD = 80
    frame_logs = []
    frame_count = 0
    start_time = time.time()
    last_sample_time = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # Sample at the specified interval
            if elapsed - last_sample_time < args.interval:
                continue
            last_sample_time = elapsed

            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_enhanced = model._clahe.apply(gray)
            faces = model.face_cascade.detectMultiScale(
                gray_enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
            )

            if len(faces) == 0:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  [{ts}] Frame {frame_count:03d}: no face detected (skipped)")
                frame_count += 1
                continue

            # Pick largest face
            areas = [w * h for (x, y, w, h) in faces]
            idx = int(np.argmax(areas))
            x, y, w, h = faces[idx]

            # Crop with 20% margin
            margin = 0.2
            mx, my = int(w * margin), int(h * margin)
            fh, fw = frame.shape[:2]
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(fw, x + w + mx)
            y2 = min(fh, y + h + my)
            raw_crop = frame[y1:y2, x1:x2].copy()

            # Check if CLAHE enhancement will be triggered
            crop_gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
            raw_mean = float(crop_gray.mean())
            raw_std = float(crop_gray.std())
            clahe_triggered = raw_mean <= BRIGHTNESS_THRESHOLD

            # Enhance (same logic as video_model._enhance_face)
            if clahe_triggered:
                lab = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.equalizeHist(l)
                lab = cv2.merge([l, a, b])
                processed_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                processed_crop = raw_crop.copy()

            processed_gray = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2GRAY)
            proc_mean = float(processed_gray.mean())
            proc_std = float(processed_gray.std())

            # Save crops
            raw_path = output_dir / f"raw_crop_{frame_count:03d}.jpg"
            proc_path = output_dir / f"processed_crop_{frame_count:03d}.jpg"
            cv2.imwrite(str(raw_path), raw_crop)
            cv2.imwrite(str(proc_path), processed_crop)

            # Run inference
            probs = model._classify_face(raw_crop)
            predicted_idx = int(np.argmax(probs))
            predicted_class = model.classes[predicted_idx]
            confidence = float(probs[predicted_idx])

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{ts}] Frame {frame_count:03d}: {predicted_class} ({confidence:.3f})"
                  f"  [D={probs[0]:.3f} A={probs[1]:.3f} L={probs[2]:.3f} C={probs[3]:.3f}]"
                  f"  raw_mean={raw_mean:.1f} proc_mean={proc_mean:.1f}"
                  f"  CLAHE={'YES' if clahe_triggered else 'no'}")

            frame_logs.append({
                "frame": frame_count,
                "timestamp": elapsed,
                "probs": probs.copy(),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "proc_mean": proc_mean,
                "proc_std": proc_std,
                "clahe_triggered": clahe_triggered,
            })

            frame_count += 1

    except KeyboardInterrupt:
        print("\n  Capture interrupted by user.")
    finally:
        cap.release()

    # --- Summary report ---
    print()
    print("=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    total_sampled = frame_count
    faces_detected = len(frame_logs)

    print(f"Total frames sampled:   {total_sampled}")
    print(f"Faces detected:         {faces_detected}")
    print(f"Face detection rate:    {faces_detected / max(total_sampled, 1) * 100:.1f}%")
    print()

    if faces_detected == 0:
        print("No faces detected. Cannot produce prediction summary.")
        print("Possible causes:")
        print("  - Camera is IR-only (check raw frames in output directory)")
        print("  - No person in front of camera")
        print(f"  - Face too small or too far from camera")
        print()
        print(f"Output directory: {output_dir}")
        return

    all_probs = np.array([log["probs"] for log in frame_logs])
    classes = ["DISTRESS", "ANGRY", "ALERT", "CALM"]

    # Per-class statistics
    avg_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    print("Per-class probability statistics:")
    print(f"  {'Class':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, cls in enumerate(classes):
        print(f"  {cls:<10} {avg_probs[i]:>8.4f} {std_probs[i]:>8.4f}"
              f" {np.min(all_probs[:, i]):>8.4f} {np.max(all_probs[:, i]):>8.4f}")
    print()

    # Top prediction distribution
    top_preds = [log["predicted_class"] for log in frame_logs]
    print("Top prediction distribution:")
    for cls in classes:
        count = top_preds.count(cls)
        pct = count / faces_detected * 100
        bar = "#" * int(pct / 2)
        print(f"  {cls:<10} {count:>4} ({pct:>5.1f}%)  {bar}")
    print()

    # Pixel statistics
    raw_means = [log["raw_mean"] for log in frame_logs]
    proc_means = [log["proc_mean"] for log in frame_logs]
    clahe_count = sum(1 for log in frame_logs if log["clahe_triggered"])

    print("Pixel statistics (grayscale 0-255):")
    print(f"  Avg raw pixel mean:       {np.mean(raw_means):.1f}")
    print(f"  Avg processed pixel mean: {np.mean(proc_means):.1f}")
    print(f"  CLAHE triggered:          {clahe_count}/{faces_detected}"
          f" ({clahe_count / faces_detected * 100:.1f}%)")
    print()

    # Stability indicator
    max_std = np.max(std_probs)
    if max_std > 0.20:
        stability = "UNSTABLE (std > 0.20 -- predictions fluctuate significantly)"
    elif max_std > 0.10:
        stability = "MODERATE (std 0.10-0.20 -- some variation)"
    else:
        stability = "STABLE (std < 0.10 -- consistent predictions)"
    print(f"Prediction stability:   {stability}")
    print()

    # Save log to text file
    log_path = output_dir / "diagnostic_log.txt"
    with open(log_path, "w") as f:
        f.write(f"SilentCare Video Diagnostic Log\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Duration: {args.duration}s, Device: {args.device}\n")
        f.write(f"Frames sampled: {total_sampled}, Faces detected: {faces_detected}\n\n")
        f.write(f"{'Frame':>5} {'Time':>6} {'Class':<10} {'Conf':>6}"
                f" {'DISTR':>7} {'ANGRY':>7} {'ALERT':>7} {'CALM':>7}"
                f" {'RawMn':>6} {'PrcMn':>6} {'CLAHE':>5}\n")
        f.write("-" * 85 + "\n")
        for log in frame_logs:
            p = log["probs"]
            f.write(f"{log['frame']:>5} {log['timestamp']:>6.1f} {log['predicted_class']:<10}"
                    f" {log['confidence']:>6.3f}"
                    f" {p[0]:>7.4f} {p[1]:>7.4f} {p[2]:>7.4f} {p[3]:>7.4f}"
                    f" {log['raw_mean']:>6.1f} {log['proc_mean']:>6.1f}"
                    f" {'YES' if log['clahe_triggered'] else 'no':>5}\n")

    print(f"Detailed log saved to:  {log_path}")
    print(f"Face crops saved to:    {output_dir}")
    print(f"Total files:            {faces_detected * 2} images + 1 log")


if __name__ == "__main__":
    main()
