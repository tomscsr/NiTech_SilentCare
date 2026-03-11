# SilentCare

Real-time emotional monitoring system for non-verbal individuals.
Combined audio + video analysis with multimodal fusion, intelligent alerts, and a web dashboard.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [ML Pipeline](#ml-pipeline)
7. [Multimodal Fusion](#multimodal-fusion)
8. [Alert System](#alert-system)
9. [Dashboard](#dashboard)
10. [Human-in-the-Loop Feedback](#human-in-the-loop-feedback)
11. [Database](#database)
12. [Configuration](#configuration)
13. [Datasets and Training](#datasets-and-training)
14. [Tests](#tests)
15. [Known Limitations](#known-limitations)

---

## Overview

SilentCare is a research prototype designed to detect the emotional states of people who cannot communicate verbally (dependent elderly, non-verbal patients, infants). The system continuously captures ambient audio and webcam video, analyses each segment through two independent deep learning models, fuses predictions, and triggers alerts when persistent distress is detected.

### 4 Emotional Classes

| Class     | Index | Description                              |
|-----------|-------|------------------------------------------|
| DISTRESS  | 0     | Distress (crying, fear, disgust, sadness) |
| ANGRY     | 1     | Anger                                    |
| ALERT     | 2     | Alert (surprise, agitation)              |
| CALM      | 3     | Calm (neutral, happy)                    |

---

## Architecture

```
                         run.py
                           |
              +------------+------------+
              |                         |
     CaptureService              AnalysisPipeline
     (capture_service.py)        (analysis_pipeline.py)
       |           |                |          |
   Audio thread  Video thread   AudioModel  VideoModel
   (sounddevice) (OpenCV)       (YAMNet+    (ResNet50)
                                 Keras)
       |           |                |          |
       +-----+-----+          Fusion (weighted avg
             |                  + agreement boost)
        Segment Queue                  |
             |                   AlertManager
             +-----> Pipeline -----> (alert_manager.py)
                       |                  |
                    Database           SSE Broadcast
                  (SQLite WAL)            |
                       |            Flask Routes
                       +--------> (routes.py)
                                      |
                                  Dashboard
                                 (index.html)
```

---

## Project Structure

```
SilentCare/
|-- run.py                          # Main entry point
|-- requirements.txt                # Python dependencies
|-- silentcare.db                   # SQLite database (created on first launch)
|
|-- silentcare/                     # Main package
|   |-- app/
|   |   |-- config.py               # Centralised configuration (thresholds, weights, paths)
|   |   |-- routes.py               # API routes + SSE + MJPEG
|   |
|   |-- core/
|   |   |-- capture_service.py      # Threaded audio + video capture with circular buffer
|   |   |-- analysis_pipeline.py    # Analysis pipeline (inference + fusion)
|   |   |-- alert_manager.py        # Anti-false-positive logic and alerts
|   |   |-- database.py             # Thread-safe SQLite layer
|   |   |-- feedback_service.py     # Human-in-the-loop feedback service
|   |   |-- offline_extractor.py    # Frame + audio extraction from uploaded videos
|   |   |-- offline_pipeline.py     # Offline analysis (realtime / complete modes)
|   |
|   |-- ml/
|   |   |-- audio_model.py          # YAMNet + Keras classification head
|   |   |-- video_model.py          # ResNet50 + face detection (ViT available via use_vit=True)
|   |
|   |-- training/
|       |-- train_audio.py          # Audio model training script
|       |-- train_video.py          # Video model training script (ResNet50)
|
|-- model/
|   |-- Audio_SilentCare_model.h5   # Audio classification head (Keras)
|   |-- audio_silentcare_classes.npy
|   |-- Video_SilentCare_model.pth  # Video backbone (ResNet50, production model)
|   |-- Video_SilentCare_temporal.pth
|
|-- templates/
|   |-- index.html                  # Live monitoring dashboard
|   |-- offline.html                # Offline video analysis page
|
|-- static/
|   |-- css/dashboard.css           # Dashboard styles (dark theme)
|   |-- js/dashboard.js             # Client logic (SSE, Chart.js, waveform)
|
|-- scripts/
|   |-- prepare_audio_dataset.py    # Audio dataset preparation
|   |-- finetune_from_feedback.py   # Fine-tuning from human feedback
|
|-- tests/                          # Test suite (132 tests)
|
|-- data/                           # Datasets (not included in repo)
|   |-- audio_dataset/              # 1876 audio files (4 classes)
|   |-- feedback/                   # Human feedback data (generated at runtime)
```

---

## Installation

### Prerequisites

- Python 3.9+
- Webcam (IR cameras like Windows Hello are supported)
- Microphone (optional -- the system works in video-only mode)

### Setup

```bash
cd SilentCare
pip install -r requirements.txt

# Model weights are included in the repository. No additional download required.
```

### Main Dependencies

| Library          | Usage                                          |
|------------------|------------------------------------------------|
| tensorflow       | Audio model (YAMNet + Keras)                   |
| tensorflow-hub   | YAMNet embedding extraction                    |
| torch            | Video model (ResNet50)                         |
| transformers     | ViT evaluation only (optional, not required for production) |
| opencv-python    | Video capture + face detection                 |
| sounddevice      | Real-time audio capture                        |
| librosa          | Audio processing (resampling)                  |
| noisereduce      | Audio noise reduction (spectral gating)         |
| flask            | Web server + API                               |
| numpy            | Numerical computation                          |
| chart.js (CDN)   | Real-time charts (dashboard)                   |

---

## Usage

### Launch

```bash
python run.py                        # Full mode (audio + video, port 5000)
python run.py --no-audio             # Video only
python run.py --no-video             # Audio only
python run.py --port 8080            # Custom port
python run.py --no-capture           # Dashboard only (no capture, for testing)
python run.py --audio-device 2       # Specific audio device
python run.py --video-device 1       # Specific camera (default: 0)
python run.py --debug                # Flask debug mode
```

### Starting the Monitor

1. Run `python run.py`
2. Open `http://localhost:5000` in a browser
3. Click **START** to begin monitoring
4. Results appear in real time on the dashboard
5. Click **STOP** to end the session

### Typical Console Output

```
============================================================
SilentCare - Emotional Monitoring System
============================================================
[Init] Database: .../silentcare.db
[Init] Capture: audio=ON, video=ON
[Init] Loading ML models...
[Pipeline] Audio model loaded.
[VideoModel] Loading ResNet50 from model/Video_SilentCare_model.pth
[VideoModel] Model loaded successfully.
[Pipeline] Video model loaded (ResNet50).

[Ready] Dashboard: http://localhost:5000
[Ready] Press Ctrl+C to stop.
```

---

## ML Pipeline

### Audio Model: YAMNet + Classification Head

**File**: `silentcare/ml/audio_model.py`

**Architecture**:
1. **YAMNet** (TensorFlow Hub, frozen) extracts 1024-dim embeddings every 0.48s
2. **Temporal pooling**: Mean + Max + Std -> 3072-dim feature vector
3. **Classification head** (Keras):
   - Dense(256, ReLU) -> BatchNorm -> Dropout(0.4)
   - Dense(128, ReLU) -> Dropout(0.3)
   - Dense(4, Softmax)

**Input**: mono waveform (any sample rate, internally resampled to 16 kHz for YAMNet)

**Output**: probabilities over 4 classes [DISTRESS, ANGRY, ALERT, CALM]

**Silence gate**: if audio RMS < 0.01, the segment is ignored (the model is trained on vocal expressions, not silence).

### Audio Preprocessing (AudioPreprocessor)

**File**: `silentcare/ml/audio_preprocessor.py`

Before the waveform reaches YAMNet, a preprocessing stage handles three tasks:

1. **Noise reduction**: spectral gating via `noisereduce` (prop_decrease=0.75) using the first 0.5s of each segment as the noise profile. Reduces stationary environmental noise (traffic, HVAC, background music) without degrading vocal quality.
2. **Voice Activity Detection (VAD)**: energy-based frame analysis (20ms frames, dynamic RMS threshold). If fewer than 15% of frames are voiced, the segment is flagged as non-voice and audio inference is skipped, preventing environmental sounds (sirens, alarms) from being misclassified as emotional vocalisations.
3. **Normalisation**: peak normalisation to 0.95 to ensure consistent input amplitude across different microphones and environments.

All three steps are individually configurable via `config.py` (`AUDIO_NOISE_REDUCTION`, `AUDIO_VAD_ENABLED`).

### Video Model: ResNet50

**File**: `silentcare/ml/video_model.py`

**Architecture**:
1. **Face detection**: Haar cascade (OpenCV) on CLAHE-enhanced grayscale
2. **Enhancement**: LAB histogram equalization for dark / IR camera images
3. **Crop**: largest detected face + 20% margin
4. **Classification**: ResNet50 backbone (ImageNet pretrained) with custom head:
   - Linear(2048 -> 512, ReLU) -> Dropout(0.5) -> Linear(512 -> 4, Softmax)

**Input**: 224x224 RGB face crop, ImageNet-normalised

**Output**: probabilities over 4 SilentCare classes directly (no FER remapping at inference)

The FER-to-SilentCare mapping (7 emotions -> 4 classes) was used only during **training data preparation** from RAF-DB. At inference time, ResNet50 outputs 4-class probabilities directly.

A ViT model (`trpakov/vit-face-expression`) is accessible via `use_vit=True` for comparison purposes but is not used in production.

**Multi-frame averaging**: for each segment, 15 frames are analysed individually. Predictions are averaged with linear weighting (more recent frames weighted higher, from 0.5 to 1.0).

**No face detected**: if no face is found in any frame, the video modality is ignored and fusion falls back to 100% audio.

### IR Camera / Low-Light Support

Face detection uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on grayscale before the Haar cascade, enabling detection even on infrared cameras (Windows Hello) or in low-light conditions. The face crop is further enhanced via LAB histogram equalization before sending to ResNet50.

---

## Multimodal Fusion

**File**: `silentcare/core/analysis_pipeline.py` -- method `_fuse_predictions()`

The fusion combines audio and video predictions in 3 steps:

### Step 1: Adaptive Weighted Average

Default weights: **30% audio, 70% video**.

If either modality's peak confidence is below `VIDEO_MIN_CONFIDENCE` (0.45), its weight is reduced proportionally:

```
If video_conf < 0.45:
    video_weight = 0.70 * (video_conf / 0.45)
    audio_weight = 1.0 - video_weight

If audio_conf < 0.45:
    audio_weight = 0.30 * (audio_conf / 0.45)
    video_weight = 1.0 - audio_weight
```

This protects against domain shift: when one model is uncertain, the other takes over.

### Step 2: Agreement Boost

If both modalities agree on the same dominant class, the fused probability for that class is multiplied by **1.3x**, then renormalised.

### Step 3: Uncertainty Threshold

If the maximum fused probability is below **0.30**, the result defaults to CALM (too uncertain to act).

### Degraded Cases

| Situation               | Behaviour                          |
|-------------------------|------------------------------------|
| Audio + Video present   | Adaptive weighted fusion           |
| Audio only (no face)    | 100% audio                         |
| Video only (silence)    | 100% video                         |
| Neither audio nor video | Default to CALM [0, 0, 0, 1]      |

---

## Alert System

**File**: `silentcare/core/alert_manager.py`

The alert system includes several anti-false-positive mechanisms:

### Confidence Thresholds

| Class    | Confidence Threshold |
|----------|---------------------|
| DISTRESS | 0.60                |
| ANGRY    | 0.50                |
| ALERT    | 0.55                |
| CALM     | No alert            |

### Consecutive Segments

An alert only fires if the same non-CALM class is detected for **3 consecutive segments** (~9-15 seconds depending on segmentation rate).

### DISTRESS-CALM Margin

For DISTRESS specifically, confidence must exceed CALM probability by at least **0.25**. This prevents false positives when the face is neutral (domain shift between posed photos and webcam).

### Cooldown

After an alert fires, the same emotion cannot re-trigger for **30 seconds**.

### Severity Levels

| Severity | Condition                    |
|----------|------------------------------|
| LOW      | 1 consecutive segment        |
| MEDIUM   | 2 consecutive segments       |
| HIGH     | 3+ consecutive segments      |

---

## Dashboard

**Files**: `templates/index.html`, `static/css/dashboard.css`, `static/js/dashboard.js`

The dashboard provides a real-time monitoring interface with: current emotion display (icon + class + confidence), probability bars for each modality (audio/video/fused), a live MJPEG video feed with CLAHE enhancement, an audio waveform oscilloscope, a Chart.js history of the last 60 data points, an alert feed with severity/acknowledgment/reporting, and a system quality panel with feedback statistics and CSV export.

### REST API

| Method | Endpoint                            | Description                         |
|--------|--------------------------------------|-------------------------------------|
| GET    | `/`                                  | Dashboard page                      |
| GET    | `/api/status`                        | System state (running, models)      |
| GET    | `/api/segments?limit=20`             | Recent analysed segments            |
| GET    | `/api/alerts?limit=50`               | Recent alerts                       |
| POST   | `/api/alerts/<id>/ack`               | Acknowledge an alert                |
| GET    | `/api/stats`                         | Session statistics                  |
| POST   | `/api/start`                         | Start monitoring                    |
| POST   | `/api/stop`                          | Stop monitoring                     |
| GET    | `/api/audio_devices`                 | List audio devices                  |
| POST   | `/api/audio_devices`                 | Switch microphone `{device_id: int}`|
| GET    | `/api/video_feed`                    | Live MJPEG stream                   |
| GET    | `/api/audio_data`                    | Audio samples (200-point JSON)      |
| GET    | `/api/stream`                        | SSE stream (events: segment, alert) |
| POST   | `/api/feedback/false_alert`          | Report a false alert                |
| POST   | `/api/feedback/missed_detection`     | Report a missed detection           |
| POST   | `/api/feedback/wrong_classification` | Report a wrong classification       |
| GET    | `/api/feedback`                      | List feedback entries               |
| GET    | `/api/feedback/stats`                | Feedback statistics                 |
| GET    | `/api/feedback/export`               | Export feedback as CSV              |
|        |                                      |                                     |
| GET    | `/offline`                           | Offline analysis page               |
| POST   | `/api/offline/upload`                | Upload MP4 for analysis             |
| GET    | `/api/offline/info/<job_id>`         | Video metadata                      |
| POST   | `/api/offline/analyze/<job_id>`      | Start offline analysis              |
| GET    | `/api/offline/status/<job_id>`       | Analysis progress                   |
| GET    | `/api/offline/results/<job_id>`      | Full analysis results               |
| POST   | `/api/offline/control/<job_id>`      | Pause / Resume / Stop               |
| GET    | `/api/offline/stream/<job_id>`       | SSE stream (offline realtime mode)  |

---

## Human-in-the-Loop Feedback

**Files**: `silentcare/core/feedback_service.py`, `scripts/finetune_from_feedback.py`

Operators can correct system errors directly from the dashboard, accumulating labelled training data for model improvement.

### Report Types

| Type                   | Description                                         | Source          |
|------------------------|-----------------------------------------------------|-----------------|
| FALSE_ALERT            | The system triggered an alert for nothing           | Alert           |
| MISSED_DETECTION       | The system failed to detect a real emotion          | Segment         |
| WRONG_CLASSIFICATION   | Correct detection but wrong class                   | Alert or Segment|

### Fine-Tuning from Feedback

```bash
# Display a report of available feedback
python scripts/finetune_from_feedback.py

# Fine-tune audio (Keras head on YAMNet)
python scripts/finetune_from_feedback.py --audio-only --confirm

# Fine-tune video (ResNet50 4-class head)
python scripts/finetune_from_feedback.py --video-only --confirm

# Both models + mix with original data
python scripts/finetune_from_feedback.py --confirm --mix-original
```

**Safeguards**: refuses if < 10 feedback samples or if any class has 0 samples; automatic model backup before overwriting; before/after accuracy and per-class F1 report.

**Parameters**:
- Audio: lr=1e-4, early stopping patience=5, stratified 80/20 split
- Video: lr=1e-5, ResNet50 backbone partially frozen, 4-class head trained

---

## Database

**File**: `silentcare/core/database.py`

SQLite with WAL mode (Write-Ahead Logging) and thread-local connections in autocommit for inter-thread visibility.

4 tables: **sessions**, **segments** (audio/video/fused probability vectors), **alerts** (severity, acknowledgment), **feedback** (operator corrections with saved media).

---

## Configuration

**File**: `silentcare/app/config.py`

All parameters are centralised in this single file.

### Fusion

| Parameter              | Value | Description                                  |
|------------------------|-------|----------------------------------------------|
| AUDIO_WEIGHT           | 0.30  | Audio weight in fusion                       |
| VIDEO_WEIGHT           | 0.70  | Video weight in fusion                       |
| AGREEMENT_BOOST        | 1.3   | Multiplier when audio and video agree        |
| UNCERTAINTY_THRESHOLD  | 0.30  | Below this, default to CALM                  |
| VIDEO_MIN_CONFIDENCE   | 0.45  | Below this, video weight is reduced          |
| AUDIO_MIN_RMS          | 0.01  | Below this, audio is treated as silence      |

### Alerts

| Parameter                      | Value | Description                              |
|--------------------------------|-------|------------------------------------------|
| THRESHOLD_DISTRESS             | 0.60  | Confidence threshold for DISTRESS        |
| THRESHOLD_ANGRY                | 0.50  | Confidence threshold for ANGRY           |
| THRESHOLD_ALERT                | 0.55  | Confidence threshold for ALERT           |
| CONSECUTIVE_SEGMENTS_FOR_ALERT | 3     | Consecutive segments required             |
| ALERT_COOLDOWN_SECONDS         | 30    | Cooldown between same-type alerts        |
| DISTRESS_CALM_MARGIN           | 0.25  | Minimum gap DISTRESS - CALM              |

### Capture

| Parameter              | Value | Description                               |
|------------------------|-------|-------------------------------------------|
| AUDIO_SAMPLE_RATE      | 22050 | Sampling rate (Hz)                        |
| SEGMENT_DURATION_S     | 5     | Segment duration (seconds)                |
| SEGMENT_OVERLAP_S      | 2     | Overlap between segments                  |
| SEGMENT_STEP_S         | 3     | Interval between segments (5 - 2)         |
| VIDEO_FPS_TARGET       | 15    | Target video capture FPS                  |
| VIDEO_FRAMES_PER_SEGMENT | 15  | Frames extracted per segment              |

### Model Paths

| Parameter          | Value                               |
|--------------------|-------------------------------------|
| AUDIO_MODEL_PATH   | `model/Audio_SilentCare_model.h5`   |
| AUDIO_CLASSES_PATH | `model/audio_silentcare_classes.npy` |
| VIDEO_MODEL_PATH   | `model/Video_SilentCare_model.pth`  |

The production video model is ResNet50 (`Video_SilentCare_model.pth`). The ViT HuggingFace model (`trpakov/vit-face-expression`) was evaluated but is not used in production (FER-2013 domain mismatch with RAF-DB, 115 ms inference exceeds the 100 ms real-time threshold).

---

## Datasets and Training

### Audio Dataset

**Preparation**: `scripts/prepare_audio_dataset.py`

| Source        | Classes used                                                    | Description             |
|---------------|-----------------------------------------------------------------|-------------------------|
| Donate-a-Cry  | DISTRESS                                                        | Infant crying sounds    |
| ESC-50        | DISTRESS (crying_baby), CALM (laughing)                         | Environmental sounds    |
| VIVAE         | DISTRESS (Fear), ANGRY, ALERT (Surprise), CALM (Happy)         | Vocal expressions       |

**Total**: 1876 files across 4 classes (minimum 400 per class).

### Video Dataset

| Source | Description                                                  |
|--------|--------------------------------------------------------------|
| RAF-DB | Real-world Affective Faces Database (aligned photos)         |

**RAF-DB -> SilentCare mapping** (used for training data preparation):

| RAF-DB Label | Code | SilentCare Class |
|--------------|------|------------------|
| Surprise     | 1    | ALERT            |
| Fear         | 2    | DISTRESS         |
| Disgust      | 3    | DISTRESS         |
| Happy        | 4    | CALM             |
| Sad          | 5    | DISTRESS         |
| Angry        | 6    | ANGRY            |
| Neutral      | 7    | CALM             |

### Training Results

| Model          | Accuracy | F1 Macro | DISTRESS F1 |
|----------------|----------|----------|-------------|
| Audio          | 71.0%    | 0.699    | 0.838       |
| Video (ResNet50)| 76.2%   | 0.766    | --          |
| **Fusion**     | **89.6%**| **0.895**| --          |

---

## Tests

### Running the Tests

```bash
cd SilentCare
python -m pytest tests/ -v
```

### Test Suite (132 tests)

| File                | Tests | Description                                        |
|---------------------|-------|----------------------------------------------------|
| test_step4.py       | 21    | Unit tests: Database, AlertManager, Fusion         |
| test_integration.py | 9     | Integration: end-to-end pipeline with mocks        |
| test_flask_api.py   | 16    | API: all Flask endpoints                           |
| test_stability.py   | 4     | Stability: 300 segments, memory, concurrent writes |
| test_feedback.py    | 34    | Feedback: DB CRUD, buffer, service, WAV, API       |
| test_offline.py     | 33    | Offline: extraction, pipeline, API endpoints       |
| test_audio_preprocessor.py | 15 | Preprocessor: noise reduction, VAD, normalisation |
| **Total**           |**132**| **All passing**                                    |

---

## Known Limitations

### IR Camera / Windows Hello

The default camera (device 0) may be an infrared camera on modern laptops. Images are very dark (mean ~14/255). SilentCare compensates with CLAHE on grayscale for face detection and LAB histogram equalization on the face crop before classification.

### Video Domain Shift

ResNet50 was trained on RAF-DB (posed studio photos) and may generalise less well to real-time webcam conditions with subtle, spontaneous expressions. The anti-false-positive mechanisms (DISTRESS-CALM margin, consecutive segments, cooldown) mitigate this gap.

### Audio Silence

The audio model is trained on vocal expressions (crying, screaming, laughing). On silence or ambient noise, predictions are unreliable. The RMS gate (threshold 0.01) disables audio analysis when no vocal signal is present.

### DISTRESS Over-Representation

3 of 7 FER emotions (fear, disgust, sad) map to DISTRESS. The DISTRESS-CALM margin of 0.25 and the high confidence threshold of 0.60 compensate for this bias.
