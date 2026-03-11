"""
SilentCare - Configuration and constants.
All thresholds and parameters are centralized here.
"""

# ============================================
# Emotion classes
# ============================================
EMOTION_CLASSES = ["DISTRESS", "ANGRY", "ALERT", "CALM"]
NUM_CLASSES = 4

# Class indices
CLASS_DISTRESS = 0
CLASS_ANGRY = 1
CLASS_ALERT = 2
CLASS_CALM = 3

# ============================================
# Fusion weights
# ============================================
AUDIO_WEIGHT = 0.30
VIDEO_WEIGHT = 0.70
AGREEMENT_BOOST = 1.3
UNCERTAINTY_THRESHOLD = 0.30
VIDEO_MIN_CONFIDENCE = 0.45  # Below this, video weight is reduced (domain shift protection)
AUDIO_MIN_RMS = 0.01         # Below this RMS, audio is silence -> skip audio analysis

# ============================================
# Alert thresholds
# ============================================
THRESHOLD_DISTRESS = 0.60
THRESHOLD_ANGRY = 0.50
THRESHOLD_ALERT = 0.55

ALERT_THRESHOLDS = {
    "DISTRESS": THRESHOLD_DISTRESS,
    "ANGRY": THRESHOLD_ANGRY,
    "ALERT": THRESHOLD_ALERT,
}

# ============================================
# Anti-false-positive logic
# ============================================
CONSECUTIVE_SEGMENTS_FOR_ALERT = 3  # Must detect on N consecutive segments
ALERT_COOLDOWN_SECONDS = 30         # Seconds between same-type alerts
SEVERITY_WINDOW = 3                 # Number of segments for severity calc
DISTRESS_CALM_MARGIN = 0.25        # DISTRESS must exceed CALM by this margin

# Severity levels
SEVERITY_LOW = "LOW"        # 1 segment
SEVERITY_MEDIUM = "MEDIUM"  # 2 segments
SEVERITY_HIGH = "HIGH"      # 3+ consecutive segments

# ============================================
# Capture settings
# ============================================
AUDIO_SAMPLE_RATE = 22050
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"

SEGMENT_DURATION_S = 5     # Segment length in seconds
SEGMENT_OVERLAP_S = 2      # Overlap between segments
SEGMENT_STEP_S = SEGMENT_DURATION_S - SEGMENT_OVERLAP_S  # 3s between segments

VIDEO_FPS_TARGET = 15      # Target frames per second for capture
VIDEO_FRAMES_PER_SEGMENT = 15  # Frames extracted per 10s segment

# ============================================
# Model paths
# ============================================
AUDIO_MODEL_PATH = "model/Audio_SilentCare_model.h5"
AUDIO_CLASSES_PATH = "model/audio_silentcare_classes.npy"
VIDEO_MODEL_PATH = "model/Video_SilentCare_model.pth"

# ============================================
# Database
# ============================================
DATABASE_PATH = "silentcare.db"

# ============================================
# Flask / Server
# ============================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ============================================
# Dataset paths (configurable)
# ============================================
AUDIO_DATASET_DIR = "data/audio_dataset"
RAF_DB_DIR = "data/RAF-DB"
VIVAE_DIR = ""  # To be set by user

# ============================================
# Training
# ============================================
AUDIO_TRAIN_EPOCHS = 50
AUDIO_TRAIN_BATCH_SIZE = 32
AUDIO_TRAIN_LR = 0.001

VIDEO_TRAIN_EPOCHS = 30
VIDEO_TRAIN_BATCH_SIZE = 32
VIDEO_TRAIN_LR = 0.0001

# ============================================
# Offline analysis
# ============================================
OFFLINE_UPLOAD_DIR = "data/offline"
OFFLINE_MAX_FILE_SIZE_MB = 500
