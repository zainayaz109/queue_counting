"""
Production Configuration for Unified Queue Counting System.
Defines RTSP stream endpoints, model weights, normalized PINK/RED/GREEN regions,
Square AOI cropping parameters (Aspect Ratio ~ 1:1), dwell thresholds, and grace periods.
"""
import os
import numpy as np

# ==============================================================================
# 1. VIDEO SOURCE & MODEL CONFIGURATION
# ==============================================================================
RTSP_URL = "rtsp://ABANA:abana123@10.64.109.32:554/Streaming/Channels/1501"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "yolo26m.pt"))
TRACKER_TYPE = os.path.abspath(os.path.join(os.path.dirname(__file__), "custom_bytetrack.yaml"))
TARGET_FPS = 20.0
IMG_SIZE = 960
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.25
CONF_THRESH = 0.25
IOU_THRESH = 0.50

# ==============================================================================
# 2. SQUARE AOI CROP COORDINATES (Aspect Ratio ~ 1:1 for undistorted YOLO prediction)
# Full Image: 1024x766 (or native 2560x1920)
# ==============================================================================
ROI_COORDS = {
    "x_min_ratio": 0.0000,   # Far left edge of camera frame (0 px)
    "x_max_ratio": 0.7178,   # Right edge of aisle approach (~735 px in 1024 / ~1837 px in 2560)
    "y_min_ratio": 0.0200,   # Aisle background & ceiling lights (~15 px in 766 / ~38 px in 1920)
    "y_max_ratio": 0.9771    # Full foreground floor (Exact 1:1 Aspect Ratio: ~748 px in 766 / ~1876 px in 1920)
}

# ==============================================================================
# 3. EXACT THREE UNIFIED QUEUE REGIONS (Normalized to Square AOI [0.0 -> 1.0])
# ==============================================================================

# 🌸 PINK (region_1): Arrival & Waiting Floor Area in Main Aisle
PINK_ZONE_NORM = np.array([
    [0.6000, 0.4692],   # Top-left
    [0.9074, 0.4787],   # Top-right
    [0.9809, 0.6215],   # Bottom-right
    [0.5741, 0.6202]    # Bottom-left
], dtype=np.float32)

# 🔴 RED (region_2): Transition & Kiosk Corridor Area
RED_ZONE_NORM = np.array([
    [0.4354, 0.3797],   # Top-left (meeting Green)
    [0.6000, 0.4692],   # Top-right (meeting Pink)
    [0.5741, 0.6202],   # Bottom-right (meeting Pink)
    [0.4000, 0.4483]    # Bottom-left (meeting Green)
], dtype=np.float32)

# 🟢 GREEN (region_3): Checkout 1 Counter
GREEN_ZONE_NORM = np.array([
    [0.25, 0.3881],   # Top-left
    [0.4354, 0.3797],   # Top-right (meeting Red)
    [0.4000, 0.4483],   # Bottom-right (meeting Red)
    [0.21, 0.4470]    # Bottom-left
], dtype=np.float32)

# ==============================================================================
# 3b. OPTIONAL YELLOW REGION (Checkout 1 Bagging / Packing Area at far left)
# ==============================================================================
ENABLE_YELLOW_ZONE = False  # Toggle True to include Yellow Zone in queue counting, False to disable

# 🟡 YELLOW (region_4): Customer Bagging / Packing Area at far left
YELLOW_ZONE_NORM = np.array([
    [0.0951, 0.4470],   # Top-left (meeting Green counter edge)
    [0.1436, 0.4492],   # Top-right (towards cashier belt)
    [0.0621, 0.5289],   # Bottom-right (lower packing shelf)
    [0.0373, 0.5034]    # Bottom-left (outer bagging edge)
], dtype=np.float32)

# ==============================================================================
# 4. TEMPORAL & DWELL THRESHOLDS
# ==============================================================================
# Continuous dwell duration (seconds) inside combined QUEUE_ROI (PINK ∪ RED ∪ GREEN) to become QUEUE
QUEUE_DWELL_SECONDS = 5.0

# Grace period (seconds) for a CANDIDATE who temporarily steps out of QUEUE_ROI before dwell resets
CANDIDATE_EXIT_GRACE_SECONDS = 3.0

# Grace period (seconds) for a QUEUE member who temporarily steps out or whose track drops
QUEUE_EXIT_GRACE_SECONDS = 5.0

# Track confirmation frames before creating a new entity
TRACK_CONFIRMATION_FRAMES = 8

# Track loss grace and reassociation window (seconds)
TRACK_LOST_GRACE_SECONDS = 3.0
TRACK_REASSOCIATION_SECONDS = 5.0
SPATIAL_CONTINUITY_TOLERANCE = 0.35

# ==============================================================================
# 5. ALERT THRESHOLDS
# ==============================================================================
QUEUE_LENGTH_ALERT_THRESHOLD = 3
QUEUE_LENGTH_SUSTAINED_SEC = 15.0
EXCESSIVE_WAIT_ALERT_SEC = 240.0
ALERT_COOLDOWN_SEC = 60.0
