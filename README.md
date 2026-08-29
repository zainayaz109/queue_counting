# Production Queue Management & Wait-Time Analytics Engine

A real-time AI vision system for automated retail checkout queue counting, customer wait-time tracking, and real-time WebSocket alerting.

Integrates **YOLO26m**, **SPS-Queue (Spatial Presence & Slot Architecture)** with **1-to-1 Hungarian Bipartite Tracking**, and a **FastAPI Real-Time Web Dashboard & WebSocket Alert Server**.

---

## Key Features

1. **SPS-Queue Spatial Slot Architecture**:
   - Decouples queue dwell and wait times from volatile low-level track IDs.
   - **5.0-Second Occlusion & Barrier Grace Buffer**: When a customer bends down to pack bags, reaches into a cart, or is blocked by counter pillars, their queue slot stays alive and reconnects with **zero dwell or wait-time resets**.
2. **1-to-1 Optimal Hungarian Bipartite Matching**:
   - Eliminates track jumping and ID fluctuation when multiple customers stand close to each other in queue.
3. **4-Zone Unified Queue Geometry**:
   - Continuous forward progression across **PINK** (Arrival/Aisle) $\to$ **RED** (Corridor) $\to$ **GREEN** (Checkout 1 Counter) $\to$ **YELLOW** (Bagging / Packing Area).
   - `PINK + RED + GREEN + YELLOW = ONE CONTINUOUS QUEUE`.
4. **Exact 1:1 Square AOI Crop**:
   - Crops a square Area of Interest ($1837\text{ px} \times 1838\text{ px}$) so YOLO $960\times 960$ inference runs with zero aspect-ratio distortion or stretching.
5. **Real-Time WebSocket Alerts**:
   - Broadcasts instant JSON alert payloads to connected clients/webhooks whenever queue length meets or exceeds the startup threshold (`--alert-threshold N`).
6. **Anti-Inflation Body Clamping**:
   - Multiple bounding boxes on the same physical person are fused into strictly 1 queue slot.

---

## Directory Structure

```
queue_counting/
├── config.py                 # Queue zones, AOI ratios, model path, and thresholds
├── server.py                 # FastAPI Production Server, MJPEG stream, and WebSocket alerts
├── main.py                   # Standalone video & RTSP processing pipeline
├── batch_process.py          # Batch processor for input video chunks
├── draw_regions.py           # Calibration visualization tool (generates ref_regions.png)
├── queue_slot_manager.py     # SPS-Queue Spatial Slot Engine & 1-to-1 Hungarian Matcher
├── queue_state_machine.py    # Queue State Machine & lifecycle manager
├── queue_analytics.py        # Analytics & threshold alert engine
├── occlusion_manager.py      # Occlusion and temporal buffer manager
├── visualizer.py             # HUD overlays, zone rendering, and gold occlusion badges
├── test_pipeline.py          # Automated 13-test integration suite
├── ref.png                   # Reference full-frame image
├── ref_regions.png           # Calibrated zone overlays with vertex labels
├── ref_regions_crop.png      # Calibrated AOI zoomed crop
├── inputs/                   # Input video chunks folder
└── output/                   # Processed annotated videos folder
```

---

## Quick Start Guide

### 1. Run the Production Server with Live WebSocket Alerts
Start the server and specify the queue alert threshold (e.g., alert when $\ge 3$ people):

```bash
python server.py --alert-threshold 3 --port 8000
```

Optional arguments:
```bash
python server.py --source "rtsp://..." --alert-threshold 3 --sustained-sec 5.0 --port 8000
```

Open your browser at:
👉 **http://localhost:8000**

---

### 2. Run Single Video File or Live RTSP Pipeline

```bash
# Process a local video and save annotated MP4:
python main.py --source sample.mp4 --save-video output.mp4 --stride 2

# Process live RTSP feed with visual GUI window:
python main.py --display
```

---

### 3. Run Batch Processing on Multiple Video Chunks
Processes all video files in `inputs/` and outputs optimized 720p H.264 MP4s to `output/`:

```bash
python batch_process.py
```

---

### 4. Calibrate & Visualize Queue Boundaries
Modify any zone coordinates in `config.py`, then run:

```bash
python draw_regions.py
```
Open `ref_regions.png` to inspect labeled vertex points (`P0..P3`, `R0..R3`, `G0..G3`, `Y0..Y3`) and exact coordinates overlaid on the camera view.

---

### 5. Run the Automated Test Suite (13 Tests)

```bash
python test_pipeline.py
```

---

## Real-Time WebSocket Alerts API

### Endpoint: `ws://localhost:8000/ws/alerts`

Connect any external notification service, dashboard, or IoT device over WebSocket.

#### Payload Broadcasted when Queue Exceeds Threshold:
```json
{
    "type": "QUEUE_ALERT",
    "timestamp": 1724945123.4,
    "time_str": "16:08:30",
    "queue_length": 4,
    "alert_threshold": 3,
    "is_alert": true,
    "message": "⚠️ HIGH QUEUE ALERT: 4 customers in queue (Threshold: 3)",
    "max_wait_sec": 48.5,
    "avg_wait_sec": 31.2,
    "occupants_count": 4,
    "inference_fps": 11.5
}
```

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Real-Time Dark-Theme HTML5 Dashboard |
| `GET` | `/video_feed` | Low-latency MJPEG live annotated video stream |
| `WS` | `/ws/alerts` | Live WebSocket stream for real-time queue length alerts |
| `GET` | `/api/stats` | Returns JSON metrics (`queue_length`, `max_wait_sec`, `occupants`, `FPS`) |
| `POST` | `/api/reset` | Resets active queue tracking slots |

---

## Zone Geometry Reference (`config.py`)

| Zone | Name | Description |
| :--- | :--- | :--- |
| 🌸 **PINK** | `region_1` | Customer Arrival & Waiting Floor Area in Main Aisle |
| 🔴 **RED** | `region_2` | Transition & Kiosk Corridor Area |
| 🟢 **GREEN** | `region_3` | Checkout 1 Counter & Conveyor Belt |
| 🟡 **YELLOW** | `region_4` | Customer Bagging / Packing Area (toggle `ENABLE_YELLOW_ZONE = True`) |

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `../yolo26m.pt` | Path to YOLO detection weights |
| `IMG_SIZE` | `960` | Input resolution for inference |
| `QUEUE_DWELL_SECONDS` | `5.0` | Continuous dwell seconds required to promote to `QUEUE` |
| `QUEUE_EXIT_GRACE_SECONDS` | `5.0` | Occlusion / exit grace buffer before queue removal |
| `CANDIDATE_EXIT_GRACE_SECONDS` | `3.0` | Grace period for candidates stepping outside |
| `ENABLE_YELLOW_ZONE` | `True` | Toggle to include Yellow Bagging Area in queue counting |
| `QUEUE_LENGTH_ALERT_THRESHOLD` | `2` | Default alert threshold for queue length |
| `QUEUE_LENGTH_SUSTAINED_SEC` | `15.0` | Sustained seconds before length alert fires |
| `EXCESSIVE_WAIT_ALERT_SEC` | `240.0` | Max wait time alert threshold |
