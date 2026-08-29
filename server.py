"""
FastAPI Server & Real-time Web Dashboard for Checkout 1 Queue & Wait-Time Analytics.
Provides live MJPEG video streaming, REST API endpoints, and Real-Time WebSocket Alerts.
"""
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # Suppress low-level FFmpeg decoder warnings

import cv2
import time
import json
import asyncio
import threading
import sys
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RTSP_URL, MODEL_PATH, IMG_SIZE, CONF_THRESH, IOU_THRESH, TRACKER_TYPE,
    ROI_COORDS, QUEUE_LENGTH_ALERT_THRESHOLD, QUEUE_LENGTH_SUSTAINED_SEC,
    EXCESSIVE_WAIT_ALERT_SEC
)
from stream_reader import ThreadedRTSPStream
from tracker import YOLOv26Tracker
from occlusion_manager import OcclusionManager
from queue_state_machine import QueueStateMachine
from queue_analytics import QueueAnalytics
from visualizer import QueueVisualizer

# Global Application Settings (Configured on Startup)
APP_CONFIG = {
    "source": RTSP_URL,
    "model_path": MODEL_PATH,
    "alert_threshold": QUEUE_LENGTH_ALERT_THRESHOLD,  # Alerts trigger when queue > N (or >= N)
    "sustained_sec": QUEUE_LENGTH_SUSTAINED_SEC
}

class WebSocketConnectionManager:
    """Manages active WebSocket client connections and broadcasts real-time alert events."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] Client connected ({len(self.active_connections)} total clients).")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket] Client disconnected ({len(self.active_connections)} remaining).")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts a JSON message to all connected clients asynchronously."""
        if not self.active_connections:
            return

        dead_connections = []
        msg_text = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(msg_text)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            if dead in self.active_connections:
                self.active_connections.remove(dead)

ws_manager = WebSocketConnectionManager()
main_event_loop = None

class QueuePipelineService:
    def __init__(self):
        self.stream = None
        self.tracker = None
        self.state_machine = None
        self.analytics = None
        self.visualizer = None
        self.latest_annotated_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.stats = {
            "queue_length": 0,
            "max_wait_sec": 0.0,
            "avg_wait_sec": 0.0,
            "active_alerts": [],
            "occupants": []
        }
        self.inference_fps = 0.0
        self.stream_fps = 0.0
        self.roi_w = 0
        self.roi_h = 0
        self.last_alert_state = False

    def start(self, source=None, model_path=None, alert_threshold=None, sustained_sec=None):
        src = source or APP_CONFIG["source"]
        model = model_path or APP_CONFIG["model_path"]
        threshold = alert_threshold if alert_threshold is not None else APP_CONFIG["alert_threshold"]
        sustained = sustained_sec if sustained_sec is not None else APP_CONFIG["sustained_sec"]

        print(f"[Queue Server] Connecting to RTSP stream: {src} ...")
        print(f"[Queue Server] Alert Threshold: > {threshold} people in queue (Sustained >= {sustained}s)")
        
        self.stream = ThreadedRTSPStream(src).start()
        self.running = True
        
        # Start AI worker thread
        threading.Thread(target=self._ai_worker, args=(model, threshold, sustained), daemon=True).start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()

    def _ai_worker(self, model_path, alert_threshold, sustained_sec):
        print(f"[Queue Server] Initializing YOLO & SPS-Queue (Model: {model_path}, ImgSize: {IMG_SIZE})...")
        self.tracker = YOLOv26Tracker(
            model_path=model_path,
            tracker_type=TRACKER_TYPE,
            img_size=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH
        )
        self.occlusion_mgr = OcclusionManager()
        self.state_machine = QueueStateMachine(occlusion_manager=self.occlusion_mgr)
        self.analytics = QueueAnalytics(
            length_threshold=alert_threshold,
            sustained_duration=sustained_sec
        )
        self.visualizer = QueueVisualizer()

        # Wait for first valid frame to compute dimensions
        full_h, full_w = 1920, 2560
        while self.running:
            ret, ts, frame = self.stream.read_latest(timeout=1.0)
            if ret and frame is not None:
                full_h, full_w = frame.shape[:2]
                break
            time.sleep(0.1)

        roi_x1 = int(full_w * ROI_COORDS["x_min_ratio"])
        roi_x2 = int(full_w * ROI_COORDS["x_max_ratio"])
        roi_y1 = int(full_h * ROI_COORDS["y_min_ratio"])
        roi_y2 = int(full_h * ROI_COORDS["y_max_ratio"])
        self.roi_w = roi_x2 - roi_x1
        self.roi_h = roi_y2 - roi_y1

        print(f"[Queue Server] AI Worker Active! AOI Crop: {self.roi_w}x{self.roi_h} (Square 1:1)")

        fps_timer = time.time()
        fps_count = 0
        log_timer = time.time()
        ws_broadcast_timer = time.time()
        stride_count = 0
        cached_detections = []

        while self.running:
            ret, ts, frame = self.stream.read_latest(timeout=0.5)
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi_frame.size == 0:
                continue

            # Run detection with stride 2 for maximum FPS
            stride_count += 1
            if (stride_count % 2) == 0 or not cached_detections:
                cached_detections = self.tracker.track_roi(roi_frame)

            # Update SPS-Queue State Machine
            active_entities = self.state_machine.process_frame(cached_detections, timestamp=ts)

            # Compute Analytics & Alerts
            metrics = self.analytics.compute_analytics(active_entities, current_time=ts)

            # Build occupants breakdown list for dashboard & webhooks
            occupants = []
            for e in active_entities:
                occupants.append({
                    "slot_id": getattr(e, "slot_id", f"Q{getattr(e, 'track_id', '?')}"),
                    "zone": getattr(e, "zone", "GREEN"),
                    "state": getattr(e, "state", "QUEUE"),
                    "wait_time_sec": round(e.get_wait_time(ts), 1),
                    "dwell_sec": round(getattr(e, "cumulative_dwell_sec", 0.0), 1)
                })

            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                self.inference_fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now
                self.stream_fps = self.stream.fps

            # Check for WebSocket Alert Broadcast condition:
            # Condition: queue_length exceeds threshold (e.g. >= alert_threshold)
            has_queue_alert = (metrics["queue_length"] >= alert_threshold) or bool(metrics["active_alerts"])
            
            # Broadcast WebSocket updates every second or immediately when an alert triggers
            if (now - ws_broadcast_timer >= 1.0) or (has_queue_alert and not self.last_alert_state):
                ws_broadcast_timer = now
                self.last_alert_state = has_queue_alert
                
                alert_payload = {
                    "type": "QUEUE_ALERT" if has_queue_alert else "QUEUE_UPDATE",
                    "timestamp": now,
                    "time_str": time.strftime("%H:%M:%S"),
                    "queue_length": metrics["queue_length"],
                    "alert_threshold": alert_threshold,
                    "is_alert": has_queue_alert,
                    "max_wait_sec": round(metrics["max_wait_sec"], 1),
                    "avg_wait_sec": round(metrics["avg_wait_sec"], 1),
                    "active_alerts": metrics["active_alerts"],
                    "occupants_count": len(occupants),
                    "inference_fps": round(self.inference_fps, 1),
                    "message": f"⚠️ HIGH QUEUE ALERT: {metrics['queue_length']} customers in queue (Threshold: {alert_threshold})" if has_queue_alert else f"Queue normal ({metrics['queue_length']} customers)"
                }

                # Schedule async broadcast on FastAPI main event loop
                if main_event_loop and main_event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws_manager.broadcast(alert_payload), main_event_loop)

            # Telemetry logging every 3 seconds
            if now - log_timer >= 3.0:
                log_timer = now
                print(f"[Telemetry] RTSP: {self.stream_fps:.1f} FPS | Local AI: {self.inference_fps:.1f} FPS | "
                      f"Queue Count: {metrics['queue_length']}/{alert_threshold} | Max Wait: {metrics['max_wait_sec']:.1f}s | "
                      f"Active Alerts: {len(metrics['active_alerts'])}")

            # Render Visual HUD
            vis_frame = self.visualizer.render(
                roi_frame=roi_frame,
                active_entities=active_entities,
                metrics=metrics,
                fps=self.inference_fps,
                timestamp=ts
            )

            with self.lock:
                self.stats = {
                    "queue_length": metrics["queue_length"],
                    "max_wait_sec": round(metrics["max_wait_sec"], 1),
                    "avg_wait_sec": round(metrics["avg_wait_sec"], 1),
                    "active_alerts": metrics["active_alerts"],
                    "occupants": occupants
                }
                self.latest_annotated_frame = vis_frame


pipeline_service = QueuePipelineService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    # Startup
    pipeline_service.start(
        source=APP_CONFIG["source"],
        model_path=APP_CONFIG["model_path"],
        alert_threshold=APP_CONFIG["alert_threshold"],
        sustained_sec=APP_CONFIG["sustained_sec"]
    )
    yield
    # Shutdown
    pipeline_service.stop()

app = FastAPI(
    title="Checkout 1 Queue Management & Wait-Time Analytics",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_frames():
    """MJPEG stream generator."""
    while True:
        with pipeline_service.lock:
            frame = pipeline_service.latest_annotated_frame

        if frame is not None:
            # Scale frame for smooth web streaming
            out_w = 960
            out_h = int(frame.shape[0] * (out_w / float(frame.shape[1])))
            frame_resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            ret, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 fps web streaming cap

@app.get("/video_feed")
def video_feed():
    """Live MJPEG video feed of the AI annotated stream."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket endpoint that sends live alerts whenever more than N people are in the queue.
    Clients receive instant JSON payloads:
    {
        "type": "QUEUE_ALERT",
        "queue_length": 4,
        "alert_threshold": 2,
        "is_alert": true,
        "message": "⚠️ HIGH QUEUE ALERT: 4 customers in queue (Threshold: 2)",
        "timestamp": 1724945123.4,
        "max_wait_sec": 45.2
    }
    """
    await ws_manager.connect(websocket)
    # Send initial welcome & current state
    init_msg = {
        "type": "INIT",
        "alert_threshold": APP_CONFIG["alert_threshold"],
        "queue_length": pipeline_service.stats.get("queue_length", 0),
        "message": f"Connected to Real-time Queue Alerts (Threshold: > {APP_CONFIG['alert_threshold']} customers)"
    }
    await websocket.send_text(json.dumps(init_msg))

    try:
        while True:
            # Keep-alive receive loop
            data = await websocket.receive_text()
            # If client sends ping, respond with pong
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "PONG", "time": time.time()}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

# Backward-compatible WebSocket alias
@app.websocket("/ws")
async def websocket_alias(websocket: WebSocket):
    await websocket_alerts_endpoint(websocket)

@app.get("/api/stats")
def get_stats():
    """Returns current real-time queue counting statistics and FPS."""
    return {
        "status": "active",
        "queue_length": pipeline_service.stats.get("queue_length", 0),
        "alert_threshold": APP_CONFIG["alert_threshold"],
        "is_alert": pipeline_service.stats.get("queue_length", 0) >= APP_CONFIG["alert_threshold"],
        "max_wait_sec": pipeline_service.stats.get("max_wait_sec", 0.0),
        "avg_wait_sec": pipeline_service.stats.get("avg_wait_sec", 0.0),
        "active_alerts": pipeline_service.stats.get("active_alerts", []),
        "occupants": pipeline_service.stats.get("occupants", []),
        "inference_fps": round(pipeline_service.inference_fps, 1),
        "stream_fps": round(pipeline_service.stream_fps, 1),
    }

@app.post("/api/reset")
def reset_counters():
    """Resets the queue tracking state."""
    if pipeline_service.state_machine:
        pipeline_service.state_machine.slot_manager.slots.clear()
    if pipeline_service.analytics:
        pipeline_service.analytics.reset_metrics()
    return {"status": "success", "message": "Queue state reset."}

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Modern Dark-Theme HTML5 Real-Time Queue Analytics Dashboard with Live WebSockets."""
    threshold = APP_CONFIG["alert_threshold"]
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkout 1 - Queue Management & Real-Time Alerts</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0b1120; color: #f8fafc; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; padding-bottom: 18px; border-bottom: 1px solid #1e293b; margin-bottom: 20px; }}
        h1 {{ font-size: 22px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 10px; }}
        .header-badges {{ display: flex; gap: 10px; align-items: center; }}
        .status-badge {{ background: #16a34a; color: white; padding: 5px 14px; border-radius: 9999px; font-size: 13px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }}
        .ws-badge {{ background: #0284c7; color: white; padding: 5px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
        .ws-badge.alert {{ background: #dc2626; animation: pulse 1s infinite; }}
        .threshold-badge {{ background: #475569; color: #f8fafc; padding: 5px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
            100% {{ opacity: 1; }}
        }}
        .grid {{ display: grid; grid-template-columns: 1.8fr 1.2fr; gap: 20px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px; }}
        .stat-box {{ background: #0f172a; padding: 16px; border-radius: 10px; border-left: 4px solid #38bdf8; }}
        .stat-box.queue {{ border-left-color: #22c55e; }}
        .stat-box.queue.alert {{ border-left-color: #ef4444; background: rgba(239, 68, 68, 0.1); }}
        .stat-box.max-wait {{ border-left-color: #f59e0b; }}
        .stat-box.avg-wait {{ border-left-color: #06b6d4; }}
        .stat-box.alerts {{ border-left-color: #ef4444; }}
        .stat-box.ai-fps {{ border-left-color: #a855f7; }}
        .stat-title {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }}
        .stat-val {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
        .video-container {{ position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #000; aspect-ratio: 1/1; }}
        .video-container img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
        .fps-badge {{ display: flex; justify-content: space-between; margin-top: 15px; font-size: 13px; color: #94a3b8; padding-top: 10px; border-top: 1px solid #334155; }}
        .fps-badge span {{ color: #f8fafc; font-weight: 600; }}
        .alert-banner {{ background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; display: none; }}
        .alert-banner.active {{ display: block; animation: pulse 1.5s infinite; }}
        .alert-title {{ color: #f87171; font-weight: 700; font-size: 15px; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }}
        .occupants-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        .occupants-table th {{ text-align: left; padding: 8px 10px; background: #0f172a; color: #94a3b8; font-weight: 600; border-radius: 4px; }}
        .occupants-table td {{ padding: 10px; border-bottom: 1px solid #334155; }}
        .badge {{ padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 11px; display: inline-block; }}
        .badge.QUEUE {{ background: #16a34a; color: white; }}
        .badge.CANDIDATE {{ background: #0284c7; color: white; }}
        .badge.OCCLUDED {{ background: #d97706; color: white; }}
        .zone-pill {{ padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
        .zone-pill.GREEN {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; }}
        .zone-pill.RED {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
        .zone-pill.PINK {{ background: rgba(217, 70, 239, 0.2); color: #e879f9; }}
        .zone-pill.YELLOW {{ background: rgba(234, 179, 8, 0.2); color: #facc15; }}
        .ws-log {{ max-height: 180px; overflow-y: auto; list-style: none; margin-top: 10px; font-family: monospace; font-size: 12px; }}
        .ws-item {{ padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; background: #0f172a; display: flex; justify-content: space-between; }}
        .ws-item.alert {{ border-left: 3px solid #ef4444; color: #fca5a5; }}
        .ws-item.normal {{ border-left: 3px solid #22c55e; color: #86efac; }}
        button {{ background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }}
        button:hover {{ background: #b91c1c; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🛒 Checkout 1 &mdash; Queue Management & Real-Time Alerts</h1>
            <div class="header-badges">
                <span class="threshold-badge">🔔 Alert Threshold: &ge; {threshold} People</span>
                <span id="ws_status_badge" class="ws-badge">● WS CONNECTING...</span>
                <span class="status-badge">● LIVE STREAM</span>
                <button onclick="resetQueue()">Reset</button>
            </div>
        </header>

        <!-- KPI Metrics Grid -->
        <div class="stats-grid">
            <div class="stat-box queue" id="card_queue_box">
                <div class="stat-title">Current Queue Length</div>
                <div class="stat-val" id="queue_len" style="color: #22c55e;">0</div>
            </div>
            <div class="stat-box max-wait">
                <div class="stat-title">Max Wait Time</div>
                <div class="stat-val" id="max_wait" style="color: #f59e0b;">0.0s</div>
            </div>
            <div class="stat-box avg-wait">
                <div class="stat-title">Average Wait Time</div>
                <div class="stat-val" id="avg_wait" style="color: #06b6d4;">0.0s</div>
            </div>
            <div class="stat-box alerts">
                <div class="stat-title">Active Alerts</div>
                <div class="stat-val" id="active_alerts_count" style="color: #ef4444;">0</div>
            </div>
            <div class="stat-box ai-fps">
                <div class="stat-title">⚡ AI Inference FPS</div>
                <div class="stat-val" id="card_inf_fps" style="color: #a855f7;">0.0</div>
            </div>
        </div>

        <div class="grid">
            <!-- Left: Live Video Stream -->
            <div class="card">
                <h3 style="margin-bottom: 12px; color: #f8fafc;">Live 4-Zone AI Queue Detection</h3>
                <div class="video-container">
                    <img src="/video_feed" alt="Live Queue Video Feed" />
                </div>
                <div class="fps-badge">
                    <div>AI Engine: <span id="inf_fps">0.0</span> FPS</div>
                    <div>RTSP Ingestion: <span id="cam_fps">0.0</span> FPS</div>
                    <div>Model: <span>YOLO26m (Medium)</span></div>
                </div>
            </div>

            <!-- Right: Active Occupants & WebSocket Alerts -->
            <div class="card">
                <!-- Flashing WebSocket Alert Banner -->
                <div id="alert_box" class="alert-banner">
                    <div class="alert-title">🚨 HIGH QUEUE ALERT TRIGGERED</div>
                    <div id="alert_text" style="font-size: 13px; color: #fca5a5;"></div>
                </div>

                <h3 style="margin-bottom: 8px; color: #f8fafc;">Active Queue Occupants</h3>
                <table class="occupants-table">
                    <thead>
                        <tr>
                            <th>Slot</th>
                            <th>Zone</th>
                            <th>Status</th>
                            <th>Dwell</th>
                            <th>Wait Time</th>
                        </tr>
                    </thead>
                    <tbody id="occupants_body">
                        <tr><td colspan="5" style="text-align: center; color: #64748b;">No active occupants in queue ROI</td></tr>
                    </tbody>
                </table>

                <h3 style="margin-top: 20px; margin-bottom: 6px; color: #f8fafc; font-size: 14px;">📡 Live WebSocket Alert Stream</h3>
                <ul class="ws-log" id="ws_log_list">
                    <li style="color: #64748b; padding: 4px;">Connecting to WebSocket /ws/alerts...</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        const threshold = {threshold};
        let ws;

        function initWebSocket() {{
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = protocol + '//' + window.location.host + '/ws/alerts';
            ws = new WebSocket(wsUrl);

            const wsBadge = document.getElementById('ws_status_badge');

            ws.onopen = function() {{
                wsBadge.innerText = '● WS CONNECTED';
                wsBadge.style.background = '#16a34a';
                addWsLog('Connected to WebSocket alerts endpoint (/ws/alerts)', 'normal');
            }};

            ws.onmessage = function(event) {{
                try {{
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                }} catch (e) {{
                    console.error('WS Parse Error:', e);
                }}
            }};

            ws.onclose = function() {{
                wsBadge.innerText = '● WS DISCONNECTED (Retrying...)';
                wsBadge.style.background = '#dc2626';
                setTimeout(initWebSocket, 2000);
            }};
        }}

        function handleWebSocketMessage(data) {{
            if (data.type === 'INIT') {{
                addWsLog(data.message, 'normal');
                return;
            }}

            const queueLen = data.queue_length;
            const isAlert = data.is_alert || (queueLen >= threshold);

            // Update UI Counters
            document.getElementById('queue_len').innerText = queueLen;
            if (data.max_wait_sec !== undefined) document.getElementById('max_wait').innerText = data.max_wait_sec + 's';
            if (data.avg_wait_sec !== undefined) document.getElementById('avg_wait').innerText = data.avg_wait_sec + 's';
            if (data.inference_fps !== undefined) {{
                document.getElementById('inf_fps').innerText = data.inference_fps;
                document.getElementById('card_inf_fps').innerText = data.inference_fps;
            }}

            const alertBox = document.getElementById('alert_box');
            const alertText = document.getElementById('alert_text');
            const queueCard = document.getElementById('card_queue_box');
            const wsBadge = document.getElementById('ws_status_badge');

            if (isAlert) {{
                alertBox.classList.add('active');
                alertText.innerHTML = `<strong>${{data.message}}</strong><br>Max Wait Time: ${{data.max_wait_sec}}s`;
                queueCard.classList.add('alert');
                wsBadge.classList.add('alert');
                wsBadge.innerText = '🚨 WS ALERT: QUEUE EXCEEDED';
                document.getElementById('active_alerts_count').innerText = Math.max(1, (data.active_alerts ? data.active_alerts.length : 1));
            }} else {{
                alertBox.classList.remove('active');
                queueCard.classList.remove('alert');
                wsBadge.classList.remove('alert');
                wsBadge.innerText = '● WS CONNECTED';
                wsBadge.style.background = '#16a34a';
                document.getElementById('active_alerts_count').innerText = 0;
            }}

            if (data.type === 'QUEUE_ALERT') {{
                addWsLog(`[${{data.time_str}}] ${{data.message}}`, 'alert');
            }}
        }}

        function addWsLog(msg, type) {{
            const list = document.getElementById('ws_log_list');
            const li = document.createElement('li');
            li.className = 'ws-item ' + (type || 'normal');
            li.innerHTML = `<span>${{msg}}</span><span>${{new Date().toLocaleTimeString()}}</span>`;
            list.insertBefore(li, list.firstChild);
            if (list.children.length > 25) list.removeChild(list.lastChild);
        }}

        // REST polling backup for occupant details table
        async function fetchStats() {{
            try {{
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('cam_fps').innerText = data.stream_fps;

                // Occupants Table
                const tbody = document.getElementById('occupants_body');
                if (data.occupants && data.occupants.length > 0) {{
                    tbody.innerHTML = data.occupants.map(o => `
                        <tr>
                            <td><strong>${{o.slot_id}}</strong></td>
                            <td><span class="zone-pill ${{o.zone}}">${{o.zone}}</span></td>
                            <td><span class="badge ${{o.state}}">${{o.state}}</span></td>
                            <td>${{o.dwell_sec}}s</td>
                            <td><strong style="color: ${{o.wait_time_sec > 60 ? '#f87171' : '#4ade80'}};">${{o.wait_time_sec}}s</strong></td>
                        </tr>
                    `).join('');
                }} else {{
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #64748b;">No active occupants in queue ROI</td></tr>';
                }}
            }} catch (err) {{}}
        }}

        initWebSocket();
        setInterval(fetchStats, 1000);

        async function resetQueue() {{
            if (confirm("Reset current queue metrics and slots?")) {{
                await fetch('/api/reset', {{ method: 'POST' }});
                fetchStats();
            }}
        }}
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Checkout 1 Queue Analytics Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8090, help="Port to listen on (default: 8090)")
    parser.add_argument("--source", type=str, default=RTSP_URL, help="RTSP URL or video file path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to YOLO model weights")
    parser.add_argument("--alert-threshold", type=int, default=QUEUE_LENGTH_ALERT_THRESHOLD,
                        help=f"Alert threshold for queue length (default: {QUEUE_LENGTH_ALERT_THRESHOLD})")
    parser.add_argument("--sustained-sec", type=float, default=QUEUE_LENGTH_SUSTAINED_SEC,
                        help=f"Sustained duration in seconds before alert triggers (default: {QUEUE_LENGTH_SUSTAINED_SEC}s)")

    args = parser.parse_args()

    APP_CONFIG["source"] = args.source
    APP_CONFIG["model_path"] = args.model
    APP_CONFIG["alert_threshold"] = args.alert_threshold
    APP_CONFIG["sustained_sec"] = args.sustained_sec

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
