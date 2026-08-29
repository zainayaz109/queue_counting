"""
Threaded, zero-latency RTSP stream capture engine with auto-reconnect watchdog.
"""
import cv2
import threading
import queue
import time
import os

class ThreadedRTSPStream:
    """
    High-performance RTSP stream consumer.
    Automatically drops stale network frames and keeps only the latest frame in memory,
    completely preventing buffer lag buildup over high-latency networks.
    """
    def __init__(self, rtsp_url, retry_interval=2.0):
        self.rtsp_url = rtsp_url
        self.retry_interval = retry_interval
        self.queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.connected = False
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        return self

    def _worker_loop(self):
        # Configure FFmpeg backend flags for low latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "buffer_size;102400|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "max_delay;500000|"
            "reorder_queue_size;0"
        )

        while self.running:
            print(f"[RTSP Stream] Connecting to {self.rtsp_url}...")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"[RTSP Stream] Connection failed. Retrying in {self.retry_interval}s...")
                self.connected = False
                time.sleep(self.retry_interval)
                continue

            self.connected = True
            print("[RTSP Stream] Successfully connected and streaming.")
            
            fps_timer = time.time()
            fps_counter = 0

            while self.running and cap.isOpened():
                # Grab grabs the newest frame from the network stream immediately
                grabbed = cap.grab()
                if not grabbed:
                    # Check for stream freeze/stall
                    if time.time() - self.last_frame_time > 4.0:
                        print("[RTSP Stream] Stream stall detected (>4s no frames). Reconnecting...")
                        break
                    time.sleep(0.005)
                    continue

                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    continue

                now = time.time()
                self.last_frame_time = now
                self.frame_count += 1
                fps_counter += 1

                if now - fps_timer >= 1.0:
                    self.fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # Discard old frame if queue is full to ensure zero latency
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                self.queue.put((now, frame))

            cap.release()
            self.connected = False
            if self.running:
                print(f"[RTSP Stream] Connection lost. Reconnecting in {self.retry_interval}s...")
                time.sleep(self.retry_interval)

    def read_latest(self, timeout=1.0):
        """
        Retrieves the latest available frame.
        Returns: (success: bool, timestamp: float, frame: np.ndarray)
        """
        try:
            timestamp, frame = self.queue.get(timeout=timeout)
            return True, timestamp, frame
        except queue.Empty:
            return False, 0.0, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("[RTSP Stream] Stream stopped.")
