"""
Main Production Queue Management & Wait-Time Analytics Application.
Supports Live Zero-Lag RTSP Streams and Local Offline Video Files.
Integrates YOLOv26 Person Detection, SPS-Queue Spatial Presence Engine,
Queue Analytics, and HUD Visualization.
"""
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # Suppress low-level FFmpeg decoder warnings

import cv2
import time
import argparse
import sys

from config import (
    RTSP_URL, MODEL_PATH, IMG_SIZE, CONF_THRESH, IOU_THRESH, TRACKER_TYPE,
    ROI_COORDS
)

from stream_reader import ThreadedRTSPStream
from tracker import YOLOv26Tracker
from occlusion_manager import OcclusionManager
from queue_state_machine import QueueStateMachine
from queue_analytics import QueueAnalytics
from visualizer import QueueVisualizer

def parse_time_str(time_val):
    """Converts seconds (int/float) or string format ('MM:SS', 'HH:MM:SS') to float seconds."""
    if isinstance(time_val, (int, float)):
        return float(time_val)
    if not time_val:
        return 0.0
    parts = str(time_val).strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0
    return 0.0

def run_pipeline(source=RTSP_URL, model_path=MODEL_PATH, start_time=0, duration=0, 
                 display=False, save_video=None, tracker_type=TRACKER_TYPE, device=None,
                 stride=2, out_width=960):

    is_video_file = os.path.isfile(source) or (not source.startswith("rtsp://") and not source.startswith("http://"))
    start_sec = parse_time_str(start_time)
    print("==================================================================")
    print("      PRODUCTION QUEUE MANAGEMENT & WAIT-TIME ANALYTICS ENGINE    ")
    print("==================================================================")
    print(f"[*] Input Source:   {'[Local Video File] ' if is_video_file else '[Live RTSP Stream] '}{source}")
    print(f"[*] AI Model:       {model_path}")
    print(f"[*] Engine:         SPS-Queue (Spatial Presence & Slot Architecture)")
    print(f"[*] Stride:         {stride} (1 detection every {stride} frames)")
    if is_video_file and start_sec > 0:
        print(f"[*] Start Offset:   {start_sec:.1f}s ({int(start_sec//60):02d}:{int(start_sec%60):02d})")
    if duration > 0:
        print(f"[*] Duration:       {duration:.1f}s")
    print(f"[*] Mode:           {'Display Window' if display else 'Headless Processing'}")
    if save_video:
        print(f"[*] Saving Output:  {save_video}")
    print("==================================================================\n")

    # 1. Initialize Video Source
    cap = None
    stream = None

    if is_video_file:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[!] ERROR: Failed to open video file '{source}'")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle Start Offset
        if start_sec > 0:
            start_frame_idx = int(start_sec * source_fps)
            start_frame_idx = min(start_frame_idx, total_video_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            print(f"[+] Seeking to frame {start_frame_idx}/{total_video_frames} ({start_sec:.1f}s)")

        print(f"[+] Video Properties: {full_w}x{full_h} @ {source_fps:.1f} FPS | Total Frames: {total_video_frames}")
    else:
        stream = ThreadedRTSPStream(source, max_reconnect_attempts=10).start()
        full_w = 2560
        full_h = 1920
        source_fps = 20.0
        total_video_frames = 0
        print(f"[+] Native Stream Resolution: {full_w} x {full_h}")

    # 2. Compute AOI Bounding Box
    roi_x1 = int(full_w * ROI_COORDS["x_min_ratio"])
    roi_x2 = int(full_w * ROI_COORDS["x_max_ratio"])
    roi_y1 = int(full_h * ROI_COORDS["y_min_ratio"])
    roi_y2 = int(full_h * ROI_COORDS["y_max_ratio"])
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1
    
    # Output Resolution (Crisp 720p scaling for fast processing and small file size)
    out_w = out_width if out_width > 0 else roi_w
    out_h = int(roi_h * (out_w / float(roi_w)))
    # Ensure even dimensions for video codecs
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)

    print(f"[+] Extracted AOI Dimensions: {roi_w} x {roi_h} -> Output Video Res: {out_w} x {out_h}")

    # 3. Initialize AI & Analytics Components
    tracker = YOLOv26Tracker(
        model_path=model_path,
        tracker_type=tracker_type,
        img_size=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        device=device
    )

    occlusion_mgr = OcclusionManager()
    state_machine = QueueStateMachine(occlusion_manager=occlusion_mgr)
    analytics = QueueAnalytics()
    visualizer = QueueVisualizer()

    # 4. Optional Video Writer
    video_writer = None
    if save_video:
        os.makedirs(os.path.dirname(os.path.abspath(save_video)), exist_ok=True)
        # Try avc1 first, fallback to mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(save_video, fourcc, source_fps, (out_w, out_h))
            if not video_writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(save_video, fourcc, source_fps, (out_w, out_h))
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(save_video, fourcc, source_fps, (out_w, out_h))

        print(f"[+] Initialized VideoWriter to: {save_video} ({out_w}x{out_h}) @ {source_fps:.1f} FPS")

    # 5. Main Processing Loop
    print("\n[+] Queue Analytics Processing Loop ACTIVE. Press Ctrl+C or 'q' to stop.\n")
    start_time = time.time()
    fps_timer = time.time()
    fps_frames = 0
    inference_fps = 0.0
    processed_count = 0
    frame_idx = 0
    cached_detections = []

    try:
        while True:
            if duration > 0 and (time.time() - start_time) >= duration:
                print(f"[+] Reached duration limit of {duration} seconds.")
                break

            if is_video_file:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[+] Reached end of video file.")
                    break
                frame_ts = start_sec + (frame_idx / source_fps)
                frame_idx += 1
            else:
                ret, frame_ts, frame = stream.read_latest(timeout=0.1)
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

            # Crop frame to AOI
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi_frame.size == 0:
                continue

            # Run YOLO Person Detection on stride frames (re-use cache on stride skips)
            if (processed_count % max(1, stride)) == 0:
                cached_detections = tracker.track_roi(roi_frame)

            # Update State Machine
            active_entities = state_machine.process_frame(cached_detections, timestamp=frame_ts)

            # Compute Queue Analytics & Alerts
            metrics = analytics.compute_analytics(active_entities, current_time=frame_ts)

            # FPS calculation
            fps_frames += 1
            processed_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                inference_fps = fps_frames / (now - fps_timer)
                fps_frames = 0
                fps_timer = now

            # Visualization
            vis_frame = visualizer.render(roi_frame, active_entities, metrics, fps=inference_fps, timestamp=frame_ts)

            # Output writing (scaled for fast writing & compact file size)
            if video_writer:
                if (out_w, out_h) != (roi_w, roi_h):
                    vis_out = cv2.resize(vis_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    vis_out = vis_frame
                video_writer.write(vis_out)

            # Display
            if display:
                cv2.imshow("Production Queue Management - Checkout 1", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] User interrupted display window.")
                    break

            if processed_count % 200 == 0:
                pct = f"({(frame_idx/total_video_frames*100):.1f}%)" if is_video_file and total_video_frames else ""
                print(f"[{time.strftime('%H:%M:%S')}] Frame: {processed_count:5d} {pct} | "
                      f"Queue Count: {metrics['queue_length']} | "
                      f"Max Wait: {metrics['max_wait_sec']:.1f}s | "
                      f"FPS: {inference_fps:.1f}")

    except KeyboardInterrupt:
        print("\n[*] Processing interrupted by user (KeyboardInterrupt).")

    finally:
        total_time = time.time() - start_time
        avg_fps = processed_count / max(0.001, total_time)
        print("\n==================================================================")
        print("                   PROCESSING SUMMARY REPORT                      ")
        print("==================================================================")
        print(f"[*] Total Processed Frames: {processed_count}")
        print(f"[*] Total Elapsed Time:     {total_time:.2f} seconds")
        print(f"[*] Average Throughput:     {avg_fps:.2f} FPS")
        print("==================================================================")

        if is_video_file and cap is not None:
            cap.release()
        if stream is not None:
            stream.stop()
        if video_writer is not None:
            video_writer.release()
            file_size_mb = os.path.getsize(save_video) / (1024 * 1024) if os.path.exists(save_video) else 0.0
            print(f"[+] Video successfully saved to: {save_video} ({file_size_mb:.1f} MB)")

        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Queue Management & Wait-Time Analytics")
    parser.add_argument("--source", type=str, default=RTSP_URL, help="RTSP URL or path to local video file")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to YOLOv26 person detection model")
    parser.add_argument("--start-time", type=str, default="0", help="Start offset in seconds or 'MM:SS' / 'HH:MM:SS'")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration to process in seconds (0 = full)")
    parser.add_argument("--display", action="store_true", help="Display visual window in GUI environment")
    parser.add_argument("--save-video", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--tracker", type=str, default=TRACKER_TYPE, help="Tracker config yaml or type")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', 'cuda', 'cpu')")
    parser.add_argument("--stride", type=int, default=2, help="Inference frame stride (1=every frame, 2=every 2nd frame)")
    parser.add_argument("--out-width", type=int, default=960, help="Output video width (e.g. 960 for 720p)")

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        model_path=args.model,
        start_time=args.start_time,
        duration=args.duration,
        display=args.display,
        save_video=args.save_video,
        tracker_type=args.tracker,
        device=args.device,
        stride=args.stride,
        out_width=args.out_width
    )
