"""
YOLOv26 Detection & ByteTrack multi-object tracking module.
Extracts bottom-center foot contact points for queue spatial evaluation.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import torch

from config import MODEL_PATH, TRACKER_TYPE, IMG_SIZE, CONF_THRESH, IOU_THRESH

class YOLOv26Tracker:
    def __init__(self, model_path=MODEL_PATH, tracker_type=TRACKER_TYPE, 
                 img_size=IMG_SIZE, conf=CONF_THRESH, iou=IOU_THRESH, device=None):
        if device is not None:
            self.device = device
        else:
            # Auto-detect: Prioritize NVIDIA CUDA GPU, then Apple Silicon MPS, fallback to CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
        print(f"[YOLOv26 Tracker] Initializing model '{model_path}' (imgsz={img_size}) on device: {str(self.device).upper()}")
        
        self.model = YOLO(model_path)
        self.tracker_type = tracker_type
        self.img_size = img_size
        self.conf = conf
        self.iou = iou

    def track_roi(self, roi_frame):
        """
        Runs YOLOv26 person detection and ByteTrack multi-object tracking on the ROI frame.
        Returns a list of dicts: [
            {
                'track_id': int,
                'bbox': [x1, y1, x2, y2],
                'conf': float,
                'center': (cx, cy),
                'bottom_center': (bx, by),
                'norm_foot': (nx, ny),
                'height': float,
                'width': float
            }
        ]
        """
        results = self.model.track(
            source=roi_frame,
            persist=True,
            classes=[0],           # 0 = Person class only
            tracker=self.tracker_type,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes))
        track_ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else [-1] * len(boxes)

        roi_h, roi_w = roi_frame.shape[:2]

        for bbox, conf_score, track_id in zip(boxes, confs, track_ids):
            if track_id == -1:
                continue
                
            x1, y1, x2, y2 = bbox
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bx = float((x1 + x2) / 2.0)
            by = float(y2)  # Ground contact foot point

            # Normalized relative to ROI frame [0.0 -> 1.0]
            nx = float(np.clip(bx / max(1, roi_w), 0.0, 1.0))
            ny = float(np.clip(by / max(1, roi_h), 0.0, 1.0))

            detections.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(conf_score),
                "center": (cx, cy),
                "bottom_center": (bx, by),
                "norm_foot": (nx, ny),
                "height": float(y2 - y1),
                "width": float(x2 - x1)
            })

        return detections
