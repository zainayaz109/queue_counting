"""
Unified Queue Visualizer & HUD Overlay Engine.
Renders translucent Pink, Red, Green queue regions, active bounding boxes with state badges,
and an informative top HUD analytics dashboard.
"""
import cv2
import numpy as np
import time

import config
from config import (
    PINK_ZONE_NORM,
    RED_ZONE_NORM,
    GREEN_ZONE_NORM
)

class QueueVisualizer:
    def __init__(self):
        # Zone Colors (BGR)
        self.COLOR_PINK = (255, 0, 255)   # Magenta/Pink
        self.COLOR_RED = (0, 0, 255)      # Red
        self.COLOR_GREEN = (0, 255, 0)    # Green
        self.COLOR_YELLOW = (0, 255, 255) # Yellow

        # State Colors (BGR)
        self.STATE_COLORS = {
            "QUEUE": (0, 255, 0),                 # Bright Green
            "CANDIDATE": (255, 255, 0),             # Cyan
            "TEMPORARILY_OUTSIDE": (0, 200, 255),  # Orange/Yellow
            "OCCLUDED": (0, 200, 255),             # Gold/Yellow
            "NOT_QUEUE": (140, 140, 140)           # Gray (Cashier/Outside)
        }

    def render(self, roi_frame, active_entities, metrics, fps=0.0, timestamp=None):
        """
        Renders regions, detections, state badges, and top HUD onto the ROI frame.
        """
        vis = roi_frame.copy()
        h, w = vis.shape[:2]
        now = timestamp or time.time()

        # 1. Render Zone Polygons with Translucent Blend
        vis = self._draw_zones(vis, w, h)

        # 2. Render Entity Bounding Boxes, Foot Points & Badges
        for entity in active_entities:
            self._draw_entity(vis, entity, now)

        # 3. Render Top HUD Bar
        vis = self._draw_hud(vis, metrics, fps)

        # 4. Render Active Alerts Banner
        if metrics.get("active_alerts"):
            vis = self._draw_alerts(vis, metrics["active_alerts"])

        return vis

    def _draw_zones(self, vis, w, h):
        overlay = vis.copy()

        def to_px(norm_pts):
            return np.array([[int(p[0] * w), int(p[1] * h)] for p in norm_pts], dtype=np.int32)

        px_pink = to_px(PINK_ZONE_NORM)
        px_red = to_px(RED_ZONE_NORM)
        px_green = to_px(GREEN_ZONE_NORM)

        # Fill Polygons
        cv2.fillPoly(overlay, [px_pink], self.COLOR_PINK)
        cv2.fillPoly(overlay, [px_red], self.COLOR_RED)
        cv2.fillPoly(overlay, [px_green], self.COLOR_GREEN)

        enable_yellow = getattr(config, "ENABLE_YELLOW_ZONE", False) and hasattr(config, "YELLOW_ZONE_NORM")
        if enable_yellow:
            px_yellow = to_px(config.YELLOW_ZONE_NORM)
            cv2.fillPoly(overlay, [px_yellow], self.COLOR_YELLOW)

        # Blend with frame
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        # Poly Outlines
        cv2.polylines(vis, [px_pink], True, self.COLOR_PINK, 2)
        cv2.polylines(vis, [px_red], True, self.COLOR_RED, 2)
        cv2.polylines(vis, [px_green], True, self.COLOR_GREEN, 2)
        if enable_yellow:
            cv2.polylines(vis, [px_yellow], True, self.COLOR_YELLOW, 2)

        # Zone Text Labels
        cv2.putText(vis, "PINK (region_1)", (px_pink[0][0] + 10, px_pink[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "RED (region_2)", (px_red[0][0] + 5, px_red[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "GREEN (region_3)", (px_green[0][0] + 5, px_green[0][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        if enable_yellow:
            cv2.putText(vis, "YELLOW (4)", (px_yellow[0][0] + 2, px_yellow[0][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def _draw_entity(self, vis, entity, now):
        # Do not render stale bounding boxes for lost/missing tracks
        if (now - entity.last_seen_time) > 1.0 or not entity.is_active:
            return

        bbox = entity.last_bbox
        if not bbox or sum(bbox) == 0:
            return

        x1, y1, x2, y2 = [int(v) for v in bbox]
        state = entity.state
        color = self.STATE_COLORS.get(state, (140, 140, 140))
        if (now - entity.last_seen_time) > 0.3:
            color = (0, 215, 255) # Gold/Yellow for brief occlusion

        # 1. Bounding Box
        thickness = 2 if state in ["QUEUE", "CANDIDATE"] else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # 2. Foot-point ground contact dot
        if entity.last_position_px:
            fx, fy = int(entity.last_position_px[0]), int(entity.last_position_px[1])
            if fx > 0 and fy > 0:
                cv2.circle(vis, (fx, fy), 4, (0, 0, 255), -1)
                cv2.circle(vis, (fx, fy), 5, (255, 255, 255), 1)

        # 3. Badge Text
        if state == "QUEUE":
            wait_sec = entity.get_wait_time(now)
            label = f"{entity.entity_id} | QUEUE | Wait: {wait_sec:.0f}s"
        elif state == "CANDIDATE":
            label = f"{entity.entity_id} | CANDIDATE | Dwell: {entity.cumulative_dwell_sec:.1f}s"
        elif state == "TEMPORARILY_OUTSIDE" or state == "OCCLUDED":
            wait_sec = entity.get_wait_time(now)
            label = f"{entity.entity_id} | OCCLUDED | Wait: {wait_sec:.0f}s"
        else:
            # NOT_QUEUE (e.g. Cashier / Helper outside zones)
            label = f"T{entity.current_track_id} | NOT_QUEUE"

        # Badge Background Box
        font_scale = 0.38 if state != "NOT_QUEUE" else 0.32
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        bx1 = max(0, x1)
        by1 = max(0, y1 - th - 6)
        bx2 = bx1 + tw + 8
        by2 = by1 + th + 6

        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 1)
        cv2.putText(vis, label, (bx1 + 4, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, vis, metrics, fps):
        h, w = vis.shape[:2]

        # Top Bar Background
        cv2.rectangle(vis, (0, 0), (w, 32), (15, 15, 15), -1)
        cv2.line(vis, (0, 32), (w, 32), (50, 50, 50), 1)

        queue_count = metrics.get("queue_length", 0)
        cand_count = metrics.get("candidate_count", 0)
        max_wait = metrics.get("max_wait_sec", 0.0)
        avg_wait = metrics.get("avg_wait_sec", 0.0)

        # Queue Count Badge (Bright Green)
        count_text = f"CHECKOUT 1 QUEUE: {queue_count}"
        cv2.putText(vis, count_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)

        # Candidates
        cand_text = f"Candidates: {cand_count}"
        cv2.putText(vis, cand_text, (280, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        # Wait Times
        wait_text = f"Max Wait: {max_wait:.0f}s | Avg Wait: {avg_wait:.0f}s"
        cv2.putText(vis, wait_text, (430, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # FPS
        fps_text = f"{fps:.1f} FPS"
        cv2.putText(vis, fps_text, (w - 85, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        return vis

    def _draw_alerts(self, vis, alerts):
        h, w = vis.shape[:2]
        y_offset = 55
        for alert in alerts:
            (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(vis, (15, y_offset - th - 6), (25 + tw, y_offset + 6), (0, 0, 180), -1)
            cv2.rectangle(vis, (15, y_offset - th - 6), (25 + tw, y_offset + 6), (0, 0, 255), 2)
            cv2.putText(vis, f"ALERT: {alert}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 32
        return vis
