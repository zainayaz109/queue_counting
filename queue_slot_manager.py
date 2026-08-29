"""
Spatial Presence & Slot-Based Queue Management Engine (SPS-Queue).
Decouples queue dwell and wait times from volatile low-level Track IDs.
Uses 1-to-1 Optimal Hungarian Bipartite Matching to prevent ID fluctuation across nearby persons.
"""
import time
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

import config
from config import (
    PINK_ZONE_NORM,
    RED_ZONE_NORM,
    GREEN_ZONE_NORM,
    QUEUE_DWELL_SECONDS,
    CANDIDATE_EXIT_GRACE_SECONDS,
    QUEUE_EXIT_GRACE_SECONDS
)

def calculate_bbox_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class QueueSlot:
    """
    Represents a distinct physical customer space / occupant in the queue.
    Maintains continuous dwell and wait times across track ID switches and brief occlusions.
    """
    _slot_counter = 0

    def __init__(self, initial_pos_norm, initial_px, bbox, initial_zone, initial_track_id, timestamp):
        QueueSlot._slot_counter += 1
        self.slot_id = f"Q{QueueSlot._slot_counter}"
        self.entity_id = self.slot_id  # Compatible alias
        self.zone = initial_zone
        self.current_zone = initial_zone
        self.origin_region = initial_zone
        
        # Spatial Position (EMA smoothed anchor)
        self.anchor_norm = initial_pos_norm
        self.last_position_norm = initial_pos_norm
        self.last_position_px = initial_px or (0, 0)
        self.last_bbox = bbox or [0, 0, 0, 0]
        self.current_track_id = initial_track_id
        self.associated_track_ids = [initial_track_id] if initial_track_id else []
        self.previous_track_ids = self.associated_track_ids
        
        # State: 'CANDIDATE', 'QUEUE', 'OCCLUDED'
        self.state = "CANDIDATE"
        self.is_active = True
        
        # Temporal metrics
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.candidate_start_time = timestamp
        self.cumulative_dwell_sec = 0.0
        self.last_dwell_update_time = timestamp
        self.occluded_since_ts = None
        self.outside_since_ts = None
        self.queue_enter_time = None

    @classmethod
    def reset_counter(cls):
        cls._slot_counter = 0

    def get_wait_time(self, current_time=None):
        if not self.queue_enter_time:
            return 0.0
        now = current_time or time.time()
        return max(0.0, now - self.queue_enter_time)

    def update_matched(self, foot_norm, foot_px, bbox, zone, track_id, timestamp):
        """
        Updates the slot when a detection matches this spatial customer position.
        """
        dt = timestamp - self.last_seen_time
        dt = min(max(dt, 0.0), 0.5)

        # Smooth anchor position (EMA filter)
        alpha = 0.4
        self.anchor_norm = (
            alpha * foot_norm[0] + (1 - alpha) * self.anchor_norm[0],
            alpha * foot_norm[1] + (1 - alpha) * self.anchor_norm[1]
        )
        self.last_position_norm = self.anchor_norm
        self.last_position_px = foot_px
        self.last_bbox = bbox
        self.zone = zone
        self.current_zone = zone
        self.last_seen_time = timestamp
        self.occluded_since_ts = None
        self.outside_since_ts = None
        self.is_active = True

        if track_id not in self.associated_track_ids:
            self.associated_track_ids.append(track_id)
        self.current_track_id = track_id

        # Accumulate dwell
        self.cumulative_dwell_sec += dt
        if self.state in ["CANDIDATE", "OCCLUDED"]:
            if self.cumulative_dwell_sec >= QUEUE_DWELL_SECONDS:
                self.state = "QUEUE"
                if not self.queue_enter_time:
                    self.queue_enter_time = timestamp
                print(f"[SPS-Queue] {self.slot_id} [Track {track_id}] -> Promoted to QUEUE (Dwell >= {QUEUE_DWELL_SECONDS}s)")
            else:
                self.state = "CANDIDATE"

    def mark_occluded(self, timestamp):
        """
        Marks slot as temporarily occluded (e.g. customer bent down or blocked by cart/counter).
        """
        if self.occluded_since_ts is None:
            self.occluded_since_ts = timestamp
            self.outside_since_ts = timestamp

    def is_expired(self, current_time):
        """
        Checks if slot has been missing longer than the grace period.
        """
        if self.occluded_since_ts is None:
            return False
        grace_limit = QUEUE_EXIT_GRACE_SECONDS if self.state == "QUEUE" else CANDIDATE_EXIT_GRACE_SECONDS
        return (current_time - self.occluded_since_ts) > grace_limit


class SpatialSlotManager:
    """
    Manages all physical queue slots across PINK, RED, and GREEN regions.
    Uses 1-to-1 Optimal Hungarian Bipartite Assignment to prevent slot fluctuation.
    """
    def __init__(self, slot_radius=0.08, iou_thresh=0.20):
        self.slot_radius = slot_radius
        self.iou_thresh = iou_thresh
        self.slots = {}  # slot_id -> QueueSlot

    def get_zone_at_point(self, norm_point):
        pt = (float(norm_point[0]), float(norm_point[1]))
        if getattr(config, "ENABLE_YELLOW_ZONE", False) and hasattr(config, "YELLOW_ZONE_NORM"):
            if cv2.pointPolygonTest(config.YELLOW_ZONE_NORM, pt, False) >= 0:
                return "YELLOW"
        if cv2.pointPolygonTest(GREEN_ZONE_NORM, pt, False) >= 0:
            return "GREEN"
        if cv2.pointPolygonTest(RED_ZONE_NORM, pt, False) >= 0:
            return "RED"
        if cv2.pointPolygonTest(PINK_ZONE_NORM, pt, False) >= 0:
            return "PINK"
        return "OUTSIDE"

    def is_in_queue_roi(self, zone):
        valid_zones = ["PINK", "RED", "GREEN"]
        if getattr(config, "ENABLE_YELLOW_ZONE", False):
            valid_zones.append("YELLOW")
        return zone in valid_zones

    def process_frame(self, detections, timestamp=None):
        """
        Processes detections using 1-to-1 Hungarian Matching and spatial slot clustering.
        Returns: list of active QueueSlot objects.
        """
        current_time = timestamp or time.time()
        enable_yellow = getattr(config, "ENABLE_YELLOW_ZONE", False)
        
        # 1. Classify detections into queue ROI vs outside
        queue_dets = []
        outside_dets = []

        for det in detections:
            foot_norm = det["norm_foot"]
            zone = self.get_zone_at_point(foot_norm)
            det["zone"] = zone
            
            # Check for Direct LEFT -> GREEN entrance filter (Passersby outside counter)
            is_direct_left = (not enable_yellow and zone == "GREEN" and foot_norm[0] < 0.15)

            if self.is_in_queue_roi(zone) and not is_direct_left:
                queue_dets.append(det)
            else:
                outside_dets.append(det)

        # 2. Optimal 1-to-1 Hungarian Matching (Bipartite Assignment)
        active_slots_list = [s for s in self.slots.values() if s.is_active]
        matched_slot_ids = set()
        unmatched_det_indices = set(range(len(queue_dets)))

        if len(active_slots_list) > 0 and len(queue_dets) > 0:
            cost_matrix = np.full((len(active_slots_list), len(queue_dets)), 1000.0, dtype=np.float32)

            for i, slot in enumerate(active_slots_list):
                for j, det in enumerate(queue_dets):
                    foot_norm = det["norm_foot"]
                    bbox = det["bbox"]
                    track_id = det["track_id"]

                    dist = np.sqrt((foot_norm[0] - slot.anchor_norm[0])**2 + (foot_norm[1] - slot.anchor_norm[1])**2)
                    iou = calculate_bbox_iou(bbox, slot.last_bbox)
                    is_same_track = (track_id == slot.current_track_id or track_id in slot.associated_track_ids)

                    # Cost scoring
                    if is_same_track:
                        cost_matrix[i, j] = dist * 0.1  # Highest priority for identical Track ID
                    elif dist <= self.slot_radius or iou >= self.iou_thresh:
                        cost_matrix[i, j] = dist - (0.4 * iou)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 500.0:
                    slot = active_slots_list[r]
                    det = queue_dets[c]
                    slot.update_matched(
                        det["norm_foot"], det["bottom_center"], det["bbox"],
                        det["zone"], det["track_id"], current_time
                    )
                    matched_slot_ids.add(slot.slot_id)
                    unmatched_det_indices.discard(c)

        # 3. Create New Slots for Unmatched Detections inside Queue ROI
        for j in unmatched_det_indices:
            det = queue_dets[j]
            foot_norm = det["norm_foot"]
            bbox = det["bbox"]
            zone = det["zone"]
            track_id = det["track_id"]
            foot_px = det["bottom_center"]

            # Filter out duplicate bounding boxes on an already matched slot
            is_duplicate = False
            for matched_sid in matched_slot_ids:
                s = self.slots[matched_sid]
                dist = np.sqrt((foot_norm[0] - s.anchor_norm[0])**2 + (foot_norm[1] - s.anchor_norm[1])**2)
                iou = calculate_bbox_iou(bbox, s.last_bbox)
                if dist < 0.04 or (dist < 0.06 and iou > 0.40):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            new_slot = QueueSlot(
                initial_pos_norm=foot_norm,
                initial_px=foot_px,
                bbox=bbox,
                initial_zone=zone,
                initial_track_id=track_id,
                timestamp=current_time
            )
            self.slots[new_slot.slot_id] = new_slot
            matched_slot_ids.add(new_slot.slot_id)
            print(f"[SPS-Queue] Created new {new_slot.slot_id} at {zone} [Track {track_id}] (CANDIDATE)")

        # 4. Handle Unmatched Slots (Enter Occlusion Grace Buffer)
        for slot_id, slot in list(self.slots.items()):
            if slot_id not in matched_slot_ids:
                slot.mark_occluded(current_time)
                if slot.is_expired(current_time):
                    slot.is_active = False
                    slot.state = "NOT_QUEUE"
                    print(f"[SPS-Queue] Slot {slot.slot_id} grace expired -> REMOVED from queue.")
                    del self.slots[slot_id]

        # 5. Spatial Deduplication / Slot Fusion (Anti-Inflation)
        # Tight threshold: only fuse if foot distance < 0.04 AND bbox IoU > 0.50
        self._fuse_overlapping_slots(current_time)

        # 6. Format output entities
        active_slots = [s for s in self.slots.values() if s.is_active]
        return active_slots

    def _fuse_overlapping_slots(self, current_time):
        """
        Fuses duplicate slots that are literally overlapping on the exact same person.
        """
        sids = list(self.slots.keys())
        merged = set()

        for i in range(len(sids)):
            sid1 = sids[i]
            if sid1 in merged or sid1 not in self.slots:
                continue
            s1 = self.slots[sid1]
            if not s1.is_active:
                continue

            for j in range(i + 1, len(sids)):
                sid2 = sids[j]
                if sid2 in merged or sid2 not in self.slots:
                    continue
                s2 = self.slots[sid2]
                if not s2.is_active:
                    continue

                dist = np.sqrt((s1.anchor_norm[0] - s2.anchor_norm[0])**2 + (s1.anchor_norm[1] - s2.anchor_norm[1])**2)
                iou = calculate_bbox_iou(s1.last_bbox, s2.last_bbox)

                # Strict condition: only fuse if virtually identical location (same human body)
                if dist < 0.04 or (dist < 0.06 and iou > 0.50):
                    s1.cumulative_dwell_sec = max(s1.cumulative_dwell_sec, s2.cumulative_dwell_sec)
                    if s2.queue_enter_time:
                        if s1.queue_enter_time:
                            s1.queue_enter_time = min(s1.queue_enter_time, s2.queue_enter_time)
                        else:
                            s1.queue_enter_time = s2.queue_enter_time

                    if s2.state == "QUEUE":
                        s1.state = "QUEUE"

                    for tid in s2.associated_track_ids:
                        if tid not in s1.associated_track_ids:
                            s1.associated_track_ids.append(tid)

                    s2.is_active = False
                    s2.state = "NOT_QUEUE"
                    merged.add(sid2)
                    print(f"[SPS-Queue] Fused duplicate slot {sid2} into {sid1} (State: {s1.state})")

        for sid in merged:
            if sid in self.slots:
                del self.slots[sid]
