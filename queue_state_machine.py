"""
Unified 3-Zone Queue State Machine powered by SPS-Queue (Spatial Presence & Slot Engine).
Manages customer lifecycle across PINK, RED, and GREEN regions.
Decouples dwell and wait times from volatile low-level Track IDs.
"""
import time
import numpy as np
import cv2

from config import (
    PINK_ZONE_NORM,
    RED_ZONE_NORM,
    GREEN_ZONE_NORM,
    QUEUE_DWELL_SECONDS,
    CANDIDATE_EXIT_GRACE_SECONDS,
    QUEUE_EXIT_GRACE_SECONDS
)

from queue_slot_manager import SpatialSlotManager, QueueSlot

class QueueStateMachine:
    def __init__(self, occlusion_manager=None, slot_radius=0.18, iou_thresh=0.25):
        self.slot_manager = SpatialSlotManager(slot_radius=slot_radius, iou_thresh=iou_thresh)
        self.active_entities = self.slot_manager.slots

    def get_zone_at_point(self, norm_point):
        return self.slot_manager.get_zone_at_point(norm_point)

    def is_in_queue_roi(self, zone):
        return self.slot_manager.is_in_queue_roi(zone)

    def process_frame(self, detections, timestamp=None):
        """
        Processes frame detections via Spatial Presence & Slot Manager.
        """
        active_slots = self.slot_manager.process_frame(detections, timestamp=timestamp)
        self.active_entities = self.slot_manager.slots
        return active_slots
