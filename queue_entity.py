"""
Queue Entity & Tracking History Data Models.
Decouples raw ByteTrack IDs from persistent logical Customer IDs (Q1, Q2, ...),
maintains cumulative dwell across combined QUEUE_ROI, and manages state lifecycles.
"""
from collections import deque
import numpy as np
import time

class TrackHistory:
    """
    Stores historical observations for a single ByteTrack ID.
    Tracks entry zone, trajectory, and recent presence.
    """
    def __init__(self, track_id, max_history=60):
        self.track_id = track_id
        self.max_history = max_history
        self.positions_norm = deque(maxlen=max_history)  # (nx, ny)
        self.positions_px = deque(maxlen=max_history)    # (x, y)
        self.timestamps = deque(maxlen=max_history)      # float seconds
        self.zones = deque(maxlen=max_history)           # zone names
        self.first_zone = None
        self.first_seen = None
        self.last_seen = None
        self.total_frames = 0

    def update(self, pos_px, pos_norm, bbox, timestamp, zone):
        if self.first_seen is None:
            self.first_seen = timestamp
            self.first_zone = zone

        self.last_seen = timestamp
        self.last_bbox = bbox
        self.total_frames += 1
        self.positions_norm.append(pos_norm)
        self.positions_px.append(pos_px)
        self.timestamps.append(timestamp)
        self.zones.append(zone)

    @property
    def last_position_px(self):
        return self.positions_px[-1] if self.positions_px else (0, 0)

    @property
    def last_position_norm(self):
        return self.positions_norm[-1] if self.positions_norm else (0.0, 0.0)

    @property
    def current_zone(self):
        return self.zones[-1] if self.zones else "OUTSIDE"

    @property
    def delta_x_norm(self):
        if len(self.positions_norm) < 2:
            return 0.0
        return self.positions_norm[-1][0] - self.positions_norm[0][0]


class QueueEntity:
    """
    Persistent logical Customer Queue Entity.
    Minimal States:
      - 'NOT_QUEUE'
      - 'CANDIDATE'
      - 'QUEUE'
      - 'TEMPORARILY_OUTSIDE'
    """
    _entity_counter = 0

    def __init__(self, initial_track_id, origin_region, initial_pos_norm, first_seen_time, initial_px=None, bbox=None):
        QueueEntity._entity_counter += 1
        self.entity_id = f"Q{QueueEntity._entity_counter}"
        self.current_track_id = initial_track_id
        self.previous_track_ids = []
        self.origin_region = origin_region
        
        # State
        self.state = "NOT_QUEUE"
        self.is_active = True

        # Temporal tracking
        self.first_seen_time = first_seen_time
        self.last_seen_time = first_seen_time
        self.candidate_start_time = None
        self.cumulative_dwell_sec = 0.0
        self.last_dwell_update_time = first_seen_time
        self.outside_since_ts = None
        self.queue_enter_time = None

        # Spatial tracking
        self.last_position_norm = initial_pos_norm
        self.last_position_px = initial_px or (0, 0)
        self.last_bbox = bbox or [0, 0, 0, 0]
        self.current_zone = origin_region
        self.last_position_norm = initial_pos_norm
        self.last_position_px = (0, 0)
        self.last_bbox = None
        self.current_zone = origin_region

    def reassociate(self, new_track_id, new_pos_norm, timestamp):
        """
        Seamlessly reassociates a new ByteTrack ID with this existing QueueEntity.
        """
        self.previous_track_ids.append(self.current_track_id)
        self.current_track_id = new_track_id
        self.last_seen_time = timestamp
        self.last_position_norm = new_pos_norm
        self.outside_since_ts = None
        self.is_active = True

    def get_wait_time(self, current_time):
        """
        Returns the duration (seconds) spent as a confirmed QUEUE member.
        """
        if self.queue_enter_time is None:
            return 0.0
        return max(0.0, current_time - self.queue_enter_time)

    def reset_candidate(self, timestamp):
        """
        Resets candidate dwell accumulation when grace period is exceeded.
        """
        self.state = "NOT_QUEUE"
        self.cumulative_dwell_sec = 0.0
        self.candidate_start_time = None
        self.outside_since_ts = None
        self.last_dwell_update_time = timestamp

    @classmethod
    def reset_counter(cls):
        cls._entity_counter = 0
