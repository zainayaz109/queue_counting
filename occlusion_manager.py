"""
Occlusion Management & Reassociation Engine.
Seamlessly reassociates dropped/lost ByteTrack IDs for active queue members and candidates
without duplicate counting or restarting wait/dwell timers.
"""
import numpy as np
import time

from config import (
    TRACK_REASSOCIATION_SECONDS,
    SPATIAL_CONTINUITY_TOLERANCE
)

class PendingOcclusionRecord:
    def __init__(self, entity, last_pos_norm, timestamp):
        self.entity = entity
        self.entity_id = entity.entity_id
        self.old_track_id = entity.current_track_id
        self.last_position_norm = last_pos_norm
        self.timestamp = timestamp
        self.saved_state = entity.state
        self.saved_dwell = entity.cumulative_dwell_sec

    def is_expired(self, current_time):
        return (current_time - self.timestamp) > TRACK_REASSOCIATION_SECONDS


class OcclusionManager:
    """
    Manages pending occlusion states for queue members and candidates, matching emerging tracks.
    """
    def __init__(self, reassociation_window=TRACK_REASSOCIATION_SECONDS,
                 tolerance=SPATIAL_CONTINUITY_TOLERANCE):
        self.reassociation_window = reassociation_window
        self.tolerance = tolerance
        self.pending_records = {}  # entity_id -> PendingOcclusionRecord

    def register_occlusion(self, entity, last_pos_norm, timestamp):
        """
        Registers a lost track for an active QUEUE or CANDIDATE member.
        """
        if entity.state not in ["QUEUE", "CANDIDATE", "TEMPORARILY_OUTSIDE"]:
            return False

        record = PendingOcclusionRecord(entity, last_pos_norm, timestamp)
        self.pending_records[entity.entity_id] = record
        print(f"[Occlusion Manager] Registered pending occlusion for {entity.entity_id} "
              f"(Track {entity.current_track_id}, State: {entity.state}) at t={timestamp:.2f}")
        return True

    def attempt_reassociation(self, new_track_id, new_pos_norm, timestamp):
        """
        Attempts to match a newly appearing track with an existing pending queue entity or candidate.
        Returns: matched QueueEntity or None.
        """
        self.cleanup_expired(timestamp)

        if not self.pending_records:
            return None

        candidates = []
        for entity_id, record in list(self.pending_records.items()):
            dt = timestamp - record.timestamp
            if dt > self.reassociation_window:
                continue

            p_last = record.last_position_norm
            dist = np.sqrt((new_pos_norm[0] - p_last[0])**2 + (new_pos_norm[1] - p_last[1])**2)

            if dist <= self.tolerance:
                score = dist + (0.05 * dt)
                candidates.append((score, record))

        if not candidates:
            return None

        # Select closest matching candidate
        candidates.sort(key=lambda x: x[0])
        best_score, best_record = candidates[0]

        matched_entity = best_record.entity
        matched_entity.reassociate(new_track_id, new_pos_norm, timestamp)
        
        # Restore appropriate state
        if best_record.saved_state == "QUEUE" or best_record.saved_state == "TEMPORARILY_OUTSIDE":
            matched_entity.state = "QUEUE"
            print(f"[Occlusion Manager] SUCCESSFUL REASSOCIATION: New Track {new_track_id} -> {matched_entity.entity_id} "
                  f"(QUEUE Restored, Wait Time Preserved: {matched_entity.get_wait_time(timestamp):.1f}s)")
        else:
            # Preserve accumulated candidate dwell
            matched_entity.state = "CANDIDATE"
            matched_entity.cumulative_dwell_sec = best_record.saved_dwell
            matched_entity.candidate_start_time = timestamp - best_record.saved_dwell
            print(f"[Occlusion Manager] SUCCESSFUL REASSOCIATION: New Track {new_track_id} -> {matched_entity.entity_id} "
                  f"(CANDIDATE Restored, Dwell Preserved: {matched_entity.cumulative_dwell_sec:.1f}s)")

        matched_entity.is_active = True
        del self.pending_records[matched_entity.entity_id]

        return matched_entity

    def cleanup_expired(self, current_time):
        """
        Removes expired pending records and transitions entity to NOT_QUEUE.
        """
        expired_ids = [eid for eid, rec in self.pending_records.items() if rec.is_expired(current_time)]
        for eid in expired_ids:
            rec = self.pending_records[eid]
            rec.entity.state = "NOT_QUEUE"
            rec.entity.is_active = False
            print(f"[Occlusion Manager] Expired pending occlusion for {eid} -> NOT_QUEUE.")
            del self.pending_records[eid]
