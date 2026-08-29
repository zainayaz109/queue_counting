"""
Queue Analytics & Alert Engine.
Computes real-time queue length, customer wait times, and evaluates alert conditions.
"""
import time
from config import (
    QUEUE_LENGTH_ALERT_THRESHOLD,
    QUEUE_LENGTH_SUSTAINED_SEC,
    EXCESSIVE_WAIT_ALERT_SEC,
    ALERT_COOLDOWN_SEC
)

class QueueAnalytics:
    def __init__(self,
                 length_threshold=QUEUE_LENGTH_ALERT_THRESHOLD,
                 sustained_duration=QUEUE_LENGTH_SUSTAINED_SEC,
                 max_wait=EXCESSIVE_WAIT_ALERT_SEC,
                 cooldown=ALERT_COOLDOWN_SEC):
        self.length_threshold = length_threshold
        self.sustained_duration = sustained_duration
        self.max_wait = max_wait
        self.cooldown = cooldown

        self.threshold_exceeded_start_time = None
        self.last_length_alert_time = 0.0
        self.last_wait_alert_time = 0.0

    def compute_analytics(self, active_entities, current_time=None):
        """
        Computes queue metrics across active entities:
        queue_count = unique active QUEUE members
        """
        now = current_time or time.time()

        queue_members = [e for e in active_entities if e.state == "QUEUE"]
        candidates = [e for e in active_entities if e.state == "CANDIDATE"]
        temp_outside = [e for e in active_entities if e.state == "TEMPORARILY_OUTSIDE"]

        queue_count = len(queue_members)
        wait_times = [e.get_wait_time(now) for e in queue_members]
        max_wait = max(wait_times) if wait_times else 0.0
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0

        active_alerts = []

        # 1. Queue Length Alert Evaluation (Length >= threshold for >= sustained_duration)
        if queue_count >= self.length_threshold:
            if self.threshold_exceeded_start_time is None:
                self.threshold_exceeded_start_time = now
            
            exceeded_duration = now - self.threshold_exceeded_start_time
            if exceeded_duration >= self.sustained_duration:
                if (now - self.last_length_alert_time) >= self.cooldown:
                    alert_msg = f"QUEUE LENGTH ALERT: {queue_count} customers waiting (Sustained for {exceeded_duration:.1f}s)"
                    active_alerts.append(alert_msg)
                    self.last_length_alert_time = now
                    print(f"\n[ALERT TRIGGERED] {alert_msg}\n")
        else:
            self.threshold_exceeded_start_time = None

        # 2. Excessive Wait Alert Evaluation
        if max_wait >= self.max_wait:
            if (now - self.last_wait_alert_time) >= self.cooldown:
                alert_msg = f"EXCESSIVE WAIT ALERT: Max wait time reached {max_wait:.1f}s (Threshold: {self.max_wait}s)"
                active_alerts.append(alert_msg)
                self.last_wait_alert_time = now
                print(f"\n[ALERT TRIGGERED] {alert_msg}\n")

        return {
            "queue_length": queue_count,
            "candidate_count": len(candidates),
            "temp_outside_count": len(temp_outside),
            "max_wait_sec": max_wait,
            "avg_wait_sec": avg_wait,
            "active_alerts": active_alerts,
            "queue_entities": queue_members
        }
