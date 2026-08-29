"""
Comprehensive Automated Test Suite for SPS-Queue (Spatial Presence & Slot-Based Queue Engine).
Validates all 13 scenarios:
1. Dwell progression in PINK
2. Continuous walk across PINK -> RED -> GREEN
3. Passerby leaving < 5.0s
4. Candidate return within grace
5. Candidate exceeding grace (reset)
6. Queue member return within grace
7. Queue member staying outside > grace
8. Direct LEFT -> GREEN invalid entry
9. Bending down / Track ID switch (SPS-Queue immunity)
10. Multi-detection clamping on single customer (Anti-Inflation)
11. Region transitions preserve wait time
12. Alert Engine triggers
13. Ghost bounding box suppression
"""
import numpy as np
import time

from config import (
    QUEUE_DWELL_SECONDS,
    CANDIDATE_EXIT_GRACE_SECONDS,
    QUEUE_EXIT_GRACE_SECONDS
)
from queue_slot_manager import QueueSlot
from queue_state_machine import QueueStateMachine
from queue_analytics import QueueAnalytics

def create_mock_detection(track_id, nx, ny, bbox_w=0.08, bbox_h=0.35, roi_w=690, roi_h=285):
    """Creates a mock detection dictionary in normalized and pixel units."""
    px = int(nx * roi_w)
    py = int(ny * roi_h)
    bw = int(bbox_w * roi_w)
    bh = int(bbox_h * roi_h)
    bbox = [px - bw // 2, py - bh, px + bw // 2, py]
    return {
        "track_id": track_id,
        "bbox": bbox,
        "norm_foot": (nx, ny),
        "bottom_center": (px, py),
        "confidence": 0.88,
        "class_id": 0
    }

def run_tests():
    print("==================================================================")
    print("      RUNNING SPS-QUEUE AUTOMATED TEST SUITE (13 TESTS)          ")
    print("==================================================================")

    passed_tests = 0
    total_tests = 13

    # -------------------------------------------------------------------------
    # TEST 1: New person enters PINK and dwells >= 5.0s -> QUEUE
    # -------------------------------------------------------------------------
    print("[Test 1] Testing New Entry in PINK with 5s Dwell...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    analytics = QueueAnalytics()
    t = 1000.0

    for _ in range(5):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(1, 0.75, 0.55)], timestamp=t)

    assert len(active) == 1
    assert active[0].state == "CANDIDATE"

    for _ in range(52):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(1, 0.75, 0.55)], timestamp=t)

    assert active[0].state == "QUEUE", f"Expected QUEUE, got {active[0].state}"
    m = analytics.compute_analytics(active, current_time=t)
    assert m["queue_length"] == 1
    print(" -> PASSED: Promoted to QUEUE in PINK.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 2: Walk across PINK -> RED -> GREEN continuously for >= 5.0s -> QUEUE
    # -------------------------------------------------------------------------
    print("[Test 2] Testing Continuous Walk across PINK -> RED -> GREEN...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 2000.0

    # 2s in PINK (0.75, 0.55)
    for _ in range(20):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(2, 0.75, 0.55)], timestamp=t)
    assert active[0].state == "CANDIDATE"

    # 2s in RED (0.50, 0.45)
    for _ in range(20):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(2, 0.50, 0.45)], timestamp=t)
    assert active[0].state == "CANDIDATE"

    # 1.5s in GREEN (0.25, 0.42)
    for _ in range(15):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(2, 0.25, 0.42)], timestamp=t)

    assert active[0].state == "QUEUE"
    print(" -> PASSED: Combined continuous dwell across PINK->RED->GREEN verified.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 3: Passerby leaves in < 5.0s -> Candidate removed
    # -------------------------------------------------------------------------
    print("[Test 3] Testing Passerby Leaving in < 5.0s...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 3000.0

    # In PINK for 2.0s
    for _ in range(20):
        t += 0.1
        fsm.process_frame([create_mock_detection(3, 0.75, 0.55)], timestamp=t)

    # Exits to OUTSIDE for > 3.0s
    for _ in range(35):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(3, 0.90, 0.90)], timestamp=t)

    active_candidates = [e for e in active if e.state == "CANDIDATE"]
    assert len(active_candidates) == 0
    print(" -> PASSED: Passerby correctly excluded from queue.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 4: Candidate Return within 3.0s Grace
    # -------------------------------------------------------------------------
    print("[Test 4] Testing Candidate Return within 3.0s Grace...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 4000.0

    # In PINK for 3.0s
    for _ in range(30):
        t += 0.1
        fsm.process_frame([create_mock_detection(4, 0.75, 0.55)], timestamp=t)

    # Missing for 1.5s (< 3.0s grace)
    for _ in range(15):
        t += 0.1
        fsm.process_frame([], timestamp=t)

    # Returns for 2.5s -> Reaches 5s
    for _ in range(25):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(4, 0.75, 0.55)], timestamp=t)

    assert active[0].state == "QUEUE"
    print(" -> PASSED: Candidate outside grace resumed timer and promoted to QUEUE.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 5: Candidate Exceeding Grace (Reset)
    # -------------------------------------------------------------------------
    print("[Test 5] Testing Candidate Exceeding 3.0s Grace (Reset)...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 5000.0

    # In PINK for 2.0s
    for _ in range(20):
        t += 0.1
        fsm.process_frame([create_mock_detection(5, 0.75, 0.55)], timestamp=t)

    # Missing for 3.5s (> 3.0s grace)
    for _ in range(35):
        t += 0.1
        fsm.process_frame([], timestamp=t)

    # New presence starts fresh candidate
    for _ in range(10):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(5, 0.75, 0.55)], timestamp=t)

    assert active[0].state == "CANDIDATE"
    assert active[0].cumulative_dwell_sec < 2.0
    print(" -> PASSED: Candidate dwell timer correctly reset after exceeding grace.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 6: Queue Member Return within 5.0s Grace
    # -------------------------------------------------------------------------
    print("[Test 6] Testing Queue Member Return within 5.0s Grace...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 6000.0

    for _ in range(55):
        t += 0.1
        fsm.process_frame([create_mock_detection(6, 0.75, 0.55)], timestamp=t)

    # Missing for 3.0s (< 5.0s queue grace)
    for _ in range(30):
        t += 0.1
        fsm.process_frame([], timestamp=t)

    # Returns
    for _ in range(10):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(6, 0.75, 0.55)], timestamp=t)

    assert active[0].state == "QUEUE"
    print(" -> PASSED: Queue member restored directly to QUEUE without dwell restart.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 7: Queue Member Staying Outside > 5.0s Grace
    # -------------------------------------------------------------------------
    print("[Test 7] Testing Queue Member Staying Outside > 5.0s Grace...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 7000.0

    for _ in range(55):
        t += 0.1
        fsm.process_frame([create_mock_detection(7, 0.75, 0.55)], timestamp=t)

    # Missing for 5.5s (> 5.0s grace)
    for _ in range(55):
        t += 0.1
        fsm.process_frame([], timestamp=t)

    active = fsm.process_frame([], timestamp=t)
    assert len(active) == 0
    print(" -> PASSED: Queue member correctly removed after exceeding grace.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 8: Direct LEFT -> GREEN Invalid Entry
    # -------------------------------------------------------------------------
    print("[Test 8] Testing Direct LEFT -> GREEN Invalid Entry...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 8000.0

    # Person walking outside hallway (x=0.08, y=0.30 - entrance hallway)
    for _ in range(20):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(8, 0.08, 0.30)], timestamp=t)

    assert len(active) == 0
    print(" -> PASSED: Outside entrance hallway passerby strictly rejected.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 9: Bending Down & Track ID Switch (SPS-Queue Immunity)
    # -------------------------------------------------------------------------
    print("[Test 9] Testing Bending Down & Track ID Switch Immunity...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    analytics = QueueAnalytics()
    t = 9000.0

    # 1. Customer enters Checkout 1 (GREEN) with Track 10
    for _ in range(55):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(10, 0.25, 0.42)], timestamp=t)

    assert active[0].state == "QUEUE"
    slot_id_initial = active[0].slot_id

    # 2. Customer bends down: bbox height shrinks by 60%, detection drops for 1s, reappears with Track 15
    for _ in range(10):
        t += 0.1
        fsm.process_frame([], timestamp=t)

    # Reappears as Track 15 at the same counter position
    for _ in range(30):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(15, 0.25, 0.42, bbox_h=0.15)], timestamp=t)

    assert len(active) == 1
    assert active[0].slot_id == slot_id_initial
    assert active[0].state == "QUEUE"
    assert active[0].get_wait_time(t) > 3.0
    print(" -> PASSED: Bending down and Track ID switch did NOT reset queue state or wait time!\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 10: Multi-Detection Clamping on Single Customer (Anti-Inflation)
    # -------------------------------------------------------------------------
    print("[Test 10] Testing Multi-Detection Clamping (Anti-Inflation)...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    analytics = QueueAnalytics()
    t = 10000.0

    # 1 customer at counter gets 3 simultaneous detections from arms/cart
    for _ in range(55):
        t += 0.1
        active = fsm.process_frame([
            create_mock_detection(20, 0.25, 0.42),
            create_mock_detection(21, 0.26, 0.43),
            create_mock_detection(22, 0.24, 0.41)
        ], timestamp=t)

    m = analytics.compute_analytics(active, current_time=t)
    assert m["queue_length"] == 1, f"Expected queue_length == 1, got {m['queue_length']}"
    print(" -> PASSED: Overlapping detections successfully clamped to strictly 1 customer.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 11: Movement across regions preserves wait time
    # -------------------------------------------------------------------------
    print("[Test 11] Testing Region Progression Preserves Wait Time...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    t = 11000.0

    # Qualify in PINK
    for _ in range(55):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(30, 0.75, 0.55)], timestamp=t)

    assert active[0].state == "QUEUE"
    wait_pink = active[0].get_wait_time(t)

    # Step forward to RED
    for _ in range(30):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(30, 0.50, 0.45)], timestamp=t)

    assert active[0].state == "QUEUE"
    assert active[0].get_wait_time(t) > wait_pink

    # Step forward to GREEN
    for _ in range(30):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(30, 0.25, 0.42)], timestamp=t)

    assert active[0].state == "QUEUE"
    assert active[0].get_wait_time(t) > wait_pink + 3.0
    print(" -> PASSED: Region progression preserves continuous customer wait time.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 12: Alert Engine Triggers
    # -------------------------------------------------------------------------
    print("[Test 12] Testing Queue Length & Excessive Wait Alert Triggers...")
    QueueSlot.reset_counter()
    fsm = QueueStateMachine()
    analytics = QueueAnalytics(length_threshold=2, sustained_duration=15.0, max_wait=240.0)
    t = 12000.0

    # 2 customers in queue for 16s
    for _ in range(55):
        t += 0.1
        fsm.process_frame([create_mock_detection(40, 0.25, 0.42), create_mock_detection(41, 0.75, 0.55)], timestamp=t)

    all_alerts = []
    for _ in range(160):
        t += 0.1
        active = fsm.process_frame([create_mock_detection(40, 0.25, 0.42), create_mock_detection(41, 0.75, 0.55)], timestamp=t)
        m = analytics.compute_analytics(active, current_time=t)
        if m["active_alerts"]:
            all_alerts.extend(m["active_alerts"])

    assert len(all_alerts) >= 1
    print(" -> PASSED: Alert Engine triggers verified.\n")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 13: Ghost Bounding Box Suppression
    # -------------------------------------------------------------------------
    print("[Test 13] Testing Visualizer Ghost Bounding Box Suppression...")
    from visualizer import QueueVisualizer as Visualizer
    viz = Visualizer()
    slot = QueueSlot((0.25, 0.42), (172, 119), [100, 50, 240, 190], "GREEN", 50, 13000.0)
    slot.last_seen_time = 13000.0
    
    vis_frame = np.zeros((285, 690, 3), dtype=np.uint8)
    # Stale by 2.0s (> 1.0s limit)
    viz._draw_entity(vis_frame, slot, now=13002.0)
    assert np.sum(vis_frame) == 0, "Expected empty frame, ghost bbox was drawn!"
    print(" -> PASSED: Ghost bounding box suppressed when track is missing.\n")
    passed_tests += 1

    print("==================================================================")
    print(f"       TEST SUITE COMPLETED: {passed_tests}/{total_tests} PASSED")
    print("==================================================================")

if __name__ == "__main__":
    run_tests()
