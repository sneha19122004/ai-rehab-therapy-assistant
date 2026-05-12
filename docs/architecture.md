# System Architecture

## Module Dependency Map

```
main.py
 ├── config.py              ← All constants, ACTION_MAP, landmark indices
 ├── user_input.py          ← CLI: builds task sequence from user input
 ├── landmark_detectors.py  ← Initializes MediaPipe pose & hand landmarkers
 ├── angle_processor.py     ← Per-frame logic: angles, rep counting, accuracy
 │    ├── geometry_utils.py ← 3D angle math, smoothing, visibility check
 │    └── comm_utils.py     ← UDP socket: broadcasts rep counts
 ├── video_utils.py         ← Demo video playback, ROM extraction, end-board
 │    └── landmark_detectors.py
 ├── ui_utils.py            ← OpenCV drawing: skeleton, landmarks, overlay text
 ├── audio_utils.py         ← pyttsx3 TTS engine
 ├── db_logger.py           ← MongoDB bulk-insert session logger
 └── landmark_refiner.py    ← Temporal smoothing + bone-length constraint refiner

romvidtest.py              ← Standalone: extract ROM from reference videos → JSON
comprehensive_calibration.py ← Standalone: full-body ROM calibration with live feed
udp_receiver.py            ← IoT: receives sensor packets, validates, stores to MongoDB
udp_sender.py              ← IoT: simulates sensor sending (for testing)
setup_indexes.py           ← One-time: creates MongoDB indexes
schema.py                  ← Pydantic model for IoT sensor data validation
accuracy_analyzer.py       ← Post-session ROM accuracy analysis utilities
```

## Data Flow

```
Camera Feed
    │
    ▼
MediaPipe (LIVE_STREAM mode)
    │  pose_callback()  hand_callback()
    ▼
Global Result Buffers (thread-safe lock)
    │
    ▼
process_landmarks_for_actions()   ← angle_processor.py
    │
    ├─► 3D Angle Calculation      ← geometry_utils.py
    ├─► Temporal Smoothing        ← HISTORY deque
    ├─► Rep Detection             ← is_at_max + is_rep_complete flags
    ├─► Accuracy %                ← user_ROM / reference_ROM × 100
    └─► UDP Broadcast             ← comm_utils.py → Node-RED
    │
    ▼
OpenCV Frame Overlay              ← ui_utils.py
    │
    ▼
MongoDB Session Log               ← db_logger.py
```

## Phase State Machine

```
DEMO_VIDEO (0)
    │  elapsed > VIDEO_DEMO_DURATION
    ▼
EXERCISE_TIMER (1)
    │
    ├── sub_phase: TRIAL_ROUND (0)
    │       Records min/max angles for auto-threshold calibration
    │       Duration: TRIAL_DURATION (15s)
    │
    └── sub_phase: EXERCISE_COUNT (1)
            Counts reps using threshold targets
            Duration: part_duration (task_duration / total_parts)
            │
            └── On all parts complete → display_end_board → next task
```

## Rep Counting Logic

```
smoothed_angle > MAX_TARGET - ANGLE_BUFFER  →  is_at_max = True
smoothed_angle < MIN_TARGET + ANGLE_BUFFER  AND is_at_max  →  rep complete
    │
    ├── rep_count += 1
    ├── Send UDP: "REP_COUNT|{code}|{count}"
    ├── Append to rep_history
    └── Reset flags
```
