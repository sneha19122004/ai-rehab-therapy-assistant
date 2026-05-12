# AI Rehab Therapy Assistant

An AI-powered physiotherapy rehabilitation assistant that uses **MediaPipe** pose and hand landmark detection to guide users through exercises, measure their Range of Motion (ROM), and track rep counts with real-time accuracy feedback.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Rehab Therapy Assistant                  │
├─────────────┬──────────────────────────┬────────────────────┤
│  Demo Phase │      Trial Phase          │   Exercise Phase   │
│  ─────────  │  ──────────────────────  │  ────────────────  │
│  Play ref   │  User performs exercise   │  Count reps with   │
│  video      │  to establish ROM         │  real-time angle   │
│  showing    │  baseline. Targets        │  tracking & TTS    │
│  ideal form │  auto-calculated.         │  audio feedback    │
└─────────────┴──────────────────────────┴────────────────────┘
```

---

## Features

- **Real-time Pose & Hand Tracking** using MediaPipe Landmarker Tasks API
- **Reference Video ROM Extraction** — automatically determines ideal ROM from demonstration videos
- **Multi-joint Support** — shoulder, elbow, wrist, and individual finger joints
- **Trial Round** — user warms up and the system auto-calibrates target angles
- **Rep Counting** — detects full extension and flexion cycles per joint
- **Accuracy Percentage** — compares user ROM vs reference video ROM
- **Text-to-Speech feedback** — announces "Good Job!" on rep completion
- **MongoDB logging** — session data persisted for progress tracking
- **UDP messaging** — sends rep count to Node-RED or other integrations
- **End-of-session scoreboard** — visual summary of all reps, ROM, and time

---

## Reference Exercise Videos

The system ships with 8 reference exercise videos used to extract ideal ROM:

| File | Exercise | Joint Code |
|------|----------|------------|
| `Shoulder Internal Rotation (R).mp4` | Shoulder Internal Rotation (R) | `R-SH` |
| `Shoulder Internal Rotation (L).mp4` | Shoulder Internal Rotation (L) | `L-SH` |
| `Shoulder Flexion with elbow at 90.mp4` | Shoulder Flexion with elbow at 90° | `R-SH` / `L-SH` |
| `Shoulder Flexion with elbow at 120.mp4` | Shoulder Flexion with elbow at 120° | `R-SH` / `L-SH` |
| `Shoulder Flexion at 90.mp4` | Shoulder Flexion at 90° | `R-SH` / `L-SH` |
| `Shoulder Flexion beyond 90.mp4` | Shoulder Flexion beyond 90° | `R-SH` / `L-SH` |
| `Reaching Exercise.mp4` | Reaching Exercise | `R-EL` / `L-EL` |
| `Finger Flexion & Extension.mp4` | Finger Flexion & Extension| `R-WI`, `R-FINGERS` |
| `Elbow Flexion & Extension.mp4` | Elbow Flexion & Extension| `R-EL`, `L-EL` |

## Project Structure

```
ai-rehab-therapy-assistant/
│
├── src/                          # Core application source code
│   ├── main.py                   # Entry point — orchestrates the full session loop
│   ├── config.py                 # All constants, landmark indices, ACTION_MAP
│   ├── angle_processor.py        # Per-frame angle calculation & rep counting logic
│   ├── geometry_utils.py         # 3D angle math and smoothing utilities
│   ├── landmark_detectors.py     # MediaPipe pose & hand landmarker initialization
│   ├── landmark_refiner.py       # Temporal smoothing & bone-length constraints
│   ├── ui_utils.py               # OpenCV skeleton/landmark drawing functions
│   ├── video_utils.py            # Demo video playback, end-board display, ROM extraction
│   ├── user_input.py             # CLI task sequence builder
│   ├── audio_utils.py            # pyttsx3 TTS engine setup
│   ├── comm_utils.py             # UDP socket for rep-count broadcasting
│   ├── db_logger.py              # MongoDB bulk-insert session logger
│   ├── schema.py                 # Pydantic validation schema for sensor data
│   ├── udp_receiver.py           # UDP listener + MongoDB writer (IoT integration)
│   ├── udp_sender.py             # UDP sender for sensor simulation/testing
│   ├── romvidtest.py             # Standalone ROM extraction script from reference videos
│   ├── accuracy_analyzer.py      # Post-session accuracy analysis utilities
│   ├── comprehensive_calibration.py  # Full-body ROM calibration tool
│   └── setup_indexes.py          # MongoDB index setup script (run once)
│
├── reference_videos/             # Place reference exercise .mp4 files here
│   └── .gitkeep
│
├── Video_ROM_Records/            # Auto-generated JSON ROM records from reference videos
│   └── .gitkeep
│
├── Exercise_ROM_Records/         # Auto-generated JSON ROM records from user sessions
│   └── .gitkeep
│
├── docs/
│   └── architecture.md           # System architecture notes
│
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Webcam
- MongoDB (optional, for session logging)

### 1. Clone the Repository

```bash
git clone https://github.com/sneha19122004/ai-rehab-therapy-assistant.git
cd ai-rehab-therapy-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download MediaPipe Models

Download the MediaPipe Task model files and place them at the paths set in `src/config.py`:

- **Pose Landmarker:** [`pose_landmarker.task`](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models)
- **Hand Landmarker:** [`hand_landmarker.task`](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models)

Update `src/config.py`:
```python
MODEL_POSE_PATH = 'path/to/pose_landmarker.task'
MODEL_HAND_PATH = 'path/to/hand_landmarker.task'
```

### 5. Add Reference Videos

Copy your reference `.mp4` exercise videos into `reference_videos/` and update the `VIDEO_PATHS` list in `src/config.py`.

### 6. (Optional) Set Up MongoDB

```bash
# Start MongoDB locally
mongod

# Create indexes (run once)
python src/setup_indexes.py
```

---

## Running the Application

### Main Rehab Session

```bash
python src/main.py
```

You will be prompted to define your exercise sequence in the terminal:

```
--- Define Sequential Tasks (CODE1;CODE2;...,DURATION_sec) ---
Available Codes: R-SH, L-SH, R-EL, L-EL, R-WI, L-WI, R-FINGERS, L-FINGERS

Task 1 (or DONE): R-EL,60
Task 2 (or DONE): L-EL,60
Task 3 (or DONE): DONE
```

**Format:** `JOINT_CODE(s),DURATION_IN_SECONDS`  
**Minimum duration:** 30 seconds per task

### Extract ROM from Reference Videos

```bash
python src/romvidtest.py
```

### Run Full-Body Calibration

```bash
python src/comprehensive_calibration.py
```

---

## Joint Codes

| Code | Joint | Type |
|------|-------|------|
| `R-SH` | Right Shoulder Flex/Extension | Pose |
| `L-SH` | Left Shoulder Flex/Extension | Pose |
| `R-EL` | Right Elbow Flexion | Pose |
| `L-EL` | Left Elbow Flexion | Pose |
| `R-WI` | Right Wrist Flexion | Hybrid (Pose + Hand) |
| `L-WI` | Left Wrist Flexion | Hybrid (Pose + Hand) |
| `R-FINGERS` | Right Hand Finger Flexion | Hand |
| `L-FINGERS` | Left Hand Finger Flexion | Hand |

---

## How It Works

### Phase 1 — Demo
The system plays a reference video demonstrating the correct exercise form. ROM is extracted from the video using MediaPipe's VIDEO mode with temporal smoothing.

### Phase 2 — Trial Round (15 seconds)
The user performs the exercise freely. The system records the min/max angle to auto-set rep-counting thresholds. If the range is too small (< 10°), the user is prompted to input manual thresholds.

### Phase 3 — Exercise Count
Rep counting begins. A rep is counted when:
1. The joint angle exceeds `MAX_TARGET - ANGLE_BUFFER` → `is_at_max = True`
2. The joint angle drops below `MIN_TARGET + ANGLE_BUFFER` → rep complete

Accuracy is calculated as:
```
accuracy_pct = min(100, (user_ROM / reference_ROM) × 100)
```

### Session End
An OpenCV scoreboard displays reps, duration, ROM range, and accuracy per exercise. Data is bulk-inserted into MongoDB.

---

## IoT / Node-RED Integration

Rep counts are broadcast over UDP on `127.0.0.1:5006` in the format:

```
REP_COUNT|JOINT_CODE|COUNT
# Example: REP_COUNT|R-EL|5
```

A `udp_receiver.py` is also included for receiving and storing IoT sensor data (e.g., wearable devices) from `127.0.0.1:5005`.

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `mediapipe` | Pose & hand landmark detection |
| `opencv-python` | Camera capture, frame rendering |
| `numpy` | Vector math for angle calculations |
| `pymongo` | MongoDB session data logging |
| `pyttsx3` | Text-to-speech feedback |
| `pydantic` | Data validation for IoT sensor schema |

---

## Roadmap

- [ ] GUI configuration screen (replace CLI input)
- [ ] Web dashboard for session history (Flask/FastAPI)
- [ ] Support for lower-body exercises (knee, ankle)
- [ ] Export session reports to PDF
- [ ] Real-time accuracy graph during exercise

---
## License

MIT License — see [LICENSE](LICENSE) for details.
