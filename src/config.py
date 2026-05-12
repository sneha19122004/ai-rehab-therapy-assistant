import numpy as np
from collections import deque
ROM_CALIBRATION = "ROM_CALIBRATION"
# Model Paths
MODEL_POSE_PATH = 'C:/Users/sneha/pose_landmarker.task'
MODEL_HAND_PATH = 'C:/Users/sneha/hand_landmarker.task'

S_L, E_L, W_L, S_R, E_R, W_R = 11, 13, 15, 12, 14, 16
H_L, K_L, A_L, H_R, K_R, A_R = 23, 25, 27, 24, 26, 28
F2_L, F2_R = 19, 20

# Hand Landmarks
HAND_LM_WRIST = 0
HAND_LM_F2 = 5
HAND_LM_INDEX_MCP = 5
HAND_LM_INDEX_PIP = 8
HAND_LM_MIDDLE_MCP = 9
HAND_LM_MIDDLE_PIP = 12
HAND_LM_RING_MCP = 13
HAND_LM_RING_PIP = 16
HAND_LM_PINKY_MCP = 17
HAND_LM_PINKY_PIP = 20
HAND_LM_THUMB_MCP = 2
HAND_LM_THUMB_IP = 4

MP_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (9, 10), (10, 11), (11, 12),           # Middle finger
    (0, 9), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (5, 9), (9, 13), (13, 17)              # Palm/Metacarpals
]

HAND_FINGER_ACTIONS = {
    'FI1': {'name': 'Index Flex', 'indices': (HAND_LM_WRIST, HAND_LM_INDEX_MCP, HAND_LM_INDEX_PIP)},
    'FI2': {'name': 'Middle Flex', 'indices': (HAND_LM_WRIST, HAND_LM_MIDDLE_MCP, HAND_LM_MIDDLE_PIP)},
    'FI3': {'name': 'Ring Flex', 'indices': (HAND_LM_WRIST, HAND_LM_RING_MCP, HAND_LM_RING_PIP)},
    'FI4': {'name': 'Pinky Flex', 'indices': (HAND_LM_WRIST, HAND_LM_PINKY_MCP, HAND_LM_PINKY_PIP)},
    'FI5': {'name': 'Thumb Flex', 'indices': (HAND_LM_WRIST, HAND_LM_THUMB_MCP, HAND_LM_THUMB_IP)},
}

HAND_TRACKING_DETAILS = {
    'R': {'type': 'hand_r', 'prefix': 'RS_', 'actions': HAND_FINGER_ACTIONS},
    'L': {'type': 'hand_l', 'prefix': 'LS_', 'actions': HAND_FINGER_ACTIONS},
}

ACTION_MAP = {
    'R-SH': {'name': 'Right Shoulder Flex/Ext', 'indices': (H_R, S_R, E_R), 'hist_key': 'RS_SH', 'type': 'pose'},
    'L-SH': {'name': 'Left Shoulder Flex/Ext', 'indices': (H_L, S_L, E_L), 'hist_key': 'LS_SH', 'type': 'pose'},
    'R-EL': {'name': 'Right Elbow Flex', 'indices': (S_R, E_R, W_R), 'hist_key': 'RS_EL', 'type': 'pose'},
    'L-EL': {'name': 'Left Elbow Flex', 'indices': (S_L, E_L, W_L), 'hist_key': 'LS_EL', 'type': 'pose'},
    'R-WI': {'name': 'Right Wrist Flex', 'indices': (E_R, W_R, F2_R), 'hist_key': 'RS_WR', 'type': 'hybrid_wi'},
    'L-WI': {'name': 'Left Wrist Flex', 'indices': (E_L, W_L, F2_L), 'hist_key': 'LS_WR', 'type': 'hybrid_wi'},
    'R-FINGERS': {'name': 'Right Hand/Fingers', 'type': 'hand_r', 'is_group': True},
    'L-FINGERS': {'name': 'Left Hand/Fingers', 'type': 'hand_l', 'is_group': True},
}
REFINEMENT_SETTINGS = {
    'window_size': 5,  # Temporal smoothing window
    'min_calibration_frames': 15,
    'apply_constraints': True,
    'temporal_smoothing': True,
    'rom_smoothing_window': 30
}

# UI colors for refinement status
COLORS = {
    'calibrating': (0, 165, 255),  # Orange
    'calibrated': (0, 255, 0),     # Green
    'raw_angle': (255, 255, 0),    # Yellow
    'refined_angle': (0, 255, 255) # Cyan
}
VIDEO_PATHS = [r'C:\Users\sneha\Downloads\Shoulder Internal Rotation (R).mp4', r'C:\Users\sneha\Downloads\Shoulder Internal Rotation (L).mp4', r'C:\Users\sneha\Downloads\Shoulder Flexion with elbow at 90.mp4',r'C:\Users\sneha\Downloads\Shoulder Flexion with elbow at 120.mp4',r'C:\Users\sneha\Downloads\Shoulder flexion at 90.mp4',r'C:\Users\sneha\Downloads\Shoulder flexion beyond 90.mp4',r'C:\Users\sneha\Downloads\Reaching Exercise.mp4',r'C:\Users\sneha\Downloads\Finger Flexion & Extension.mp4',r'C:\Users\sneha\Downloads\Elbow flexion and extension.mp4']
VIDEO_DEMO_DURATION = 5
TRIAL_DURATION = 15
FEEDBACK_DISPLAY_DURATION = 5
LANDMARK_COLOR = (0, 255, 0)  # Green Landmarks
CONNECTION_COLOR = (255, 255, 255)  # White Connections
HAND_COLOR = (255, 0, 0)  # Blue for Hands

MP_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (27, 29), (29, 31), (26, 28), (28, 30), (30, 32)
]

DEMO_VIDEO, EXERCISE_TIMER = 0, 1
TRIAL_ROUND, EXERCISE_COUNT = 0, 1
MIN_REP_ANGLE_THRESHOLD = 10
ANGLE_BUFFER = 10
TARGET_EXPANSION_BUFFER = 5.0
HISTORY = {}
FONT_SCALE_MAIN = 0.65
FONT_SCALE_DETAIL = 0.55
TEXT_SPACING = 20
SMOOTHING_WINDOW_SIZE = 10

def initialize_tts():
    import pyttsx3  # Import here to avoid dependency if not needed
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.9)
    print("Text-to-Speech Engine initialized successfully.")
    return tts_engine
