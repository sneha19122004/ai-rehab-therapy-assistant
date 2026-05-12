import mediapipe as mp
from config import MODEL_POSE_PATH, MODEL_HAND_PATH

def initialize_detectors(pose_callback, hand_callback):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    HandLandmarker = mp.tasks.vision.HandLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode.LIVE_STREAM
    
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_POSE_PATH),
        running_mode=VisionRunningMode,
        result_callback=pose_callback,
        min_pose_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_HAND_PATH),
        running_mode=VisionRunningMode,
        result_callback=hand_callback,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    
    return pose_landmarker, hand_landmarker