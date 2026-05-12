import cv2
import time
import mediapipe as mp
import numpy as np
import threading
import sys
import json
from datetime import datetime

# --- Configuration & Constants ---
CALIBRATION_DURATION_S = 15  # Time for full body/hand movement
OUTPUT_FILENAME = "calibration_rom_results.json"
MIN_RANGE_THRESHOLD = 10.0 # Minimum degrees of movement to be considered a valid ROM

# Mediapipe Initializations
# We only import mp.tasks for the landmarker functionality
# We need to manually define the landmark indices because mp.solutions is unavailable.

# --- Landmark Index Mapping (Manual Definition) ---
# POSE Landmarks (0-32):
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16
LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_KNEE = 25; RIGHT_KNEE = 26
LEFT_ANKLE = 27; RIGHT_ANKLE = 28

# HAND Landmarks (0-20):
WRIST = 0
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11
THUMB_MCP = 2; THUMB_IP = 3

# Define the connections needed for drawing the skeleton (Manual recreation of MediaPipe POSE_CONNECTIONS and HAND_CONNECTIONS subset)
POSE_CONNECTIONS_TO_DRAW = [
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)
]
HAND_CONNECTIONS_TO_DRAW = [
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP),
    (WRIST, THUMB_MCP), (THUMB_MCP, THUMB_IP)
    # Add other connections (Ring, Pinky, etc.) if needed
]


# Global storage for asynchronous results
POSE_RESULTS_GLOBAL = None
HAND_RESULTS_GLOBAL = None
RESULTS_LOCK = threading.Lock()
# Store the latest calculated angles for display
LATEST_ANGLES = {} 


# --- Comprehensive List of Angles to Monitor ---
CALIBRATION_ACTIONS = [
    # POSE - ARM JOINTS (Elbow Flexion)
    {'name': 'L_Elbow_Flexion', 'landmarks': [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'POSE'},
    {'name': 'R_Elbow_Flexion', 'landmarks': [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'POSE'},
    
    # POSE - SHOULDER FLEXION/EXTENSION (Uses Hip as anchor)
    {'name': 'L_Shoulder_Flexion', 'landmarks': [LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'POSE'},
    {'name': 'R_Shoulder_Flexion', 'landmarks': [RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'POSE'},
    
    # POSE - LEG JOINTS (Knee Flexion)
    {'name': 'L_Knee_Flexion', 'landmarks': [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'POSE'},
    {'name': 'R_Knee_Flexion', 'landmarks': [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'POSE'},
    
    # HAND JOINTS (Example: Index and Middle finger PIP Flexion)
    {'name': 'Index_PIP_Flexion', 'landmarks': [INDEX_MCP, INDEX_PIP, INDEX_DIP], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'HAND'}, 
    {'name': 'Middle_PIP_Flexion', 'landmarks': [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'HAND'}, 
    {'name': 'Thumb_IP_Flexion', 'landmarks': [WRIST, THUMB_MCP, THUMB_IP], 'min_angle': 180.0, 'max_angle': 0.0, 'type': 'HAND'}
]

# --- Drawing & Calculation Helpers ---

def _get_normalized_landmark_coords(lms, frame_width, frame_height):
    """Converts a list of normalized landmarks to a dictionary of pixel (x, y) coordinates."""
    coords = {}
    for i, lm in enumerate(lms):
        coords[i] = (int(lm.x * frame_width), int(lm.y * frame_height))
    return coords

def draw_landmarks_and_angles(frame, pose_lms, hand_lms_list, frame_width, frame_height):
    """Draws skeleton, points, and angle values on the frame."""
    
    # 1. POSE LANDMARKS AND SKELETON
    if pose_lms:
        pose_coords = _get_normalized_landmark_coords(pose_lms, frame_width, frame_height)
        
        # Draw Connections
        for start_idx, end_idx in POSE_CONNECTIONS_TO_DRAW:
            if start_idx in pose_coords and end_idx in pose_coords:
                cv2.line(frame, pose_coords[start_idx], pose_coords[end_idx], (0, 255, 0), 2)
                
        # Draw Points (Joints)
        for idx, (x, y) in pose_coords.items():
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1) # Blue dot

        # Draw POSE Angles
        for action in CALIBRATION_ACTIONS:
            if action['type'] == 'POSE':
                action_name = action['name']
                if action_name in LATEST_ANGLES:
                    # Angle is displayed near the vertex joint (middle landmark)
                    vertex_idx = action['landmarks'][1]
                    if vertex_idx in pose_coords:
                        angle_text = f"{LATEST_ANGLES[action_name]:.0f} deg"
                        x, y = pose_coords[vertex_idx]
                        cv2.putText(frame, angle_text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. HAND LANDMARKS AND SKELETON
    if hand_lms_list:
        for hand_index, hand_lms in enumerate(hand_lms_list):
            handedness = "L" if hand_index == 0 else "R" # Simple label for display
            hand_coords = _get_normalized_landmark_coords(hand_lms, frame_width, frame_height)
            
            # Draw Connections
            for start_idx, end_idx in HAND_CONNECTIONS_TO_DRAW:
                if start_idx in hand_coords and end_idx in hand_coords:
                    cv2.line(frame, hand_coords[start_idx], hand_coords[end_idx], (0, 255, 255), 1) # Cyan line
                    
            # Draw Points (Joints)
            for idx, (x, y) in hand_coords.items():
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1) # Magenta dot

            # Draw HAND Angles
            for action in CALIBRATION_ACTIONS:
                if action['type'] == 'HAND':
                    action_name = f"{handedness}_{action['name']}" # e.g., "L_Index_PIP_Flexion"
                    if action_name in LATEST_ANGLES:
                        # Angle is displayed near the vertex joint (middle landmark)
                        vertex_idx = action['landmarks'][1]
                        if vertex_idx in hand_coords:
                            angle_text = f"{LATEST_ANGLES[action_name]:.0f} deg"
                            x, y = hand_coords[vertex_idx]
                            cv2.putText(frame, angle_text, (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


# --- API Callbacks and Angle Calculation (Unchanged) ---

def pose_callback(result, output_image, timestamp_ms):
    global POSE_RESULTS_GLOBAL
    with RESULTS_LOCK:
        POSE_RESULTS_GLOBAL = result

def hand_callback(result, output_image, timestamp_ms):
    global HAND_RESULTS_GLOBAL
    with RESULTS_LOCK:
        HAND_RESULTS_GLOBAL = result

def calculate_angle(a, b, c):
    A = np.array([a.x, a.y, a.z]); B = np.array([b.x, b.y, b.z]); C = np.array([c.x, c.y, c.z])
    BA = A - B; BC = C - B
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA); magnitude_BC = np.linalg.norm(BC)
    if magnitude_BA == 0 or magnitude_BC == 0: return None
    cosine_angle = dot_product / (magnitude_BA * magnitude_BC)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

def initialize_detectors():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    HandLandmarker = mp.tasks.vision.HandLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # --- PATH FIX: Using Absolute Path (UPDATED TO C:/Users/sneha/) ---
    BASE_MODEL_PATH = "C:/Users/sneha/" 
    
    # The user updated their path to 'pose_landmarker.task' (not 'heavy')
    POSE_MODEL_FILE = BASE_MODEL_PATH + 'pose_landmarker.task' 
    HAND_MODEL_FILE = BASE_MODEL_PATH + 'hand_landmarker.task'
    # -----------------------------------------------------------------
    
    try:
        # 1. Pose Landmarker
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_FILE), 
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=pose_callback
        )
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        # 2. Hand Landmarker
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_FILE),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=hand_callback,
            num_hands=2
        )
        hand_landmarker = HandLandmarker.create_from_options(hand_options)
        
        return pose_landmarker, hand_landmarker
        
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Mediapipe detectors. Please ensure the files are at {BASE_MODEL_PATH}. Error: {e}")
        sys.exit(1)

# --- Main Calibration Function ---

def run_comprehensive_calibration():
    global LATEST_ANGLES
    print(f"--- Starting Comprehensive Range of Motion Calibration ({CALIBRATION_DURATION_S}s) ---")
    print("Please move ALL joints (arms, legs, hands) through their maximum range for the duration.")
    
    pose_landmarker, hand_landmarker = initialize_detectors()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open video stream (webcam).")
        sys.exit(1)
        
    start_time = time.time()
    
    rom_data = {action['name']: {'min_angle': 180.0, 'max_angle': 0.0, 'type': action['type']} for action in CALIBRATION_ACTIONS}
    
    with pose_landmarker, hand_landmarker:
        while cap.isOpened() and (time.time() - start_time) < CALIBRATION_DURATION_S:
            ret, frame = cap.read()
            if not ret: break

            current_time = time.time()
            elapsed_time = current_time - start_time
            remaining_time = CALIBRATION_DURATION_S - elapsed_time
            frame_height, frame_width, _ = frame.shape

            # Prepare frame and send for asynchronous detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_time_ms = int(current_time * 1000)
            
            pose_landmarker.detect_async(mp_image, frame_time_ms)
            hand_landmarker.detect_async(mp_image, frame_time_ms)

            # --- Process Latest Results (Angle Calculation and ROM Update) ---
            pose_results, hand_results = None, None
            with RESULTS_LOCK:
                pose_results = POSE_RESULTS_GLOBAL
                hand_results = HAND_RESULTS_GLOBAL
            
            # Clear previous angles and calculate the new ones
            LATEST_ANGLES.clear()

            # 1. Process POSE Angles
            pose_lms_to_draw = None
            if pose_results and pose_results.pose_landmarks:
                pose_lms = pose_results.pose_landmarks[0]
                pose_lms_to_draw = pose_lms
                
                for action in CALIBRATION_ACTIONS:
                    if action['type'] == 'POSE':
                        action_name = action['name']
                        try:
                            p1 = pose_lms[action['landmarks'][0]]
                            p2 = pose_lms[action['landmarks'][1]] 
                            p3 = pose_lms[action['landmarks'][2]]
                            angle = calculate_angle(p1, p2, p3)
                            
                            if angle is not None:
                                LATEST_ANGLES[action_name] = angle # Store for visualization
                                rom_data[action_name]['min_angle'] = min(rom_data[action_name]['min_angle'], angle)
                                rom_data[action_name]['max_angle'] = max(rom_data[action_name]['max_angle'], angle)
                        except Exception:
                            pass
                            
            # 2. Process HAND Angles
            hand_lms_list_to_draw = None
            if hand_results and hand_results.hand_landmarks:
                hand_lms_list_to_draw = hand_results.hand_landmarks
                
                for hand_index, hand_lms in enumerate(hand_results.hand_landmarks):
                    handedness = hand_results.handedness[hand_index][0].category_name
                    
                    for action in CALIBRATION_ACTIONS:
                        if action['type'] == 'HAND':
                            # Use a simple "L" or "R" for the visualization key
                            handedness_key = "L" if handedness == "Left" else "R"
                            action_name = f"{handedness_key}_{action['name']}"
                            
                            # Initialize if needed (for tracking ROM)
                            if action_name not in rom_data:
                                rom_data[action_name] = {'min_angle': 180.0, 'max_angle': 0.0, 'type': 'HAND'}

                            try:
                                p1 = hand_lms[action['landmarks'][0]]
                                p2 = hand_lms[action['landmarks'][1]] 
                                p3 = hand_lms[action['landmarks'][2]]
                                angle = calculate_angle(p1, p2, p3)

                                if angle is not None:
                                    LATEST_ANGLES[action_name] = angle # Store for visualization
                                    rom_data[action_name]['min_angle'] = min(rom_data[action_name]['min_angle'], angle)
                                    rom_data[action_name]['max_angle'] = max(rom_data[action_name]['max_angle'], angle)
                            except Exception:
                                pass

            # 3. DRAW VISUALIZATION
            draw_landmarks_and_angles(
                frame, 
                pose_lms_to_draw, 
                hand_lms_list_to_draw, 
                frame_width, 
                frame_height
            )

            # --- Display Feedback ---
            cv2.putText(frame, f"CALIBRATION IN PROGRESS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"TIME LEFT: {int(remaining_time)}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Comprehensive ROM Calibration Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- Cleanup and Final Output (Unchanged) ---
    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Calibration Complete ---")
    final_calibration_results = {}
    
    # Filter and format the results
    for name, data in rom_data.items():
        min_a, max_a = data['min_angle'], data['max_angle']
        range_achieved = max_a - min_a
        
        if range_achieved > MIN_RANGE_THRESHOLD:
            print(f"✅ {name}: Range {range_achieved:.2f} deg (Min: {min_a:.2f} deg / Max: {max_a:.2f} deg)")
            final_calibration_results[name] = {
                'min_angle_achieved': round(min_a, 2),
                'max_angle_achieved': round(max_a, 2),
                'range': round(range_achieved, 2),
                'type': data['type']
            }
        else:
            if min_a != 180.0 and max_a != 0.0:
                print(f"⚠️ {name}: Range TOO SMALL ({range_achieved:.2f} deg). Target movement not detected.")
    
    # Save results to JSON file
    output = {
        "timestamp": datetime.now().isoformat(),
        "duration_s": CALIBRATION_DURATION_S,
        "min_range_threshold": MIN_RANGE_THRESHOLD,
        "results": final_calibration_results
    }
    
    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"\n--- RESULTS SAVED TO {OUTPUT_FILENAME} ---")
    
    return output

if __name__ == "__main__":
    run_comprehensive_calibration()