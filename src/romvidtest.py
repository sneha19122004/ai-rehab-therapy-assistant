import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
# Import from your config.py
from config import ACTION_MAP, MODEL_POSE_PATH, MP_POSE_CONNECTIONS
from geometry_utils import calculate_angle_3d  # ADD THIS
from landmark_refiner import UnifiedLandmarkRefiner
def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def save_session_data(data, folder="Video_ROM_Records"):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if 'multi_results' in data:
        for res in data['multi_results']:
            filename = os.path.join(folder, f"{res['joint_code']}_ref.json")
            with open(filename, 'w') as f:
                json.dump(res, f, indent=4)
            print(f"File written: {filename}")
    elif 'joint_code' in data:
        filename = os.path.join(folder, f"{data['joint_code']}_ref.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"File written: {filename}")
    # Save a separate file for each joint found in the video
    for res in data['multi_results']:
        filename = f"{folder}/{res['joint_code']}_ref.json"
        with open(filename, 'w') as f:
            json.dump(res, f, indent=4)
        print(f"✅ Saved ROM for {res['joint_code']}")
def track_joint_rom(video_path, joint_codes, duration=10):
    # Ensure joint_codes is a list even if a single string is passed
    if isinstance(joint_codes, str):
        joint_codes = [joint_codes]

    # Filter valid actions from ACTION_MAP
    active_actions = []
    for code in joint_codes:
        if code in ACTION_MAP:
            active_actions.append({
                'code': code,
                'indices': ACTION_MAP[code]['indices'],
                'name': ACTION_MAP[code]['name'],
                'all_angles': [],
                'min_angle': 180.0,
                'max_angle': 0.0
            })

    if not active_actions:
        return None

    # ENHANCEMENT: Initialize refiner for video analysis
    refiner = UnifiedLandmarkRefiner(window_size=7)
    
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_POSE_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(video_path)
    start_time = time.time()

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            elapsed = time.time() - start_time
            if elapsed > duration: break

            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, int(elapsed * 1000))

            if result.pose_landmarks:
                raw_lms = result.pose_landmarks[0]
                h, w, _ = frame.shape
                
                # ENHANCEMENT: Apply refinement to video analysis
                smoothed_lms = refiner.smooth_pose(raw_lms)
                refiner.calibrate_bone_lengths(smoothed_lms)
                
                # Use refined landmarks for better tracking
                processed_lms = smoothed_lms if smoothed_lms else raw_lms

                # --- SKELETON DRAWING ---
                for conn in MP_POSE_CONNECTIONS:
                    if conn[0] < len(processed_lms) and conn[1] < len(processed_lms):
                        p1, p2 = processed_lms[conn[0]], processed_lms[conn[1]]
                        cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (255, 255, 255), 1)

                # --- MULTI-JOINT PROCESSING ---
                y_offset = 40
                for action in active_actions:
                    idx1, idx2, idx3 = action['indices']
                    
                    if (idx1 < len(processed_lms) and 
                        idx2 < len(processed_lms) and 
                        idx3 < len(processed_lms)):
                        
                        p_a, p_b, p_c = processed_lms[idx1], processed_lms[idx2], processed_lms[idx3]
                        
                        # Highlight joints for this action
                        for p in [p_a, p_b, p_c]:
                            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 8, (0, 0, 255), -1)

                        angle = calculate_angle_3d(
                            (p_a.x, p_a.y, p_a.z), 
                            (p_b.x, p_b.y, p_b.z), 
                            (p_c.x, p_c.y, p_c.z)
                        )
                        
                        if not np.isnan(angle):
                            # Apply temporal refinement to angle
                            refined_angle = refiner.refine_rom_angle(action['code'], angle)
                            
                            action['all_angles'].append(refined_angle)
                            action['min_angle'] = min(action['min_angle'], refined_angle)
                            action['max_angle'] = max(action['max_angle'], refined_angle)

                        # Show refined angle in UI
                        refiner_status = refiner.get_status()
                        calib_status = "✓" if refiner_status['is_calibrated'] else f"({refiner_status['calibration_frames']}/{refiner_status['min_calibration_frames']})"
                        
                        cv2.putText(frame, f"{action['code']}: {int(refined_angle)}° {calib_status} | ROM: {int(action['max_angle'] - action['min_angle'])}", 
                                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_offset += 30

            cv2.imshow('Multi-ROM Tracker - Visual Debug', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # --- COMPILE RESULTS WITH REFINEMENT METRICS ---
    results = []
    for action in active_actions:
        final_rom = 0
        if len(action['all_angles']) > 10:
            action['all_angles'].sort()
            robust_min = action['all_angles'][int(len(action['all_angles']) * 0.05)]
            robust_max = action['all_angles'][int(len(action['all_angles']) * 0.95)]
            final_rom = robust_max - robust_min
        else:
            final_rom = action['max_angle'] - action['min_angle']

        results.append({
            "joint_code": action['code'],
            "total_rom": round(final_rom, 2),
            "refinement_applied": True,
            "min_angle": round(action['min_angle'], 2),
            "max_angle": round(action['max_angle'], 2),
            "frame_count": len(action['all_angles'])
        })

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video_path": video_path,
        "duration_analyzed": duration,
        "refinement_settings": {
            "window_size": refiner.window_size,
            "calibration_complete": refiner.is_calibrated
        },
        "multi_results": results
    }
if __name__ == "__main__":
    # 1. Run the AI Tracking
    session_result = track_joint_rom(r'C:\Users\sneha\Downloads\A1.mp4', 'R-SH')
    
    # 2. Save the results if successful
    if session_result:
        save_session_data(session_result)