import cv2
import time
import sys
import threading
import json
import os
import numpy as np  # ADDED
import mediapipe as mp  # ADDED
from comm_utils import send_rep_count
from config import *
from ui_utils import draw_skeleton_and_vertices
from user_input import get_monitoring_sequence
from video_utils import play_demo_video, display_end_board
from landmark_detectors import initialize_detectors
from angle_processor import process_landmarks_for_actions
from db_logger import log_session_data
from romvidtest import track_joint_rom

GLOBAL_SESSION_HISTORY = []
POSE_RESULTS_GLOBAL = None
HAND_RESULTS_GLOBAL = None
FRAME_TIME_MS_GLOBAL = 0
RESULTS_LOCK = threading.Lock()
SCOREBOARD = 3

def pose_callback(result, output_image, timestamp_ms):
    global POSE_RESULTS_GLOBAL, FRAME_TIME_MS_GLOBAL
    with RESULTS_LOCK:
        POSE_RESULTS_GLOBAL = result
        FRAME_TIME_MS_GLOBAL = timestamp_ms

def hand_callback(result, output_image, timestamp_ms):
    global HAND_RESULTS_GLOBAL
    with RESULTS_LOCK:
        HAND_RESULTS_GLOBAL = result

def save_exercise_rom(task_name, actions, part_num):
    folder = "Exercise_ROM_Records"
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{folder}/ROM_Part{part_num}_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "task_name": task_name,
        "part": part_num,
        "results": []
    }

    for a in actions:
        total_rom = a['max_angle'] - a['min_angle']
        data["results"].append({
            "code": a['code'],
            "min_reached": round(a['min_angle'], 2),
            "max_reached": round(a['max_angle'], 2),
            "rom": round(total_rom, 2),
            "reps_done": a['rep_count']
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"📂 [RECORDED] Exercise ROM saved to {filename}")  

def main():
    try:
        tts_engine = initialize_tts()
    except Exception as e:
        print(f"Warning: Could not initialize Text-to-Speech engine (pyttsx3). Speech output will be disabled. Error: {e}")
        tts_engine = None

    MONITORING_SEQUENCE = get_monitoring_sequence()
    print(f"Sequence defined with {len(MONITORING_SEQUENCE)} tasks.")
    cap = cv2.VideoCapture(0)
    pose_landmarker, hand_landmarker = initialize_detectors(pose_callback, hand_callback)
    current_task_index, total_tasks = 0, len(MONITORING_SEQUENCE)
    current_task_set = MONITORING_SEQUENCE[0]
    current_task_set['session_history'] = []
    task_phase, phase_start_time, exercise_part = DEMO_VIDEO, time.time(), 0
    exercise_sub_phase = TRIAL_ROUND
    last_good_job_time = 0
    frame_time_ms = 0
    
    # Track which videos have been analyzed to avoid repeating demo
    analyzed_videos = set()
    
    try:
        with pose_landmarker, hand_landmarker:
            while cap.isOpened():
                current_time = time.time()
                elapsed_time = current_time - phase_start_time
                frame = None
                if task_phase == DEMO_VIDEO:
                    primary_action = current_task_set['actions'][0]
                    joint_code = primary_action['code'].upper()
                    target_idx = None
                    video_path = None
                    if "SH" in joint_code:
                        current_task_set['total_parts'] = 4
                        base_idx = 0  # Starts at A1.mp4
                    elif "EL" in joint_code:
                        current_task_set['total_parts'] = 3
                        base_idx = 4  # Starts at EL.mp4
                    elif "WI" in joint_code:
                        current_task_set['total_parts'] = 1
                        base_idx = 7  # Starts at WI1.mp4
                    else:
                        current_task_set['total_parts'] = 1
                        base_idx = None
                    current_task_set['part_duration'] = current_task_set['duration'] / current_task_set['total_parts']
                    current_task_set['is_shoulder_task'] = True # Force the split-timer logic to run
                    should_play_demo = False
                    
                    # For finger exercises, skip demo video entirely
                    is_finger_exercise = any(f in joint_code for f in ["FI", "TH", "IN", "MI", "RI", "LI"])
                    
                    if not is_finger_exercise and base_idx is not None:
                        target_idx = base_idx + exercise_part
                        if target_idx < len(VIDEO_PATHS):
                            video_path = VIDEO_PATHS[target_idx]
                            should_play_demo = True
                            
                    if video_path and os.path.exists(video_path):
                        video_name = os.path.basename(video_path).replace('.mp4', '')
                        ref_file = os.path.join("Video_ROM_Records", f"{primary_action['code']}_ref.json")
                        
                        # Check if video has already been analyzed in this session
                        if joint_code in analyzed_videos:
                            print(f"✅ Video already analyzed for {joint_code}. Skipping demo.")
                            should_play_demo = False
                        elif not os.path.exists(ref_file):
                            print(f"🚀 ANALYZING: {video_path}")
                            from romvidtest import track_joint_rom, save_session_data
                            all_joint_codes = [a['code'] for a in current_task_set['actions']]
                            try:
                                video_result = track_joint_rom(video_path, joint_code, duration=10)
                                if video_result:
                                    save_session_data(video_result, folder="Video_ROM_Records")
                                    analyzed_videos.add(joint_code)
                                    should_play_demo = False  # Don't play demo after analysis
                            except Exception as e:
                                print(f"⚠️ Analysis failed for {video_path}: {e}")
                                should_play_demo = False
                        else:
                            # FILE EXISTS: We don't need to analyze, so we CAN play the demo
                            print(f"✅ Analysis found for {joint_code}. Playing demo.")
                            should_play_demo = True
                            analyzed_videos.add(joint_code)
                            
                        for action in current_task_set['actions']:
                            action_ref = os.path.join("Video_ROM_Records", f"{action['code']}_ref.json")
                            
                            # We check if the file exists BEFORE opening it to avoid [Errno 2]
                            if os.path.exists(action_ref):
                                try:
                                    with open(action_ref, 'r') as f:
                                        data = json.load(f) # Call this ONLY once
                                        # Now use the 'data' variable to get the value
                                        action['ref_rom'] = data.get('total_rom', 160.0)
                                        print(f"✅ Loaded {action['code']} ROM: {action['ref_rom']}")
                                except Exception as e:
                                    print(f"⚠️ Error reading JSON: {e}")
                                    action['ref_rom'] = 160.0
                                                                        
                    # Play demo only if it's not a finger exercise and should_play_demo is True
                    if not is_finger_exercise and should_play_demo and video_path and os.path.exists(video_path):
                        play_demo_video(video_path, VIDEO_DEMO_DURATION)
                    else:
                        print(f"⏩ No video demo played for {joint_code}. Starting Trial phase.")
                        
                    for action in current_task_set['actions']:
                        action_ref = f"Video_ROM_Records/{action['code']}_ref.json"
                        rom_val = 160.0 # Default
                        if os.path.exists(action_ref):
                            try:
                                with open(action_ref, 'r') as f:
                                    data = json.load(f)
                                    # Look for total_rom in the new multi_results format or old format
                                    rom_val = data.get('total_rom', 160.0)
                                    action['ref_rom'] = rom_val
                                    print(f"✅ {action['code']} Target ROM: {rom_val}")
                            except Exception as e:
                                print(f"⚠️ Error reading JSON for {action['code']}: {e}")
                                action['ref_rom'] = 160.0
                        else:
                            # If the file isn't there, don't crash, just use a default
                            print(f"⚠️ Reference file {action_ref} not found. Using default 160.0")
                        action['ref_rom'] = rom_val    
                        action.update({
                            'min_angle': 180.0, 'max_angle': 0.0,
                            'min_target': 180.0, 'max_target': 0.0,
                            'rep_count': 0, 'is_at_max': False, 'is_rep_complete': False,
                            'rep_min_angle': 180.0, 'rep_max_angle': 0.0, 'rep_history': []
                        })
                    
                        if action.get('hist_key') in HISTORY: 
                            HISTORY[action['hist_key']].clear()

                    # PHASE TRANSITION (Only happens AFTER all actions are initialized)
                    exercise_sub_phase = TRIAL_ROUND
                    task_phase = EXERCISE_TIMER
                    phase_start_time = time.time() 
                    continue
                    
                if task_phase == EXERCISE_TIMER:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_h, frame_w = frame.shape[:2]
                    frame_time_ms = int(time.time() * 1000)
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    pose_landmarker.detect_async(mp_image, frame_time_ms)
                    hand_landmarker.detect_async(mp_image, frame_time_ms)

                    elapsed_time = current_time - phase_start_time
                    part_duration = current_task_set['part_duration']
                    remaining_time = max(0, part_duration - elapsed_time)

                    # 1. LIVE ACCURACY (Updating every frame for the UI)
                    for action in current_task_set['actions']:
                        ref_rom = action.get('ref_rom', 160.0)
                        user_rom = action['max_angle'] - action['min_angle']
                        action['accuracy_pct'] = min(100, int((user_rom / max(1, ref_rom)) * 100))

                    # 2. TRIAL -> COUNTING TRANSITION
                    # 2. TRIAL -> COUNTING TRANSITION
                    if exercise_sub_phase == TRIAL_ROUND and elapsed_time >= TRIAL_DURATION:
                        exercise_sub_phase = EXERCISE_COUNT
                        phase_start_time = time.time()
                        
                        print(f"\n🎯 TRANSITIONING FROM TRIAL TO COUNTING PHASE")
                        print(f"Processing {len(current_task_set['actions'])} action(s)...")
                        
                        # Track which actions got manual input
                        manual_count = 0
                        auto_count = 0
                        
                        # Process each action
                        for action in current_task_set['actions']:
                            action_code = action['code']
                            current_min = action['min_angle']
                            current_max = action['max_angle']
                            RANGE = current_max - current_min
                            
                            print(f"\n{action_code}: Current range = {RANGE:.1f}deg (min={current_min:.1f}, max={current_max:.1f})")
                            
                            if RANGE < MIN_REP_ANGLE_THRESHOLD:
                                print(f"⚠️ Range too small (< {MIN_REP_ANGLE_THRESHOLD}°)")
                                
                                # Manual input for this action
                                while True:
                                    try:
                                        user_input = input(f"Enter thresholds for {action_code} (format: min,max): ").strip()
                                        
                                        if not user_input:
                                            print("Input cannot be empty. Please enter thresholds.")
                                            continue
                                        
                                        min_str, max_str = user_input.split(',')
                                        min_val = float(min_str.strip())
                                        max_val = float(max_str.strip())
                                        
                                        if min_val >= max_val:
                                            print("Error: Min must be less than Max")
                                            continue
                                        
                                        # Set manual thresholds
                                        action['min_target'] = min_val
                                        action['max_target'] = max_val
                                    
                                        
                                        manual_count += 1
                                        print(f"✅ {action_code}: Manual thresholds = {min_val:.1f}°/{max_val:.1f}°")
                                        break
                                        
                                    except ValueError:
                                        print("Error: Please enter numbers in format 'min,max' (e.g., '30,150')")
                                    except Exception as e:
                                        print(f"Error: {e}")
                            else:
                                
                                min_target = current_min + TARGET_EXPANSION_BUFFER
                                max_target = current_max - TARGET_EXPANSION_BUFFER
                                   
                                auto_count += 1
                        
                        # Initialize ALL actions for counting phase
                        print(f"\n📊 Initializing counting phase...")
                        for action in current_task_set['actions']:
                            action.update({
                                'rep_min_angle': 180.0, 
                                'rep_max_angle': 0.0, 
                                'is_at_max': False, 
                                'is_rep_complete': False,
                                'rep_history': []
                            })
                            print(f"  {action['code']}: Targets = {action['min_target']:.0f}°/{action['max_target']:.0f}°")
                        
                        # Final status message
                        if manual_count > 0:
                            print(f"\n--- COUNTING STARTED with {manual_count} manual + {auto_count} auto thresholds ---")
                        else:
                            print(f"\n--- COUNTING STARTED with AUTO thresholds for all {len(current_task_set['actions'])} actions ---")
                        
                        continue  # Skip to next frame to avoid old display
                    if elapsed_time >= part_duration:
                        # Capture data for the board BEFORE moving to next part
                        # Get actual max/min values from the action
                        action = current_task_set['actions'][0]
                        actual_max = action['max_angle']
                        actual_min = action['min_angle']
                        
                        # Format threshold as max/min
                        threshold_str = f"{round(actual_max, 1)}/{round(actual_min, 1)}"
                        if actual_max <= 0 or actual_min >= 180:
                            threshold_str = "Not Set"
                            
                        part_summary = {
                            "part_num": exercise_part + 1,
                            "name": current_task_set['name'],
                            "count": current_task_set['actions'][0]['rep_count'],
                            "threshold": threshold_str,  # Changed to show max/min
                            "duration": int(part_duration),
                            "max_val": round(actual_max, 1),
                            "min_val": round(actual_min, 1)
                        }
                        current_task_set['session_history'].append(part_summary)
                        save_exercise_rom(current_task_set['name'], current_task_set['actions'], exercise_part + 1)

                        if tts_engine:
                            tts_engine.say(f"Exercise {exercise_part + 1} complete.")
                            tts_engine.runAndWait()

                        exercise_part += 1    
                        if exercise_part < current_task_set['total_parts']:
                            # Reset counts for the NEXT part
                            for a in current_task_set['actions']: a.update({'rep_count': 0, 'min_angle': 180.0, 'max_angle': 0.0})
                            task_phase = DEMO_VIDEO
                        else:
                            # ALL PARTS FINISHED: Show the board with the accumulated history
                            display_end_board(current_task_set) 
                            current_task_index += 1
                            if current_task_index < total_tasks:
                                next_task = MONITORING_SEQUENCE[current_task_index]
                                next_task['session_history'] = current_task_set['session_history']
                                current_task_set = next_task
                                exercise_part = 0
                                task_phase = DEMO_VIDEO
                            else:
                                task_phase = SCOREBOARD
                        
                        phase_start_time = time.time()
                        exercise_sub_phase = TRIAL_ROUND
                        continue

                    # 4. RENDER UI ELEMENTS
                    display_lines, active_pose_indices, active_hand_map, trigger_good_job = process_landmarks_for_actions(
                        current_task_set, exercise_sub_phase, POSE_RESULTS_GLOBAL, HAND_RESULTS_GLOBAL, RESULTS_LOCK
                    )

                    # Draw Skeleton WITH HAND LANDMARKS - FIX THIS PART
                    with RESULTS_LOCK:
                        pose_lms = POSE_RESULTS_GLOBAL.pose_landmarks[0] if POSE_RESULTS_GLOBAL and POSE_RESULTS_GLOBAL.pose_landmarks else None
                        hand_lms_list = HAND_RESULTS_GLOBAL.hand_landmarks if HAND_RESULTS_GLOBAL and HAND_RESULTS_GLOBAL.hand_landmarks else []
                        hand_handedness_list = HAND_RESULTS_GLOBAL.handedness if HAND_RESULTS_GLOBAL and HAND_RESULTS_GLOBAL.handedness else []

                    # The key change: Pass hand landmarks and handedness to the function
                    draw_skeleton_and_vertices(
                        frame, 
                        pose_lms, 
                        hand_lms_list,  # ADD THIS
                        hand_handedness_list,  # ADD THIS
                        MP_POSE_CONNECTIONS, 
                        active_pose_indices=active_pose_indices,
                        active_hand_map=active_hand_map  # This is already in your call
                    )
                    # --- UI LABELS (Restored Trial/Counting/Threshold) ---
                    mode_text = "TRIAL MODE" if exercise_sub_phase == TRIAL_ROUND else "COUNTING MODE"
                    cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    cv2.putText(frame, f"TIME: {int(remaining_time)}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Display individual joint lines (includes angles and reps)
                    y_off = 130
                    for line in display_lines:
                        cv2.putText(frame, line, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_off += 30

                    # Accuracy Panel (Top Right)
                    primary_code = current_task_set['actions'][0]['code'].upper()
                    if not any(f in primary_code for f in ["FI", "TH", "IN", "MI", "RI", "LI"]):
                        acc = current_task_set['actions'][0].get('accuracy_pct', 0)
                        cv2.rectangle(frame, (frame_w - 230, 20), (frame_w - 20, 120), (40, 40, 40), -1)
                        cv2.putText(frame, f"MATCH: {acc}%", (frame_w - 210, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    cv2.imshow('Mediapipe Landmarker Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if tts_engine:
            try: tts_engine.stop()
            except: pass

        cap.release(); cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()