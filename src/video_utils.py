import cv2

import time

import numpy as np  # ADDED: Needed for display_end_board

import mediapipe as mp

from config import FONT_SCALE_MAIN, FONT_SCALE_DETAIL, TEXT_SPACING, VIDEO_DEMO_DURATION

from landmark_detectors import initialize_detectors

from angle_processor import calculate_angle_3d



def play_demo_video(video_path, demo_duration):

    cap_demo = cv2.VideoCapture(video_path)

    if not cap_demo.isOpened():

        print(f"Error: Could not open video file {video_path}. Skipping demo.")

        return

   

    fps = cap_demo.get(cv2.CAP_PROP_FPS)

    target_frames = int(fps * demo_duration)

    frame_count = 0

    start_time = time.time()

   

    while cap_demo.isOpened() and frame_count < target_frames and (time.time() - start_time) < demo_duration + 1:

        ret_demo, frame_demo = cap_demo.read()

        if not ret_demo:

            break

       

        cv2.putText(frame_demo, f"DEMO {frame_count+1}/{target_frames}", (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_MAIN, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Mediapipe Landmarker Feed', frame_demo)

        delay = int(1000/fps)

       

        if cv2.waitKey(delay) & 0xFF == ord('q'):

            break

        frame_count += 1  # FIXED: was "frame_count +="

    cap_demo.release()


def display_end_board(task_set):
    w, h = 1100, 800  # Even wider for better spacing
    board = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Title with gradient background
    cv2.rectangle(board, (0, 0), (w, 100), (30, 30, 60), -1)
    cv2.putText(board, "FINAL EXERCISE RESULTS", (w//2 - 200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(board, f"{task_set['name']} | {len(task_set.get('session_history', []))} Exercises", 
               (w//2 - 150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1, cv2.LINE_AA)
    
    session_history = task_set.get('session_history', [])
    if not session_history:
        cv2.putText(board, "No exercise data available.", (w//2 - 120, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2, cv2.LINE_AA)
        cv2.imshow('Mediapipe Landmarker Feed', board)
        cv2.waitKey(0)
        return
    
    # Setup columns
    margin = 60
    col_width = (w - 3 * margin) // 2
    col1_x = margin
    col2_x = col1_x + col_width + margin
    
    # Column headers
    header_y = 120
    for col_x, col_title in [(col1_x, "LEFT COLUMN"), (col2_x, "RIGHT COLUMN")]:
        cv2.putText(board, col_title, (col_x, header_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1, cv2.LINE_AA)
    
    y_start = header_y + 30
    box_height = 130
    box_spacing = 20
    
    for idx, part in enumerate(session_history):
        # Determine column and position
        if idx % 2 == 0:
            col_x = col1_x
            row = idx // 2
        else:
            col_x = col2_x
            row = idx // 2
        
        box_y = y_start + row * (box_height + box_spacing)
        
        # Skip if box goes beyond screen
        if box_y + box_height > h - 150:
            cv2.putText(board, f"... {len(session_history) - idx} more exercises", 
                       (col_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 1, cv2.LINE_AA)
            break
        
        # Draw exercise box
        cv2.rectangle(board, (col_x, box_y), (col_x + col_width, box_y + box_height), 
                     (50, 50, 50), -1)  # Fill
        cv2.rectangle(board, (col_x, box_y), (col_x + col_width, box_y + box_height), 
                     (100, 100, 150), 2)  # Border
        
        # Exercise number and name (top row)
        cv2.putText(board, f"Ex {part['part_num']}: {part['name'][:15]}", 
                   (col_x + 10, box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 100), 1, cv2.LINE_AA)
        
        # Row 1: Reps and Duration
        cv2.putText(board, f"Reps: {part['count']}", (col_x + 10, box_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(board, f"Time: {part['duration']}s", (col_x + col_width//2, box_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1, cv2.LINE_AA)
        
        # Row 2: Threshold
        cv2.putText(board, f"Target: {part['threshold']}", (col_x + 10, box_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1, cv2.LINE_AA)
        
        # Row 3: Range and ROM
        cv2.putText(board, f"Range: {part['min_val']:.0f}°-{part['max_val']:.0f}°", 
                   (col_x + 10, box_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA)
        rom = part['max_val'] - part['min_val']
        cv2.putText(board, f"ROM: {rom:.0f}°", (col_x + col_width//2, box_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 255), 1, cv2.LINE_AA)
    
    # Summary at bottom
    summary_y = h - 120
    cv2.rectangle(board, (margin, summary_y), (w - margin, h - 40), (40, 40, 70), -1)
    cv2.rectangle(board, (margin, summary_y), (w - margin, h - 40), (100, 100, 200), 2)
    
    total_reps = sum(p['count'] for p in session_history)
    total_time = sum(p['duration'] for p in session_history)
    
    cv2.putText(board, "SUMMARY:", (margin + 20, summary_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(board, f"Total Exercises: {len(session_history)}", (margin + 200, summary_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(board, f"Total Reps: {total_reps}", (margin + 450, summary_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(board, f"Total Time: {total_time}s", (margin + 650, summary_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1, cv2.LINE_AA)
    
    # Instructions
    cv2.putText(board, "Press any key to continue...", (w//2 - 100, h - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('Mediapipe Landmarker Feed', board)
    cv2.waitKey(0)

def get_video_rom(video_path, actions_list):

    """Processes a video file to determine the Min/Max angle range (ROM)

    for the specified actions, simulating the ideal performance.

   

    Returns: A dictionary of {'action_code': {'min': angle, 'max': angle}, ...}

    """

    rom_results = {a['code']: {'min': 180.0, 'max': 0.0} for a in actions_list}

   

    # DUMMY CALLBACKS (Required by MediaPipe for Async even if unused)

    def dummy_pose_callback(result, output_image, timestamp_ms): pass

    def dummy_hand_callback(result, output_image, timestamp_ms): pass

   

    # Initialize detector instance using your existing function

    pose_landmarker, hand_landmarker = initialize_detectors(dummy_pose_callback, dummy_hand_callback)

    cap = cv2.VideoCapture(video_path)



    if not cap.isOpened():

        print(f"ERROR: Cannot open video file: {video_path}")

        return {}



    with pose_landmarker, hand_landmarker:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret: break

           

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

           

            # Run detection synchronously for offline video processing

            pose_results = pose_landmarker.detect(mp_image)

           

            if pose_results and pose_results.pose_landmarks:

                pose_lms = pose_results.pose_landmarks[0]

               

                # Calculate angles for all required actions

                for action in actions_list:

                    action_code = action['code']

                    # Only process POSE actions for simplicity, assuming angle calculation exists

                    if action['type'] == 'POSE':

                        lms_indices = action['landmarks'] # [p1, p2 (vertex), p3]

                       

                        try:

                            p1 = pose_lms[lms_indices[0]]

                            p2 = pose_lms[lms_indices[1]]

                            p3 = pose_lms[lms_indices[2]]

                           

                            # Use your project's calculate_angle function

                            angle = calculate_angle_3d(p1, p2, p3)

                           

                            if angle is not None:

                                rom_results[action_code]['min'] = min(rom_results.get(action_code, {}).get('min', 180.0), angle)

                                rom_results[action_code]['max'] = max(rom_results.get(action_code, {}).get('max', 0.0), angle)

                        except IndexError:

                            continue



    cap.release()

   

    final_rom = {}

    for code, data in rom_results.items():

        # Only return results where actual movement was registered (max > min)

        if data['max'] > data['min']:

            final_rom[code] = data

           

    return final_rom