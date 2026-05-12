import numpy as np

from collections import deque

from geometry_utils import calculate_angle_3d, smooth_angle, get_safe_visibility

from config import HISTORY, ACTION_MAP, HAND_TRACKING_DETAILS, MIN_REP_ANGLE_THRESHOLD, ANGLE_BUFFER, SMOOTHING_WINDOW_SIZE,TRIAL_ROUND,EXERCISE_COUNT

from comm_utils import send_rep_count

def process_landmarks_for_actions(current_task_set, exercise_sub_phase, pose_results, hand_results, results_lock):
    with results_lock:
        pose_results_local = pose_results
        hand_results_local = hand_results

    display_lines = []
    active_pose_indices = set()
    active_hand_map = {'left': set(), 'right': set()}

    # FIXED: Handle both cases - pose_results could be a result object OR a list of landmarks
    if pose_results_local and hasattr(pose_results_local, 'pose_landmarks'):
        # It's a MediaPipe result object
        pose_lms = pose_results_local.pose_landmarks[0] if pose_results_local.pose_landmarks else None
    else:
        # It's already a list of landmarks (from your callback)
        pose_lms = pose_results_local

    # Rest of the code remains the same...
    hand_lms_list = hand_results_local.hand_landmarks if hand_results_local and hasattr(hand_results_local, 'hand_landmarks') else []
    hand_handedness_list = hand_results_local.handedness if hand_results_local and hasattr(hand_results_local, 'handedness') else []

    # ... rest of your function ...


    hand_map = {}

    for hand_idx, hand_type_list in enumerate(hand_handedness_list):

        classification = hand_type_list[0].category_name[0].lower()

        hand_map[classification] = hand_idx



    for action in current_task_set['actions']:

        if action.get('is_group'):

            continue



        i1, i2, i3 = action['indices']

        hist_key = action['hist_key']

        action_type = action['type']



        is_visible = False

        p_a, p_b, p_c = None, None, None



        if action_type == 'pose':

            if pose_lms and len(pose_lms) > max(i1, i2, i3):

                p_a = pose_lms[i1]

                p_b = pose_lms[i2]

                p_c = pose_lms[i3]

                vis_check = get_safe_visibility(p_a) > 0.5 and get_safe_visibility(p_b) > 0.5 and get_safe_visibility(p_c) > 0.5

                if vis_check:

                    is_visible = True

                    active_pose_indices.update([i1, i2, i3])

        elif action_type.startswith('hand'):

            target_hand = action_type.split('_')[-1]

            target_hand_full = 'right' if target_hand == 'r' else 'left'

            if target_hand in hand_map:

                hand_idx = hand_map[target_hand]

                current_lms = hand_lms_list[hand_idx]

                if len(current_lms) > max(i1, i2, i3):

                    p_a = current_lms[i1]

                    p_b = current_lms[i2]

                    p_c = current_lms[i3]

                    is_visible = True

                    active_hand_map[target_hand_full].update([i1, i2, i3])

        elif action_type == 'hybrid_wi':

            target_hand = action['code'][0].lower()

            target_hand_full = 'right' if target_hand == 'r' else 'left'

            if pose_lms and target_hand in hand_map:

                p_a = pose_lms[i1]

                p_b = pose_lms[i2]

                vis12_check = get_safe_visibility(p_a) > 0.5 and get_safe_visibility(p_b) > 0.5

                hand_idx = hand_map[target_hand]

                hand_lms = hand_lms_list[hand_idx]

                if vis12_check and len(hand_lms) > i3:

                    p_c = hand_lms[i3]

                    is_visible = True

                    active_pose_indices.update([i1, i2])

                    active_hand_map[target_hand_full].add(i3)

       

        if is_visible:

            angle = calculate_angle_3d((p_a.x, p_a.y, p_a.z), (p_b.x, p_b.y, p_b.z), (p_c.x, p_c.y, p_c.z))



            if hist_key not in HISTORY:

                HISTORY[hist_key] = deque(maxlen=SMOOTHING_WINDOW_SIZE)

            smoothed_angle = smooth_angle(HISTORY[hist_key], angle)

            action['min_angle'] = min(action['min_angle'], smoothed_angle)

            action['max_angle'] = max(action['max_angle'], smoothed_angle)

            user_current_rom = action['max_angle'] - action['min_angle']

            video_ref_rom = action.get('ref_rom', 160.0) # Loaded in main.py

           

            # Calculate percentage (capped at 100)

            accuracy_pct = min(100, int((user_current_rom / video_ref_rom) * 100))

            action['accuracy_pct'] = accuracy_pct # Save for the UI drawing in main.py

            if exercise_sub_phase == TRIAL_ROUND:

                action['min_angle'] = min(action['min_angle'], smoothed_angle)

                action['max_angle'] = max(action['max_angle'], smoothed_angle)

                display_lines.append(f"{action['code']}: {int(smoothed_angle)}deg (TRIAL)")

            elif exercise_sub_phase == EXERCISE_COUNT:

                MIN_TARGET = action['min_target']

                MAX_TARGET = action['max_target']

                RANGE = MAX_TARGET - MIN_TARGET



                action['rep_min_angle'] = min(action['rep_min_angle'], smoothed_angle)

                action['rep_max_angle'] = max(action['rep_max_angle'], smoothed_angle)



                if RANGE < MIN_REP_ANGLE_THRESHOLD and exercise_sub_phase == EXERCISE_COUNT:
                    # Check if this is the first frame after manual input
                    # We can check if targets are reasonable (not the default 180/0 values)
                    if MIN_TARGET < 170 and MAX_TARGET > 10:  # Not default values
                        # Probably manual thresholds were set, don't show warning
                        display_lines.append(f"{action['code']}: {int(smoothed_angle)}deg")
                        display_lines.append(f"-> C:{action['rep_count']} (MANUAL: {int(MIN_TARGET)}/{int(MAX_TARGET)})")
                    else:
                        display_lines.append(f"{action['code']}: {int(smoothed_angle)}deg")
                        display_lines.append("-> **RANGE TOO SMALL**")



                else:

                    if smoothed_angle > MAX_TARGET - ANGLE_BUFFER:

                        action['is_at_max'] = True



                    if smoothed_angle < MIN_TARGET + ANGLE_BUFFER and action['is_at_max']:

                        action['is_rep_complete'] = True



                   



                    status = "IN PROGRESS"

                    if action['is_at_max']:

                        status = "MAX ACHIEVED"

                    if action['is_rep_complete']:

                        status = "WAITING SYNC"



                    display_lines.append(f"{action['code']}: {int(smoothed_angle)}° ({status})")

                    display_lines.append(f"-> C:{action['rep_count']} (TARGET: {int(MIN_TARGET)}/{int(MAX_TARGET)})")



        else:

            display_lines.append(f"{action['code']}: LOST")

            MIN_TARGET, MAX_TARGET = action.get('min_target', 0), action.get('max_target', 0)

            display_lines.append(f"-> C:{action['rep_count']} (T:{int(MIN_TARGET)}/{int(MAX_TARGET)})")



    trigger_good_job = False

    if exercise_sub_phase == EXERCISE_COUNT:

        all_actions_complete = True

        for action in current_task_set['actions']:

            if not action['is_rep_complete']:

                all_actions_complete = False

                break



        if all_actions_complete and len(current_task_set['actions']) > 0:

            trigger_good_job = True

            for action in current_task_set['actions']:

                action['rep_count'] += 1

                send_rep_count(action['rep_count'], action['code'])

                action['rep_history'].append((action['rep_min_angle'], action['rep_max_angle']))

                action['is_at_max'] = False

                action['is_rep_complete'] = False

                action['rep_min_angle'], action['rep_max_angle'] = 180.0, 0.0



    return display_lines, active_pose_indices, active_hand_map, trigger_good_job
    