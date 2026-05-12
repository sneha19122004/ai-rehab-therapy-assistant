import sys
from config import ACTION_MAP, HAND_TRACKING_DETAILS, HISTORY, SMOOTHING_WINDOW_SIZE, MIN_REP_ANGLE_THRESHOLD
from collections import deque

def prompt_manual_range(action_code, min_val_trial, max_val_trial):
    print("\n" + "="*50)
    print(f"** ACTION: {action_code} **")
    print(f"Trial Range: MIN {int(min_val_trial)}° to MAX {int(max_val_trial)}° (Range: {int(max_val_trial - min_val_trial)}°)")
    print(f"WARNING: Range ({int(max_val_trial - min_val_trial)}°) is below the required threshold ({MIN_REP_ANGLE_THRESHOLD}°).")
    print("Please manually set your target MIN and MAX angles to resume counting.")
    
    while True:
        try:
            min_input = input(f"Enter target MIN angle for {action_code} (e.g., {int(min_val_trial)}): ")
            max_input = input(f"Enter target MAX angle for {action_code} (e.g., {int(max_val_trial)}): ")
            new_min = float(min_input)
            new_max = float(max_input)
            
            if new_max <= new_min:
                print("Error: MAX angle must be greater than MIN angle. Try again.")
                continue
            
            if (new_max - new_min) < MIN_REP_ANGLE_THRESHOLD:
                print(f"Error: Manual range ({new_max - new_min:.1f}°) must be greater than the threshold ({MIN_REP_ANGLE_THRESHOLD}°). Try again.")
                continue
            
            print(f"Range updated for {action_code}: MIN {int(new_min)}°, MAX {int(new_max)}°")
            print("="*50)
            return new_min, new_max
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except EOFError:
            sys.exit(1)

def get_monitoring_sequence():
    global HISTORY
    
    print("--- Define Sequential Tasks (CODE1;CODE2;...,DURATION_sec) ---")
    print("Available Codes:", ", ".join(ACTION_MAP.keys()))
    
    sequence, task_num, all_hist_keys = [], 1, set()
    default_state = {
        'rep_count': 0, 'is_at_max': False, 'is_rep_complete': False,
        'min_angle': 180.0, 'max_angle': 0.0, 'min_target': 180.0,
        'max_target': 0.0, 'rep_min_angle': 180.0, 'rep_max_angle': 0.0,
        'rep_history': []
    }
    
    while True:
        try:
            user_input = input(f"Task {task_num} (or DONE): ").upper()
        except EOFError:
            sys.exit(1)
        
        if user_input == 'DONE':
            break
        
        parts = user_input.split(',')
        if len(parts) != 2:
            print("Invalid format. Use: CODES,DURATION")
            continue
        
        raw_codes = parts[0].strip().split(';')
        
        # Parse duration FIRST, before using it in the loop
        try:
            duration = int(parts[1].strip())
        except ValueError:
            print("Invalid duration. Must be an integer.")
            continue
        
        if duration < 30:
            print("Duration must be 30 seconds or more.")
            continue
        
        tracking_list, is_shoulder_task = [], False
        
        for code in raw_codes:
            if code not in ACTION_MAP:
                continue
            
            action_info = ACTION_MAP[code]
            is_shoulder_task |= (code in ['R-SH', 'L-SH'])
            
            if action_info.get('is_group'):
                # Extract hand key from codes like "R-FINGERS" or "L-FINGERS"
                hand_key = code.split('-')[0]  # Get 'R' or 'L' from "R-FINGERS"
                details = HAND_TRACKING_DETAILS[hand_key]
                
                for key_suffix, detail in details['actions'].items():
                    full_code = f"{hand_key}-{key_suffix}"
                    hist_key = details['prefix'] + key_suffix
                    all_hist_keys.add(hist_key)
                    
                    action_state = default_state.copy()
                    action_state.update(detail)
                    action_state.update({
                        'code': full_code,
                        'hist_key': hist_key,
                        'type': details['type'],
                        'base_code': code,  # Store the original group code like "R-FINGERS"
                        'task_duration': duration
                    })
                    tracking_list.append(action_state)
            else:
                hist_key = action_info['hist_key']
                all_hist_keys.add(hist_key)
                
                action_state = default_state.copy()
                action_state.update(action_info)
                action_state['code'] = code
                action_state['base_code'] = code  # Same as code for non-grouped actions
                action_state['task_duration'] = duration
                tracking_list.append(action_state)
        
        if not tracking_list:
            print("No valid actions defined in this task.")
            continue
        
        total_parts = 3 if is_shoulder_task else 1
        part_duration = duration // total_parts
        task_name = " & ".join(raw_codes)
        sequence.append({
            'name': task_name,
            'actions': tracking_list,
            'duration': duration,
            'part_duration': part_duration,
            'total_parts': total_parts,
            'is_shoulder_task': is_shoulder_task
        })
        task_num += 1
    
    if not sequence:
        print("No tasks defined. Exiting.")
        sys.exit(1)
    
    HISTORY = {key: deque(maxlen=SMOOTHING_WINDOW_SIZE) for key in all_hist_keys}
    return sequence