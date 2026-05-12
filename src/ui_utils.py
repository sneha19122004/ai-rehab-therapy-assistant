import cv2
from config import LANDMARK_COLOR, CONNECTION_COLOR, HAND_COLOR, MP_POSE_CONNECTIONS, MP_HAND_CONNECTIONS
from geometry_utils import get_safe_visibility

def draw_skeleton_and_vertices(
    frame, 
    pose_lms, 
    hand_lms_list, 
    handedness_list, 
    pose_connections, 
    active_pose_indices=set(),          # Default added
    active_hand_map={'left': set(), 'right': set()}, # Default added
    pose_color=LANDMARK_COLOR,          # Renamed from 'color'
    conn_color=CONNECTION_COLOR,        # New argument
    hand_color=HAND_COLOR,              # Assuming you want to be able to override HAND_COLOR
    thickness=2
):
    h, w, _ = frame.shape
    
    # Draw pose connections and landmarks
    if pose_lms:
        for i, j in pose_connections:
            if (i < len(pose_lms) and j < len(pose_lms) and 
                get_safe_visibility(pose_lms[i]) > 0.5 and 
                get_safe_visibility(pose_lms[j]) > 0.5):
                p1 = (int(pose_lms[i].x * w), int(pose_lms[i].y * h))
                p2 = (int(pose_lms[j].x * w), int(pose_lms[j].y * h))
                cv2.line(frame, p1, p2, conn_color, thickness)
        
        for idx, lm in enumerate(pose_lms):
            if get_safe_visibility(lm) > 0.5:
                center = (int(lm.x * w), int(lm.y * h))
                if idx in active_pose_indices:
                    draw_color, draw_size = (0, 255, 255), 8
                else:
                    draw_color, draw_size = pose_color, 5
                cv2.circle(frame, center, draw_size, draw_color, -1)
    
    # Draw hand connections and landmarks
    if hand_lms_list:
        for hand_idx, hand_lms in enumerate(hand_lms_list):
            hand_type_classification = handedness_list[hand_idx][0].category_name
            hand_type = hand_type_classification[0].lower() if hand_type_classification else ''
            for i, j in MP_HAND_CONNECTIONS:
                if i < len(hand_lms) and j < len(hand_lms):
                    p1 = (int(hand_lms[i].x * w), int(hand_lms[i].y * h))
                    p2 = (int(hand_lms[j].x * w), int(hand_lms[j].y * h))
                    cv2.line(frame, p1, p2, HAND_COLOR, thickness - 1)
            
            active_hand_lms = active_hand_map.get(hand_type, set())
            for idx, lm in enumerate(hand_lms):
                center = (int(lm.x * w), int(lm.y * h))
                if idx in active_hand_lms:
                    draw_color, draw_size = (0, 255, 255), 8
                else:
                    draw_color, draw_size = hand_color, 3 # Use the hand_color argument
                cv2.circle(frame, center, draw_size, draw_color, -1)