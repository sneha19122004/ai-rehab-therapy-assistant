import numpy as np
from collections import deque

def calculate_angle_3d(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

def smooth_angle(history_deque, new_angle):
    history_deque.append(new_angle)
    return np.mean(history_deque)

def get_safe_visibility(lm):
    if hasattr(lm, 'visibility') and lm.visibility is not None:
        return lm.visibility
    if hasattr(lm, 'presence') and lm.presence is not None:
        return lm.presence
    return 0.0