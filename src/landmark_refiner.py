# landmark_refiner.py
import numpy as np
from collections import deque
import threading

class UnifiedLandmarkRefiner:
    def __init__(self, window_size=5):
        """
        Enhanced refiner that works with your existing system
        """
        # Temporal smoothing (compatible with your existing)
        self.window_size = window_size
        self.pose_history = {}
        
        # Anatomical constraints (enhancement)
        self.bone_lengths = {}
        self.min_calibration_frames = 15
        self.calibration_frames = 0
        self.is_calibrated = False
        
        # ROM tracking enhancement
        self.angle_history = {}
        self.rom_history = {}
        self.lock = threading.Lock()
    
    def smooth_pose(self, pose_landmarks_list):
        """Your existing smoothing function - enhanced with weighted average"""
        if not pose_landmarks_list:
            return None
        
        refined_lms = []
        
        for i, lm in enumerate(pose_landmarks_list):
            if i not in self.pose_history:
                self.pose_history[i] = deque(maxlen=self.window_size)
            
            self.pose_history[i].append([lm.x, lm.y, lm.z])
            avg = np.mean(self.pose_history[i], axis=0)
            
            # Reconstruct the object structure
            refined_lms.append(type('obj', (object,), {
                'x': avg[0], 
                'y': avg[1], 
                'z': avg[2], 
                'visibility': getattr(lm, 'visibility', 1.0)
            }))
        
        return refined_lms
    
    def calibrate_bone_lengths(self, landmarks):
        """Calibrate anatomical constraints"""
        if self.calibration_frames < self.min_calibration_frames and landmarks:
            # Track bone pairs for calibration
            bone_pairs = [
                (11, 13, "left_upper_arm"),
                (13, 15, "left_forearm"),
                (12, 14, "right_upper_arm"),
                (14, 16, "right_forearm"),
                (23, 25, "left_upper_leg"),
                (25, 27, "left_lower_leg"),
                (24, 26, "right_upper_leg"),
                (26, 28, "right_lower_leg")
            ]
            
            for p1_idx, p2_idx, bone_name in bone_pairs:
                if p1_idx < len(landmarks) and p2_idx < len(landmarks):
                    p1 = landmarks[p1_idx]
                    p2 = landmarks[p2_idx]
                    
                    # Calculate 3D distance
                    dist_3d = np.sqrt(
                        (p2.x - p1.x)**2 + 
                        (p2.y - p1.y)**2 + 
                        (p2.z - p1.z)**2
                    )
                    
                    # Update bone length (keep maximum observed)
                    if bone_name not in self.bone_lengths or dist_3d > self.bone_lengths[bone_name]:
                        self.bone_lengths[bone_name] = dist_3d
            
            self.calibration_frames += 1
            if self.calibration_frames >= self.min_calibration_frames:
                self.is_calibrated = True
                print(f"✅ Bone calibration complete. Tracked {len(self.bone_lengths)} bones.")
    
    def apply_bone_constraint(self, parent_idx, child_idx, bone_name, landmarks):
        """Your existing function - enhanced with safety checks"""
        if not landmarks or len(landmarks) <= max(parent_idx, child_idx):
            return None
            
        p1 = landmarks[parent_idx]
        p2 = landmarks[child_idx]

        # Calculate 2D distance on screen
        dist_2d = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
        
        # Calibration: Remember the longest we've ever seen this bone
        if bone_name not in self.bone_lengths or dist_2d > self.bone_lengths[bone_name]:
            self.bone_lengths[bone_name] = dist_2d
        
        max_L = self.bone_lengths[bone_name]
        
        # Solve for Z
        z_depth_sq = max(0, max_L**2 - dist_2d**2)
        z_reconstructed = np.sqrt(z_depth_sq)
        
        # Determine direction
        direction = 1 if hasattr(p2, 'z') and hasattr(p1, 'z') and p2.z > p1.z else -1
        
        # Apply the reconstructed depth
        return type('obj', (object,), {
            'x': p2.x,
            'y': p2.y,
            'z': p1.z + (z_reconstructed * direction),
            'visibility': getattr(p2, 'visibility', 1.0)
        })
    
    def refine_rom_angle(self, action_code, raw_angle):
        """Apply temporal smoothing to ROM angles"""
        with self.lock:
            if action_code not in self.angle_history:
                self.angle_history[action_code] = deque(maxlen=10)
            
            self.angle_history[action_code].append(raw_angle)
            
            # Apply weighted temporal smoothing
            if len(self.angle_history[action_code]) >= 3:
                angles = list(self.angle_history[action_code])
                weights = np.linspace(0.3, 1.0, len(angles))
                weights = weights / weights.sum()
                refined_angle = np.average(angles, weights=weights)
            else:
                refined_angle = raw_angle
            
            return refined_angle
    
    def reset(self):
        """Reset for new exercise - compatible with your existing reset_history"""
        self.pose_history.clear()
        self.hand_history.clear() if hasattr(self, 'hand_history') else None
        self.angle_history.clear()
        self.rom_history.clear()
        self.calibration_frames = 0
        self.is_calibrated = False
        self.bone_lengths.clear()
    
    def get_status(self):
        """Get calibration status for UI"""
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_frames': self.calibration_frames,
            'min_calibration_frames': self.min_calibration_frames,
            'tracked_bones': len(self.bone_lengths)
        }