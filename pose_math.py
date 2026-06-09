"""Pose comparison math for the 1D trial.

Two numbers from a live pose vs a target pose:
  - linear distance (meters)
  - angular distance (degrees)
"""

import numpy as np


def linear_distance(live_pose: np.ndarray, target_pose: np.ndarray) -> float:
    """Straight-line distance between the two positions, in meters."""
    live_position = live_pose[:3, 3]
    target_position = target_pose[:3, 3]
    return float(np.linalg.norm(live_position - target_position))


def angular_distance(live_pose: np.ndarray, target_pose: np.ndarray) -> float:
    """Single rotation angle between the two orientations, in degrees."""
    R_live = live_pose[:3, :3]
    R_target = target_pose[:3, :3]

    R_diff = R_target @ R_live.T

    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)   # guard arccos against nan

    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))