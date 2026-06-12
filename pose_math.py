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


def component_errors(live_pose: np.ndarray, target_pose: np.ndarray) -> dict:
    """Return target-local XYZ translation and intrinsic XYZ rotation errors.

    Translation is in meters. Roll, pitch, and yaw are in degrees.
    """
    R_live = live_pose[:3, :3]
    R_target = target_pose[:3, :3]

    translation = R_target.T @ (live_pose[:3, 3] - target_pose[:3, 3])
    rotation = R_target.T @ R_live

    # Decompose rotation = Rx(roll) @ Ry(pitch) @ Rz(yaw).
    sin_pitch = float(np.clip(rotation[0, 2], -1.0, 1.0))
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > 1e-8:
        roll = np.arctan2(-rotation[1, 2], rotation[2, 2])
        yaw = np.arctan2(-rotation[0, 1], rotation[0, 0])
    else:
        roll = np.arctan2(rotation[2, 1], rotation[1, 1])
        yaw = 0.0

    angles = np.degrees([roll, pitch, yaw])
    return {
        "x": float(translation[0]),
        "y": float(translation[1]),
        "z": float(translation[2]),
        "roll": float(angles[0]),
        "pitch": float(angles[1]),
        "yaw": float(angles[2]),
    }


def workspace_component_errors(
    live_pose: np.ndarray,
    target_pose: np.ndarray,
    reference_pose: np.ndarray,
) -> dict:
    """Return independent component differences in one fixed reference frame."""
    live = component_errors(live_pose, reference_pose)
    target = component_errors(target_pose, reference_pose)
    result = {
        name: live[name] - target[name]
        for name in ("x", "y", "z")
    }
    for name in ("roll", "pitch", "yaw"):
        result[name] = (live[name] - target[name] + 180.0) % 360.0 - 180.0
    return result
