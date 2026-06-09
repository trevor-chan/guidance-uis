"""Shared constants and target-generation helpers (competition and future study mode)."""

import math
import random
import numpy as np

CUBE_SIZE     = 0.5   # metres, side length (cube centred on calibration origin)
HOLD_DURATION = 1.0   # seconds of continuous match required to register a hit
LINEAR_TOL    = 0.01  # metres, 1 cm linear tolerance for competition targets


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


_ROT_FNS = [_rot_x, _rot_y, _rot_z]


def _random_target_pose(origin: np.ndarray) -> np.ndarray:
    """Random target within a 0.5 m cube centred on the calibration origin.

    Each axis offset is ±0.25 m (local X/Y/Z), so the calibration pose is the
    centroid and targets spread in all directions equally.  Orientation is
    randomised up to 30° off the calibration orientation on a randomly chosen axis.
    """
    half = CUBE_SIZE / 2
    local_pos = np.array([
        random.uniform(-half, half),   # left / right  (local X)
        random.uniform(-half, half),   # up   / down   (local Y)
        random.uniform(-half, half),   # fore / aft    (local Z)
    ])
    world_pos = origin[:3, 3] + origin[:3, :3] @ local_pos
    target = np.eye(4, dtype=float)
    angle = random.uniform(-math.radians(30), math.radians(30))
    target[:3, :3] = random.choice(_ROT_FNS)(angle) @ origin[:3, :3]
    target[:3, 3]  = world_pos
    return target
