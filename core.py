"""Shared constants and target-generation helpers (competition and study mode)."""

import math
import random
import numpy as np

CUBE_SIZE     = 0.5   # metres, side length (cube centred on calibration origin)
HOLD_DURATION = 1.0   # seconds of continuous match required to register a hit
LINEAR_TOL    = 0.01  # metres, 1 cm linear tolerance for competition targets

# Box prism dimensions (study mode).  Basis convention: bX=+Y_t, bY=−Z_t, bZ=−X_t.
BOX_HALF_XY = 0.125   # ±12.5 cm lateral and elevation (box X and Y)
BOX_Z_MIN   = 0.02    # near face: 2 cm from tip along box +Z
BOX_Z_MAX   = 0.12    # far face: 12 cm from tip along box +Z


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
    centroid and targets spread in all directions equally.  Orientation is a
    uniform random rotation from SO(3) (Haar measure via QR decomposition)
    composed with the calibration orientation.
    """
    half = CUBE_SIZE / 2
    local_pos = np.array([
        random.uniform(-half, half),   # left / right  (local X)
        random.uniform(-half, half),   # up   / down   (local Y)
        random.uniform(-half, half),   # fore / aft    (local Z)
    ])
    world_pos = origin[:3, 3] + origin[:3, :3] @ local_pos
    q, r = np.linalg.qr(np.random.randn(3, 3))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    target = np.eye(4, dtype=float)
    target[:3, :3] = q @ origin[:3, :3]
    target[:3, 3]  = world_pos
    return target


def study_target_from_params(
    box_origin: np.ndarray,
    u: float,
    v: float,
    w: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Build a study-target pose inside the box prism from explicit parameters.

    box_origin is the raw tip pose from get_pose() stored in BOX_ORIGIN.
    Returns a 4×4 tip-frame pose; HEAD_OFFSET_M must NOT be applied again.

    Position: (u, v, w) are metres along the box basis — u/v lateral and
    elevation (±BOX_HALF_XY), w depth (BOX_Z_MIN–BOX_Z_MAX) — expressed in
    world frame via the box basis.

    Orientation: intrinsic XYZ perturbation of the set-box orientation —
    roll/pitch/yaw in radians, composed as R_perturb = Rx(roll) @ Ry(pitch)
    @ Rz(yaw), matching pose_math.component_errors decomposition.
    """
    bX =  box_origin[:3, 1].copy()   # lateral      (+transducer Y)
    bY = -box_origin[:3, 2].copy()   # elevation    (−transducer Z)
    bZ = -box_origin[:3, 0].copy()   # depth        (−transducer X, away from tip)
    tip = box_origin[:3, 3]

    pos = tip + u * bX + v * bY + w * bZ

    R_perturb = _rot_x(roll) @ _rot_y(pitch) @ _rot_z(yaw)
    R_target  = box_origin[:3, :3] @ R_perturb

    target = np.eye(4, dtype=float)
    target[:3, :3] = R_target
    target[:3, 3]  = pos
    return target


def random_study_target(box_origin: np.ndarray) -> np.ndarray:
    """Random study target inside the box prism with bounded orientation.

    Position: uniform inside the prism (±BOX_HALF_XY lateral/elevation,
    BOX_Z_MIN–BOX_Z_MAX depth). Orientation: roll ∈ [−π, π), pitch ∈
    [−60°, 60°], yaw ∈ [−60°, 60°]. See study_target_from_params for the
    exact composition.
    """
    u = random.uniform(-BOX_HALF_XY, BOX_HALF_XY)
    v = random.uniform(-BOX_HALF_XY, BOX_HALF_XY)
    w = random.uniform(BOX_Z_MIN, BOX_Z_MAX)

    _60deg = math.radians(60)
    roll  = random.uniform(-math.pi, math.pi)
    pitch = random.uniform(-_60deg, _60deg)
    yaw   = random.uniform(-_60deg, _60deg)

    return study_target_from_params(box_origin, u, v, w, roll, pitch, yaw)


def fixed_study_target(box_origin: np.ndarray, target_id: str) -> np.ndarray:
    """Study target for a fixed, committed target id (e.g. "T5").

    Looks up box-local params from study.targets.STUDY_TARGETS so every
    participant sees the identical target for a given target_id.
    """
    from study.targets import STUDY_TARGETS

    params = STUDY_TARGETS[target_id]
    return study_target_from_params(
        box_origin,
        params["u"], params["v"], params["w"],
        params["roll"], params["pitch"], params["yaw"],
    )
