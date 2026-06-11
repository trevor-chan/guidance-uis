"""Tracker-to-transducer calibration used by the real SteamVR pose source."""

from dataclasses import dataclass
import math

import numpy as np


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _rotation_from_euler_zyx_deg(angles: tuple[float, float, float]) -> np.ndarray:
    z, y, x = np.deg2rad(angles)
    return _rot_z(z) @ _rot_y(y) @ _rot_x(x)


def _rotation_from_rotvec_deg(rotvec_deg: tuple[float, float, float]) -> np.ndarray:
    rotvec = np.deg2rad(np.asarray(rotvec_deg, dtype=float))
    angle = float(np.linalg.norm(rotvec))
    if angle == 0.0:
        return np.eye(3, dtype=float)

    x, y, z = rotvec / angle
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class CalibrationConfig:
    """Fixed PAL HD3 tracker-to-transducer calibration."""

    tracker_to_transducer_rotation_euler_zyx_deg: tuple[float, float, float] = (
        -90.0, -45.0, 0.0
    )
    tracker_to_transducer_translation_m: tuple[float, float, float] = (
        0.0, -0.09, 0.13
    )
    tracker_world_basis_diag: tuple[float, float, float] = (-1.0, 1.0, -1.0)
    transducer_local_basis_diag: tuple[float, float, float] = (1.0, -1.0, -1.0)
    extrinsic_correction_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    extrinsic_correction_rotvec_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    virtual_source_offset_phased_m: tuple[float, float, float] = (-0.002, 0.0, -0.005)
    virtual_source_offset_linear_m: tuple[float, float, float] = (-0.001, 0.002, 0.007)
    transducer_center_offset_m: tuple[float, float, float] = (-0.08, 0.0, 0.0)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _rotation_from_euler_zyx_deg(
            self.tracker_to_transducer_rotation_euler_zyx_deg
        )

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.tracker_to_transducer_translation_m, dtype=float)

    @property
    def tracker_world_basis(self) -> np.ndarray:
        return np.diag(self.tracker_world_basis_diag).astype(float)

    @property
    def transducer_local_basis(self) -> np.ndarray:
        return np.diag(self.transducer_local_basis_diag).astype(float)

    @property
    def extrinsic_correction_rotation(self) -> np.ndarray:
        return _rotation_from_rotvec_deg(self.extrinsic_correction_rotvec_deg)

    @property
    def extrinsic_correction_translation(self) -> np.ndarray:
        return np.asarray(self.extrinsic_correction_translation_m, dtype=float)

    def get_virtual_source_offset(self, probe_element: str = "phased") -> np.ndarray:
        if probe_element == "phased":
            offset = self.virtual_source_offset_phased_m
        elif probe_element == "linear":
            offset = self.virtual_source_offset_linear_m
        else:
            raise ValueError(
                f"Unknown probe element: {probe_element}. Expected 'phased' or 'linear'."
            )
        return np.asarray(offset, dtype=float)

    @property
    def transducer_center_offset(self) -> np.ndarray:
        return np.asarray(self.transducer_center_offset_m, dtype=float)


DEFAULT_CALIBRATION = CalibrationConfig()


def compute_transducer_from_tracker(
    tracker_pose: np.ndarray,
    calibration: CalibrationConfig = DEFAULT_CALIBRATION,
    probe_element: str = "phased",
    center_offset_m: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a 4x4 SteamVR tracker pose to the rendered transducer-center pose."""
    tracker_pose = np.asarray(tracker_pose, dtype=float)
    if tracker_pose.shape != (4, 4):
        raise ValueError("tracker_pose must be a 4x4 homogeneous transform")

    R_wt = tracker_pose[:3, :3]
    t_wt = tracker_pose[:3, 3]

    R_wc_recorded = R_wt @ calibration.rotation_matrix
    t_wc_recorded = t_wt + R_wt @ calibration.translation

    R_wc = (
        calibration.tracker_world_basis
        @ R_wc_recorded
        @ calibration.transducer_local_basis
    )
    t_wc = calibration.tracker_world_basis @ t_wc_recorded

    # These corrections are expressed in the current transducer-local frame.
    t_wc = t_wc + R_wc @ calibration.extrinsic_correction_translation
    R_wc = R_wc @ calibration.extrinsic_correction_rotation
    t_wc = t_wc + R_wc @ calibration.get_virtual_source_offset(probe_element)

    # Calibration terminates at the imaging plane. The OBJ origin is its center,
    # 8 cm farther along negative local X by default.
    center_offset = (
        calibration.transducer_center_offset
        if center_offset_m is None
        else np.asarray(center_offset_m, dtype=float)
    )
    t_wc = t_wc + R_wc @ center_offset

    result = np.eye(4, dtype=float)
    result[:3, :3] = R_wc
    result[:3, 3] = t_wc
    return result
