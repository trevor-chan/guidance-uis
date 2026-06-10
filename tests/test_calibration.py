import math
import unittest

import numpy as np

from calibration import (
    CalibrationConfig,
    compute_transducer_from_tracker,
)


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class CalibrationTests(unittest.TestCase):
    def test_center_shift_is_negative_eight_cm_in_corrected_local_x(self):
        tracker_pose = np.eye(4)
        tracker_pose[:3, :3] = rot_z(0.4) @ rot_y(-0.2) @ rot_x(0.1)
        tracker_pose[:3, 3] = [0.7, 1.1, -0.3]

        centered = compute_transducer_from_tracker(tracker_pose)
        imaging_plane = compute_transducer_from_tracker(
            tracker_pose, center_offset_m=np.zeros(3)
        )

        world_delta = centered[:3, 3] - imaging_plane[:3, 3]
        local_delta = centered[:3, :3].T @ world_delta
        np.testing.assert_allclose(
            local_delta, CalibrationConfig().transducer_center_offset, atol=1e-12
        )

    def test_extrinsic_then_virtual_then_center_composition(self):
        config = CalibrationConfig(
            tracker_to_transducer_rotation_euler_zyx_deg=(12.0, -8.0, 3.0),
            tracker_to_transducer_translation_m=(0.02, -0.03, 0.04),
            tracker_world_basis_diag=(1.0, 1.0, 1.0),
            transducer_local_basis_diag=(1.0, 1.0, 1.0),
            extrinsic_correction_translation_m=(0.01, 0.02, -0.01),
            extrinsic_correction_rotvec_deg=(4.0, -2.0, 3.0),
            virtual_source_offset_phased_m=(-0.002, 0.003, -0.005),
        )
        tracker_pose = np.eye(4)
        tracker_pose[:3, :3] = rot_y(0.3) @ rot_x(-0.1)
        tracker_pose[:3, 3] = [0.2, -0.5, 0.8]

        actual = compute_transducer_from_tracker(tracker_pose, config)

        R = tracker_pose[:3, :3] @ config.rotation_matrix
        t = tracker_pose[:3, 3] + tracker_pose[:3, :3] @ config.translation
        t = t + R @ config.extrinsic_correction_translation
        R = R @ config.extrinsic_correction_rotation
        t = t + R @ config.get_virtual_source_offset("phased")
        t = t + R @ config.transducer_center_offset

        np.testing.assert_allclose(actual[:3, :3], R, atol=1e-12)
        np.testing.assert_allclose(actual[:3, 3], t, atol=1e-12)

    def test_linear_and_phased_offsets_are_selectable(self):
        tracker_pose = np.eye(4)
        phased = compute_transducer_from_tracker(tracker_pose, probe_element="phased")
        linear = compute_transducer_from_tracker(tracker_pose, probe_element="linear")
        self.assertFalse(np.allclose(phased[:3, 3], linear[:3, 3]))


if __name__ == "__main__":
    unittest.main()
