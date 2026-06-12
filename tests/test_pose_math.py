import math
import unittest

import numpy as np

from pose_math import component_errors, workspace_component_errors


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class ComponentErrorTests(unittest.TestCase):
    def test_translation_is_reported_in_target_local_frame(self):
        target = np.eye(4)
        target[:3, :3] = rot_z(math.radians(90))
        target[:3, 3] = [0.3, -0.1, 0.5]

        live = target.copy()
        local_delta = np.array([0.02, -0.03, 0.04])
        live[:3, 3] += target[:3, :3] @ local_delta

        errors = component_errors(live, target)
        np.testing.assert_allclose(
            [errors["x"], errors["y"], errors["z"]], local_delta, atol=1e-12
        )

    def test_roll_pitch_yaw_components(self):
        target = np.eye(4)
        target[:3, :3] = rot_z(0.3) @ rot_y(-0.2)
        expected = np.array([12.0, -9.0, 17.0])

        live = target.copy()
        live[:3, :3] = target[:3, :3] @ (
            rot_x(math.radians(expected[0]))
            @ rot_y(math.radians(expected[1]))
            @ rot_z(math.radians(expected[2]))
        )

        errors = component_errors(live, target)
        np.testing.assert_allclose(
            [errors["roll"], errors["pitch"], errors["yaw"]],
            expected,
            atol=1e-10,
        )

    def test_workspace_errors_compare_independent_reference_components(self):
        reference = np.eye(4)
        reference[:3, :3] = rot_z(0.4)
        target = reference.copy()
        target[:3, 3] += reference[:3, :3] @ np.array([0.1, -0.2, 0.05])
        target[:3, :3] = reference[:3, :3] @ (
            rot_x(0.2) @ rot_y(-0.1) @ rot_z(0.3)
        )
        live = target.copy()
        live[:3, 3] += reference[:3, :3] @ np.array([0.01, 0.02, -0.03])
        live[:3, :3] = reference[:3, :3] @ (
            rot_x(0.2) @ rot_y(-0.1) @ rot_z(0.3 + math.radians(2))
        )

        errors = workspace_component_errors(live, target, reference)
        np.testing.assert_allclose(
            [errors["x"], errors["y"], errors["z"]],
            [0.01, 0.02, -0.03],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [errors["roll"], errors["pitch"], errors["yaw"]],
            [0.0, 0.0, 2.0],
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
