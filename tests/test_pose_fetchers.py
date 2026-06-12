import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from calibration import compute_transducer_from_tracker
from core import _rot_x, _rot_y, _rot_z
from pose_fetcher import TrackerPoseFetcher
from pose_math import component_errors
from server import FakePoseFetcher


class _Pose:
    bPoseIsValid = True
    mDeviceToAbsoluteTracking = (
        (1.0, 0.0, 0.0, 0.12),
        (0.0, 0.0, -1.0, 0.34),
        (0.0, 1.0, 0.0, -0.56),
    )


class _System:
    def __init__(self, openvr):
        self.openvr = openvr

    def getTrackedDeviceClass(self, index):
        if index == 2:
            return self.openvr.TrackedDeviceClass_GenericTracker
        return 0

    def getStringTrackedDeviceProperty(self, index, prop):
        return "TEST-TRACKER"

    def getDeviceToAbsoluteTrackingPose(self, universe, prediction, count):
        poses = [types.SimpleNamespace(bPoseIsValid=False) for _ in range(count)]
        poses[2] = _Pose()
        return poses


def _fake_openvr():
    module = types.ModuleType("openvr")
    module.VRApplication_Other = 3
    module.k_unMaxTrackedDeviceCount = 4
    module.TrackedDeviceClass_GenericTracker = 3
    module.TrackingUniverseStanding = 1
    module.Prop_SerialNumber_String = 1000
    module.init = lambda application: _System(module)
    module.shutdown = lambda: None
    return module


class PoseFetcherTests(unittest.TestCase):
    def test_tracker_fetcher_returns_calibrated_transducer_center(self):
        openvr = _fake_openvr()
        with patch.dict(sys.modules, {"openvr": openvr}):
            fetcher = TrackerPoseFetcher()
            fetcher.connect()
            actual = fetcher.get_pose()

        raw = np.eye(4)
        raw[:3, :4] = np.asarray(_Pose.mDeviceToAbsoluteTracking)
        expected = compute_transducer_from_tracker(raw)
        np.testing.assert_allclose(actual, expected)

    def test_fake_translation_keys_use_requested_local_axes(self):
        fetcher = FakePoseFetcher()
        fetcher.connect()
        pairs = (("d", "a", 0), ("w", "s", 1), ("q", "e", 2))
        reference = np.eye(4)
        reference[:3, :3] = _rot_z(0.7) @ _rot_y(-0.4)

        for positive, negative, axis in pairs:
            initial = fetcher.get_pose()
            before = component_errors(initial, reference)
            fetcher.nudge(positive, reference)
            moved = fetcher.get_pose()
            after = component_errors(moved, reference)
            expected = reference[:3, :3][:, axis] * 0.01
            np.testing.assert_allclose(
                moved[:3, 3] - initial[:3, 3], expected, atol=1e-12
            )
            changed = np.array([
                after[name] - before[name] for name in ("x", "y", "z")
            ])
            expected_components = np.zeros(3)
            expected_components[axis] = 0.01
            np.testing.assert_allclose(changed, expected_components, atol=1e-12)

            fetcher.nudge(negative, reference)
            np.testing.assert_allclose(fetcher.get_pose(), initial, atol=1e-12)

    def test_fake_rotation_keys_use_requested_local_axes(self):
        fetcher = FakePoseFetcher()
        fetcher.connect()
        pairs = (("u", "o"), ("i", "k"), ("j", "l"))
        reference = np.eye(4)
        reference[:3, :3] = _rot_x(-0.3) @ _rot_z(0.5)

        for axis, (positive, negative) in enumerate(pairs):
            initial = fetcher.get_pose()
            before = component_errors(initial, reference)
            fetcher.nudge(positive, reference)
            after = component_errors(fetcher.get_pose(), reference)
            changed = np.array([
                after[name] - before[name] for name in ("roll", "pitch", "yaw")
            ])
            expected = np.zeros(3)
            expected[axis] = 2.0
            np.testing.assert_allclose(changed, expected, atol=1e-10)

            fetcher.nudge(negative, reference)
            restored = component_errors(fetcher.get_pose(), reference)
            np.testing.assert_allclose(
                [restored[name] for name in ("roll", "pitch", "yaw")],
                [before[name] for name in ("roll", "pitch", "yaw")],
                atol=1e-10,
            )

    def test_fake_random_start_stays_inside_calibrated_cube(self):
        fetcher = FakePoseFetcher()
        fetcher.connect()
        origin = fetcher.get_pose()

        fetcher.randomize(origin)
        randomized = fetcher.get_pose()
        local_position = origin[:3, :3].T @ (
            randomized[:3, 3] - origin[:3, 3]
        )

        self.assertTrue(np.all(np.abs(local_position) <= 0.25))
        self.assertFalse(np.allclose(randomized, origin))


if __name__ == "__main__":
    unittest.main()
