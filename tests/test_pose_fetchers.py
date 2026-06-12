import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from calibration import compute_transducer_from_tracker
from pose_fetcher import TrackerPoseFetcher
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
        pairs = (("1", "q", 0), ("2", "w", 1), ("3", "e", 2))

        for positive, negative, axis in pairs:
            initial = fetcher.get_pose()
            fetcher.nudge(positive)
            moved = fetcher.get_pose()
            expected = fetcher._control_frame[:, axis] * 0.01
            np.testing.assert_allclose(
                moved[:3, 3] - initial[:3, 3], expected, atol=1e-12
            )

            fetcher.nudge(negative)
            np.testing.assert_allclose(fetcher.get_pose(), initial, atol=1e-12)

    def test_fake_rotation_keys_use_requested_local_axes(self):
        fetcher = FakePoseFetcher()
        fetcher.connect()
        pairs = (("4", "r"), ("5", "t"), ("6", "y"))

        for positive, negative in pairs:
            initial = fetcher.get_pose()
            fetcher.nudge(positive)
            self.assertFalse(np.allclose(fetcher.get_pose(), initial))
            fetcher.nudge(negative)
            np.testing.assert_allclose(fetcher.get_pose(), initial, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
