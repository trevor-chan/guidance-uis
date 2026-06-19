"""Pose source interface and implementations.

A pose is a 4x4 numpy homogeneous transform. get_pose() returns one,
or None if no valid pose is currently available.

All poses returned by get_pose() are in scanning-head / tip coordinates.
HEAD_OFFSET_M is applied in the base-class get_pose() wrapper so every
consumer (1D/2D/3D, competition, study, set-box) inherits tip coords
automatically.  Subclasses implement _raw_pose() (transducer-center coords)
and must not apply the offset themselves.
"""

from abc import ABC, abstractmethod
import numpy as np

from calibration import DEFAULT_CALIBRATION, compute_transducer_from_tracker

# Distance from transducer-center OBJ origin to scanning-head tip along
# local +X (red axis).  Validated correct sign/magnitude on the rig in
# --setbox mode.  Change here to recalibrate; zero to revert to OBJ-center.
HEAD_OFFSET_M = 0.07


class LivePoseFetcher(ABC):
    """Interface for anything that supplies live poses on demand.

    External callers always use get_pose(), which returns tip coordinates.
    Subclasses implement _raw_pose() (transducer-center / OBJ-origin coords).
    """

    source_mode = "unknown"
    source_label = "Unknown pose source"

    @abstractmethod
    def connect(self) -> None:
        """Set up the pose source. Call once before get_pose()."""
        pass

    @abstractmethod
    def _raw_pose(self) -> np.ndarray | None:
        """Return the raw transducer-center pose, or None if invalid.

        Do NOT call directly from outside this class hierarchy.  Use get_pose().
        """
        pass

    def get_pose(self) -> np.ndarray | None:
        """Return current tip pose (4x4), or None if invalid.

        Applies HEAD_OFFSET_M along the transducer's local +X axis to convert
        from transducer-center (_raw_pose) to scanning-head tip coordinates.
        This is the single point where the offset is applied; all consumers
        receive tip coordinates without any further adjustment.
        """
        pose = self._raw_pose()
        if pose is None:
            return None
        result = pose.copy()
        result[:3, 3] += pose[:3, 0] * HEAD_OFFSET_M
        return result

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the pose source cleanly."""
        pass


class TrackerPoseFetcher(LivePoseFetcher):
    """Real fetcher: wraps the OpenVR/SteamVR tracker. Runs on the lab rig."""

    source_mode = "tracker"
    source_label = "SteamVR tracker"

    def __init__(self, calibration=DEFAULT_CALIBRATION, probe_element: str = "phased"):
        self.vr_system = None
        self.device_index = None
        self.device_serial = None
        self.calibration = calibration
        self.probe_element = probe_element

    def connect(self) -> None:
        import openvr
        self.vr_system = openvr.init(openvr.VRApplication_Other)

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if self.vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
                self.device_index = i
                try:
                    self.device_serial = self.vr_system.getStringTrackedDeviceProperty(
                        i, openvr.Prop_SerialNumber_String
                    )
                except Exception:
                    self.device_serial = None
                if self.device_serial:
                    self.source_label = f"SteamVR tracker {self.device_serial}"
                return

        openvr.shutdown()
        raise RuntimeError("No generic tracker found. Is it powered on and tracked by SteamVR?")

    def _raw_pose(self) -> np.ndarray | None:
        import openvr
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        pose = poses[self.device_index]
        if not pose.bPoseIsValid:
            return None

        m = pose.mDeviceToAbsoluteTracking  # 3x4
        tracker_pose = np.array([
            [m[0][0], m[0][1], m[0][2], m[0][3]],
            [m[1][0], m[1][1], m[1][2], m[1][3]],
            [m[2][0], m[2][1], m[2][2], m[2][3]],
            [0.0,     0.0,     0.0,     1.0],
        ])
        return compute_transducer_from_tracker(
            tracker_pose, self.calibration, self.probe_element
        )

    def disconnect(self) -> None:
        import openvr
        openvr.shutdown()


class FakePoseFetcher(LivePoseFetcher):
    """Fake fetcher: produces poses without hardware. Runs on the laptop."""
    pass
