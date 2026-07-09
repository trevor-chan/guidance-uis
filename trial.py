"""1D trial: match a target pose, end on success or timeout.

No calibration, no reference frame (1D case). Owns a pose fetcher,
compares each live pose to a hardcoded target, ends when within
tolerance or after a timeout.
"""

import time
import numpy as np

from pose_fetcher import LivePoseFetcher
from pose_math import angular_distance, component_errors, linear_distance


TARGET_POSE = np.array([
    [ 0.866, 0.0, 0.5,    0.30],
    [ 0.0,   1.0, 0.0,    0.10],
    [-0.5,   0.0, 0.866, -0.40],
    [ 0.0,   0.0, 0.0,    1.0],
])

LINEAR_TOLERANCE = 0.005    # meters (5 mm)
ANGULAR_TOLERANCE = 5.0     # degrees
TIMEOUT_SECONDS = 90.0


class Trial:
    """Runs one target-match test against a live pose source.

    timeout_seconds=None (default) reads the module-level TIMEOUT_SECONDS on
    every step, so tests that monkeypatch it still affect already-constructed
    instances. Pass math.inf explicitly to disable the timeout entirely (used
    by PracticeActivity, which must never auto-end on time).
    """

    def __init__(self, fetcher: LivePoseFetcher, target_pose: np.ndarray = TARGET_POSE,
                 linear_tol: float = LINEAR_TOLERANCE, angular_tol: float = ANGULAR_TOLERANCE,
                 timeout_seconds: float | None = None):
        self.fetcher = fetcher
        self.target_pose = target_pose
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def start(self) -> None:
        """Mark the trial start time. Call once before stepping."""
        self.start_time = time.perf_counter()

    def step(self) -> dict:
        """Read one live pose, compare to target, report current state.

        Returns a dict with the two distances, whether it's a match,
        whether it timed out, and elapsed time.
        """
        live_pose = self.fetcher.get_pose()
        elapsed = time.perf_counter() - self.start_time
        timeout = self.timeout_seconds if self.timeout_seconds is not None else TIMEOUT_SECONDS

        if live_pose is None:
            return {
                "linear": None,
                "angular": None,
                "matched": False,
                "timed_out": elapsed >= timeout,
                "elapsed": elapsed,
                "live_pose": None,
                "components": None,
                "component_aligned": {name: False for name in (
                    "x", "y", "z", "roll", "pitch", "yaw"
                )},
            }

        lin = linear_distance(live_pose, self.target_pose)
        ang = angular_distance(live_pose, self.target_pose)
        matched = lin <= self.linear_tol and ang <= self.angular_tol
        components = component_errors(live_pose, self.target_pose)
        component_aligned = {
            name: abs(value) <= (
                self.linear_tol if name in ("x", "y", "z") else self.angular_tol
            )
            for name, value in components.items()
        }

        return {
            "linear": lin,
            "angular": ang,
            "matched": matched,
            "timed_out": elapsed >= timeout,
            "elapsed": elapsed,
            "live_pose": live_pose.tolist(),
            "components": components,
            "component_aligned": component_aligned,
        }
