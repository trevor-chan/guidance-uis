"""1D trial: match a target pose, end on success or timeout.

No calibration, no reference frame (1D case). Owns a pose fetcher,
compares each live pose to a hardcoded target, ends when within
tolerance or after a timeout.
"""

import time
import numpy as np

from pose_fetcher import LivePoseFetcher
from pose_math import linear_distance, angular_distance


TARGET_POSE = np.array([
    [ 0.866, 0.0, 0.5,    0.30],
    [ 0.0,   1.0, 0.0,    0.10],
    [-0.5,   0.0, 0.866, -0.40],
    [ 0.0,   0.0, 0.0,    1.0],
])

LINEAR_TOLERANCE = 0.005    # meters (5 mm)
ANGULAR_TOLERANCE = 5.0     # degrees
TIMEOUT_SECONDS = 60.0


class Trial:
    """Runs one target-match test against a live pose source."""

    def __init__(self, fetcher: LivePoseFetcher, target_pose: np.ndarray = TARGET_POSE,
                 linear_tol: float = LINEAR_TOLERANCE, angular_tol: float = ANGULAR_TOLERANCE):
        self.fetcher = fetcher
        self.target_pose = target_pose
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
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

        if live_pose is None:
            return {
                "linear": None,
                "angular": None,
                "matched": False,
                "timed_out": elapsed >= TIMEOUT_SECONDS,
                "elapsed": elapsed,
                "live_pose": None,
            }

        lin = linear_distance(live_pose, self.target_pose)
        ang = angular_distance(live_pose, self.target_pose)
        matched = lin <= self.linear_tol and ang <= self.angular_tol

        return {
            "linear": lin,
            "angular": ang,
            "matched": matched,
            "timed_out": elapsed >= TIMEOUT_SECONDS,
            "elapsed": elapsed,
            "live_pose": live_pose.tolist(),
        }
