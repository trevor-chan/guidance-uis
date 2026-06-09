"""Study activity classes: base interface plus three concrete activities.

Pull model: every activity exposes start() + step() → dict.
step() reads exactly one pose and returns plain data; the caller owns the loop.
"""

from abc import ABC, abstractmethod
import time
import numpy as np

from pose_fetcher import LivePoseFetcher
from trial import Trial
from core import LINEAR_TOL, HOLD_DURATION


class Activity(ABC):
    """Base for all study steps.

    Lifecycle:
        activity.start()          # once, before the first step
        while True:
            data = activity.step()
            if data["done"]:
                break
    """

    @abstractmethod
    def start(self) -> None:
        """Reset internal state. Must be called once before the first step()."""

    @abstractmethod
    def step(self) -> dict:
        """Advance by one pose-read. Always includes 'done': bool in the return dict."""


# ── Calibration ───────────────────────────────────────────────────────────────

class CalibrationActivity(Activity):
    """Captures the current tracker pose as the calibration origin (cube centre).

    Competition-style one-shot capture: the moment a valid pose is available,
    that pose becomes the origin. For 1D the reference frame is identity, so no
    further transform is applied.
    """

    def __init__(self, fetcher: LivePoseFetcher) -> None:
        self._fetcher = fetcher
        self._done = False
        self._origin: np.ndarray | None = None

    def start(self) -> None:
        self._done = False
        self._origin = None

    def step(self) -> dict:
        if self._done:
            return {"done": True, "origin": self._origin}
        pose = self._fetcher.get_pose()
        if pose is not None:
            self._origin = pose.copy()
            self._done = True
        return {"done": self._done, "origin": self._origin}


# ── Trial ─────────────────────────────────────────────────────────────────────

class TrialActivity(Activity):
    """Wraps Trial with 1-second continuous-hold-to-register on top.

    Finish conditions (whichever fires first):
      - hold_duration (default 1 s) of continuous match → achieved=True
      - Trial's 60 s timeout                            → achieved=False

    hold_progress (0.0–1.0) is reported every step so a renderer can show a
    fill bar without any extra bookkeeping outside this class.
    """

    def __init__(
        self,
        fetcher: LivePoseFetcher,
        target_pose: np.ndarray,
        linear_tol: float = LINEAR_TOL,
        angular_tol: float = 5.0,
        hold_duration: float = HOLD_DURATION,
    ) -> None:
        # angular_tol defaults to 5° — matches trial.ANGULAR_TOLERANCE.
        self._trial = Trial(fetcher, target_pose, linear_tol=linear_tol, angular_tol=angular_tol)
        self._hold_duration = hold_duration
        self._hold_start: float | None = None
        self._achieved = False
        self._done = False

    @property
    def target_pose(self) -> np.ndarray:
        return self._trial.target_pose

    def start(self) -> None:
        self._trial.start()
        self._hold_start = None
        self._achieved = False
        self._done = False

    def step(self) -> dict:
        if self._done:
            return {
                "done": True,
                "achieved": self._achieved,
                "hold_progress": 1.0 if self._achieved else 0.0,
                "linear": None,
                "angular": None,
                "matched": False,
                "timed_out": not self._achieved,
                "elapsed": None,
            }

        state = self._trial.step()
        now = time.monotonic()

        if state["matched"]:
            if self._hold_start is None:
                self._hold_start = now
            hold_dur = now - self._hold_start
            hold_progress = min(1.0, hold_dur / self._hold_duration)
            if hold_progress >= 1.0:
                self._achieved = True
                self._done = True
        else:
            self._hold_start = None
            hold_progress = 0.0

        if state["timed_out"] and not self._done:
            self._done = True

        return {
            "done": self._done,
            "achieved": self._achieved,
            "hold_progress": 1.0 if self._achieved else hold_progress,
            "linear": state["linear"],
            "angular": state["angular"],
            "matched": state["matched"],
            "timed_out": state["timed_out"],
            "elapsed": state["elapsed"],
        }


# ── Preference ────────────────────────────────────────────────────────────────

class PreferenceActivity(Activity):
    """Collects a 1–5 preference rating.

    Headless stub for Step 2: returns done=True with rating=None on the first
    step. Step 3 wires real user input through the WebSocket transport; at that
    point set_rating() can be called externally before or during stepping.
    """

    def __init__(self) -> None:
        self._done = False
        self._rating: int | None = None

    def start(self) -> None:
        self._done = False
        self._rating = None

    def step(self) -> dict:
        self._done = True
        return {"done": True, "rating": self._rating}

    def set_rating(self, rating: int) -> None:
        """Inject a rating from outside (Step 3 transport hook)."""
        self._rating = rating
