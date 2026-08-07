"""Study activity classes: base interface plus three concrete activities.

Pull model: every activity exposes start() + step() → dict.
step() reads exactly one pose and returns plain data; the caller owns the loop.
"""

from abc import ABC, abstractmethod
import math
import time
import numpy as np

from pose_fetcher import LivePoseFetcher
from trial import Trial, LINEAR_TOLERANCE
from core import HOLD_DURATION


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

COUNTDOWN_SECONDS = 3.0  # pre-trial 3-2-1 countdown, measured server-side (see start()/step())


class TrialActivity(Activity):
    """Wraps Trial with a pre-trial countdown, then 1-second continuous-
    hold-to-register on top.

    Finish conditions (whichever fires first, both measured from the END of
    the countdown, not from start()):
      - hold_duration (default 1 s) of continuous match → achieved=True
      - Trial's 90 s timeout                            → achieved=False

    start() does NOT start Trial's clock immediately -- it only stamps when
    the countdown began. step() reports countdown_remaining (and leaves
    elapsed/live_pose/etc at their "nothing happening yet" defaults) until
    COUNTDOWN_SECONDS has elapsed, then calls self._trial.start() for real on
    that exact tick. This is the single point where the trial clock and the
    90s timeout window both begin -- see trial.py Trial.start()/step(). The
    countdown is timed with time.perf_counter(), the same clock Trial itself
    uses, so there's one continuous clock source across the handoff.

    hold_progress (0.0–1.0) is reported every step so a renderer can show a
    fill bar without any extra bookkeeping outside this class.
    """

    def __init__(
        self,
        fetcher: LivePoseFetcher,
        target_pose: np.ndarray,
        linear_tol: float = LINEAR_TOLERANCE,
        angular_tol: float = 5.0,
        hold_duration: float = HOLD_DURATION,
        label: str | None = None,
        noise: float | None = None,
        latency_ms: float | None = None,
        perceived_ms: float | None = None,
        precision_linear_mm: float | None = None,
        precision_angular_deg: float | None = None,
        countdown_seconds: float = COUNTDOWN_SECONDS,
    ) -> None:
        # angular_tol defaults to 5° — matches trial.ANGULAR_TOLERANCE.
        self._trial = Trial(fetcher, target_pose, linear_tol=linear_tol, angular_tol=angular_tol)
        self._hold_duration = hold_duration
        self._hold_start: float | None = None
        self._achieved = False
        self._done = False
        self._countdown_seconds = countdown_seconds
        self._countdown_start: float | None = None
        self._trial_started = False
        self.label = label  # e.g. "T5" for fixed study targets; None otherwise
        # Per-trial ramp metadata for the noise/latency/precision experiments'
        # scrambled blocks (see SequenceGenerator.make_block's trial_overrides).
        # None for every other experiment/trial — server.py and persistence
        # both read these straight off the active TrialActivity so display
        # perturbation and the recorded trial_* columns never disagree.
        self.noise = noise
        self.latency_ms = latency_ms
        self.perceived_ms = perceived_ms
        self.precision_linear_mm = precision_linear_mm
        self.precision_angular_deg = precision_angular_deg

    @property
    def target_pose(self) -> np.ndarray:
        return self._trial.target_pose

    def start(self) -> None:
        # Trial's clock is deliberately NOT started here -- only once the
        # countdown elapses, in step() below. Re-entering start() (e.g. a
        # pause/resume rebuild) restarts the countdown from the top, which is
        # correct: a fresh attempt gets its own full 3-2-1.
        self._countdown_start = time.perf_counter()
        self._trial_started = False
        self._hold_start = None
        self._achieved = False
        self._done = False

    def _meta(self) -> dict:
        return {
            "label": self.label,
            "noise": self.noise,
            "latency_ms": self.latency_ms,
            "perceived_ms": self.perceived_ms,
            "precision_linear_mm": self.precision_linear_mm,
            "precision_angular_deg": self.precision_angular_deg,
            "linear_tol": self._trial.linear_tol,
            "angular_tol": self._trial.angular_tol,
        }

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
                "live_pose": None,
                "target_pose": self.target_pose.tolist(),
                "components": None,
                "countdown_remaining": None,
                **self._meta(),
            }

        if not self._trial_started:
            remaining = self._countdown_seconds - (time.perf_counter() - self._countdown_start)
            if remaining > 0:
                return {
                    "done": False,
                    "achieved": False,
                    "hold_progress": 0.0,
                    "linear": None,
                    "angular": None,
                    "matched": False,
                    "timed_out": False,
                    "elapsed": None,
                    "live_pose": None,
                    "target_pose": self.target_pose.tolist(),
                    "components": None,
                    "countdown_remaining": remaining,
                    **self._meta(),
                }
            # Countdown just finished on this tick -- start the REAL clock
            # now, at t=0. Both the time-to-match clock and the 90s timeout
            # window (both owned by Trial, see trial.py) begin here.
            self._trial.start()
            self._trial_started = True

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
            "live_pose": state["live_pose"],
            "target_pose": self.target_pose.tolist(),
            "components": state["components"],
            "countdown_remaining": None,
            **self._meta(),
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


# ── Practice ──────────────────────────────────────────────────────────────────

class PracticeActivity(Activity):
    """Practice phase: live reticle with box-origin pose as target.

    No timeout — runs indefinitely until request_end() is called from outside
    (e.g. the user presses Ready on the study UI). Match is displayed but does
    NOT end the phase — the user is encouraged to explore freely.
    """

    def __init__(self, fetcher: LivePoseFetcher, target_pose: np.ndarray) -> None:
        self._trial = Trial(fetcher, target_pose, timeout_seconds=math.inf)
        self._end_requested = False
        self._done = False

    @property
    def target_pose(self) -> np.ndarray:
        return self._trial.target_pose

    def start(self) -> None:
        self._trial.start()
        self._end_requested = False
        self._done = False

    def request_end(self) -> None:
        """Signal from server recv_loop that the user pressed Ready."""
        self._end_requested = True

    def step(self) -> dict:
        if self._done:
            return {
                "done": True, "achieved": False, "hold_progress": 0.0,
                "linear": None, "angular": None, "matched": False,
                "timed_out": False, "elapsed": None,
                "live_pose": None, "target_pose": self.target_pose.tolist(),
                "components": None,
            }
        state = self._trial.step()
        if self._end_requested:
            self._done = True
        return {
            "done": self._done,
            "achieved": False,
            "hold_progress": 0.0,
            "linear": state["linear"],
            "angular": state["angular"],
            "matched": state["matched"],
            "timed_out": False,
            "elapsed": state["elapsed"],
            "live_pose": state["live_pose"],
            "target_pose": self.target_pose.tolist(),
            "components": state["components"],
        }
