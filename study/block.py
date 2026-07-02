"""Block: runs one study condition as a fixed activity sequence.

Structure: CalibrationActivity → TrialActivity × N → PreferenceActivity.

Trial activities are created lazily after calibration completes, because their
targets depend on the captured origin pose. The caller provides a
trial_factory(origin) → list[TrialActivity] callback; Block expands its
activity list the moment calibration returns done=True.
"""

from typing import Callable, Optional
import numpy as np

from .activities import Activity, CalibrationActivity, TrialActivity, PreferenceActivity, PracticeActivity


class Block:
    """Runs one condition: [Calibration?] → [Practice?] → Trial×N → [Preference?].

    calibration=None skips the calibration step; trials are expanded immediately
    from fallback_origin at start() time.
    practice=None skips the practice phase.
    preference=None skips the preference phase (e.g. learning_curve/noise/latency).
    rebuild_trial(trial_index) -> TrialActivity, if supplied, is used by
    resume_current_trial() to build a fresh attempt for the current trial slot
    (e.g. a new random target) after a participant-triggered pause/cancel. When
    omitted, resume_current_trial() just restarts the same TrialActivity.
    """

    def __init__(
        self,
        calibration: CalibrationActivity | None,
        trial_factory: Callable[[np.ndarray], list[TrialActivity]],
        preference: Optional[PreferenceActivity],
        practice: PracticeActivity | None = None,
        fallback_origin: np.ndarray | None = None,
        rebuild_trial: Callable[[int], TrialActivity] | None = None,
    ) -> None:
        self._calibration = calibration
        self._practice = practice
        self._trial_factory = trial_factory
        self._preference = preference
        self._fallback_origin = fallback_origin
        self._rebuild_trial = rebuild_trial
        self._activities: list[Activity] = []
        self._idx = 0
        self._done = False
        self._origin: np.ndarray | None = None
        self._paused = False

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def origin(self) -> np.ndarray | None:
        """Calibration origin (or fallback_origin when calibration is skipped)."""
        return self._origin

    def start(self) -> None:
        self._idx = 0
        self._done = False
        self._origin = None
        self._paused = False
        if self._calibration is not None:
            # Lazy expansion: start with calibration only; trials added after origin known.
            self._activities = [self._calibration]
        else:
            # No calibration: expand immediately from fallback_origin.
            origin = self._fallback_origin
            self._origin = origin
            trials = self._trial_factory(origin) if origin is not None else []
            practice_part = [self._practice] if self._practice is not None else []
            preference_part = [self._preference] if self._preference is not None else []
            self._activities = practice_part + trials + preference_part
        self._activities[0].start()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def current_activity(self) -> Activity | None:
        """The activity currently being stepped, or None if the block is done."""
        if self._idx >= len(self._activities):
            return None
        return self._activities[self._idx]

    @property
    def current_trial_index(self) -> int | None:
        """0-based index among TrialActivity instances, or None if not in a trial."""
        a = self.current_activity
        if not isinstance(a, TrialActivity):
            return None
        return sum(1 for i in range(self._idx)
                   if isinstance(self._activities[i], TrialActivity))

    def step(self) -> dict:
        """Step the current activity. Returns block-level state dict.

        Keys:
          block_done     bool   — True once all activities have finished
          activity_index int    — index of the activity that just stepped
          activity_type  str    — "calibration" | "trial" | "preference" |
                                   "paused" | "done"
          trial_index    int|None — 0-based trial number (None outside a trial)
          data           dict   — the activity's own step() return value
        """
        if self._done:
            return {
                "block_done": True,
                "activity_index": self._idx,
                "activity_type": "done",
                "trial_index": None,
                "data": {},
            }

        current_idx = self._idx

        if self._paused:
            # Cancelled by pause_current_trial(): do not step the trial's
            # clock until resume_current_trial() rebuilds/restarts it.
            return {
                "block_done": False,
                "activity_index": current_idx,
                "activity_type": "paused",
                "trial_index": self._trial_index_at(current_idx),
                "data": {"done": False},
            }

        current = self._activities[current_idx]
        data = current.step()

        if data["done"]:
            # Calibration finished: expand full activity list now that origin is known.
            if current is self._calibration:
                origin = data.get("origin")
                if origin is not None:
                    self._origin = origin
                    trials = self._trial_factory(origin)
                    practice_part = [self._practice] if self._practice is not None else []
                    preference_part = [self._preference] if self._preference is not None else []
                    self._activities = (
                        [self._calibration] + practice_part + trials + preference_part
                    )

            self._idx += 1
            if self._idx >= len(self._activities):
                self._done = True
            else:
                self._activities[self._idx].start()

        return {
            "block_done": self._done,
            "activity_index": current_idx,
            "activity_type": self._activity_type_at(current_idx),
            "trial_index": self._trial_index_at(current_idx),
            "data": data,
        }

    def pause_current_trial(self) -> bool:
        """Cancel the trial currently in progress.

        Its clock/hold progress are discarded — no further steps are taken
        until resume_current_trial() restarts it as a fresh attempt. No-op
        (returns False) unless a TrialActivity is actively running.
        """
        if self._done or self._paused:
            return False
        if not isinstance(self.current_activity, TrialActivity):
            return False
        self._paused = True
        return True

    def resume_current_trial(self) -> None:
        """Restart the paused trial slot as a brand-new attempt.

        Uses rebuild_trial() to draw a fresh target (e.g. a new random target
        for learning_curve) when one was supplied to __init__; otherwise just
        restarts the same TrialActivity from scratch.
        """
        if not self._paused:
            return
        self._paused = False
        if self._rebuild_trial is not None:
            trial_idx = self._trial_index_at(self._idx)
            if trial_idx is not None:
                self._activities[self._idx] = self._rebuild_trial(trial_idx)
        self._activities[self._idx].start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _activity_type_at(self, idx: int) -> str:
        if idx >= len(self._activities):
            return "done"
        a = self._activities[idx]
        if isinstance(a, CalibrationActivity):
            return "calibration"
        if isinstance(a, PracticeActivity):
            return "practice"
        if isinstance(a, TrialActivity):
            return "trial"
        if isinstance(a, PreferenceActivity):
            return "preference"
        return "unknown"

    def _trial_index_at(self, idx: int) -> int | None:
        if idx >= len(self._activities):
            return None
        a = self._activities[idx]
        if not isinstance(a, TrialActivity):
            return None
        return sum(1 for i in range(idx)
                   if isinstance(self._activities[i], TrialActivity))
