"""Block: runs one study condition as a fixed activity sequence.

Structure: CalibrationActivity → TrialActivity × N → PreferenceActivity.

Trial activities are created lazily after calibration completes, because their
targets depend on the captured origin pose. The caller provides a
trial_factory(origin) → list[TrialActivity] callback; Block expands its
activity list the moment calibration returns done=True.
"""

from typing import Callable
import numpy as np

from .activities import Activity, CalibrationActivity, TrialActivity, PreferenceActivity, PracticeActivity


class Block:
    """Runs one condition: [Calibration?] → [Practice?] → Trial×N → Preference.

    calibration=None skips the calibration step; trials are expanded immediately
    from fallback_origin at start() time.
    practice=None skips the practice phase.
    """

    def __init__(
        self,
        calibration: CalibrationActivity | None,
        trial_factory: Callable[[np.ndarray], list[TrialActivity]],
        preference: PreferenceActivity,
        practice: PracticeActivity | None = None,
        fallback_origin: np.ndarray | None = None,
    ) -> None:
        self._calibration = calibration
        self._practice = practice
        self._trial_factory = trial_factory
        self._preference = preference
        self._fallback_origin = fallback_origin
        self._activities: list[Activity] = []
        self._idx = 0
        self._done = False
        self._origin: np.ndarray | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def origin(self) -> np.ndarray | None:
        """Calibration origin (or fallback_origin when calibration is skipped)."""
        return self._origin

    def start(self) -> None:
        self._idx = 0
        self._done = False
        self._origin = None
        if self._calibration is not None:
            # Lazy expansion: start with calibration only; trials added after origin known.
            self._activities = [self._calibration]
        else:
            # No calibration: expand immediately from fallback_origin.
            origin = self._fallback_origin
            self._origin = origin
            trials = self._trial_factory(origin) if origin is not None else []
            practice_part = [self._practice] if self._practice is not None else []
            self._activities = practice_part + trials + [self._preference]
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
          activity_type  str    — "calibration" | "trial" | "preference" | "done"
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
                    self._activities = (
                        [self._calibration] + practice_part + trials + [self._preference]
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
