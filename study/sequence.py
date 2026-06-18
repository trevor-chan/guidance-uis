"""SequenceGenerator and SequenceRunner: top-level study orchestration.

SequenceGenerator builds the list of Block objects for a session.
SequenceRunner owns the fetcher lifecycle, steps through all blocks in order,
and calls DataArchiver after each activity completes.
"""

from typing import Optional
import numpy as np

from pose_fetcher import LivePoseFetcher
from core import random_study_target, LINEAR_TOL, HOLD_DURATION

from .activities import CalibrationActivity, TrialActivity, PreferenceActivity, PracticeActivity
from .block import Block
from .archiver import DataArchiver
from .reference_frame import ReferenceFrame


# ── SequenceGenerator ─────────────────────────────────────────────────────────

class SequenceGenerator:
    """Builds the block list for one study session.

    Targets are generated inside the trial_factory closure so they are only
    created once the calibration origin is known (after Block drives calibration).
    """

    def __init__(
        self,
        fetcher: LivePoseFetcher,
        n_trials: int = 7,
        frame: Optional[ReferenceFrame] = None,
        box_origin: Optional[np.ndarray] = None,
    ) -> None:
        self._fetcher = fetcher
        self._n_trials = n_trials
        self._frame = frame or ReferenceFrame()
        self._box_origin = box_origin

    def make_1d_block(self) -> Block:
        """One block: Calibration → 7 random trials → Preference."""
        calibration = CalibrationActivity(self._fetcher)
        preference = PreferenceActivity()
        n = self._n_trials
        fetcher = self._fetcher
        box_origin = self._box_origin   # always sample from set-box pose, not calibration result

        def trial_factory(_origin: np.ndarray) -> list[TrialActivity]:
            return [
                TrialActivity(fetcher, random_study_target(box_origin))
                for _ in range(n)
            ]

        return Block(calibration, trial_factory, preference)

    def make_block(
        self,
        modality: str,
        frame: str,
        origin: np.ndarray,
        n_trials: int = 3,
    ) -> Block:
        """Build a single study block seeded with a pre-set box origin.

        Calibration is included only for user/patient frames (where the
        camera reference pose must be captured from the live tracker).
        All other modes (1d, transducer) skip calibration and seed trials
        directly from the supplied origin.
        """
        needs_calibration = frame == "user"   # patient/transducer/1d: no calibration step
        calibration = CalibrationActivity(self._fetcher) if needs_calibration else None
        practice    = PracticeActivity(self._fetcher, origin)
        preference  = PreferenceActivity()
        n       = n_trials
        fetcher = self._fetcher

        def trial_factory(_effective_origin: np.ndarray) -> list[TrialActivity]:
            # Always sample from the set-box pose; calibration sets camera only.
            return [
                TrialActivity(fetcher, random_study_target(origin))
                for _ in range(n)
            ]

        return Block(
            calibration,
            trial_factory,
            preference,
            practice=practice,
            fallback_origin=origin,
        )

    def make_blocks(self) -> list[Block]:
        """Block list for the 1D study (single block; extend for multi-condition)."""
        return [self.make_1d_block()]


# ── SequenceRunner ────────────────────────────────────────────────────────────

class SequenceRunner:
    """Runs one participant's study session: owns the fetcher, steps all blocks.

    Usage:
        runner = SequenceRunner(fetcher, n_trials=7, archiver=archiver)
        runner.start()
        while not runner.done:
            data = runner.step()
            # inspect data, update display, etc.
        runner.stop()

    DataArchiver is called automatically after each activity completes.
    """

    def __init__(
        self,
        fetcher: LivePoseFetcher,
        n_trials: int = 7,
        archiver: Optional[DataArchiver] = None,
        frame: Optional[ReferenceFrame] = None,
        box_origin: Optional[np.ndarray] = None,
    ) -> None:
        self._fetcher = fetcher
        self._archiver = archiver
        generator = SequenceGenerator(fetcher, n_trials=n_trials, frame=frame, box_origin=box_origin)
        self._blocks = generator.make_blocks()
        self._block_idx = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def start(self) -> None:
        self._fetcher.connect()
        self._block_idx = 0
        self._done = False
        if self._blocks:
            self._blocks[0].start()

    def step(self) -> dict:
        """Step the current block. Returns runner-level state dict.

        Keys:
          runner_done  bool  — True once every block has finished
          block_index  int   — index of the block that just stepped
          data         dict  — Block.step() return value
        """
        if self._done:
            return {"runner_done": True, "block_index": self._block_idx, "data": {}}

        current_block_idx = self._block_idx
        block = self._blocks[current_block_idx]
        block_data = block.step()

        # Archive when an activity has just finished (data["done"] == True).
        if self._archiver and block_data["data"].get("done"):
            act_type = block_data["activity_type"]
            if act_type == "calibration":
                origin = block_data["data"].get("origin")
                if origin is not None:
                    self._archiver.save_calibration(current_block_idx, origin)
            elif act_type == "trial":
                t_idx = block_data["trial_index"]
                if t_idx is not None:
                    self._archiver.save_trial(current_block_idx, t_idx, block_data["data"])
            elif act_type == "preference":
                self._archiver.save_preference(
                    current_block_idx, block_data["data"].get("rating")
                )

        if block_data["block_done"]:
            self._block_idx += 1
            if self._block_idx >= len(self._blocks):
                self._done = True
                if self._archiver:
                    self._archiver.finalize()
            else:
                self._blocks[self._block_idx].start()

        return {
            "runner_done": self._done,
            "block_index": current_block_idx,
            "data": block_data,
        }

    def stop(self) -> None:
        self._fetcher.disconnect()
