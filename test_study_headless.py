"""Headless regression test for the study architecture (Step 2).

Verifies:
  SequenceRunner → Block → CalibrationActivity → TrialActivity×7 → PreferenceActivity

Scenarios exercised:
  - Trial 0–5: scripted fetcher returns a pose 100 m away → times out at 1.5 s each
  - Trial 6   : scripted fetcher returns the trial's own target_pose → hold achieved in ~1 s

No websockets, no browser, no keyboard.

Run from the project root:
    python test_study_headless.py
Expected wall-clock time: ~10 s.
"""

import time
import numpy as np

# Monkeypatch trial timeout BEFORE importing any study module so every Trial
# instance in the session uses the patched constant.
import trial as trial_module
_REAL_TIMEOUT = trial_module.TIMEOUT_SECONDS
trial_module.TIMEOUT_SECONDS = 1.5   # 1.5 s per trial instead of 60 s

from pose_fetcher import LivePoseFetcher
from study.activities import CalibrationActivity, TrialActivity, PreferenceActivity
from study.sequence import SequenceRunner
from study.archiver import NoOpArchiver


# ── Scripted fetcher ──────────────────────────────────────────────────────────

class ScriptedFetcher(LivePoseFetcher):
    """Returns whatever pose is set via set_pose(). No hardware required."""

    def __init__(self, initial_pose: np.ndarray | None = None) -> None:
        self._pose = initial_pose

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def set_pose(self, pose: np.ndarray | None) -> None:
        self._pose = pose

    def get_pose(self) -> np.ndarray | None:
        return self._pose.copy() if self._pose is not None else None


# ── Fixed poses ───────────────────────────────────────────────────────────────

ORIGIN_POSE = np.eye(4, dtype=float)
ORIGIN_POSE[:3, 3] = [0.5, 0.5, 0.5]      # arbitrary lab position for calibration

FAR_POSE = np.eye(4, dtype=float)
FAR_POSE[:3, 3] = [100.0, 100.0, 100.0]   # 100 m away — never within 1 cm tolerance


# ── Main test ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Step 2 headless regression test")
    print("=" * 60)
    print(f"  trial.TIMEOUT_SECONDS patched: {_REAL_TIMEOUT} s → {trial_module.TIMEOUT_SECONDS} s")
    print(f"  hold_duration (default):        {1.0} s")
    print(f"  Trials 0–5 will time out; trial 6 will achieve via 1 s hold.")
    print()

    fetcher = ScriptedFetcher(FAR_POSE)
    archiver = NoOpArchiver()
    runner = SequenceRunner(fetcher, n_trials=7, archiver=archiver)
    runner.start()

    # Grab block reference for introspection (current_activity, current_trial_index).
    block = runner._blocks[0]

    ACHIEVE_TRIAL = 6          # 0-indexed; the 7th and final trial
    achieve_target_set = False # latch: print once when we switch to the matching pose

    step_count = 0
    t0 = time.perf_counter()

    last_act_type: str | None = None
    last_trial_idx: int | None = None
    last_hold_milestone = -1.0  # track 0 % / 25 % / 50 % / 75 % / 100 % milestones

    while not runner.done:
        step_count += 1

        # ── Set fetcher pose for this step ────────────────────────────────────
        current = block.current_activity
        if isinstance(current, CalibrationActivity):
            fetcher.set_pose(ORIGIN_POSE)
        elif isinstance(current, TrialActivity):
            trial_idx = block.current_trial_index
            if trial_idx == ACHIEVE_TRIAL:
                # Feed the trial's own target back through the fetcher.
                if not achieve_target_set:
                    tgt = current.target_pose
                    print(f"  Trial {ACHIEVE_TRIAL}: switching to matching pose "
                          f"(pos={tgt[:3, 3].round(4)})")
                    achieve_target_set = True
                fetcher.set_pose(current.target_pose)
            else:
                fetcher.set_pose(FAR_POSE)
        # PreferenceActivity: fetcher pose irrelevant.

        data = runner.step()
        bd = data["data"]        # block-level dict
        ad = bd.get("data", {})  # activity-level dict
        act_type = bd.get("activity_type", "")
        trial_idx = bd.get("trial_index")

        # ── Print on activity transitions ─────────────────────────────────────
        if act_type != last_act_type or trial_idx != last_trial_idx:
            elapsed_total = time.perf_counter() - t0
            if act_type == "calibration":
                print(f"[{elapsed_total:6.2f}s] CALIBRATION started")
            elif act_type == "trial":
                print(f"[{elapsed_total:6.2f}s] TRIAL {trial_idx} started")
                last_hold_milestone = -1.0
            elif act_type == "preference":
                print(f"[{elapsed_total:6.2f}s] PREFERENCE started")
            last_act_type = act_type
            last_trial_idx = trial_idx

        # ── Print hold milestones during the achieve trial ────────────────────
        if act_type == "trial" and trial_idx == ACHIEVE_TRIAL and not ad.get("done"):
            hp = ad.get("hold_progress", 0.0)
            milestone = int(hp * 4) * 0.25   # 0.00, 0.25, 0.50, 0.75
            if milestone > last_hold_milestone:
                elapsed_total = time.perf_counter() - t0
                print(f"  hold_progress = {hp:.2f}  [{elapsed_total:.2f}s]")
                last_hold_milestone = milestone

        # ── Print on activity completion ──────────────────────────────────────
        if ad.get("done"):
            elapsed_total = time.perf_counter() - t0
            if act_type == "calibration":
                origin = ad.get("origin")
                print(f"  → Calibration: origin pos = {origin[:3, 3]}  [{elapsed_total:.2f}s]")
            elif act_type == "trial":
                achieved = ad.get("achieved")
                elapsed = ad.get("elapsed")
                hp = ad.get("hold_progress", 0.0)
                status = "ACHIEVED (hold)" if achieved else "TIMED OUT"
                elapsed_str = f"{elapsed:.3f}s" if elapsed is not None else "—"
                print(f"  → Trial {trial_idx}: {status}  "
                      f"trial_elapsed={elapsed_str}  hold_progress={hp:.2f}  "
                      f"[{elapsed_total:.2f}s total]")
            elif act_type == "preference":
                print(f"  → Preference: rating = {ad.get('rating')}  [{elapsed_total:.2f}s]")

        if data["runner_done"]:
            break

        time.sleep(0.001)   # ~1 kHz step rate (matches real ~30 Hz with margin)

    runner.stop()

    wall = time.perf_counter() - t0
    print()
    print(f"Run complete: {step_count:,} steps  {wall:.2f} s wall clock")
    print()

    # ── Archiver records ──────────────────────────────────────────────────────
    print("Archiver records:")
    for r in archiver.records:
        if r["type"] == "calibration":
            print(f"  calibration   block={r['block']}  "
                  f"origin_pos={r['origin'][:3, 3].round(4)}")
        elif r["type"] == "trial":
            res = r["result"]
            status = "achieved" if res["achieved"] else "timed_out"
            elapsed = res.get("elapsed")
            elapsed_str = f"{elapsed:.3f}s" if elapsed is not None else "—"
            print(f"  trial         block={r['block']}  trial={r['trial']}  "
                  f"status={status}  elapsed={elapsed_str}")
        elif r["type"] == "preference":
            print(f"  preference    block={r['block']}  rating={r['rating']}")
    print()

    # ── Assertions ────────────────────────────────────────────────────────────
    records = archiver.records
    calib_records = [r for r in records if r["type"] == "calibration"]
    trial_records = [r for r in records if r["type"] == "trial"]
    pref_records  = [r for r in records if r["type"] == "preference"]

    assert len(calib_records) == 1, \
        f"Expected 1 calibration record, got {len(calib_records)}"
    assert len(trial_records) == 7, \
        f"Expected 7 trial records, got {len(trial_records)}"
    assert len(pref_records) == 1, \
        f"Expected 1 preference record, got {len(pref_records)}"

    achieved_list   = [r for r in trial_records if r["result"]["achieved"]]
    timed_out_list  = [r for r in trial_records if not r["result"]["achieved"]]
    assert len(achieved_list) == 1, \
        f"Expected exactly 1 achieved trial, got {len(achieved_list)}"
    assert len(timed_out_list) == 6, \
        f"Expected 6 timed-out trials, got {len(timed_out_list)}"
    assert achieved_list[0]["trial"] == ACHIEVE_TRIAL, (
        f"Expected trial {ACHIEVE_TRIAL} to achieve, "
        f"but trial {achieved_list[0]['trial']} achieved"
    )

    # Calibration origin must match what we fed the fetcher.
    np.testing.assert_array_almost_equal(
        calib_records[0]["origin"][:3, 3], ORIGIN_POSE[:3, 3],
        decimal=6, err_msg="Calibration origin does not match ORIGIN_POSE"
    )

    print("All assertions passed.")


if __name__ == "__main__":
    main()
