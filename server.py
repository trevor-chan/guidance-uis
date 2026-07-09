"""WebSocket bridge: runs Trial at ~30 Hz in competition or study mode."""

import argparse
import asyncio
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
import websockets

from pose_fetcher import LivePoseFetcher, TrackerPoseFetcher, HEAD_OFFSET_M
from trial import Trial, TARGET_POSE, LINEAR_TOLERANCE
from pose_math import angular_distance, component_errors, workspace_component_errors
from core import (CUBE_SIZE, HOLD_DURATION, LINEAR_TOL,
                  BOX_HALF_XY, BOX_Z_MIN, BOX_Z_MAX,
                  _rot_x, _rot_y, _rot_z, _random_target_pose)
from study.sequence import SequenceRunner, SequenceGenerator
from study.archiver import NoOpArchiver
from study.activities import CalibrationActivity, PreferenceActivity, PracticeActivity
from study.storage import (
    ConditionRecorder,
    ExperimentRepository,
    StorageConfig,
    create_data_store,
)

HOST = "localhost"
PORT = 8765
HTTP_PORT = 8000
STEP_INTERVAL = 1 / 30

TRANS_STEP = 0.01             # 1 cm per keypress
ROT_STEP   = math.radians(2)  # 2° per keypress

# (dof 0-2 = x/y/z translation, dof 3-5 = roll/pitch/yaw rotation)
_KEY_MAP = {
    '1': (0, +1), '2': (1, +1), '3': (2, +1),
    '4': (3, +1), '5': (4, +1), '6': (5, +1),
    'q': (0, -1), 'w': (1, -1), 'e': (2, -1),
    'r': (3, -1), 't': (4, -1), 'y': (5, -1),
}

GAME_DURATION      = 180.0   # 3 minutes
N_STUDY_TRIALS     = 7       # competition / legacy headless runner
STUDY_BLOCK_TRIALS = 3       # trials per multi-mode study block

# Captured by set_box command from launcher.html; persists across WS connections.
BOX_ORIGIN: np.ndarray | None = None


def _apply_display_noise(pose: np.ndarray, magnitude: float | None) -> np.ndarray:
    """Return a display-only copy of pose perturbed by fresh Gaussian noise.

    magnitude is the NOISE experiment's ramp level, applied as BOTH a
    position std-dev (mm -> m) and an orientation std-dev (degrees),
    resampled every call. magnitude 0/None returns pose unperturbed — every
    other experiment passes 0, so their displayed pose is never touched.
    """
    if not magnitude:
        return pose
    noisy = pose.copy()
    position_std_m = magnitude / 1000.0
    noisy[:3, 3] = noisy[:3, 3] + np.random.normal(0.0, position_std_m, size=3)
    angle_std_rad = math.radians(magnitude)
    rx, ry, rz = np.random.normal(0.0, angle_std_rad, size=3)
    R_noise = _rot_x(rx) @ _rot_y(ry) @ _rot_z(rz)
    noisy[:3, :3] = R_noise @ pose[:3, :3]
    return noisy


LATENCY_BUFFER_MAXLEN = 60  # 2s of history @ 30Hz — covers the 1125ms max condition with margin


def _apply_display_latency(
    pose: np.ndarray, latency_ms: float | None, buffer: "deque[tuple[float, np.ndarray]]"
) -> np.ndarray:
    """Return a display-only pose delayed by latency_ms via a timestamped buffer.

    Same principle as _apply_display_noise: only the DISPLAYED pose is touched.
    Every call pushes (now, pose) onto buffer, then returns the newest buffered
    pose whose timestamp is >= latency_ms old. latency_ms 0/None returns pose
    immediately, unbuffered — every other experiment passes 0, so their
    displayed pose is never delayed. Early in a block, before the buffer holds
    latency_ms worth of history, returns the oldest pose available rather than
    blocking.
    """
    if not latency_ms:
        return pose
    now = time.monotonic()
    buffer.append((now, pose.copy()))
    cutoff = now - latency_ms / 1000.0
    selected = buffer[0][1]
    for ts, buffered_pose in buffer:
        if ts <= cutoff:
            selected = buffered_pose
        else:
            break
    return selected


class FakePoseFetcher(LivePoseFetcher):
    """Keyboard-driven pose: starts offset from TARGET_POSE so the bars are away from matched.

    Keys 1/2/3 translate along local X/Y/Z; 4/5/6 rotate roll/pitch/yaw.
    q/w/e and r/t/y are the negative counterparts respectively.
    Nudges use the active target frame so each key changes one displayed component.
    """

    source_mode = "fake"
    source_label = "Fake keyboard controls"

    def connect(self):
        self._pose = np.eye(4, dtype=float)
        # Start ~20 cm away in position and ~20° off in orientation.
        # Use all three axes so every rotation control is visibly offset.
        self._pose[:3, 3] = TARGET_POSE[:3, 3] + np.array([0.15, 0.10, -0.08])
        R_offset = _rot_x(math.radians(12)) @ _rot_y(math.radians(12)) @ _rot_z(math.radians(12))
        self._pose[:3, :3] = R_offset @ TARGET_POSE[:3, :3]

    def _raw_pose(self):
        return self._pose.copy()

    def randomize(self, origin: np.ndarray) -> None:
        """Place the fake probe at a random pose inside the calibrated cube.

        origin is a tip pose.  Stores the raw (transducer-center) pose so that
        get_pose() (which adds the head offset) lands within the cube.
        """
        tip_target = _random_target_pose(origin)
        self._pose = tip_target.copy()
        self._pose[:3, 3] -= tip_target[:3, 0] * HEAD_OFFSET_M

    def nudge(self, key: str, reference_pose: np.ndarray | None = None) -> None:
        if key not in _KEY_MAP:
            return
        dof, sign = _KEY_MAP[key]
        reference = TARGET_POSE if reference_pose is None else reference_pose

        if dof < 3:
            # Move in the fixed calibrated frame so one key changes one screen axis.
            self._pose[:3, 3] += reference[:3, :3][:, dof] * sign * TRANS_STEP
        else:
            # Update one displayed Euler component, then reconstruct the live
            # orientation. Post-multiplying a delta at an arbitrary orientation
            # would make multiple displayed components change at once.
            errors = component_errors(self._pose, reference)
            angles = np.radians([
                errors["roll"], errors["pitch"], errors["yaw"]
            ])
            angles[dof - 3] += sign * ROT_STEP
            self._pose[:3, :3] = reference[:3, :3] @ (
                _rot_x(angles[0]) @ _rot_y(angles[1]) @ _rot_z(angles[2])
            )
            print(
                f"[nudge] rot key={key!r}  "
                f"angular={angular_distance(self._pose, reference):.2f}°"
            )

    def disconnect(self):
        pass


# ── Competition mode ───────────────────────────────────────────────────────────

async def _competition_handler(websocket, fetcher, modality="1d", frame="transducer", diagnostic=False):
    trial = Trial(fetcher, linear_tol=LINEAR_TOL)
    trial.start()
    print(f"Client connected ({type(fetcher).__name__}) — competition/{modality} mode.")

    # Competition state — all mutable via commands received in recv_loop
    comp = {
        "calibrated":    False,
        "active":        False,
        "game_over":     False,
        "origin":        None,   # 4x4 ndarray: calibration pose
        "viewpoint_pose": None,  # 4x4 ndarray: locked camera pose (user/patient only)
        "hit_count":     0,
        "hold_start":    None,   # time.monotonic() when current continuous match began
        "start_time":    None,   # time.monotonic() at calibration
    }

    def _calibrate(live_pose: np.ndarray) -> None:
        comp["calibrated"] = True
        comp["origin"]     = live_pose.copy()
        comp["active"]     = True
        comp["game_over"]  = False
        comp["hit_count"]  = 0
        comp["hold_start"] = None
        comp["start_time"] = time.monotonic()
        trial.target_pose  = _random_target_pose(comp["origin"])
        if hasattr(fetcher, "randomize"):
            fetcher.randomize(comp["origin"])
        trial.start()

    def _new_target() -> None:
        if comp["calibrated"] and not comp["game_over"]:
            comp["hold_start"] = None
            trial.target_pose  = _random_target_pose(comp["origin"])
            if hasattr(fetcher, "randomize"):
                fetcher.randomize(comp["origin"])
            trial.start()

    def _reset() -> None:
        comp["calibrated"]    = False
        comp["active"]        = False
        comp["game_over"]     = False
        comp["origin"]        = None
        comp["viewpoint_pose"] = None
        comp["hit_count"]     = 0
        comp["hold_start"]    = None
        comp["start_time"]    = None
        trial.target_pose     = TARGET_POSE
        trial.start()

    # For 2D/3D: track the first valid pose as pre-calibration scene reference.
    scene_origin = fetcher.get_pose() if modality in ("2d", "3d") else None

    async def send_loop():
        nonlocal scene_origin
        _diag_frame = 0
        _DIAG_EVERY = 10  # print ~3 Hz at 30 Hz loop rate

        while True:
            state = trial.step()

            # Competition overlay
            tr = None
            hold_progress = 0.0
            if comp["active"]:
                elapsed_game = time.monotonic() - comp["start_time"]
                tr = max(0.0, GAME_DURATION - elapsed_game)
                if tr <= 0.0:
                    comp["active"]    = False
                    comp["game_over"] = True
                    comp["hold_start"] = None
                elif state["matched"]:
                    if comp["hold_start"] is None:
                        comp["hold_start"] = time.monotonic()
                    hold_dur = time.monotonic() - comp["hold_start"]
                    hold_progress = min(1.0, hold_dur / HOLD_DURATION)
                    if hold_progress >= 1.0:
                        comp["hit_count"] += 1
                        trial.target_pose  = _random_target_pose(comp["origin"])
                        if hasattr(fetcher, "randomize"):
                            fetcher.randomize(comp["origin"])
                        trial.start()
                        comp["hold_start"] = None
                else:
                    comp["hold_start"] = None
            elif comp["game_over"]:
                tr = 0.0

            # Suppress matched when competition is inactive: trial.step() measures
            # against the default TARGET_POSE before calibration (or after reset/
            # game-over), so matched=True there is unrelated to competition scoring
            # and gives a confusing MATCHED badge with no hit registered.
            if not comp["active"]:
                state["matched"] = False

            # trial.py's 90s timed_out is per-target study logic; the competition
            # uses only its own 3-minute clock.  Hide timed_out from the client
            # while a competition is running so the badge never shows "TIMED OUT".
            if comp["active"]:
                state["timed_out"] = False

            state["hold_progress"]       = hold_progress
            state["comp_calibrated"]     = comp["calibrated"]
            state["comp_active"]         = comp["active"]
            state["comp_game_over"]      = comp["game_over"]
            state["comp_hits"]           = comp["hit_count"]
            state["comp_time_remaining"] = tr
            state["mode"]                = "competition"

            if modality in ("1d", "2d", "3d"):
                # trial.step() does not include live_pose; fetch it directly so
                # the 3D renderer has the matrix and tracker_visible is correct.
                live_pose_arr = fetcher.get_pose()
                if live_pose_arr is not None:
                    state["live_pose"] = live_pose_arr.tolist()
                if scene_origin is None and live_pose_arr is not None:
                    scene_origin = live_pose_arr
                reference_pose = comp["origin"] if comp["origin"] is not None else scene_origin
                state["target_pose"]         = trial.target_pose.tolist()
                state["reference_pose"]      = (
                    reference_pose.tolist() if reference_pose is not None else None
                )
                state["source_mode"]         = fetcher.source_mode
                state["source_label"]        = fetcher.source_label
                state["stream_rate"]         = round(1 / STEP_INTERVAL)
                state["tracker_visible"]     = state["linear"] is not None
                state["cube_size"]           = CUBE_SIZE
                state["reference_frame"]     = frame
                state["viewpoint_pose"]      = (
                    comp["viewpoint_pose"].tolist()
                    if comp["viewpoint_pose"] is not None else None
                )
                if comp["origin"] is not None and live_pose_arr is not None:
                    state["live_workspace_components"] = component_errors(
                        live_pose_arr, comp["origin"]
                    )
                    state["target_workspace_components"] = component_errors(
                        trial.target_pose, comp["origin"]
                    )
                    state["workspace_component_errors"] = workspace_component_errors(
                        live_pose_arr, trial.target_pose, comp["origin"]
                    )
                    state["workspace_component_aligned"] = {
                        name: abs(value) <= (
                            LINEAR_TOL if name in ("x", "y", "z")
                            else trial.angular_tol
                        )
                        for name, value
                        in state["workspace_component_errors"].items()
                    }
                else:
                    state["live_workspace_components"] = None
                    state["target_workspace_components"] = None
                    state["workspace_component_errors"] = None
                    state["workspace_component_aligned"] = None

            if diagnostic:
                _diag_frame += 1
                if _diag_frame >= _DIAG_EVERY:
                    _diag_frame = 0
                    diag_pose = fetcher.get_pose()
                    if diag_pose is not None:
                        pos   = diag_pose[:3, 3]
                        x_ax  = diag_pose[:3, 0]
                        y_ax  = diag_pose[:3, 1]
                        z_ax  = diag_pose[:3, 2]
                        print(
                            f"[DIAG] pos:    x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}  (m)\n"
                            f"       X-axis: ({x_ax[0]:+.3f}, {x_ax[1]:+.3f}, {x_ax[2]:+.3f})\n"
                            f"       Y-axis: ({y_ax[0]:+.3f}, {y_ax[1]:+.3f}, {y_ax[2]:+.3f})\n"
                            f"       Z-axis: ({z_ax[0]:+.3f}, {z_ax[1]:+.3f}, {z_ax[2]:+.3f})"
                        )
                    else:
                        _diag_frame = _DIAG_EVERY - 1  # retry next frame

            await websocket.send(json.dumps(state))
            await asyncio.sleep(STEP_INTERVAL)

    async def recv_loop():
        async for raw in websocket:
            try:
                data = json.loads(raw)
                key  = data.get("key")
                cmd  = data.get("cmd")

                if key and hasattr(fetcher, "nudge"):
                    control_reference = (
                        comp["origin"] if comp["origin"] is not None
                        else trial.target_pose
                    )
                    fetcher.nudge(key, control_reference)

                if cmd == "calibrate":
                    live_pose = fetcher.get_pose()
                    if live_pose is not None:
                        _calibrate(live_pose)
                    else:
                        await websocket.send(json.dumps({"error": "tracker_not_visible"}))
                elif cmd == "set_viewpoint":
                    live_pose = fetcher.get_pose()
                    if live_pose is not None:
                        comp["viewpoint_pose"] = live_pose.copy()
                    else:
                        await websocket.send(json.dumps({"error": "tracker_not_visible"}))
                elif cmd == "new_target":
                    _new_target()
                elif cmd == "reset":
                    _reset()

            except (json.JSONDecodeError, AttributeError):
                pass
            except Exception as exc:
                print(f"[STUDY] data command failed: {exc}")
                try:
                    await websocket.send(json.dumps({
                        "error": "data_store_error",
                        "detail": str(exc),
                    }))
                except websockets.exceptions.ConnectionClosed:
                    pass

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosed:
        pass


# ── Multi-mode study handler ───────────────────────────────────────────────────

def _study_blank(activity_type, frame, modality, n_trials, trial_index=None):
    """Minimal study message for phases with no live reticle data."""
    return {
        "mode":                        "study",
        "activity_type":               activity_type,
        "frame":                       frame,
        "modality":                    modality,
        "linear":                      None,
        "angular":                     None,
        "hold_progress":               0.0,
        "matched":                     False,
        "timed_out":                   False,
        "elapsed":                     0.0,
        "trial_index":                 trial_index,
        "n_trials":                    n_trials,
        "target_label":                None,
        "comp_calibrated":             False,
        "live_pose":                   None,
        "target_pose":                 None,
        "reference_pose":              None,
        "reference_frame":             frame,
        "workspace_component_errors":  None,
        "workspace_component_aligned": None,
        "live_workspace_components":   None,
        "target_workspace_components": None,
        "source_mode":                 None,
        "source_label":                None,
        "stream_rate":                 round(1 / STEP_INTERVAL),
        "tracker_visible":             False,
        "cube_size":                   CUBE_SIZE,
        "viewpoint_pose":              None,
    }


async def _new_study_handler(
    websocket,
    fetcher,
    data_store: ExperimentRepository,
    n_trials=STUDY_BLOCK_TRIALS,
):
    """Multi-mode study handler.

    Lifecycle per page visit:
      launcher  → sends set_box → server sets BOX_ORIGIN → acks → launcher shows mode picker
      reticle   → sends start_block (modality, frame) → server builds Block → streams at 30 Hz
    """
    global BOX_ORIGIN

    session = {
        "block":                 None,
        "modality":              "1d",
        "frame":                 "none",
        "n_trials":              n_trials,
        "phase":                 "idle",       # idle|await_calibrate|running|await_preference|complete
        "calibrate_pending":     False,
        "practice_done_pending": False,
        "pause_pending":         False,
        "resume_pending":        False,
        "pending_rating":        None,
        "preference_act":        None,
        "practice_act":          None,
        "trial_index":           None,
        "participant_id":        None,
        "condition_option":      None,
        "condition_index":       None,
        "target_set":            None,
        "noise_magnitude":       0,
        "latency_ms":            0,
        "latency_buffer":        None,
        "modality_id":           None,
        "session_id":            None,
        "recorder":              None,
        "persistent_state":      None,
    }

    async def send_loop():
        while True:
            phase    = session["phase"]
            block    = session["block"]
            modality = session["modality"]
            frame    = session["frame"]
            n        = session["n_trials"]

            if phase == "idle":
                await asyncio.sleep(STEP_INTERVAL)
                continue

            if phase == "await_calibrate":
                if session["calibrate_pending"]:
                    session["calibrate_pending"] = False
                    calibration_data = block.step()
                    origin = calibration_data["data"].get("origin")
                    recorder = session.get("recorder")
                    if recorder and origin is not None:
                        recorder.record_calibration(origin)
                    session["phase"] = "running"
                state = _study_blank("calibration", frame, modality, n)
                live_arr = fetcher.get_pose()
                if live_arr is not None:
                    state["live_pose"]       = live_arr.tolist()
                    state["tracker_visible"] = True
                    state["source_mode"]     = fetcher.source_mode
                    state["source_label"]    = fetcher.source_label
                    if modality in ("2d", "3d"):
                        state["reference_pose"] = live_arr.tolist()
                if modality in ("2d", "3d") and BOX_ORIGIN is not None:
                    state["box_origin"] = BOX_ORIGIN.tolist()
                await websocket.send(json.dumps(state))
                await asyncio.sleep(STEP_INTERVAL)
                continue

            if phase == "running":
                if session["practice_done_pending"] and session["practice_act"] is not None:
                    session["practice_done_pending"] = False
                    session["practice_act"].request_end()

                if session["pause_pending"]:
                    session["pause_pending"] = False
                    if block.pause_current_trial():
                        recorder = session.get("recorder")
                        if recorder:
                            recorder.cancel_current_trial()

                if session["resume_pending"]:
                    session["resume_pending"] = False
                    block.resume_current_trial()

                block_data = block.step()
                act_type   = block_data["activity_type"]
                act_data   = block_data["data"]
                recorder   = session.get("recorder")
                if recorder:
                    recorder.observe_activity(
                        act_type,
                        block_data["trial_index"],
                        act_data,
                    )

                if act_type == "practice" and session["practice_act"] is None:
                    cur = block.current_activity
                    if hasattr(cur, "request_end"):
                        session["practice_act"] = cur

                if act_type in ("trial", "paused"):
                    session["trial_index"] = block_data["trial_index"]

                if isinstance(block.current_activity, PreferenceActivity):
                    session["phase"]          = "await_preference"
                    session["preference_act"] = block.current_activity
                elif block.done:
                    session["phase"] = "complete"
                    if recorder:
                        result = recorder.finish()
                        session["persistent_state"] = result["state"]

                state = _study_blank(act_type, frame, modality, n, session["trial_index"])
                state["linear"]          = act_data.get("linear")
                state["angular"]         = act_data.get("angular")
                state["matched"]         = act_data.get("matched", False)
                state["elapsed"]         = act_data.get("elapsed") or 0.0
                state["hold_progress"]   = act_data.get("hold_progress", 0.0)
                state["timed_out"]       = act_data.get("timed_out", False)
                state["comp_calibrated"] = act_type in ("practice", "trial", "paused")

                if modality in ("1d", "2d", "3d"):
                    live_arr = fetcher.get_pose()
                    display_arr = None
                    if live_arr is not None:
                        display_arr = _apply_display_latency(
                            live_arr, session["latency_ms"], session["latency_buffer"]
                        )
                        display_arr = _apply_display_noise(display_arr, session["noise_magnitude"])
                    origin   = block.origin
                    cur_act  = block.current_activity
                    target_arr = (
                        cur_act.target_pose
                        if hasattr(cur_act, "target_pose")
                        else origin
                    )
                    if act_type in ("trial", "paused"):
                        state["target_label"] = getattr(cur_act, "label", None)
                    state["source_mode"]    = fetcher.source_mode
                    state["source_label"]   = fetcher.source_label
                    state["tracker_visible"] = live_arr is not None
                    state["reference_frame"] = frame
                    if display_arr is not None:
                        state["live_pose"] = display_arr.tolist()
                    if target_arr is not None:
                        state["target_pose"] = target_arr.tolist()
                    if origin is not None:
                        state["reference_pose"] = origin.tolist()
                    if display_arr is not None and origin is not None and target_arr is not None:
                        state["live_workspace_components"]   = component_errors(display_arr, origin)
                        state["target_workspace_components"] = component_errors(target_arr, origin)
                        state["workspace_component_errors"]  = workspace_component_errors(
                            display_arr, target_arr, origin
                        )
                        state["workspace_component_aligned"] = {
                            name: abs(val) <= (
                                LINEAR_TOLERANCE if name in ("x", "y", "z") else 5.0
                            )
                            for name, val in state["workspace_component_errors"].items()
                        }
                    if BOX_ORIGIN is not None:
                        state["box_origin"] = BOX_ORIGIN.tolist()
                    if frame == "user" and block.origin is not None:
                        state["calibration_origin"] = block.origin.tolist()

                await websocket.send(json.dumps(state))
                await asyncio.sleep(STEP_INTERVAL)
                continue

            if phase == "await_preference":
                if session["pending_rating"] is not None:
                    rating = session["pending_rating"]
                    session["pending_rating"] = None
                    if session["preference_act"]:
                        session["preference_act"].set_rating(rating)
                    block.step()              # PreferenceActivity → done → block.done = True
                    recorder = session.get("recorder")
                    if recorder:
                        result = recorder.save_preference_and_finish(rating)
                        session["persistent_state"] = result["state"]
                    session["phase"] = "complete"
                state = _study_blank("preference", frame, modality, n, session["trial_index"])
                await websocket.send(json.dumps(state))
                await asyncio.sleep(STEP_INTERVAL)
                continue

            # complete
            state = _study_blank("complete", frame, modality, n)
            state["persistent_state"] = session.get("persistent_state")
            await websocket.send(json.dumps(state))
            await asyncio.sleep(STEP_INTERVAL)

    async def recv_loop():
        global BOX_ORIGIN
        async for raw in websocket:
            try:
                data = json.loads(raw)
                key  = data.get("key")
                cmd  = data.get("cmd")

                if key and hasattr(fetcher, "nudge"):
                    blk = session.get("block")
                    if blk and blk.current_activity:
                        cur = blk.current_activity
                        ref = (
                            cur.target_pose if hasattr(cur, "target_pose")
                            else blk.origin
                        )
                    else:
                        ref = None
                    fetcher.nudge(key, ref)

                if cmd == "set_box":
                    pose = fetcher.get_pose()
                    if pose is None:
                        await websocket.send(json.dumps({"error": "tracker_not_visible"}))
                        continue
                    BOX_ORIGIN = pose.copy()
                    session_id = data.get("session_id")
                    participant_id = data.get("participant_id")
                    if session_id and participant_id:
                        data_store.save_box_pose(session_id, participant_id, pose)
                    print(
                        f"[STUDY] BOX_ORIGIN set: pos=("
                        f"{pose[0,3]:+.3f}, {pose[1,3]:+.3f}, {pose[2,3]:+.3f}) m"
                    )
                    await websocket.send(json.dumps({"cmd_ack": "set_box", "mode": "study"}))

                elif cmd == "create_session":
                    state = data_store.create_session(data["session"])
                    await websocket.send(json.dumps({
                        "cmd_ack": "create_session",
                        "persistent_state": state,
                    }))

                elif cmd == "get_session_state":
                    state = data_store.get_session_state(
                        data.get("session_id", ""),
                        data.get("participant_id"),
                    )
                    await websocket.send(json.dumps({
                        "cmd_ack": "get_session_state",
                        "persistent_state": state,
                    }))

                elif cmd == "find_latest_session":
                    state = data_store.latest_session_for_participant(
                        data.get("participant_id", "")
                    )
                    await websocket.send(json.dumps({
                        "cmd_ack": "find_latest_session",
                        "persistent_state": state,
                    }))

                elif cmd == "start_block":
                    requested_session_id = data.get("session_id")
                    requested_participant_id = data.get("participant_id")
                    if BOX_ORIGIN is None:
                        saved_pose = None
                        if requested_session_id and requested_participant_id:
                            saved_pose = data_store.get_box_pose(
                                requested_session_id,
                                requested_participant_id,
                            )
                        if saved_pose is not None:
                            BOX_ORIGIN = saved_pose
                            print("[STUDY] BOX_ORIGIN restored from durable session data.")
                        else:
                            pose = fetcher.get_pose()
                            if pose is None:
                                await websocket.send(json.dumps({"error": "box_not_set"}))
                                continue
                            BOX_ORIGIN = pose.copy()
                            if requested_session_id and requested_participant_id:
                                data_store.save_box_pose(
                                    requested_session_id,
                                    requested_participant_id,
                                    pose,
                                )
                            print(
                                f"[STUDY] BOX_ORIGIN auto-set from fetcher: pos=("
                                f"{pose[0,3]:+.3f}, {pose[1,3]:+.3f}, {pose[2,3]:+.3f}) m"
                            )
                    modality_req = data.get("modality", "1d")
                    frame_req    = data.get("frame", "none")
                    session["modality"]              = modality_req
                    session["frame"]                 = frame_req
                    session["trial_index"]           = None
                    session["practice_act"]          = None
                    session["preference_act"]        = None
                    session["calibrate_pending"]     = False
                    session["practice_done_pending"] = False
                    session["pause_pending"]         = False
                    session["resume_pending"]        = False
                    session["pending_rating"]        = None
                    session["participant_id"]        = data.get("participant_id")
                    session["condition_option"]      = data.get("condition_option")
                    session["condition_index"]       = data.get("condition_index")
                    session["target_set"]            = data.get("target_set")
                    session["noise_magnitude"]        = data.get("noise_magnitude") or 0
                    session["latency_ms"]            = data.get("latency_ms") or 0
                    session["latency_buffer"]        = deque(maxlen=LATENCY_BUFFER_MAXLEN)
                    session["modality_id"]           = data.get("modality_id")
                    session["session_id"]            = requested_session_id
                    session["persistent_state"]      = None
                    block_n_trials       = data.get("n_trials") or n_trials
                    include_preference   = data.get("include_preference", True)
                    include_practice     = data.get("include_practice", True)
                    session["n_trials"]  = block_n_trials

                    if not session["session_id"] or not session["participant_id"]:
                        await websocket.send(json.dumps({
                            "error": "missing_session_metadata"
                        }))
                        continue

                    gen   = SequenceGenerator(fetcher)
                    blk   = gen.make_block(
                        modality_req, frame_req, BOX_ORIGIN, block_n_trials,
                        target_set=session["target_set"],
                        include_preference=include_preference,
                        include_practice=include_practice,
                    )
                    blk.start()
                    session["block"] = blk
                    session["recorder"] = ConditionRecorder.start(
                        data_store,
                        session["session_id"],
                        session["participant_id"],
                        int(session["condition_index"]),
                    )
                    needs_cal = isinstance(blk.current_activity, CalibrationActivity)
                    session["phase"] = "await_calibrate" if needs_cal else "running"
                    print(
                        f"[STUDY] start_block: participant={session['participant_id']} "
                        f"option={session['condition_option']} "
                        f"condition={session['condition_index']} "
                        f"target_set={session['target_set']} "
                        f"matrix_modality={session['modality_id']} "
                        f"modality={modality_req} frame={frame_req} "
                        f"{'(calibration)' if needs_cal else '(no calibration)'}"
                    )

                elif cmd == "calibrate":
                    session["calibrate_pending"] = True

                elif cmd == "practice_done":
                    session["practice_done_pending"] = True

                elif cmd == "pause":
                    session["pause_pending"] = True

                elif cmd == "resume":
                    session["resume_pending"] = True

                elif cmd == "rate":
                    rating = data.get("rating")
                    if isinstance(rating, int) and 1 <= rating <= 5:
                        session["pending_rating"] = rating

            except (json.JSONDecodeError, AttributeError):
                pass

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        recorder = session.get("recorder")
        if recorder:
            recorder.flush(session.get("trial_index"))


# ── Set-Box calibration test (throwaway — only active under --setbox flag) ─────
#
# HEAD_OFFSET_M is defined in pose_fetcher.py and applied there; get_pose()
# already returns tip (scanning-head) coordinates.  No local offset needed here.
#
# Axis mapping from transducer frame to box frame (box sits above bottom-face center):
#   transducer X (red,   col 0) → box −Z  (box +Z opposes red; volume +2 → +12 cm in box-Z)
#   transducer Y (green, col 1) → box +X  (lateral,  −12.5 → +12.5 cm)
#   transducer Z (blue,  col 2) → box −Y  (elevation negated to keep frame right-handed)
#
# Basis: bX=+Y_t, bY=−Z_t, bZ=−X_t → det([+Y_t|−Z_t|−X_t])=det([Y_t|Z_t|X_t])=+1 (RH ✓)

# BOX_HALF_XY, BOX_Z_MIN, BOX_Z_MAX imported from core above.


async def _setbox_handler(websocket, fetcher):
    """Minimal set-box test handler: stream live pose, handle set_box command."""
    state = {"box_matrix": None}

    async def send_loop():
        while True:
            live_pose_arr = fetcher.get_pose()
            msg = {
                "mode":         "setbox",
                "source_label": fetcher.source_label,
                "live_pose":    live_pose_arr.tolist() if live_pose_arr is not None else None,
                "box_matrix":   state["box_matrix"],
            }
            await websocket.send(json.dumps(msg))
            await asyncio.sleep(STEP_INTERVAL)

    async def recv_loop():
        async for raw in websocket:
            try:
                data = json.loads(raw)
                if data.get("key") and hasattr(fetcher, "nudge"):
                    fetcher.nudge(data["key"])
                if data.get("cmd") == "set_box":
                    pose = fetcher.get_pose()
                    if pose is None:
                        await websocket.send(json.dumps({"error": "tracker_not_visible"}))
                        continue
                    # get_pose() returns tip coords; pose[:3, 3] IS the scanning-head origin.
                    # Build box world matrix.
                    #   box +X ←  transducer Y (col 1, green)    lateral
                    #   box +Y ← −transducer Z (col 2, blue)     elevation (negated → RH)
                    #   box +Z ← −transducer X (col 0, red)      opposes red; volume +2→+12 cm
                    #   origin  = box centroid = tip + (BOX_Z_MIN+BOX_Z_MAX)/2 along box +Z
                    head_pos = pose[:3, 3]
                    bX =  pose[:3, 1].copy()
                    bY = -pose[:3, 2].copy()
                    bZ = -pose[:3, 0].copy()
                    centroid = head_pos + (BOX_Z_MIN + BOX_Z_MAX) / 2 * bZ
                    box_mat = np.eye(4)
                    box_mat[:3, 0] = bX
                    box_mat[:3, 1] = bY
                    box_mat[:3, 2] = bZ
                    box_mat[:3, 3] = centroid
                    state["box_matrix"] = box_mat.tolist()
                    print(
                        f"[SETBOX] head_pos: ({head_pos[0]:+.3f}, {head_pos[1]:+.3f}, {head_pos[2]:+.3f}) m\n"
                        f"         centroid: ({centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f}) m"
                    )
            except (json.JSONDecodeError, AttributeError):
                pass

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosed:
        pass


# ── Study mode ─────────────────────────────────────────────────────────────────

async def _study_handler(websocket, fetcher, runner):
    """Study-mode WebSocket session. Runner is already started (fetcher connected)."""
    print(f"Client connected ({type(fetcher).__name__}) — study mode.")

    # Phase transitions are owned exclusively by send_loop to avoid races.
    study = {
        "phase":             "await_calibrate",
        "calibrate_pending": False,   # set by recv_loop, cleared by send_loop
        "pending_rating":    None,    # int set by recv_loop, cleared by send_loop
        "preference_act":    None,    # PreferenceActivity ref for set_rating()
        "last_trial_state":  None,    # most recent TrialActivity step dict
        "trial_index":       None,    # current 0-based trial index
    }

    def _blank(activity_type):
        return {
            "mode":          "study",
            "activity_type": activity_type,
            "linear":        None,
            "angular":       None,
            "hold_progress": 0.0,
            "matched":       False,
            "timed_out":     False,
            "elapsed":       0.0,
            "trial_index":   None,
            "trial_count":   N_STUDY_TRIALS,
        }

    async def send_loop():
        while True:
            phase = study["phase"]

            if phase == "await_calibrate":
                if study["calibrate_pending"]:
                    study["calibrate_pending"] = False
                    runner.step()   # CalibrationActivity captures origin → done in one tick
                    study["phase"] = "running"
                    # Edge case: n_trials=0 → preference appears immediately
                    block = runner._blocks[runner._block_idx]
                    if isinstance(block.current_activity, PreferenceActivity):
                        study["phase"]          = "await_preference"
                        study["preference_act"] = block.current_activity
                state = _blank("calibration")

            elif phase == "running":
                runner_data = runner.step()
                block_data  = runner_data["data"]
                act_type    = block_data["activity_type"]
                act_data    = block_data["data"]

                if act_type == "trial":
                    study["trial_index"]      = block_data["trial_index"]
                    study["last_trial_state"] = act_data

                # Peek: did Block just advance to PreferenceActivity?
                block = runner._blocks[runner._block_idx]
                if isinstance(block.current_activity, PreferenceActivity):
                    study["phase"]          = "await_preference"
                    study["preference_act"] = block.current_activity
                elif runner.done:
                    study["phase"] = "complete"

                td = study["last_trial_state"] or {}
                state = {
                    "mode":          "study",
                    "activity_type": "trial",
                    "linear":        td.get("linear"),
                    "angular":       td.get("angular"),
                    "hold_progress": td.get("hold_progress", 0.0),
                    "matched":       td.get("matched", False),
                    "timed_out":     td.get("timed_out", False),
                    "elapsed":       td.get("elapsed") or 0.0,
                    "trial_index":   study["trial_index"],
                    "trial_count":   N_STUDY_TRIALS,
                }

            elif phase == "await_preference":
                if study["pending_rating"] is not None:
                    rating = study["pending_rating"]
                    study["pending_rating"] = None
                    if study["preference_act"]:
                        study["preference_act"].set_rating(rating)
                    runner.step()   # PreferenceActivity → done → runner done
                    study["phase"] = "complete"
                state = _blank("preference")

            else:  # complete
                state = _blank("complete")

            await websocket.send(json.dumps(state))
            await asyncio.sleep(STEP_INTERVAL)

    async def recv_loop():
        async for raw in websocket:
            try:
                data = json.loads(raw)
                key  = data.get("key")
                cmd  = data.get("cmd")

                if key and hasattr(fetcher, "nudge"):
                    fetcher.nudge(key)

                if cmd == "calibrate":
                    live_pose = fetcher.get_pose()
                    if live_pose is None:
                        await websocket.send(json.dumps({"error": "tracker_not_visible"}))
                    else:
                        study["calibrate_pending"] = True
                elif cmd == "rate":
                    rating = data.get("rating")
                    if isinstance(rating, int) and 1 <= rating <= 5:
                        study["pending_rating"] = rating

            except (json.JSONDecodeError, AttributeError):
                pass

    try:
        await asyncio.gather(send_loop(), recv_loop())
    except websockets.exceptions.ConnectionClosed:
        pass


# ── Transport layer ────────────────────────────────────────────────────────────

async def handler(
    websocket,
    fetcher_cls,
    mode,
    modality,
    data_store,
    frame="transducer",
    diagnostic=False,
):
    fetcher = fetcher_cls()

    if mode == "competition":
        try:
            fetcher.connect()
        except Exception as exc:
            print(f"Fetcher connect failed: {exc}")
            await websocket.close(1011, "tracker unavailable")
            return
        try:
            await _competition_handler(websocket, fetcher, modality, frame, diagnostic)
        finally:
            fetcher.disconnect()
            print("Client disconnected.")

    elif mode == "setbox":
        try:
            fetcher.connect()
        except Exception as exc:
            print(f"Fetcher connect failed: {exc}")
            await websocket.close(1011, "tracker unavailable")
            return
        print(f"Client connected ({type(fetcher).__name__}) — setbox mode.")
        try:
            await _setbox_handler(websocket, fetcher)
        finally:
            fetcher.disconnect()
            print("Client disconnected.")

    else:  # study — all modalities
        try:
            fetcher.connect()
        except Exception as exc:
            print(f"Fetcher connect failed: {exc}")
            await websocket.close(1011, "tracker unavailable")
            return
        print(f"Client connected ({type(fetcher).__name__}) — study mode.")
        try:
            await _new_study_handler(websocket, fetcher, data_store)
        finally:
            fetcher.disconnect()
            print("Client disconnected.")


async def main(
    fetcher_cls,
    mode,
    modality,
    frame="transducer",
    diagnostic=False,
    storage_config: StorageConfig | None = None,
):
    root = Path(__file__).resolve().parent
    data_store = create_data_store(
        storage_config or StorageConfig(),
        project_root=root,
    )

    async def _handler(websocket):
        await handler(
            websocket,
            fetcher_cls,
            mode,
            modality,
            data_store,
            frame,
            diagnostic,
        )

    request_handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    http_server = ThreadingHTTPServer((HOST, HTTP_PORT), request_handler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    try:
        async with websockets.serve(_handler, HOST, PORT):
            print(
                f"WebSocket: ws://{HOST}:{PORT}  "
                f"[{fetcher_cls.__name__}, {mode}, {modality}]"
            )
            print(f"1D UI:     http://{HOST}:{HTTP_PORT}/index.html")
            print(f"2D UI:     http://{HOST}:{HTTP_PORT}/index-2d.html")
            print(f"3D UI:     http://{HOST}:{HTTP_PORT}/index-3d.html")
            if mode == "setbox":
                print(f"Setbox UI: http://{HOST}:{HTTP_PORT}/setbox.html")
            if mode == "study":
                print(f"Launcher:  http://{HOST}:{HTTP_PORT}/launcher.html")
                print(
                    f"Data:      {data_store.config.layout} layout at "
                    f"{data_store.config.experiment_root}"
                )
            await asyncio.Future()
    finally:
        http_server.shutdown()
        http_server.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--study",       action="store_true", help="Run in study mode")
    g.add_argument("--competition", action="store_true", help="Run in competition mode")
    g.add_argument("--setbox",      action="store_true",
                   help="[THROWAWAY] Box calibration test: capture transducer pose → render 3D bounding box")
    p.add_argument("--modality", choices=["1d", "2d", "3d"], default="1d",
                   help="Rendering modality: 1d (bar-graph, default), 2d (reticle), or 3d (Three.js)")
    p.add_argument("--fake", action="store_true",
                   help="Use FakePoseFetcher instead of TrackerPoseFetcher (no SteamVR needed)")
    p.add_argument("--frame", choices=["user", "patient", "transducer", "hybrid"], default="transducer",
                   help="3D camera reference frame (competition --modality 3d only): "
                        "transducer=camera rides probe (world rotates), hybrid=probe rotates in place, "
                        "user/patient=locked at calib pose")
    p.add_argument("--diagnostic", action="store_true",
                   help="Print live pose (position + basis axes) to terminal at ~3 Hz for coordinate-system verification")
    p.add_argument("--data-root",
                   default=str(Path.home() / "Documents" / "visualexperiment"),
                   help="Root directory for durable study data "
                        "(default: ~/Documents/visualexperiment)")
    p.add_argument("--data-backend", choices=["sqlite"], default="sqlite",
                   help="Collection backend adapter (default: sqlite)")
    p.add_argument("--data-layout", choices=["session", "participant", "experiment"],
                   default="participant",
                   help="SQLite file layout: one file per session, participant, or experiment")
    p.add_argument("--experiment-id", default="pose-guidance-ui",
                   help="Experiment directory/name used by the data store")
    p.add_argument("--export-format", choices=["csv", "patient_trials", "none"], action="append",
                   help="Export adapter to run after each completed condition (repeatable)")
    args = p.parse_args()

    mode        = "setbox" if args.setbox else ("study" if args.study else "competition")
    modality    = args.modality
    fetcher_cls = FakePoseFetcher if args.fake else TrackerPoseFetcher
    export_formats = tuple(args.export_format or ["csv", "patient_trials"])
    storage_config = StorageConfig(
        root=Path(args.data_root),
        backend=args.data_backend,
        layout=args.data_layout,
        experiment_id=args.experiment_id,
        export_formats=export_formats,
    )

    asyncio.run(main(
        fetcher_cls,
        mode,
        modality,
        args.frame,
        args.diagnostic,
        storage_config,
    ))
