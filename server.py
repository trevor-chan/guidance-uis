"""WebSocket bridge: runs Trial at ~30 Hz in competition or study mode."""

import argparse
import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
import websockets

from pose_fetcher import LivePoseFetcher, TrackerPoseFetcher
from trial import Trial, TARGET_POSE
from pose_math import angular_distance, component_errors, workspace_component_errors
from core import (CUBE_SIZE, HOLD_DURATION, LINEAR_TOL,
                  _rot_x, _rot_y, _rot_z, _random_target_pose)
from study.sequence import SequenceRunner
from study.archiver import NoOpArchiver
from study.activities import PreferenceActivity

HOST = "localhost"
PORT = 8765
HTTP_PORT = 8000
STEP_INTERVAL = 1 / 30

TRANS_STEP = 0.01             # 1 cm per keypress
ROT_STEP   = math.radians(2)  # 2° per keypress

# (dof 0-2 = x/y/z translation, dof 3-5 = roll/pitch/yaw rotation)
_KEY_MAP = {
    'd': (0, +1), 'a': (0, -1),
    'w': (1, +1), 's': (1, -1),
    'q': (2, +1), 'e': (2, -1),
    'u': (3, +1), 'o': (3, -1),
    'i': (4, +1), 'k': (4, -1),
    'j': (5, +1), 'l': (5, -1),
}

GAME_DURATION  = 180.0   # 3 minutes
N_STUDY_TRIALS = 7


class FakePoseFetcher(LivePoseFetcher):
    """Keyboard-driven pose: starts offset from TARGET_POSE so the bars are away from matched.

    D/A, W/S, and Q/E translate along local X/Y/Z.
    U/O, I/K, and J/L rotate roll/pitch/yaw.
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

    def get_pose(self):
        return self._pose.copy()

    def randomize(self, origin: np.ndarray) -> None:
        """Place the fake probe at a random pose inside the calibrated cube."""
        self._pose = _random_target_pose(origin)

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

async def _competition_handler(websocket, fetcher, modality="1d", frame="transducer"):
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

            # trial.py's 60s timed_out is per-target study logic; the competition
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

            if modality in ("2d", "3d"):
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

async def handler(websocket, fetcher_cls, mode, modality, frame="transducer"):
    if mode == "study" and modality in ("2d", "3d"):
        msg = f"--study --modality {modality} is not yet implemented"
        print(f"Rejected connection: {msg}.")
        await websocket.close(1011, msg)
        return

    fetcher = fetcher_cls()

    if mode == "competition":
        try:
            fetcher.connect()
        except Exception as exc:
            print(f"Fetcher connect failed: {exc}")
            await websocket.close(1011, "tracker unavailable")
            return
        try:
            await _competition_handler(websocket, fetcher, modality, frame)
        finally:
            fetcher.disconnect()
            print("Client disconnected.")

    else:  # study (1d only at this point)
        runner = SequenceRunner(fetcher, n_trials=N_STUDY_TRIALS, archiver=NoOpArchiver())
        try:
            runner.start()  # connects fetcher + starts first block (CalibrationActivity)
        except Exception as exc:
            print(f"Fetcher connect failed: {exc}")
            await websocket.close(1011, "tracker unavailable")
            return
        try:
            await _study_handler(websocket, fetcher, runner)
        finally:
            runner.stop()  # disconnects fetcher
            print("Client disconnected.")


async def main(fetcher_cls, mode, modality, frame="transducer"):
    async def _handler(websocket):
        await handler(websocket, fetcher_cls, mode, modality, frame)

    root = Path(__file__).resolve().parent
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
            await asyncio.Future()
    finally:
        http_server.shutdown()
        http_server.server_close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--study",       action="store_true", help="Run in study mode")
    g.add_argument("--competition", action="store_true", help="Run in competition mode")
    p.add_argument("--modality", choices=["1d", "2d", "3d"], default="1d",
                   help="Rendering modality: 1d (bar-graph, default), 2d (reticle), or 3d (Three.js)")
    p.add_argument("--fake", action="store_true",
                   help="Use FakePoseFetcher instead of TrackerPoseFetcher (no SteamVR needed)")
    p.add_argument("--frame", choices=["user", "patient", "transducer"], default="transducer",
                   help="3D camera reference frame (competition --modality 3d only; ignored for 2d): "
                        "transducer=camera rides probe, user/patient=locked at calib pose")
    args = p.parse_args()

    mode        = "study" if args.study else "competition"
    modality    = args.modality
    fetcher_cls = FakePoseFetcher if args.fake else TrackerPoseFetcher

    if mode == "study" and modality in ("2d", "3d"):
        p.error(f"--study --modality {modality} is not yet implemented")

    asyncio.run(main(fetcher_cls, mode, modality, args.frame))
