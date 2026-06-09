"""WebSocket bridge: runs a Trial at ~30 Hz and streams state to the browser."""

import argparse
import asyncio
import json
import math
import random
import time
import numpy as np
import websockets

from pose_fetcher import LivePoseFetcher, TrackerPoseFetcher
from trial import Trial, TARGET_POSE
from pose_math import angular_distance

HOST = "localhost"
PORT = 8765
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

GAME_DURATION = 180.0   # 3 minutes
CUBE_SIZE     = 0.5     # metres, side length (cube centred on calibration origin)
HOLD_DURATION = 1.0     # seconds of continuous match required to register a hit


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


_ROT_FNS = [_rot_x, _rot_y, _rot_z]


def _random_target_pose(origin: np.ndarray) -> np.ndarray:
    """Random target within a 0.5 m cube centred on the calibration origin.

    Each axis offset is ±0.25 m (local X/Y/Z), so the calibration pose is the
    centroid and targets spread in all directions equally.  Orientation is
    randomised up to 30° off the calibration orientation on a randomly chosen axis.
    """
    half = CUBE_SIZE / 2
    local_pos = np.array([
        random.uniform(-half, half),   # left / right  (local X)
        random.uniform(-half, half),   # up   / down   (local Y)
        random.uniform(-half, half),   # fore / aft    (local Z)
    ])
    world_pos = origin[:3, 3] + origin[:3, :3] @ local_pos
    target = np.eye(4, dtype=float)
    angle = random.uniform(-math.radians(30), math.radians(30))
    target[:3, :3] = random.choice(_ROT_FNS)(angle) @ origin[:3, :3]
    target[:3, 3]  = world_pos
    return target


class FakePoseFetcher(LivePoseFetcher):
    """Keyboard-driven pose: starts offset from TARGET_POSE so the bars are away from matched.

    Keys 1-6 increase x/y/z/roll/pitch/yaw; q/w/e/r/t/y decrease them.
    Each press nudges by TRANS_STEP (1 cm) or ROT_STEP (2°).
    Rotation is kept valid by composing with an incremental rotation matrix.
    """

    def connect(self):
        self._pose = np.eye(4, dtype=float)
        # Start ~20 cm away in position and ~20° off in orientation.
        # Use all three axes so r/t/y and 4/5/6 all produce visible angular changes.
        self._pose[:3, 3] = TARGET_POSE[:3, 3] + np.array([0.15, 0.10, -0.08])
        R_offset = _rot_x(math.radians(12)) @ _rot_y(math.radians(12)) @ _rot_z(math.radians(12))
        self._pose[:3, :3] = R_offset @ TARGET_POSE[:3, :3]

    def get_pose(self):
        return self._pose.copy()

    def nudge(self, key: str) -> None:
        if key not in _KEY_MAP:
            return
        dof, sign = _KEY_MAP[key]
        if dof < 3:
            self._pose[dof, 3] += sign * TRANS_STEP
        else:
            R_delta = _ROT_FNS[dof - 3](sign * ROT_STEP)
            self._pose[:3, :3] = R_delta @ self._pose[:3, :3]
            print(f"[nudge] rot key={key!r}  angular={angular_distance(self._pose, TARGET_POSE):.2f}°")

    def disconnect(self):
        pass


async def handler(websocket, fetcher_cls):
    fetcher = fetcher_cls()
    try:
        fetcher.connect()
    except Exception as exc:
        print(f"Fetcher connect failed: {exc}")
        await websocket.close(1011, "tracker unavailable")
        return

    trial = Trial(fetcher, linear_tol=0.01)
    trial.start()
    print(f"Client connected ({fetcher_cls.__name__}) — trial started.")

    # Competition state — all mutable via commands received in recv_loop
    comp = {
        "calibrated": False,
        "active":     False,
        "game_over":  False,
        "origin":     None,   # 4x4 ndarray: calibration pose
        "hit_count":  0,
        "hold_start": None,   # time.monotonic() when current continuous match began
        "start_time": None,   # time.monotonic() at calibration
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
        trial.start()

    def _new_target() -> None:
        if comp["calibrated"] and not comp["game_over"]:
            comp["hold_start"] = None
            trial.target_pose  = _random_target_pose(comp["origin"])
            trial.start()

    def _reset() -> None:
        comp["calibrated"] = False
        comp["active"]     = False
        comp["game_over"]  = False
        comp["origin"]     = None
        comp["hit_count"]  = 0
        comp["hold_start"] = None
        comp["start_time"] = None
        trial.target_pose  = TARGET_POSE
        trial.start()

    async def send_loop():
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
                    if live_pose is not None:
                        _calibrate(live_pose)
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
    finally:
        fetcher.disconnect()
        print("Client disconnected.")


async def main(fetcher_cls):
    async def _handler(websocket):
        await handler(websocket, fetcher_cls)

    async with websockets.serve(_handler, HOST, PORT):
        print(f"Listening on ws://{HOST}:{PORT}  [{fetcher_cls.__name__}]")
        await asyncio.Future()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fake", action="store_true",
                   help="Use FakePoseFetcher instead of TrackerPoseFetcher (no SteamVR needed)")
    args = p.parse_args()

    fetcher_cls = FakePoseFetcher if args.fake else TrackerPoseFetcher
    asyncio.run(main(fetcher_cls))
