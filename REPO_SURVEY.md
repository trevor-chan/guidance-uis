# REPO SURVEY — guidance-uis

_Generated: 2026-06-16 by read-only reconnaissance agent._

---

## 1. FILE TREE

```
/Users/elebowit/Desktop/guidance-uis/
├── assets/
│   └── transducer.obj           — 3-D mesh of transducer (centimetres, loaded by index-3d.html)
├── study/
│   ├── __init__.py              — empty package marker
│   ├── activities.py            — CalibrationActivity, TrialActivity, PreferenceActivity ABC
│   ├── archiver.py              — DataArchiver ABC + NoOpArchiver in-memory stub
│   ├── block.py                 — Block: sequences one condition (Calib→Trials→Preference)
│   ├── reference_frame.py       — ReferenceFrame identity stub (extension point for 2D/3D)
│   └── sequence.py              — SequenceGenerator + SequenceRunner top-level orchestration
├── tests/
│   ├── test_calibration.py      — unit tests for tracker→transducer calibration math
│   ├── test_pose_fetchers.py    — unit tests for FakePoseFetcher nudge/randomize/tip-offset
│   └── test_pose_math.py        — unit tests for component_errors, workspace_component_errors
├── vendor/
│   ├── OBJLoader.js             — Three.js OBJ file loader (bundled copy, no npm)
│   ├── OrbitControls.js         — Three.js orbit-camera controls (bundled copy)
│   └── three.module.js          — Three.js r150-ish ES module build (bundled copy)
├── calibration.py               — CalibrationConfig dataclass + compute_transducer_from_tracker()
├── core.py                      — Shared constants (CUBE_SIZE, HOLD_DURATION, LINEAR_TOL) + helpers
├── index.html                   — 1D bar-graph UI: competition + study mode, hold indicator
├── index-2d.html                — 2D reticle UI: SVG cross-hair / depth ruler + axis-stat grid
├── index-3d.html                — 3D Three.js UI: live mesh, ghost target, displacement arcs
├── live_pose.py                 — Standalone SteamVR tracker pose printer (dev/diagnostic tool)
├── pose_fetcher.py              — LivePoseFetcher ABC + TrackerPoseFetcher + FakePoseFetcher stub
├── pose_math.py                 — linear_distance, angular_distance, component_errors, workspace_*
├── requirements.txt             — numpy, openvr, websockets (no pinned versions)
├── server.py                    — WebSocket + HTTP server; competition / study / setbox handlers
├── setbox.html                  — Throwaway set-box test UI (3D Three.js, no mesh, just axes+box)
├── test_study_headless.py       — End-to-end headless test of SequenceRunner (patches 60 s → 1.5 s)
└── trial.py                     — Trial class: one target-match loop, returns step() dicts
```

---

## 2. PER-FILE INTERFACES

### calibration.py
```python
class CalibrationConfig:           # frozen dataclass
    # Fields (all tuples):
    tracker_to_transducer_rotation_euler_zyx_deg
    tracker_to_transducer_translation_m
    tracker_world_basis_diag
    transducer_local_basis_diag
    extrinsic_correction_translation_m
    extrinsic_correction_rotvec_deg
    virtual_source_offset_phased_m
    virtual_source_offset_linear_m
    transducer_center_offset_m
    # Properties:
    def rotation_matrix(self) -> np.ndarray: ...
    def translation(self) -> np.ndarray: ...
    def tracker_world_basis(self) -> np.ndarray: ...
    def transducer_local_basis(self) -> np.ndarray: ...
    def extrinsic_correction_rotation(self) -> np.ndarray: ...
    def extrinsic_correction_translation(self) -> np.ndarray: ...
    def get_virtual_source_offset(self, probe_element: str = "phased") -> np.ndarray: ...
    def transducer_center_offset(self) -> np.ndarray: ...

DEFAULT_CALIBRATION = CalibrationConfig()

def compute_transducer_from_tracker(
    tracker_pose: np.ndarray,
    calibration: CalibrationConfig = DEFAULT_CALIBRATION,
    probe_element: str = "phased",
    center_offset_m: np.ndarray | None = None,
) -> np.ndarray: ...
```
_State: DONE. Full pipeline: tracker world basis → extrinsic correction → virtual source → center offset._

---

### core.py
```python
CUBE_SIZE     = 0.5   # metres
HOLD_DURATION = 1.0   # seconds
LINEAR_TOL    = 0.01  # metres

def _rot_x(a) -> np.ndarray: ...
def _rot_y(a) -> np.ndarray: ...
def _rot_z(a) -> np.ndarray: ...
def _random_target_pose(origin: np.ndarray) -> np.ndarray: ...
```
_State: DONE. Shared constants and random-target generator used by competition and study._

---

### pose_fetcher.py
```python
HEAD_OFFSET_M = 0.07  # metres along local +X

class LivePoseFetcher(ABC):
    source_mode: str
    source_label: str
    def connect(self) -> None: ...
    def _raw_pose(self) -> np.ndarray | None: ...
    def get_pose(self) -> np.ndarray | None: ...   # applies HEAD_OFFSET_M; external callers use this
    def disconnect(self) -> None: ...

class TrackerPoseFetcher(LivePoseFetcher):
    def __init__(self, calibration=DEFAULT_CALIBRATION, probe_element: str = "phased"): ...
    def connect(self) -> None: ...        # inits OpenVR, finds first generic tracker
    def _raw_pose(self) -> np.ndarray | None: ...  # reads SteamVR pose, applies calibration
    def disconnect(self) -> None: ...

class FakePoseFetcher(LivePoseFetcher):
    pass  # stub — actual implementation lives in server.py (subclass)
```
_State: DONE. TrackerPoseFetcher is production-ready; FakePoseFetcher ABC stub here is overridden in server.py._

---

### pose_math.py
```python
def linear_distance(live_pose: np.ndarray, target_pose: np.ndarray) -> float: ...
def angular_distance(live_pose: np.ndarray, target_pose: np.ndarray) -> float: ...
def component_errors(live_pose: np.ndarray, target_pose: np.ndarray) -> dict: ...
    # returns: {"x", "y", "z"} metres, {"roll", "pitch", "yaw"} degrees
def workspace_component_errors(
    live_pose: np.ndarray,
    target_pose: np.ndarray,
    reference_pose: np.ndarray,
) -> dict: ...
    # returns live[name] - target[name] in reference frame; angles wrapped to (-180, 180]
```
_State: DONE. Full 6-DOF error decomposition; all tested._

---

### trial.py
```python
TARGET_POSE = np.ndarray  # shape (4,4), hardcoded default target
LINEAR_TOLERANCE  = 0.005  # metres
ANGULAR_TOLERANCE = 5.0    # degrees
TIMEOUT_SECONDS   = 60.0

class Trial:
    def __init__(self, fetcher: LivePoseFetcher, target_pose: np.ndarray = TARGET_POSE,
                 linear_tol: float = LINEAR_TOLERANCE, angular_tol: float = ANGULAR_TOLERANCE): ...
    def start(self) -> None: ...
    def step(self) -> dict: ...
    # step() returns: linear, angular, matched, timed_out, elapsed, live_pose, components, component_aligned
```
_State: DONE. Core trial loop; target_pose is mutable (server assigns new random targets directly)._

---

### server.py
```python
# Module-level constants
HOST = "localhost"; PORT = 8765; HTTP_PORT = 8000; STEP_INTERVAL = 1/30
TRANS_STEP = 0.01; ROT_STEP = math.radians(2)
GAME_DURATION = 180.0; N_STUDY_TRIALS = 7

class FakePoseFetcher(LivePoseFetcher):
    source_mode = "fake"
    source_label = "Fake keyboard controls"
    def connect(self): ...
    def _raw_pose(self) -> np.ndarray: ...
    def randomize(self, origin: np.ndarray) -> None: ...   # random pose inside calibrated cube
    def nudge(self, key: str, reference_pose: np.ndarray | None = None) -> None: ...
    def disconnect(self): ...

async def _competition_handler(websocket, fetcher, modality="1d", frame="transducer", diagnostic=False): ...
async def _setbox_handler(websocket, fetcher): ...
async def _study_handler(websocket, fetcher, runner): ...
async def handler(websocket, fetcher_cls, mode, modality, frame="transducer", diagnostic=False): ...
async def main(fetcher_cls, mode, modality, frame="transducer", diagnostic=False): ...
```
_State: DONE for competition (1d/2d/3d) and study (1d only). setbox is labelled throwaway. Study 2d/3d rejected at connection._

---

### study/activities.py
```python
class Activity(ABC):
    def start(self) -> None: ...
    def step(self) -> dict: ...       # always includes "done": bool

class CalibrationActivity(Activity):
    def __init__(self, fetcher: LivePoseFetcher) -> None: ...
    def start(self) -> None: ...
    def step(self) -> dict: ...       # returns {"done", "origin"}

class TrialActivity(Activity):
    def __init__(self, fetcher, target_pose, linear_tol=LINEAR_TOL, angular_tol=5.0,
                 hold_duration=HOLD_DURATION) -> None: ...
    def target_pose(self) -> np.ndarray: ...   # property
    def start(self) -> None: ...
    def step(self) -> dict: ...       # returns {"done", "achieved", "hold_progress", ...}

class PreferenceActivity(Activity):
    def __init__(self) -> None: ...
    def start(self) -> None: ...
    def step(self) -> dict: ...       # returns {"done": True, "rating": self._rating}
    def set_rating(self, rating: int) -> None: ...
```
_State: DONE. PreferenceActivity is a documented stub (step() immediately returns done=True; rating injected by set_rating())._

---

### study/block.py
```python
class Block:
    def __init__(self, calibration: CalibrationActivity,
                 trial_factory: Callable[[np.ndarray], list[TrialActivity]],
                 preference: PreferenceActivity) -> None: ...
    def start(self) -> None: ...
    def done(self) -> bool: ...           # property
    def current_activity(self) -> Activity | None: ...   # property
    def current_trial_index(self) -> int | None: ...     # property
    def step(self) -> dict: ...
    # step() returns: {block_done, activity_index, activity_type, trial_index, data}
```
_State: DONE. Lazy trial expansion after calibration completes._

---

### study/sequence.py
```python
class SequenceGenerator:
    def __init__(self, fetcher: LivePoseFetcher, n_trials: int = 7,
                 frame: Optional[ReferenceFrame] = None) -> None: ...
    def make_1d_block(self) -> Block: ...
    def make_blocks(self) -> list[Block]: ...   # currently returns [make_1d_block()]

class SequenceRunner:
    def __init__(self, fetcher, n_trials=7, archiver=None, frame=None) -> None: ...
    def done(self) -> bool: ...   # property
    def start(self) -> None: ...
    def step(self) -> dict: ...
    # step() returns: {runner_done, block_index, data (Block.step() result)}
    def stop(self) -> None: ...
```
_State: DONE. Single-block 1D study only; archiver hooks fire after each activity._

---

### study/archiver.py
```python
class DataArchiver(ABC):
    def save_calibration(self, block_idx: int, origin: np.ndarray) -> None: ...
    def save_trial(self, block_idx: int, trial_idx: int, result: dict) -> None: ...
    def save_preference(self, block_idx: int, rating: int | None) -> None: ...
    def finalize(self) -> None: ...

class NoOpArchiver(DataArchiver):
    records: list[dict]
    def save_calibration(self, ...): ...    # appends dict to self.records
    def save_trial(self, ...): ...
    def save_preference(self, ...): ...
    def finalize(self) -> None: ...         # no-op
```
_State: PARTIAL. Interface complete; only NoOpArchiver exists — no CSV/Parquet on disk. Comment cites Python 3.14 Parquet wheel issue._

---

### study/reference_frame.py
```python
class ReferenceFrame:
    def transform(self, pose: np.ndarray) -> np.ndarray: ...   # identity pass-through
```
_State: PARTIAL stub. Identity only. Docstring names three planned subclasses (user, patient, transducer) — none implemented._

---

### index.html (1D UI)
Top-level JS functions/behaviour:
- `barColor(closeness)` — interpolates cold/hot color
- `updateBar(barId, valueId, value, maxVal, fmt)` — renders one progress bar
- `fmtTime(secs)` — formats M:SS
- `updateHold(progress)` — animates full-screen hold-indicator overlay
- `connect()` — opens WebSocket, dispatches to `renderCompetition(d)` or `renderStudy(d)`
- `renderBars(d)` — updates linear/angular bars + elapsed + MATCHED/RUNNING/TIMED-OUT badge
- `renderCompetition(d)` — scoreboard, GAME OVER, UNCALIBRATED states
- `renderStudy(d)` — phase switcher: calibration / trial (with progress label) / preference (1-5 buttons) / complete

_Renders: 1D bar-graph, competition scoreboard, study phase cards. Handles both `mode=="competition"` and `mode=="study"` from the same WS stream._

---

### index-2d.html (2D UI)
Top-level JS functions/behaviour:
- `clamp(value, min, max)` / `scaled(value, range, pixels)` — SVG mapping helpers
- `fmtTime(seconds)` — M:SS
- `setPoseVisual(elements, pose)` — positions cross, depth marker, orientation circle for one pose
- `updateReticle(components, aligned, livePose, targetPose)` — drives target (black) + current (red) SVG elements + axis-stat tiles
- `send(message)` — WS send wrapper
- `connect()` — opens WS; `onmessage` calls updateReticle + scoreboard + hold bar

_Renders: SVG 6-DOF reticle (cross translates Y/Z, depth tick for X, orientation circle for yaw/pitch, roll via cross rotation), 6-axis stat tiles (green=aligned), competition scoreboard. Competition mode only (no study branch)._

---

### index-3d.html (3D UI)
Top-level JS functions/behaviour (ES module, Three.js):
- `createPoseAxes(opacity)` — builds RGB arrow-helper group for live/ghost
- `createArc()` / `updateArcGeometry(arc, angle, axis)` / `updateArcArrow(cone, angle, axis)` — displacement-vector orange arcs (roll/pitch/yaw)
- `createDropLine(color)` / `updateDropLine(line, group)` — vertical shadow lines to ground
- `centeredModel(object, material)` — centers OBJ mesh and applies visual Z rotation
- `matrixFromRows(rows)` — JSON rows → THREE.Matrix4
- `setRelativePose(group, poseRows, referenceRows)` — sets ghost group matrix
- `fmtTime(seconds)` — M:SS
- `send(message)` / `connect()` — WS layer
- `initLockedFrame()` / `applyLockedViewpoint(refRows)` — user/patient frame camera lock
- `resize()` — responsive canvas + ortho camera update
- `updateDisplacementVectors()` — recomputes roll/pitch/yaw arcs from live vs ghost quaternion
- `animate()` — rAF loop: handles transducer / hybrid / user/patient frame modes, sway, drop lines, render

_Renders: Full 3D scene — live (white) and ghost-target (blue) transducer meshes with RGB axes, orange displacement arcs, ground shadow, bounding cube wireframe. 4 reference frames: transducer (world moves around probe), hybrid (world translates only), user/patient (camera locked). Toggle buttons for mesh/axes/displacement/shadow/sway/projection/z-align._

---

### setbox.html (Set-Box Test UI)
- `matFromRows(rows)` — JSON rows → THREE.Matrix4
- `applyBoxMatrix(mat4)` — positions/sizes yellow wireframe box, re-homes camera
- `send(msg)` / `connect()` — WS layer; keys: `B` = set_box, 1/2/3/4/5/6/q/w/e/r/t/y = nudge
- Inline `animate()` — minimal rAF render loop

_Renders: Throwaway 3D view — live transducer RGB axes + yellow box wireframe + translucent fill. Camera orbits the box centroid once set._

---

### live_pose.py
```python
def init_tracker() -> tuple[vr_system, device_index]: ...
def get_pose_matrix(vr_system, device_index) -> np.ndarray | None: ...
def main(): ...
```
_State: Standalone dev diagnostic — prints raw OpenVR 4×4 matrices at configurable Hz. Not used by server._

---

### tests/test_calibration.py
Tests `compute_transducer_from_tracker`: center-shift direction, extrinsic+virtual+center composition, phased vs linear probe element selectability.

### tests/test_pose_fetchers.py
Tests `TrackerPoseFetcher.get_pose()` tip offset, `FakePoseFetcher.nudge()` translation/rotation axes (123456/qwerty mapping), `FakePoseFetcher.randomize()` stays inside cube.

### tests/test_pose_math.py
Tests `component_errors` translation in target frame, roll/pitch/yaw decomposition, `workspace_component_errors` independent reference frame subtraction.

### test_study_headless.py
End-to-end regression: `SequenceRunner` → 7 trials (6 timeout at 1.5 s, 1 achieved). Verifies archiver records, calibration origin fidelity, PreferenceActivity stub.

---

## 3. DAY-GOAL SUBSYSTEMS

### Mode selector / mode-picker UI
**[MISSING]**
No UI for choosing modality or mode. The user selects by navigating to one of three URLs (`/index.html`, `/index-2d.html`, `/index-3d.html`) and the server is launched with CLI flags (`--competition`/`--study`, `--modality 1d/2d/3d`, `--frame`). There is no in-browser picker or switcher.

---

### Block / activity sequencing layer
**[DONE]**
- `study/sequence.py` `SequenceRunner` → `Block` → `CalibrationActivity → TrialActivity×N → PreferenceActivity`
- `study/block.py:63` `Block.step()` drives transitions; lazy trial expansion at `block.py:87-93` once calibration returns `origin`.
- `study/sequence.py:102` `SequenceRunner.step()` advances blocks, triggers archiver.
- Competition mode uses inline state machine in `server.py:162-325` (no Block abstraction).

---

### Reticle rendering

**1D** — [DONE]  
`index.html` bars card: `linear-bar` and `angular-bar` fill/color via `updateBar()`. Cold→hot interpolation. Hold indicator as full-screen overlay. File: `index.html:343-356`, `413-431`.

**2D** — [DONE]  
`index-2d.html` SVG reticle: target (black) and current (red) cross elements. `setPoseVisual()` at `index-2d.html:361-381` translates Y/Z → screen XY, maps X → depth ruler, yaw/pitch → orientation circle, roll → cross rotation. Requires `workspace_component_errors` and `live/target_workspace_components` from server.

**3D** — [DONE]  
`index-3d.html` Three.js scene: live (white) + ghost (blue translucent) transducer OBJ meshes, RGB axes, orange displacement arcs (`rollArc`, `pitchArc`, `yawArc`), ground drop lines. `updateDisplacementVectors()` at `index-3d.html:906-929`.

---

### Reference-frame transforms

**user frame** — [PARTIAL]  
`server.py:619-631` sends `reference_frame` and `viewpoint_pose` in competition/3D messages. `index-3d.html:957-987` `initLockedFrame()` and `applyLockedViewpoint()` lock camera to calibration pose. Server-side the "user" frame is only a camera mode — pose math still happens in world space. `study/reference_frame.py` identity stub is not yet subclassed.

**patient frame** — [PARTIAL]  
Identical code path as user frame (treated the same in both server and 3D UI). No distinct patient-space transform exists.

**transducer frame** — [DONE]  
`index-3d.html:1005-1033` `animate()` transducer branch: `txWorldGroup.matrix = liveGroup.matrix.invert()` each frame so probe is pinned at scene origin and world moves. Server sends `reference_frame: "transducer"` via CLI `--frame transducer`.

**hybrid frame** — [DONE]  
`index-3d.html:1034-1071` hybrid branch: translates world by `-t` of live pose but keeps rotation. Probe stays at origin with live rotation.

---

### Activities

**Calibration** — [DONE]  
`study/activities.py:38-62` `CalibrationActivity.step()` captures first valid pose as origin. Competition mode does the same inline at `server.py:126-137`.

**Practice** — [MISSING]  
No Practice activity exists. Not referenced anywhere in code.

**Trial** — [DONE]  
`study/activities.py:67-143` `TrialActivity` wraps `Trial` with hold-duration logic. `trial.py` core loop at `trial.py:43-85`. Competition uses `Trial` directly.

**Preference** — [PARTIAL]  
`study/activities.py:148-170` `PreferenceActivity` exists with `set_rating()`. Wired through `_study_handler` at `server.py:480-488` and rendered in `index.html` 1-5 buttons. Works but is documented as a "headless stub" — `step()` returns `done=True` immediately, requires caller to call `set_rating()` before stepping.

---

### Data archiving / recording
**[PARTIAL]**
`study/archiver.py` defines `DataArchiver` ABC with `save_calibration()`, `save_trial()`, `save_preference()`, `finalize()`. Only `NoOpArchiver` (in-memory list, no persistence) exists. Server uses `NoOpArchiver()` at `server.py:565`. Comment in archiver.py cites "Parquet wheel issues on Python 3.14" as reason disk persistence is deferred.

---

## 4. HOW IT RUNS NOW

**Fake / no hardware (laptop):**
```
python server.py --competition --modality 1d --fake
python server.py --competition --modality 2d --fake
python server.py --competition --modality 3d --fake [--frame transducer|hybrid|user|patient]
python server.py --study --fake                      # 1d only; 2d/3d rejected
python server.py --setbox --fake                     # throwaway box test
```
All modes also accept `--diagnostic` to print live pose at ~3 Hz to terminal.

**Real hardware (lab rig, SteamVR running):**
Same commands without `--fake`.

**Entry points:**
- `server.py __main__` → `asyncio.run(main(...))` at `server.py:632`
- HTTP server: `ThreadingHTTPServer` at `server.py:585` serves static files from repo root
- WebSocket server: `websockets.serve` at `server.py:590` on ws://localhost:8765

**Ports:**
- HTTP: http://localhost:8000
- WebSocket: ws://localhost:8765

**What the user sees on load:**
- `index.html` → "TRIAL MONITOR" — competition scoreboard (UNCALIBRATED) + linear/angular bars (both at 0)
- `index-2d.html` → "2D POSE GUIDANCE" — blank SVG reticle + UNCALIBRATED badge, connecting...
- `index-3d.html` → "3D POSE GUIDANCE" — 3D scene loading transducer.obj, connecting...
- `setbox.html` → SET BOX TEST MODE — 3D axes view, "No box set" text

**Tests:**
```
pytest tests/                     # unit tests (~instant)
python test_study_headless.py     # end-to-end regression (~10 s wall clock)
```

---

## 5. DIVERGENCE FROM HANDOFF (old 5-file description)

- **pose_fetcher.py** has grown: `HEAD_OFFSET_M = 0.07` tip-offset now applied in `get_pose()` base-class wrapper (was not present). `TrackerPoseFetcher` now takes `probe_element` param (phased/linear). `FakePoseFetcher` ABC stub here is overridden/extended in server.py with `nudge()`, `randomize()`, `connect()`, `_raw_pose()`.

- **calibration.py** is entirely new vs handoff. In the old description calibration was not a separate module. Now it has a full `CalibrationConfig` dataclass with extrinsic corrections, virtual source offsets, center offsets, and `compute_transducer_from_tracker()`.

- **pose_math.py** has new functions: `component_errors()` (6-DOF decomposition, target-local frame) and `workspace_component_errors()` (reference-frame subtraction). The old description had only `linear_distance` and `angular_distance`.

- **trial.py** is mostly as described but `TARGET_POSE` is now used only as fallback — server sets `trial.target_pose` directly on the instance. `step()` returns 3 additional keys: `live_pose`, `components`, `component_aligned`.

- **server.py** has grown dramatically: was one-handler; now has three handlers (`_competition_handler`, `_study_handler`, `_setbox_handler`), `FakePoseFetcher` subclass, `SequenceRunner` integration, modality (1d/2d/3d) dispatch, reference-frame support (`--frame`), diagnostic flag, and `set_viewpoint` command.

- **index.html** (old 1D UI) now supports both competition and study mode (branching on `d.mode`). Was competition-only in handoff description.

- **Entirely new files vs old 5-file description:** `core.py`, `index-2d.html`, `index-3d.html`, `setbox.html`, `live_pose.py`, `study/` package (5 files), `tests/` (3 test files), `test_study_headless.py`, `vendor/` (3 bundled JS files), `assets/transducer.obj`.

---

## 6. OPEN ITEMS

| Location | Type | Description |
|---|---|---|
| `study/archiver.py:1-5` | stub | `NoOpArchiver` only — no CSV/Parquet persistence. Cited reason: "Parquet wheel issues on Python 3.14" |
| `study/reference_frame.py:1-19` | stub | `ReferenceFrame` is identity only. Docstring lists `from_user_pose`, `from_patient_pose`, `from_transducer_pose` as "extension points for future steps" — none implemented |
| `study/activities.py:148-166` | stub | `PreferenceActivity.step()` returns `done=True` immediately; docstring calls it "Headless stub for Step 2: Step 3 wires real user input" |
| `server.py:529-630` | MISSING | `--study --modality 2d` and `--study --modality 3d` are explicitly rejected with `"is not yet implemented"` |
| `study/sequence.py:56-58` | partial | `SequenceGenerator.make_blocks()` returns only `[make_1d_block()]` — comment says "single block; extend for multi-condition" |
| `server.py:609-612` | throwaway | `--setbox` flag is labeled `[THROWAWAY]` in argparse help text |
| `study/activities.py:148` | MISSING | No `PracticeActivity` exists anywhere; "Practice" is not referenced in any file |

---

## 7. TWO KNOWN BUGS STATUS

### Bug (a): Random targets not updating the angular goal

**FIXED / does not apply in current code.**

In the current code, `trial.target_pose` is a mutable attribute set directly on the `Trial` instance (e.g. `server.py:134 trial.target_pose = _random_target_pose(comp["origin"])`). The `angular_tol` field on `Trial` is never tied to a specific target — it is a fixed tolerance (5.0°) set at `Trial.__init__` time and never changed. `trial.step()` always measures `angular_distance(live_pose, self.target_pose)` against whatever `self.target_pose` currently is. After `trial.target_pose = new_pose` and `trial.start()` are called (always in sequence, e.g. `server.py:134-137`, `142-145`, `187-190`), the very next `trial.step()` call compares against the new target. No stale angular goal is observable.

If the original bug was that `trial.start()` was not being called after setting a new random target (causing the timer to not reset and the target pose not to take effect), that is not present: every `trial.target_pose = ...` line is immediately followed by `trial.start()`.

---

### Bug (b): Target appears hit on both bars but doesn't register (display target vs hit-check target mismatch)

**FIXED / no mismatch observable in current code.**

In competition mode, all of the following use the **same** `trial.target_pose` reference:
1. `state["matched"]` — from `trial.step()` which reads `self.target_pose` (`trial.py:66-67`)
2. `state["target_pose"]` — sent to 3D/2D clients as `trial.target_pose.tolist()` (`server.py:227`)
3. `state["workspace_component_errors"]` and `state["workspace_component_aligned"]` — computed from `trial.target_pose` at `server.py:249,254`

The 2D UI shows axis tiles as green (aligned) based on `workspace_component_aligned`, and the competition badge shows "MATCHED" based on `data.matched`. Both originate from `trial.target_pose` in the same server frame. There is no separate "display target" vs "hit-check target"; they are the same object.

The hold-progress logic in `send_loop()` (`server.py:180-193`) also reads `state["matched"]` from the same `trial.step()` call, so the hold indicator and the actual hit registration are driven by identical data. No mismatch is present.

---
