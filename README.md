# guidance-uis
Guidance User Interface Study

Code for evaluating a range of 3D pose guidance graphical interfaces

## Run the experiment dashboard

Install the Python dependencies, then start one of the two pose modes:

```bash
# Real-time SteamVR tracker (default, 60 Hz)
python server.py --study

# Keyboard-controlled test pose
python server.py --study --fake
```

Open `http://localhost:8000/launcher.html`. The dashboard maps participant IDs
P1-P7 to the seven condition-matrix rows in the experimental protocol, captures
the target workspace, and advances through all seven modality conditions.

The matrix modalities are M1 = 1D; M2-M4 = 2D user/patient/transducer; and
M5-M7 = 3D user/patient/transducer.

### Data storage and export

The workflow records through a repository interface, so collection layout and
analysis export are independent choices. SQLite is the current collection
adapter and CSV is the current export adapter.

```bash
# Default: daily folder in Documents, one SQLite database per participant,
# plus normalized CSV exports and patient-level completed-trial exports
python server.py --study

# One SQLite database containing all sessions for each participant
python server.py --study --data-layout participant

# One master SQLite database for the entire experiment
python server.py --study --data-layout experiment

# Change output location/name or disable automatic CSV export
python server.py --study \
  --data-root ~/Documents/visualexperiment \
  --experiment-id modality-pilot \
  --export-format none
```

Data is grouped by the operator-selected data category (`real`, `practice`, or
`trash`, chosen per session in the launcher) and then by collection date under
`~/Documents/visualexperiment/<category>/YYYY-MM-DD/<experiment-id>/`. All layouts use the
same logical schema. Completed trials are committed immediately, trajectory
samples are flushed in small batches, condition reruns are retained as new
attempts, and the dashboard can restore saved progress. New collection backends
implement `ExperimentRepository`; new analysis formats implement `DataExporter`.

The normalized CSV files stay session-scoped at
`exports/<session-id>/csv/`. The patient analysis export writes one row per
completed trial at `exports/by_patient/<participant-id>/completed_trials.csv`;
it includes prefixed session, condition, run, trial, box-pose, practice,
preference, calibration, trajectory, and event data.

Fake-mode controls:

```text
D/A  X +/-
W/S  Y +/-
Q/E  Z +/-
U/O  roll +/-
I/K  pitch +/-
J/L  yaw +/-
```

For standalone competition testing, run `python server.py --competition --modality 3d`
and open `http://localhost:8000/index-3d.html`. In tracker mode, **Calibrate**
captures the current tracker transform and makes it the center and orientation
of the 50 x 50 x 50 cm workspace. The live transducer then follows the incoming
SteamVR pose, and targets are generated within +/-25 cm on each calibrated axis.

The 2D reticle interface is available at
`http://localhost:8000/index-2d.html`. It renders the current pose in red and
the target in black, with separate visual cues and alignment status for all six
translation and rotation axes.

The real tracker pose is corrected using `calibration.py`. Its fixed transform
maps the Vive tracker to the phased-array imaging plane, applies the coordinate
basis and virtual-source corrections, then shifts another 8 cm along negative
transducer-local X so the rendered pose is centered on the OBJ rather than the
front imaging plane. Fake mode does not apply this hardware calibration.
