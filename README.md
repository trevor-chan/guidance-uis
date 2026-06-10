# guidance-uis
Guidance User Interface Study

Code for evaluating a range of 3D pose guidance graphical interfaces

## Run the 3D interface

Install the Python dependencies, then start one of the two pose modes:

```bash
# Real-time SteamVR tracker (default, 60 Hz)
python server.py

# Keyboard-controlled test pose
python server.py --fake
```

Open `http://localhost:8000/index-3d.html`. In tracker mode, **Calibrate**
captures the current tracker transform and makes it the center and orientation
of the 50 x 50 x 50 cm workspace. The live transducer then follows the incoming
SteamVR pose, and targets are generated within +/-25 cm on each calibrated axis.
