"""
Live transducer pose reader.

Connects to a SteamVR generic tracker (e.g. a Vive tracker mounted on the
transducer) and prints the live pose as a 4x4 homogeneous transformation
matrix at a fixed rate.

Requires: SteamVR running, a generic tracker powered on and tracked, and
the `openvr` and `numpy` packages installed.

Usage:
    python live_pose.py              # default 20 Hz
    python live_pose.py --rate 60    # 60 Hz
"""

import argparse
import time

import numpy as np
import openvr


def init_tracker():
    """Connect to SteamVR and return (vr_system, device_index) for the first
    generic tracker found. Raises RuntimeError if none is available."""
    vr_system = openvr.init(openvr.VRApplication_Other)

    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
            serial = vr_system.getStringTrackedDeviceProperty(
                i, openvr.Prop_SerialNumber_String
            )
            print(f"-- Found tracker: device {i} ({serial})")
            return vr_system, i

    openvr.shutdown()
    raise RuntimeError("No generic tracker found. Is it powered on and tracked by SteamVR?")


def get_pose_matrix(vr_system, device_index):
    """Return the tracker pose as a 4x4 homogeneous transformation matrix,
    or None if the pose is not currently valid.

    The OpenVR pose is a 3x4 matrix (3x3 rotation + 3x1 translation). We pad
    it with a [0, 0, 0, 1] bottom row to make it a standard 4x4."""
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
    )
    pose = poses[device_index]
    if not pose.bPoseIsValid:
        return None

    m = pose.mDeviceToAbsoluteTracking  # 3x4, indexable as m[row][col]

    T = np.array([
        [m[0][0], m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1], m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2], m[2][3]],
        [0.0,     0.0,     0.0,     1.0],
    ])
    return T


def main():
    parser = argparse.ArgumentParser(description="Print live transducer pose as a 4x4 matrix.")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="Print rate in Hz (default: 20).")
    args = parser.parse_args()

    interval = 1.0 / args.rate
    vr_system, device_index = init_tracker()

    np.set_printoptions(precision=4, suppress=True)  # readable, no scientific notation
    print(f"-- Streaming at {args.rate} Hz. Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle_start = time.perf_counter()

            T = get_pose_matrix(vr_system, device_index)
            if T is None:
                print("-- pose invalid (tracker not visible?)")
            else:
                print(T)
                print()  # blank line between frames

            elapsed = time.perf_counter() - cycle_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n-- Stopping.")
    finally:
        openvr.shutdown()


if __name__ == "__main__":
    main()
