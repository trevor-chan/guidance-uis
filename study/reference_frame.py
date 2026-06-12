"""Reference-frame transform: maps world-space poses into the chosen frame.

1D (identity) case: poses pass through unchanged.

Extension points for future steps:
  - from_user_pose(cls, user_pose)     → transforms into user-centred frame
  - from_patient_pose(cls, patient_pose) → transforms into patient-centred frame
  - from_transducer_pose(cls, ...)     → transducer-relative frame
"""

import numpy as np


class ReferenceFrame:
    """Identity frame used for the 1D study. Subclass or extend for non-trivial frames."""

    def transform(self, pose: np.ndarray) -> np.ndarray:
        """Return the pose in this reference frame. Identity: no change."""
        return pose
