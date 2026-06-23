"""Data persistence interface, in-memory stub, and CSV implementation.

Parquet is intentionally absent — pyarrow wheel issues on Python 3.14.
"""

from abc import ABC, abstractmethod
import csv
import datetime
import json
import os
import subprocess

import numpy as np


class DataArchiver(ABC):
    """Abstract interface for persisting study data."""

    @abstractmethod
    def save_calibration(self, block_idx: int, origin: np.ndarray) -> None:
        """Record the calibration origin captured at the start of a block."""

    @abstractmethod
    def save_trial(self, block_idx: int, trial_idx: int, result: dict) -> None:
        """Record the outcome of one TrialActivity (achieved/timed_out, distances, elapsed)."""

    @abstractmethod
    def save_preference(self, block_idx: int, rating: int | None) -> None:
        """Record the preference rating collected at the end of a block."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush / close any open resources once all blocks have finished."""


class NoOpArchiver(DataArchiver):
    """In-memory stub: appends records to a list, persists nothing to disk."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def save_calibration(self, block_idx: int, origin: np.ndarray) -> None:
        self.records.append({
            "type": "calibration",
            "block": block_idx,
            "origin": origin.copy(),
        })

    def save_trial(self, block_idx: int, trial_idx: int, result: dict) -> None:
        self.records.append({
            "type": "trial",
            "block": block_idx,
            "trial": trial_idx,
            "result": result,
        })

    def save_preference(self, block_idx: int, rating: int | None) -> None:
        self.records.append({
            "type": "preference",
            "block": block_idx,
            "rating": rating,
        })

    def finalize(self) -> None:
        pass


class CsvArchiver(DataArchiver):
    """Writes trial results, trajectories, preferences, and run metadata to disk.

    Directory layout:
        <output_root>/<participant_id>_<timestamp>/
            trials.csv
            preferences.csv
            trajectory_b{block}_t{trial}.csv   (one per completed trial)
            metadata.json
    """

    _TRIAL_COLS = [
        "participant_id", "block_index", "modality", "frame", "trial_index",
        *[f"target_pose_{i}" for i in range(16)],
        "start_time", "end_time", "elapsed_s",
        "achieved", "matched", "timed_out",
        "final_linear_error_m", "final_angular_error_deg",
        "preference_rating",  # blank here; see preferences.csv for actual values
    ]
    _TRAJ_COLS = [
        "timestamp",
        *[f"live_pose_{i}" for i in range(16)],
        "linear_m", "angular_deg",
    ]
    _PREF_COLS = ["block_index", "rating"]

    def __init__(self, output_root: str, participant_id: str) -> None:
        self._pid = participant_id
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(output_root, f"{participant_id}_{ts}")
        os.makedirs(self._run_dir, exist_ok=True)

        self._start_ts = datetime.datetime.now().isoformat(timespec="seconds")
        self._box_origin: list | None = None
        self._block_meta: dict[int, dict] = {}  # block_idx → {modality, frame, calibration_origin}

        self._trials_f = open(
            os.path.join(self._run_dir, "trials.csv"), "w", newline=""
        )
        self._trials_w = csv.DictWriter(self._trials_f, fieldnames=self._TRIAL_COLS)
        self._trials_w.writeheader()
        self._trials_f.flush()

        self._pref_f = open(
            os.path.join(self._run_dir, "preferences.csv"), "w", newline=""
        )
        self._pref_w = csv.DictWriter(self._pref_f, fieldnames=self._PREF_COLS)
        self._pref_w.writeheader()
        self._pref_f.flush()

        print(f"[ARCHIVER] Run directory: {self._run_dir}")

    # ── Context setters called by the handler ─────────────────────────────────

    def set_block_context(self, block_idx: int, modality: str, frame: str) -> None:
        """Register modality/frame for this block; must be called before any saves."""
        self._block_meta[block_idx] = {
            "modality": modality,
            "frame": frame,
            "calibration_origin": None,
        }

    def set_box_origin(self, origin: np.ndarray) -> None:
        self._box_origin = origin.flatten().tolist()

    # ── DataArchiver interface ─────────────────────────────────────────────────

    def save_calibration(self, block_idx: int, origin: np.ndarray) -> None:
        if block_idx in self._block_meta:
            self._block_meta[block_idx]["calibration_origin"] = origin.flatten().tolist()

    def save_trial(
        self,
        block_idx: int,
        trial_idx: int,
        result: dict,
        *,
        trajectory: list[dict] | None = None,
        start_time: str = "",
        end_time: str = "",
    ) -> None:
        """Append one row to trials.csv (flushed immediately) and write trajectory CSV."""
        ctx = self._block_meta.get(block_idx, {})

        raw_tp = result.get("target_pose")
        tp_flat = np.asarray(raw_tp).flatten().tolist() if raw_tp is not None else [None] * 16

        row: dict = {
            "participant_id":          self._pid,
            "block_index":             block_idx,
            "modality":                ctx.get("modality", ""),
            "frame":                   ctx.get("frame", ""),
            "trial_index":             trial_idx,
            **{
                f"target_pose_{i}": (f"{v:.6f}" if v is not None else "")
                for i, v in enumerate(tp_flat)
            },
            "start_time":              start_time,
            "end_time":                end_time,
            "elapsed_s":               result["elapsed"] if result.get("elapsed") is not None else "",
            "achieved":                result.get("achieved", False),
            "matched":                 result.get("matched", False),
            "timed_out":               result.get("timed_out", False),
            "final_linear_error_m":    result["linear"]  if result.get("linear")  is not None else "",
            "final_angular_error_deg": result["angular"] if result.get("angular") is not None else "",
            "preference_rating":       "",
        }
        self._trials_w.writerow(row)
        self._trials_f.flush()

        if trajectory:
            traj_path = os.path.join(
                self._run_dir, f"trajectory_b{block_idx}_t{trial_idx}.csv"
            )
            with open(traj_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._TRAJ_COLS)
                w.writeheader()
                w.writerows(trajectory)

    def save_preference(self, block_idx: int, rating: int | None) -> None:
        self._pref_w.writerow({"block_index": block_idx, "rating": rating})
        self._pref_f.flush()

    def finalize(self) -> None:
        """Write metadata.json and close open file handles."""
        git_hash = ""
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            pass

        metadata = {
            "participant_id":  self._pid,
            "start_timestamp": self._start_ts,
            "end_timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit":      git_hash,
            "box_origin":      self._box_origin,
            "blocks": [
                {
                    "block_index":        idx,
                    "modality":           ctx.get("modality"),
                    "frame":              ctx.get("frame"),
                    "calibration_origin": ctx.get("calibration_origin"),
                }
                for idx, ctx in sorted(self._block_meta.items())
            ],
        }
        with open(os.path.join(self._run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self._trials_f.close()
        self._pref_f.close()
        print(f"[ARCHIVER] Finalized: {self._run_dir}")
