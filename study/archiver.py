"""Data persistence interface and in-memory stub.

Concrete CSV/Parquet implementations are deferred (Parquet wheel issues on
Python 3.14). Wire a real archiver in a later step.
"""

from abc import ABC, abstractmethod
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
