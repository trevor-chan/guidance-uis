"""Configurable experiment persistence and export adapters.

The study workflow talks only to the ExperimentRepository port. Storage layout
(one database per session, participant, or experiment) and export format are
policies selected at server startup, not decisions embedded in trial logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any, Iterable, Protocol
import uuid

import numpy as np


SCHEMA_VERSION = 1
LAYOUTS = ("session", "participant", "experiment")
CATEGORIES = ("real", "practice", "trash")
DEFAULT_CATEGORY = "practice"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return cleaned.strip("-._") or fallback


def _participant_file_stem(
    participant_id: str,
    direct_identifier: str | None = None,
) -> str:
    participant = _safe_name(participant_id, "participant")
    direct = _safe_name(direct_identifier or "", "")
    return f"{direct}_{participant}" if direct else participant


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(_jsonable(value), separators=(",", ":"), sort_keys=True)


def current_git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class StorageConfig:
    root: Path = field(
        default_factory=lambda: Path.home() / "Documents" / "visualexperiment"
    )
    backend: str = "sqlite"
    layout: str = "participant"
    experiment_id: str = "pose-guidance-ui"
    export_formats: tuple[str, ...] = ("csv", "patient_trials")

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"Unknown data layout {self.layout!r}; choose from {LAYOUTS}.")
        object.__setattr__(self, "root", self.root.expanduser())

    def experiment_root(self, category: str) -> Path:
        return (
            self.root
            / _safe_name(category, DEFAULT_CATEGORY)
            / _safe_name(self.experiment_id, "experiment")
        )

    def legacy_experiment_roots(self, category: str) -> list[Path]:
        """Pre-2026-07-09 layout nested a date folder between category and
        experiment: {root}/{category}/{date}/{experiment}. Glob for those so
        old dated-layout data stays discoverable after the date folder was
        dropped from new writes."""
        category_root = self.root / _safe_name(category, DEFAULT_CATEGORY)
        if not category_root.is_dir():
            return []
        experiment_name = _safe_name(self.experiment_id, "experiment")
        return sorted(
            path for path in category_root.glob(f"*/{experiment_name}")
            if path.is_dir()
        )

    def all_experiment_roots(self, category: str) -> list[Path]:
        """Current experiment root first, then any legacy dated roots."""
        return [self.experiment_root(category), *self.legacy_experiment_roots(category)]

    def _relative_database_path(
        self,
        session_id: str,
        participant_id: str,
        direct_identifier: str | None = None,
    ) -> Path:
        if self.layout == "session":
            return Path("sessions") / _safe_name(session_id, "session") / "experiment.sqlite"
        if self.layout == "participant":
            return (
                Path("participants")
                / _participant_file_stem(participant_id, direct_identifier)
                / "experiment.sqlite"
            )
        return Path("experiment.sqlite")

    def database_path(
        self,
        session_id: str,
        participant_id: str,
        direct_identifier: str | None = None,
        category: str = DEFAULT_CATEGORY,
    ) -> Path:
        return self.experiment_root(category) / self._relative_database_path(
            session_id, participant_id, direct_identifier
        )

    def export_root(self, session_id: str, category: str = DEFAULT_CATEGORY) -> Path:
        return (
            self.experiment_root(category)
            / "exports"
            / _safe_name(session_id, "session")
        )


class DataExporter(ABC):
    """Adapter interface for analysis-oriented exports."""

    name: str

    @abstractmethod
    def export_session(
        self,
        database_path: Path,
        session_id: str,
        output_dir: Path,
    ) -> list[Path]:
        """Export one session and return the files written."""


class CsvExporter(DataExporter):
    name = "csv"
    TABLES = (
        "sessions",
        "conditions",
        "condition_runs",
        "box_poses",
        "calibrations",
        "practice_periods",
        "trials",
        "trajectory_samples",
        "preferences",
        "events",
    )

    def export_session(
        self,
        database_path: Path,
        session_id: str,
        output_dir: Path,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            for table in self.TABLES:
                columns = [
                    row["name"]
                    for row in connection.execute(f"PRAGMA table_info({table})")
                ]
                if not columns:
                    continue
                if "session_id" in columns:
                    rows = connection.execute(
                        f"SELECT * FROM {table} WHERE session_id = ? ORDER BY rowid",
                        (session_id,),
                    ).fetchall()
                elif "run_id" in columns:
                    rows = connection.execute(
                        f"""
                        SELECT table_data.*
                        FROM {table} AS table_data
                        JOIN condition_runs AS runs ON runs.run_id = table_data.run_id
                        WHERE runs.session_id = ?
                        ORDER BY table_data.rowid
                        """,
                        (session_id,),
                    ).fetchall()
                else:
                    continue

                path = output_dir / f"{table}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(dict(row) for row in rows)
                written.append(path)
        return written


class PatientTrialsExporter(DataExporter):
    """Analysis export with one wide row per completed trial, grouped by patient."""

    name = "patient_trials"

    BASE_COLUMNS = (
        "participant_id",
        "session_id",
        "condition_index",
        "run_id",
        "trial_index",
        "completed_date",
    )

    def export_session(
        self,
        database_path: Path,
        session_id: str,
        output_dir: Path,
    ) -> list[Path]:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return []

            participant_id = str(session["participant_id"])
            patient_sessions = connection.execute(
                """
                SELECT * FROM sessions
                WHERE participant_id = ?
                ORDER BY started_at, session_id
                """,
                (participant_id,),
            ).fetchall()
            rows: list[dict[str, Any]] = []
            for patient_session in patient_sessions:
                rows.extend(
                    self._completed_trial_rows(
                        connection,
                        str(patient_session["session_id"]),
                        patient_session,
                    )
                )

        exports_root = output_dir.parent.parent
        path = (
            exports_root
            / "by_patient"
            / _participant_file_stem(participant_id, session["participant_name"])
            / "completed_trials.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._fieldnames(rows)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return [path]

    def _completed_trial_rows(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        session: sqlite3.Row,
    ) -> list[dict[str, Any]]:
        trial_rows = connection.execute(
            """
            SELECT
                trials.*,
                runs.session_id,
                runs.condition_index,
                runs.attempt_number,
                runs.status AS run_status,
                runs.started_at AS run_started_at,
                runs.completed_at AS run_completed_at
            FROM trials
            JOIN condition_runs AS runs ON runs.run_id = trials.run_id
            WHERE runs.session_id = ? AND trials.status = 'complete'
            ORDER BY runs.condition_index, runs.attempt_number, trials.trial_index
            """,
            (session_id,),
        ).fetchall()
        box_pose = connection.execute(
            "SELECT * FROM box_poses WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        rows: list[dict[str, Any]] = []
        for trial in trial_rows:
            condition = connection.execute(
                """
                SELECT * FROM conditions
                WHERE session_id = ? AND condition_index = ?
                """,
                (session_id, trial["condition_index"]),
            ).fetchone()
            practice = connection.execute(
                "SELECT * FROM practice_periods WHERE run_id = ?",
                (trial["run_id"],),
            ).fetchone()
            preference = connection.execute(
                "SELECT * FROM preferences WHERE run_id = ?",
                (trial["run_id"],),
            ).fetchone()
            calibrations = connection.execute(
                "SELECT * FROM calibrations WHERE run_id = ? ORDER BY calibration_id",
                (trial["run_id"],),
            ).fetchall()
            trajectory_samples = connection.execute(
                """
                SELECT * FROM trajectory_samples
                WHERE run_id = ? AND trial_index = ?
                ORDER BY sample_index
                """,
                (trial["run_id"], trial["trial_index"]),
            ).fetchall()
            events = connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ?
                  AND (run_id = ? OR run_id IS NULL)
                  AND (trial_index = ? OR trial_index IS NULL)
                ORDER BY event_id
                """,
                (session_id, trial["run_id"], trial["trial_index"]),
            ).fetchall()

            row: dict[str, Any] = {
                "participant_id": session["participant_id"],
                "session_id": session_id,
                "condition_index": trial["condition_index"],
                "run_id": trial["run_id"],
                "trial_index": trial["trial_index"],
                "completed_date": str(trial["ended_at"] or "")[:10],
                "trajectory_sample_count": len(trajectory_samples),
                "calibration_count": len(calibrations),
                "event_count": len(events),
                "trajectory_samples_json": _json([dict(item) for item in trajectory_samples]),
                "calibrations_json": _json([dict(item) for item in calibrations]),
                "events_json": _json([dict(item) for item in events]),
            }
            row.update(self._prefixed("session", session))
            row.update(self._prefixed("condition", condition))
            row.update(self._prefixed("run", trial, only=(
                "attempt_number",
                "run_status",
                "run_started_at",
                "run_completed_at",
            )))
            row.update(self._prefixed("box_pose", box_pose))
            row.update(self._prefixed("practice", practice))
            row.update(self._prefixed("preference", preference))
            row.update(self._prefixed("trial", trial, skip=(
                "session_id",
                "condition_index",
                "attempt_number",
                "run_status",
                "run_started_at",
                "run_completed_at",
            )))
            rows.append(row)
        return rows

    def _fieldnames(self, rows: list[dict[str, Any]]) -> list[str]:
        ordered = list(self.BASE_COLUMNS)
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        return ordered or list(self.BASE_COLUMNS)

    def _prefixed(
        self,
        prefix: str,
        row: sqlite3.Row | None,
        *,
        only: tuple[str, ...] | None = None,
        skip: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if row is None:
            return {}
        data = dict(row)
        keys = only or tuple(data.keys())
        return {
            f"{prefix}_{key}": data.get(key)
            for key in keys
            if key not in skip
        }


EXPORTERS: dict[str, type[DataExporter]] = {
    CsvExporter.name: CsvExporter,
    PatientTrialsExporter.name: PatientTrialsExporter,
}


class ExperimentRepository(Protocol):
    """Storage port consumed by the experiment workflow."""

    config: StorageConfig

    def create_session(self, payload: dict) -> dict: ...
    def get_session_state(
        self, session_id: str, participant_id: str | None = None
    ) -> dict | None: ...
    def latest_session_for_participant(self, participant_id: str) -> dict | None: ...
    def get_box_pose(
        self, session_id: str, participant_id: str
    ) -> np.ndarray | None: ...
    def save_box_pose(
        self, session_id: str, participant_id: str, pose: np.ndarray
    ) -> None: ...
    def start_condition(
        self, session_id: str, participant_id: str, condition_index: int
    ) -> str: ...
    def save_calibration(self, run_id: str, pose: np.ndarray) -> None: ...
    def start_practice(self, run_id: str) -> None: ...
    def finish_practice(self, run_id: str) -> None: ...
    def start_trial(
        self,
        run_id: str,
        trial_index: int,
        target_pose: Any,
        start_pose: Any,
        trial_meta: dict | None = None,
    ) -> None: ...
    def append_trajectory_samples(
        self, run_id: str, trial_index: int, samples: Iterable[dict]
    ) -> None: ...
    def finish_trial(
        self, run_id: str, trial_index: int, result: dict
    ) -> None: ...
    def discard_trial(self, run_id: str, trial_index: int) -> None: ...
    def save_preference(self, run_id: str, rating: int) -> None: ...
    def finish_condition(self, run_id: str) -> dict: ...
    def export_session(self, session_id: str) -> list[Path]: ...


class SqliteExperimentRepository:
    """Facade used by the server for durable experiment records.

    Every public mutation opens a short SQLite transaction and commits before
    returning. This keeps trial/block boundaries crash-recoverable and avoids
    coupling callers to a particular file layout.
    """

    def __init__(self, config: StorageConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.git_commit = current_git_commit(project_root)
        self._session_locations: dict[str, tuple[str, Path]] = {}
        self._run_locations: dict[str, tuple[Path, str, int]] = {}
        self._exporters = [
            EXPORTERS[name]()
            for name in config.export_formats
            if name != "none" and name in EXPORTERS
        ]

    # -- Session lifecycle -------------------------------------------------

    def create_session(self, payload: dict) -> dict:
        session_id = str(payload["session_id"])
        participant_id = str(payload["participant_id"])
        category = payload.get("data_category")
        if category not in CATEGORIES:
            category = DEFAULT_CATEGORY
        database_path = self.config.database_path(
            session_id,
            participant_id,
            payload.get("participant_name"),
            category=category,
        )
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_locations[session_id] = (participant_id, database_path)
        now = payload.get("started_at") or utc_now()

        with self._connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, experiment_id, participant_id, participant_name,
                    examiner_name, experiment_condition, condition_option,
                    data_category, started_at, status, git_commit, schema_version,
                    storage_layout, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    participant_name = excluded.participant_name,
                    examiner_name = excluded.examiner_name,
                    experiment_condition = excluded.experiment_condition,
                    condition_option = excluded.condition_option,
                    data_category = excluded.data_category,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session_id,
                    self.config.experiment_id,
                    participant_id,
                    payload.get("participant_name"),
                    payload.get("examiner_name"),
                    payload.get("experiment_condition", "modality"),
                    payload.get("condition_option"),
                    category,
                    now,
                    self.git_commit,
                    SCHEMA_VERSION,
                    self.config.layout,
                    _json(payload.get("metadata", {})),
                ),
            )
            for condition in payload.get("conditions", []):
                connection.execute(
                    """
                    INSERT INTO conditions (
                        session_id, condition_index, target_set, modality_id,
                        modality, frame, noise, latency_ms, perceived_ms,
                        learning_curve, precision_linear_mm, precision_angular_deg,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    ON CONFLICT(session_id, condition_index) DO UPDATE SET
                        target_set = excluded.target_set,
                        modality_id = excluded.modality_id,
                        modality = excluded.modality,
                        frame = excluded.frame,
                        noise = excluded.noise,
                        latency_ms = excluded.latency_ms,
                        perceived_ms = excluded.perceived_ms,
                        learning_curve = excluded.learning_curve,
                        precision_linear_mm = excluded.precision_linear_mm,
                        precision_angular_deg = excluded.precision_angular_deg
                    """,
                    (
                        session_id,
                        condition["condition_index"],
                        condition.get("target_set"),
                        condition.get("modality_id"),
                        condition.get("modality"),
                        condition.get("frame"),
                        condition.get("noise"),
                        condition.get("latency_ms"),
                        condition.get("perceived_ms"),
                        condition.get("learning_curve"),
                        condition.get("precision_linear_mm"),
                        condition.get("precision_angular_deg"),
                    ),
                )
            self._event(
                connection,
                session_id,
                "session_created",
                payload={"storage_layout": self.config.layout},
            )
        return self.get_session_state(session_id, participant_id)

    def get_session_state(
        self,
        session_id: str,
        participant_id: str | None = None,
    ) -> dict | None:
        database_path = self._locate_database(session_id, participant_id)
        if database_path is None or not database_path.exists():
            return None
        with self._connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            conditions = connection.execute(
                """
                SELECT conditions.*,
                       (
                           SELECT COUNT(*)
                           FROM trials
                           JOIN condition_runs
                             ON condition_runs.run_id = trials.run_id
                           WHERE condition_runs.run_id = conditions.latest_run_id
                             AND trials.status = 'complete'
                       ) AS completed_trials
                FROM conditions
                WHERE session_id = ?
                ORDER BY condition_index
                """,
                (session_id,),
            ).fetchall()
            box_pose = connection.execute(
                "SELECT pose_json FROM box_poses WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return {
                "session": dict(session),
                "conditions": [dict(row) for row in conditions],
                "has_box_pose": box_pose is not None,
                "database_path": str(database_path),
                "storage_backend": self.config.backend,
                "storage_layout": self.config.layout,
                "export_formats": [exporter.name for exporter in self._exporters],
            }

    def get_box_pose(
        self,
        session_id: str,
        participant_id: str,
    ) -> np.ndarray | None:
        database_path = self._locate_database(session_id, participant_id)
        if database_path is None:
            return None
        with self._connect(database_path) as connection:
            row = connection.execute(
                "SELECT pose_json FROM box_poses WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return np.array(json.loads(row[0]), dtype=float) if row else None

    def latest_session_for_participant(self, participant_id: str) -> dict | None:
        candidates: list[Path] = []
        for category in CATEGORIES:
            for root in self.config.all_experiment_roots(category):
                if self.config.layout == "participant":
                    candidates.extend(root.glob("participants/*/experiment.sqlite"))
                elif self.config.layout == "experiment":
                    path = root / "experiment.sqlite"
                    if path.exists():
                        candidates.append(path)
                else:
                    candidates.extend(root.glob("sessions/*/experiment.sqlite"))

        latest: tuple[str, str, Path] | None = None
        for path in candidates:
            with self._connect(path) as connection:
                row = connection.execute(
                    """
                    SELECT session_id, started_at
                    FROM sessions
                    WHERE participant_id = ?
                    ORDER BY started_at DESC
                    LIMIT 1
                    """,
                    (participant_id,),
                ).fetchone()
                if row and (latest is None or row[1] > latest[1]):
                    latest = (row[0], row[1], path)
        if latest is None:
            return None
        self._session_locations[latest[0]] = (participant_id, latest[2])
        return self.get_session_state(latest[0], participant_id)

    def save_box_pose(
        self,
        session_id: str,
        participant_id: str,
        pose: np.ndarray,
    ) -> None:
        database_path = self._require_database(session_id, participant_id)
        with self._connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO box_poses (session_id, captured_at, pose_json)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    captured_at = excluded.captured_at,
                    pose_json = excluded.pose_json
                """,
                (session_id, utc_now(), _json(pose)),
            )
            self._event(connection, session_id, "box_pose_saved")

    # -- Condition workflow ------------------------------------------------

    def start_condition(
        self,
        session_id: str,
        participant_id: str,
        condition_index: int,
    ) -> str:
        database_path = self._require_database(session_id, participant_id)
        run_id = str(uuid.uuid4())
        with self._connect(database_path) as connection:
            attempt = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0) + 1
                FROM condition_runs
                WHERE session_id = ? AND condition_index = ?
                """,
                (session_id, condition_index),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO condition_runs (
                    run_id, session_id, condition_index, attempt_number,
                    status, started_at
                ) VALUES (?, ?, ?, ?, 'in_progress', ?)
                """,
                (run_id, session_id, condition_index, attempt, utc_now()),
            )
            connection.execute(
                """
                UPDATE conditions
                SET status = 'in_progress', started_at = ?,
                    completed_at = NULL, latest_run_id = ?
                WHERE session_id = ? AND condition_index = ?
                """,
                (utc_now(), run_id, session_id, condition_index),
            )
            self._event(
                connection,
                session_id,
                "condition_started",
                condition_index=condition_index,
                run_id=run_id,
                payload={"attempt_number": attempt},
            )
        self._run_locations[run_id] = (
            database_path,
            session_id,
            condition_index,
        )
        return run_id

    def save_calibration(self, run_id: str, pose: np.ndarray) -> None:
        database_path, session_id, condition_index = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO calibrations (run_id, captured_at, pose_json)
                VALUES (?, ?, ?)
                """,
                (run_id, utc_now(), _json(pose)),
            )
            self._event(
                connection,
                session_id,
                "calibration_saved",
                condition_index=condition_index,
                run_id=run_id,
            )

    def start_practice(self, run_id: str) -> None:
        database_path, session_id, condition_index = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO practice_periods (run_id, started_at, status)
                VALUES (?, ?, 'in_progress')
                ON CONFLICT(run_id) DO NOTHING
                """,
                (run_id, utc_now()),
            )
            self._event(
                connection,
                session_id,
                "practice_started",
                condition_index=condition_index,
                run_id=run_id,
            )

    def finish_practice(self, run_id: str) -> None:
        database_path, session_id, condition_index = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.execute(
                """
                UPDATE practice_periods
                SET ended_at = ?, status = 'complete'
                WHERE run_id = ?
                """,
                (utc_now(), run_id),
            )
            self._event(
                connection,
                session_id,
                "practice_completed",
                condition_index=condition_index,
                run_id=run_id,
            )

    def start_trial(
        self,
        run_id: str,
        trial_index: int,
        target_pose: Any,
        start_pose: Any,
        trial_meta: dict | None = None,
    ) -> None:
        database_path, session_id, condition_index = self._run_location(run_id)
        meta = trial_meta or {}
        with self._connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO trials (
                    run_id, trial_index, status, started_at,
                    start_pose_json, target_pose_json,
                    trial_noise, trial_latency_ms, trial_perceived_ms,
                    trial_precision_linear_mm, trial_precision_angular_deg
                ) VALUES (?, ?, 'in_progress', ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, trial_index) DO NOTHING
                """,
                (
                    run_id,
                    trial_index,
                    utc_now(),
                    _json(start_pose),
                    _json(target_pose),
                    meta.get("noise"),
                    meta.get("latency_ms"),
                    meta.get("perceived_ms"),
                    meta.get("precision_linear_mm"),
                    meta.get("precision_angular_deg"),
                ),
            )
            self._event(
                connection,
                session_id,
                "trial_started",
                condition_index=condition_index,
                trial_index=trial_index,
                run_id=run_id,
            )

    def append_trajectory_samples(
        self,
        run_id: str,
        trial_index: int,
        samples: Iterable[dict],
    ) -> None:
        rows = list(samples)
        if not rows:
            return
        database_path, _, _ = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO trajectory_samples (
                    run_id, trial_index, sample_index, captured_at, elapsed_s,
                    live_pose_json, target_pose_json, linear_m, angular_deg,
                    components_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        trial_index,
                        sample["sample_index"],
                        sample.get("captured_at") or utc_now(),
                        sample.get("elapsed"),
                        _json(sample.get("live_pose")),
                        _json(sample.get("target_pose")),
                        sample.get("linear"),
                        sample.get("angular"),
                        _json(sample.get("components")),
                    )
                    for sample in rows
                ],
            )

    def finish_trial(
        self,
        run_id: str,
        trial_index: int,
        result: dict,
    ) -> None:
        database_path, session_id, condition_index = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.execute(
                """
                UPDATE trials
                SET status = 'complete', ended_at = ?, achieved = ?,
                    timed_out = ?, elapsed_s = ?, end_pose_json = ?,
                    final_linear_m = ?, final_angular_deg = ?
                WHERE run_id = ? AND trial_index = ?
                """,
                (
                    utc_now(),
                    int(bool(result.get("achieved"))),
                    int(bool(result.get("timed_out"))),
                    result.get("elapsed"),
                    _json(result.get("live_pose")),
                    result.get("linear"),
                    result.get("angular"),
                    run_id,
                    trial_index,
                ),
            )
            self._event(
                connection,
                session_id,
                "trial_completed",
                condition_index=condition_index,
                trial_index=trial_index,
                run_id=run_id,
                payload={
                    "achieved": bool(result.get("achieved")),
                    "timed_out": bool(result.get("timed_out")),
                },
            )

    def discard_trial(self, run_id: str, trial_index: int) -> None:
        """Delete a cancelled trial attempt (and its samples) so a later
        restart of the same slot starts from a clean row instead of being
        silently skipped by start_trial's ON CONFLICT DO NOTHING."""
        database_path, session_id, condition_index = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.execute(
                "DELETE FROM trajectory_samples WHERE run_id = ? AND trial_index = ?",
                (run_id, trial_index),
            )
            connection.execute(
                "DELETE FROM trials WHERE run_id = ? AND trial_index = ?",
                (run_id, trial_index),
            )
            self._event(
                connection,
                session_id,
                "trial_cancelled",
                condition_index=condition_index,
                trial_index=trial_index,
                run_id=run_id,
            )

    def save_preference(self, run_id: str, rating: int) -> None:
        database_path, session_id, condition_index = self._run_location(run_id)
        with self._connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO preferences (run_id, recorded_at, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    recorded_at = excluded.recorded_at,
                    rating = excluded.rating
                """,
                (run_id, utc_now(), rating),
            )
            self._event(
                connection,
                session_id,
                "preference_saved",
                condition_index=condition_index,
                run_id=run_id,
                payload={"rating": rating},
            )

    def finish_condition(self, run_id: str) -> dict:
        database_path, session_id, condition_index = self._run_location(run_id)
        now = utc_now()
        with self._connect(database_path) as connection:
            connection.execute(
                """
                UPDATE condition_runs
                SET status = 'complete', completed_at = ?
                WHERE run_id = ?
                """,
                (now, run_id),
            )
            connection.execute(
                """
                UPDATE conditions
                SET status = 'complete', completed_at = ?
                WHERE session_id = ? AND condition_index = ?
                """,
                (now, session_id, condition_index),
            )
            incomplete = connection.execute(
                """
                SELECT COUNT(*) FROM conditions
                WHERE session_id = ? AND status != 'complete'
                """,
                (session_id,),
            ).fetchone()[0]
            if incomplete == 0:
                connection.execute(
                    """
                    UPDATE sessions
                    SET status = 'complete', completed_at = ?
                    WHERE session_id = ?
                    """,
                    (now, session_id),
                )
            self._event(
                connection,
                session_id,
                "condition_completed",
                condition_index=condition_index,
                run_id=run_id,
            )

        export_error = None
        try:
            exported = self.export_session(session_id)
        except Exception as exc:
            exported = []
            export_error = str(exc)
            with self._connect(database_path) as connection:
                self._event(
                    connection,
                    session_id,
                    "export_failed",
                    condition_index=condition_index,
                    run_id=run_id,
                    payload={"error": export_error},
                )
        return {
            "state": self.get_session_state(session_id),
            "exported": [str(path) for path in exported],
            "export_error": export_error,
        }

    # -- Export ------------------------------------------------------------

    def export_session(self, session_id: str) -> list[Path]:
        database_path = self._locate_database(session_id)
        if database_path is None:
            return []
        category = self._category_from_path(database_path)
        written: list[Path] = []
        for exporter in self._exporters:
            written.extend(
                exporter.export_session(
                    database_path,
                    session_id,
                    self.config.export_root(session_id, category) / exporter.name,
                )
            )
        return written

    def _category_from_path(self, database_path: Path) -> str:
        try:
            return database_path.relative_to(self.config.root).parts[0]
        except ValueError:
            return DEFAULT_CATEGORY

    # -- SQLite helpers ----------------------------------------------------

    @contextmanager
    def _connect(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path, timeout=5.0)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            self._ensure_schema(connection)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _locate_database(
        self,
        session_id: str,
        participant_id: str | None = None,
    ) -> Path | None:
        cached = self._session_locations.get(session_id)
        if cached:
            return cached[1]
        if participant_id:
            relative = self.config._relative_database_path(session_id, participant_id)
            for category in CATEGORIES:
                for root in self.config.all_experiment_roots(category):
                    path = root / relative
                    if path.exists():
                        self._session_locations[session_id] = (participant_id, path)
                        return path
        if self.config.layout in ("experiment", "session"):
            relative = self.config._relative_database_path(
                session_id, participant_id or "participant"
            )
            for category in CATEGORIES:
                for root in self.config.all_experiment_roots(category):
                    path = root / relative
                    if path.exists():
                        return path
            return None
        for category in CATEGORIES:
            for root in self.config.all_experiment_roots(category):
                for path in root.glob("participants/*/experiment.sqlite"):
                    with self._connect(path) as connection:
                        row = connection.execute(
                            "SELECT participant_id FROM sessions WHERE session_id = ?",
                            (session_id,),
                        ).fetchone()
                        if row:
                            self._session_locations[session_id] = (row[0], path)
                            return path
        return None

    def _require_database(self, session_id: str, participant_id: str) -> Path:
        path = self._locate_database(session_id, participant_id)
        if path is None:
            raise KeyError(f"Unknown experiment session {session_id!r}.")
        return path

    def _run_location(self, run_id: str) -> tuple[Path, str, int]:
        cached = self._run_locations.get(run_id)
        if cached:
            return cached
        paths: list[Path] = []
        for category in CATEGORIES:
            for root in self.config.all_experiment_roots(category):
                if self.config.layout == "experiment":
                    paths.append(root / "experiment.sqlite")
                elif self.config.layout == "session":
                    paths.extend(root.glob("sessions/*/experiment.sqlite"))
                else:
                    paths.extend(root.glob("participants/*/experiment.sqlite"))
        for database_path in paths:
            if not database_path.exists():
                continue
            with self._connect(database_path) as connection:
                row = connection.execute(
                    """
                    SELECT session_id, condition_index
                    FROM condition_runs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if row:
                    location = (database_path, row[0], row[1])
                    self._run_locations[run_id] = location
                    return location
        raise KeyError(f"Unknown condition run {run_id!r}.")

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        session_id: str,
        event_type: str,
        *,
        condition_index: int | None = None,
        trial_index: int | None = None,
        run_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO events (
                session_id, run_id, condition_index, trial_index,
                event_type, occurred_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                run_id,
                condition_index,
                trial_index,
                event_type,
                utc_now(),
                _json(payload or {}),
            ),
        )

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                participant_name TEXT,
                examiner_name TEXT,
                experiment_condition TEXT NOT NULL,
                condition_option INTEGER,
                data_category TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                git_commit TEXT,
                schema_version INTEGER NOT NULL,
                storage_layout TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conditions (
                session_id TEXT NOT NULL,
                condition_index INTEGER NOT NULL,
                target_set TEXT,
                modality_id TEXT,
                modality TEXT,
                frame TEXT,
                noise REAL,
                latency_ms REAL,
                perceived_ms REAL,
                learning_curve TEXT,
                precision_linear_mm REAL,
                precision_angular_deg REAL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                latest_run_id TEXT,
                PRIMARY KEY (session_id, condition_index),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS condition_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                condition_index INTEGER NOT NULL,
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (session_id, condition_index)
                    REFERENCES conditions(session_id, condition_index)
            );

            CREATE TABLE IF NOT EXISTS box_poses (
                session_id TEXT PRIMARY KEY,
                captured_at TEXT NOT NULL,
                pose_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS calibrations (
                calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                pose_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES condition_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS practice_periods (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES condition_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS trials (
                run_id TEXT NOT NULL,
                trial_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                achieved INTEGER,
                timed_out INTEGER,
                elapsed_s REAL,
                start_pose_json TEXT,
                target_pose_json TEXT,
                end_pose_json TEXT,
                final_linear_m REAL,
                final_angular_deg REAL,
                trial_noise REAL,
                trial_latency_ms REAL,
                trial_perceived_ms REAL,
                trial_precision_linear_mm REAL,
                trial_precision_angular_deg REAL,
                PRIMARY KEY (run_id, trial_index),
                FOREIGN KEY (run_id) REFERENCES condition_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS trajectory_samples (
                run_id TEXT NOT NULL,
                trial_index INTEGER NOT NULL,
                sample_index INTEGER NOT NULL,
                captured_at TEXT NOT NULL,
                elapsed_s REAL,
                live_pose_json TEXT,
                target_pose_json TEXT,
                linear_m REAL,
                angular_deg REAL,
                components_json TEXT,
                PRIMARY KEY (run_id, trial_index, sample_index),
                FOREIGN KEY (run_id, trial_index)
                    REFERENCES trials(run_id, trial_index)
            );

            CREATE TABLE IF NOT EXISTS preferences (
                run_id TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                FOREIGN KEY (run_id) REFERENCES condition_runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                run_id TEXT,
                condition_index INTEGER,
                trial_index INTEGER,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_participant
                ON sessions(participant_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_runs_session_condition
                ON condition_runs(session_id, condition_index, attempt_number);
            CREATE INDEX IF NOT EXISTS idx_events_session
                ON events(session_id, occurred_at);
            """
        )
        # CREATE TABLE IF NOT EXISTS above only shapes brand-new databases;
        # existing .sqlite files predating the precision experiment need these
        # columns added explicitly. Idempotent: skips columns already present.
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conditions)")
        }
        for column in ("precision_linear_mm", "precision_angular_deg"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE conditions ADD COLUMN {column} REAL")

        # Same idempotent pattern for sessions predating the data-category feature.
        existing_session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "data_category" not in existing_session_columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN data_category TEXT")

        # Same idempotent pattern for trials predating the scrambled-block
        # noise/latency/precision ramp, where the magnitude became a per-trial
        # property instead of a per-condition one (see SequenceGenerator
        # .make_block's trial_overrides).
        existing_trial_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(trials)")
        }
        for column in (
            "trial_noise", "trial_latency_ms", "trial_perceived_ms",
            "trial_precision_linear_mm", "trial_precision_angular_deg",
        ):
            if column not in existing_trial_columns:
                connection.execute(f"ALTER TABLE trials ADD COLUMN {column} REAL")


@dataclass
class ConditionRecorder:
    """Small stateful adapter that batches trajectory writes for one block."""

    store: ExperimentRepository
    session_id: str
    participant_id: str
    condition_index: int
    run_id: str
    flush_every: int = 30
    _activity_key: tuple[str, int | None] | None = None
    _sample_index: int = 0
    _buffer: list[dict] = field(default_factory=list)

    @classmethod
    def start(
        cls,
        store: ExperimentRepository,
        session_id: str,
        participant_id: str,
        condition_index: int,
    ) -> "ConditionRecorder":
        run_id = store.start_condition(
            session_id,
            participant_id,
            condition_index,
        )
        return cls(
            store=store,
            session_id=session_id,
            participant_id=participant_id,
            condition_index=condition_index,
            run_id=run_id,
        )

    def record_calibration(self, pose: np.ndarray) -> None:
        self.store.save_calibration(self.run_id, pose)

    def observe_activity(
        self,
        activity_type: str,
        trial_index: int | None,
        data: dict,
    ) -> None:
        key = (activity_type, trial_index)
        if key != self._activity_key:
            self._begin_activity(activity_type, trial_index, data)
            self._activity_key = key

        if activity_type == "trial" and trial_index is not None:
            self._buffer.append(
                {
                    "sample_index": self._sample_index,
                    "captured_at": utc_now(),
                    "elapsed": data.get("elapsed"),
                    "live_pose": data.get("live_pose"),
                    "target_pose": data.get("target_pose"),
                    "linear": data.get("linear"),
                    "angular": data.get("angular"),
                    "components": data.get("components"),
                }
            )
            self._sample_index += 1
            if len(self._buffer) >= self.flush_every:
                self.flush(trial_index)

        if data.get("done"):
            if activity_type == "practice":
                self.store.finish_practice(self.run_id)
            elif activity_type == "trial" and trial_index is not None:
                self.flush(trial_index)
                self.store.finish_trial(self.run_id, trial_index, data)

    def cancel_current_trial(self) -> None:
        """Discard the in-progress trial attempt after a participant pause.

        Deletes any trial/trajectory rows already written for this attempt
        (see ExperimentRepository.discard_trial) and resets tracking so the
        eventual restart is recorded as a clean, brand-new trial.
        """
        if self._activity_key and self._activity_key[0] == "trial":
            trial_index = self._activity_key[1]
            if trial_index is not None:
                self.store.discard_trial(self.run_id, trial_index)
            self._buffer.clear()
            self._sample_index = 0
            self._activity_key = None

    def save_preference_and_finish(self, rating: int) -> dict:
        self.store.save_preference(self.run_id, rating)
        return self.store.finish_condition(self.run_id)

    def finish(self) -> dict:
        """Mark the condition run complete without a preference rating.

        Used by experiments (learning_curve, noise, latency) that omit the
        PreferenceActivity.
        """
        return self.store.finish_condition(self.run_id)

    def flush(self, trial_index: int | None) -> None:
        if trial_index is None or not self._buffer:
            return
        self.store.append_trajectory_samples(
            self.run_id,
            trial_index,
            self._buffer,
        )
        self._buffer.clear()

    def _begin_activity(
        self,
        activity_type: str,
        trial_index: int | None,
        data: dict,
    ) -> None:
        if activity_type == "practice":
            self.store.start_practice(self.run_id)
        elif activity_type == "trial" and trial_index is not None:
            self._sample_index = 0
            self._buffer.clear()
            self.store.start_trial(
                self.run_id,
                trial_index,
                data.get("target_pose"),
                data.get("live_pose"),
                trial_meta={
                    "noise": data.get("noise"),
                    "latency_ms": data.get("latency_ms"),
                    "perceived_ms": data.get("perceived_ms"),
                    "precision_linear_mm": data.get("precision_linear_mm"),
                    "precision_angular_deg": data.get("precision_angular_deg"),
                },
            )


REPOSITORIES: dict[str, type[SqliteExperimentRepository]] = {
    "sqlite": SqliteExperimentRepository,
}


def create_data_store(
    config: StorageConfig,
    project_root: Path,
) -> ExperimentRepository:
    """Build the configured collection backend.

    Adding another collection format requires one repository adapter and one
    registry entry; the server, runner, and dashboard remain unchanged.
    """
    try:
        repository_type = REPOSITORIES[config.backend]
    except KeyError as exc:
        raise ValueError(f"Unknown storage backend {config.backend!r}.") from exc
    return repository_type(config, project_root)
