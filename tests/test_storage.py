import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from study.storage import ConditionRecorder, StorageConfig, create_data_store


def session_payload(session_id: str, participant_id: str = "P1") -> dict:
    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "participant_name": "Test Participant",
        "examiner_name": "Test Examiner",
        "experiment_condition": "modality",
        "condition_option": 1,
        "conditions": [
            {
                "condition_index": 1,
                "target_set": "S1",
                "modality_id": "M1",
                "modality": "1d",
                "frame": "none",
            },
            {
                "condition_index": 2,
                "target_set": "S2",
                "modality_id": "M2",
                "modality": "2d",
                "frame": "user",
            },
        ],
    }


class StorageLayoutTests(unittest.TestCase):
    def test_database_path_policy_is_swappable(self):
        root = Path("/tmp/data-root")
        session_config = StorageConfig(
            root=root,
            layout="session",
            experiment_id="study",
        )
        participant_config = StorageConfig(
            root=root,
            layout="participant",
            experiment_id="study",
        )
        experiment_config = StorageConfig(
            root=root,
            layout="experiment",
            experiment_id="study",
        )

        self.assertEqual(
            session_config.database_path("abc", "P1"),
            root / "practice" / "study" / "sessions" / "abc" / "experiment.sqlite",
        )
        self.assertEqual(
            participant_config.database_path("abc", "P1"),
            root / "practice" / "study" / "participants" / "P1" / "experiment.sqlite",
        )
        self.assertEqual(
            participant_config.database_path("abc", "P1", "Direct Identifier"),
            root
            / "practice"
            / "study"
            / "participants"
            / "Direct-Identifier_P1"
            / "experiment.sqlite",
        )
        self.assertEqual(
            experiment_config.database_path("abc", "P1"),
            root / "practice" / "study" / "experiment.sqlite",
        )
        self.assertEqual(
            participant_config.database_path("abc", "P1", category="real"),
            root / "real" / "study" / "participants" / "P1" / "experiment.sqlite",
        )

    def test_all_layouts_store_the_same_logical_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for layout in ("session", "participant", "experiment"):
                with self.subTest(layout=layout):
                    config = StorageConfig(
                        root=Path(temp_dir) / layout,
                        layout=layout,
                        experiment_id="study",
                        export_formats=("none",),
                    )
                    store = create_data_store(config, Path.cwd())
                    state = store.create_session(
                        session_payload(f"{layout}-session")
                    )
                    database_path = Path(state["database_path"])
                    self.assertTrue(database_path.exists())
                    with sqlite3.connect(database_path) as connection:
                        tables = {
                            row[0]
                            for row in connection.execute(
                                "SELECT name FROM sqlite_master WHERE type = 'table'"
                            )
                        }
                    self.assertTrue(
                        {"sessions", "conditions", "trials", "trajectory_samples"}
                        <= tables
                    )


class RecordingAndExportTests(unittest.TestCase):
    def test_condition_records_and_exports_without_workflow_knowing_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("csv",),
            )
            store = create_data_store(config, Path.cwd())
            store.create_session(session_payload("session-1"))
            origin = np.eye(4)
            store.save_box_pose("session-1", "P1", origin)

            recorder = ConditionRecorder.start(
                store,
                session_id="session-1",
                participant_id="P1",
                condition_index=1,
            )
            recorder.record_calibration(origin)
            recorder.observe_activity(
                "practice",
                None,
                {"done": False},
            )
            recorder.observe_activity(
                "practice",
                None,
                {"done": True},
            )
            sample = {
                "done": False,
                "achieved": False,
                "timed_out": False,
                "elapsed": 0.1,
                "live_pose": origin.tolist(),
                "target_pose": origin.tolist(),
                "linear": 0.01,
                "angular": 2.0,
                "components": {"x": 0.01},
            }
            recorder.observe_activity("trial", 0, sample)
            recorder.observe_activity(
                "trial",
                0,
                {
                    **sample,
                    "done": True,
                    "achieved": True,
                    "elapsed": 1.1,
                    "linear": 0.001,
                    "angular": 1.0,
                },
            )
            result = recorder.save_preference_and_finish(5)

            state = result["state"]
            self.assertEqual(state["conditions"][0]["status"], "complete")
            self.assertEqual(state["conditions"][0]["completed_trials"], 1)
            self.assertTrue(state["has_box_pose"])

            database_path = Path(state["database_path"])
            with sqlite3.connect(database_path) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM trajectory_samples"
                    ).fetchone()[0],
                    2,
                )

            trial_csv = (
                config.export_root("session-1") / "csv" / "trials.csv"
            )
            self.assertTrue(trial_csv.exists())
            with trial_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["achieved"], "1")

    def test_patient_trials_export_groups_completed_trials_by_participant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("patient_trials",),
            )
            store = create_data_store(config, Path.cwd())
            store.create_session(session_payload("session-3", participant_id="P2"))
            origin = np.eye(4)
            store.save_box_pose("session-3", "P2", origin)

            recorder = ConditionRecorder.start(
                store,
                session_id="session-3",
                participant_id="P2",
                condition_index=1,
            )
            sample = {
                "done": False,
                "achieved": False,
                "timed_out": False,
                "elapsed": 0.1,
                "live_pose": origin.tolist(),
                "target_pose": origin.tolist(),
                "linear": 0.01,
                "angular": 2.0,
                "components": {"x": 0.01},
            }
            recorder.observe_activity("trial", 0, sample)
            recorder.observe_activity(
                "trial",
                0,
                {
                    **sample,
                    "done": True,
                    "achieved": True,
                    "elapsed": 1.1,
                    "linear": 0.001,
                    "angular": 1.0,
                },
            )
            recorder.save_preference_and_finish(4)

            store.create_session(session_payload("session-4", participant_id="P2"))
            store.save_box_pose("session-4", "P2", origin)
            second_recorder = ConditionRecorder.start(
                store,
                session_id="session-4",
                participant_id="P2",
                condition_index=1,
            )
            second_recorder.observe_activity("trial", 0, sample)
            second_recorder.observe_activity(
                "trial",
                0,
                {
                    **sample,
                    "done": True,
                    "achieved": False,
                    "elapsed": 2.0,
                    "linear": 0.02,
                    "angular": 3.0,
                },
            )
            second_recorder.save_preference_and_finish(2)

            patient_csv = (
                config.experiment_root("practice")
                / "exports"
                / "by_patient"
                / "Test-Participant_P2"
                / "completed_trials.csv"
            )
            self.assertTrue(patient_csv.exists())
            with patient_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["participant_id"], "P2")
            self.assertEqual(rows[0]["trial_achieved"], "1")
            self.assertEqual(rows[0]["preference_rating"], "4")
            self.assertEqual(rows[0]["trajectory_sample_count"], "2")
            self.assertEqual(rows[1]["session_id"], "session-4")
            self.assertEqual(rows[1]["trial_achieved"], "0")
            self.assertEqual(rows[1]["preference_rating"], "2")

    def test_rerun_creates_a_new_attempt_without_overwriting_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="experiment",
                experiment_id="study",
                export_formats=("none",),
            )
            store = create_data_store(config, Path.cwd())
            store.create_session(session_payload("session-2"))

            first = store.start_condition("session-2", "P1", 1)
            second = store.start_condition("session-2", "P1", 1)
            self.assertNotEqual(first, second)

            database_path = config.database_path("session-2", "P1")
            with sqlite3.connect(database_path) as connection:
                attempts = connection.execute(
                    """
                    SELECT attempt_number
                    FROM condition_runs
                    WHERE session_id = 'session-2' AND condition_index = 1
                    ORDER BY attempt_number
                    """
                ).fetchall()
            self.assertEqual(attempts, [(1,), (2,)])


class DataCategoryTests(unittest.TestCase):
    def test_create_session_routes_by_category_and_persists_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("none",),
            )
            store = create_data_store(config, Path.cwd())
            payload = session_payload("session-real")
            payload["data_category"] = "real"
            state = store.create_session(payload)

            database_path = Path(state["database_path"])
            self.assertEqual(
                database_path,
                Path(temp_dir)
                / "real"
                / "study"
                / "participants"
                / "Test-Participant_P1"
                / "experiment.sqlite",
            )
            self.assertEqual(state["session"]["data_category"], "real")

    def test_missing_or_invalid_category_defaults_to_practice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("none",),
            )
            store = create_data_store(config, Path.cwd())
            payload = session_payload("session-bogus")
            payload["data_category"] = "not-a-real-category"
            state = store.create_session(payload)
            self.assertIn(
                "practice",
                Path(state["database_path"]).parts,
            )
            self.assertEqual(state["session"]["data_category"], "practice")

    def test_resume_finds_session_regardless_of_category(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("none",),
            )
            store = create_data_store(config, Path.cwd())
            payload = session_payload("session-trash", participant_id="P9")
            payload["data_category"] = "trash"
            store.create_session(payload)

            found = store.latest_session_for_participant("P9")
            self.assertIsNotNone(found)
            self.assertEqual(found["session"]["session_id"], "session-trash")

    def test_legacy_sessions_table_without_data_category_column_is_migrated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.sqlite"
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE sessions (
                        session_id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        participant_id TEXT NOT NULL,
                        participant_name TEXT,
                        examiner_name TEXT,
                        experiment_condition TEXT NOT NULL,
                        condition_option INTEGER,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        status TEXT NOT NULL,
                        git_commit TEXT,
                        schema_version INTEGER NOT NULL,
                        storage_layout TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO sessions VALUES (
                        'old-session', 'study', 'P1', NULL, NULL, 'modality',
                        1, '2025-01-01T00:00:00Z', NULL, 'in_progress', NULL,
                        1, 'participant', '{}'
                    )
                    """
                )

            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("none",),
            )
            store = create_data_store(config, Path.cwd())
            with store._connect(db_path) as connection:
                row = connection.execute(
                    "SELECT session_id, data_category FROM sessions WHERE session_id = 'old-session'"
                ).fetchone()
            self.assertEqual(row, ("old-session", None))


class LegacyDatedLayoutTests(unittest.TestCase):
    """New writes drop the {date} folder that used to sit between the
    category bin and the experiment folder, but old dated-layout data must
    stay discoverable without moving it."""

    def _write_legacy_database(self, temp_dir, category, date, participant_id):
        config = StorageConfig(root=Path(temp_dir), layout="participant", experiment_id="study")
        legacy_path = (
            Path(temp_dir)
            / category
            / date
            / "study"
            / "participants"
            / participant_id
            / "experiment.sqlite"
        )
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        store = create_data_store(config, Path.cwd())
        with store._connect(legacy_path) as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, experiment_id, participant_id, participant_name,
                    examiner_name, experiment_condition, condition_option,
                    data_category, started_at, status, git_commit, schema_version,
                    storage_layout, metadata_json
                ) VALUES (?, 'study', ?, NULL, NULL, 'modality', 1, ?, ?, 'complete',
                          NULL, 1, 'participant', '{}')
                """,
                (
                    f"legacy-{participant_id}",
                    participant_id,
                    category,
                    f"{date}T00:00:00Z",
                ),
            )
        return legacy_path

    def test_locate_database_finds_legacy_dated_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_legacy_database(temp_dir, "practice", "2026-06-24", "P5")
            config = StorageConfig(root=Path(temp_dir), layout="participant", experiment_id="study")
            store = create_data_store(config, Path.cwd())

            state = store.get_session_state("legacy-P5", "P5")
            self.assertIsNotNone(state)
            self.assertEqual(state["session"]["participant_id"], "P5")

    def test_latest_session_for_participant_includes_legacy_dated_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_legacy_database(temp_dir, "real", "2026-05-01", "P6")
            config = StorageConfig(root=Path(temp_dir), layout="participant", experiment_id="study")
            store = create_data_store(config, Path.cwd())

            found = store.latest_session_for_participant("P6")
            self.assertIsNotNone(found)
            self.assertEqual(found["session"]["session_id"], "legacy-P6")

    def test_new_session_for_returning_participant_reuses_the_undated_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(
                root=Path(temp_dir),
                layout="participant",
                experiment_id="study",
                export_formats=("none",),
            )
            store = create_data_store(config, Path.cwd())
            store.create_session(session_payload("session-day1", participant_id="P7"))
            store.create_session(session_payload("session-day2", participant_id="P7"))

            database_path = config.database_path("session-day2", "P7", "Test Participant")
            self.assertEqual(
                database_path,
                Path(temp_dir) / "practice" / "study" / "participants" / "Test-Participant_P7" / "experiment.sqlite",
            )
            with sqlite3.connect(database_path) as connection:
                session_ids = {
                    row[0]
                    for row in connection.execute(
                        "SELECT session_id FROM sessions WHERE participant_id = 'P7'"
                    )
                }
            self.assertEqual(session_ids, {"session-day1", "session-day2"})


if __name__ == "__main__":
    unittest.main()
