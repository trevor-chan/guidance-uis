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
            collection_date="2026-06-24",
        )
        participant_config = StorageConfig(
            root=root,
            layout="participant",
            experiment_id="study",
            collection_date="2026-06-24",
        )
        experiment_config = StorageConfig(
            root=root,
            layout="experiment",
            experiment_id="study",
            collection_date="2026-06-24",
        )

        self.assertEqual(
            session_config.database_path("abc", "P1"),
            root / "2026-06-24" / "study" / "sessions" / "abc" / "experiment.sqlite",
        )
        self.assertEqual(
            participant_config.database_path("abc", "P1"),
            root / "2026-06-24" / "study" / "participants" / "P1" / "experiment.sqlite",
        )
        self.assertEqual(
            experiment_config.database_path("abc", "P1"),
            root / "2026-06-24" / "study" / "experiment.sqlite",
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
                collection_date="2026-06-24",
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
                config.experiment_root
                / "exports"
                / "by_patient"
                / "P2"
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


if __name__ == "__main__":
    unittest.main()
