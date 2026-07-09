# Experiment data architecture

The experiment workflow depends on the `ExperimentRepository` port in
`study/storage.py`. It does not construct filenames, write CSV rows, or issue
SQL directly.

## Collection backends

`create_data_store()` selects a repository adapter from `REPOSITORIES`.
`SqliteExperimentRepository` is the current adapter. A future JSONL, remote
database, or service-backed implementation can be added without changing
`server.py`, the sequence runner, or the modality pages.

All adapters expose session, condition, trial, trajectory, calibration,
practice, preference, event, state-query, and export operations.

## File-layout policy

`StorageConfig.layout` controls where the SQLite adapter resolves its database.
By default, the root is `~/Documents/visualexperiment`. Every session also
carries a `data_category` (`real`, `practice`, or `trash`) chosen by the
operator at session creation — this is threaded per-request (it is not a
server-startup flag) and is inserted as a folder above the dated experiment
folder:

- `session`: `<root>/<category>/YYYY-MM-DD/<experiment>/sessions/<session-id>/experiment.sqlite`
- `participant`: `<root>/<category>/YYYY-MM-DD/<experiment>/participants/<participant-id>/experiment.sqlite`
- `experiment`: `<root>/<category>/YYYY-MM-DD/<experiment>/experiment.sqlite`

The logical schema is identical in every layout. Therefore changing layout does
not change collection code or downstream column definitions. Session lookups
that only have a `session_id`/`participant_id` (resume, run-location) search
across all three category folders since the category isn't known in advance;
once a database file is resolved, its category is recovered from its path.

## Export adapters

Exporters implement `DataExporter` and are registered in `EXPORTERS`. CSV is
currently implemented. Export failures are recorded as events and do not undo
completed trial or condition records.

The normalized CSV exports are session-scoped even when the source database is
participant- or experiment-scoped, and live under the same category folder as
the source database:

`<root>/<category>/YYYY-MM-DD/<experiment>/exports/<session-id>/csv/`

The `patient_trials` export is analysis-oriented and grouped by participant:

`<root>/YYYY-MM-DD/<experiment>/exports/by_patient/<participant-id>/completed_trials.csv`

Each row represents one completed trial. Scalar fields from the session,
condition, run, trial, box pose, practice period, and preference tables are
prefixed into columns. Multi-row details such as trajectory samples,
calibrations, and events are included as JSON columns so the row remains one
trial wide without discarding detail.

## Recovery model

- Session and condition setup are committed immediately.
- Trajectory samples are committed in small batches.
- Trial completion is committed before the next trial begins.
- Condition completion and preference are committed together at the workflow
  boundary.
- Interrupted runs remain `in_progress`.
- Starting a completed condition creates a new `condition_runs` attempt, so
  earlier data is retained.
- The dashboard reads durable status and allows any condition to be selected.

## Adding a format

To add a collection backend:

1. Implement the `ExperimentRepository` protocol.
2. Register it in `REPOSITORIES`.
3. Add its name to the `--data-backend` CLI choices.

To add an export format:

1. Implement `DataExporter`.
2. Register it in `EXPORTERS`.
3. Add its name to the `--export-format` CLI choices.
