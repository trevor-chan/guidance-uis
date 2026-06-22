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

`StorageConfig.layout` controls where the SQLite adapter resolves its database:

- `session`: `data/<experiment>/sessions/<session-id>/experiment.sqlite`
- `participant`: `data/<experiment>/participants/<participant-id>/experiment.sqlite`
- `experiment`: `data/<experiment>/experiment.sqlite`

The logical schema is identical in every layout. Therefore changing layout does
not change collection code or downstream column definitions.

## Export adapters

Exporters implement `DataExporter` and are registered in `EXPORTERS`. CSV is
currently implemented. Export failures are recorded as events and do not undo
completed trial or condition records.

Exports are session-scoped even when the source database is participant- or
experiment-scoped:

`data/<experiment>/exports/<session-id>/<format>/`

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
