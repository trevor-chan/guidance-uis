#!/usr/bin/env python3
"""Pool experiment.sqlite files under a folder and render summary plots.

Usage:
    python3 analysis/make_plots.py <folder>

<folder> is scanned recursively for files named experiment.sqlite (e.g. point
it at a data-category bin such as .../visualexperiment/real or .../practice).
Every trial is pooled across every file found, grouped by each session's
experiment_condition. Axis values (magnitudes, thresholds, latencies, trial
counts) are always read from the data itself, never hardcoded, so changes to
the study configuration do not silently break these plots.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -- Palette (validated categorical order; see the dataviz skill) ----------

MODALITY_IDS = [f"M{i}" for i in range(1, 8)]
MODALITY_LABELS = {
    "M1": "M1\n1D",
    "M2": "M2\n2D/User",
    "M3": "M3\n2D/Patient",
    "M4": "M4\n2D/Transducer",
    "M5": "M5\n3D/User",
    "M6": "M6\n3D/Patient",
    "M7": "M7\n3D/Transducer",
}
MODALITY_COLORS = dict(
    zip(
        MODALITY_IDS,
        [
            "#2a78d6",  # M1 blue
            "#1baf7a",  # M2 aqua
            "#eda100",  # M3 yellow
            "#008300",  # M4 green
            "#4a3aa7",  # M5 violet
            "#e34948",  # M6 red
            "#e87ba4",  # M7 magenta
        ],
    )
)
COMPARE_MODES = ["M3", "M5"]  # 2D/Patient vs 3D/User: the two-series plots
COMPARE_LABELS = {"M3": "M3 (2D/Patient)", "M5": "M5 (3D/User)"}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

EXPERIMENTS = ["modality", "noise", "latency", "learning_curve", "precision"]


# -- Data loading ------------------------------------------------------------


def find_sqlite_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("experiment.sqlite"))


# Columns added by idempotent ALTER TABLE migrations (see study/storage.py
# _ensure_schema) may be absent from .sqlite files predating that migration.
# This script opens files read-only and must not mutate study data to add
# them, so missing columns are substituted with NULL at query time instead.
CONDITIONS_MIGRATED_COLUMNS = ("precision_linear_mm", "precision_angular_deg")

TRIAL_QUERY_TEMPLATE = """
SELECT
    sessions.experiment_condition AS experiment_condition,
    sessions.participant_id AS participant_id,
    conditions.modality_id AS modality_id,
    conditions.noise AS noise,
    conditions.latency_ms AS latency_ms,
    conditions.perceived_ms AS perceived_ms,
    {precision_linear_mm} AS precision_linear_mm,
    {precision_angular_deg} AS precision_angular_deg,
    condition_runs.run_id AS run_id,
    condition_runs.attempt_number AS attempt_number,
    trials.trial_index AS trial_index,
    trials.achieved AS achieved,
    trials.elapsed_s AS elapsed_s
FROM trials
JOIN condition_runs ON condition_runs.run_id = trials.run_id
JOIN conditions
    ON conditions.session_id = condition_runs.session_id
   AND conditions.condition_index = condition_runs.condition_index
JOIN sessions ON sessions.session_id = conditions.session_id
WHERE trials.status = 'complete'
"""


def trial_query_for(connection: sqlite3.Connection) -> str:
    existing = {row[1] for row in connection.execute("PRAGMA table_info(conditions)")}
    columns = {
        name: f"conditions.{name}" if name in existing else "NULL"
        for name in CONDITIONS_MIGRATED_COLUMNS
    }
    return TRIAL_QUERY_TEMPLATE.format(**columns)

PREFERENCE_QUERY = """
SELECT
    sessions.participant_id AS participant_id,
    conditions.modality_id AS modality_id,
    preferences.rating AS rating
FROM preferences
JOIN condition_runs ON condition_runs.run_id = preferences.run_id
JOIN conditions
    ON conditions.session_id = condition_runs.session_id
   AND conditions.condition_index = condition_runs.condition_index
JOIN sessions ON sessions.session_id = conditions.session_id
WHERE sessions.experiment_condition = 'modality'
"""


def load_data(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    trials: list[dict] = []
    preferences: list[dict] = []
    for path in paths:
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            with connection:
                for row in connection.execute(trial_query_for(connection)):
                    record = dict(row)
                    record["source_file"] = str(path)
                    trials.append(record)
                for row in connection.execute(PREFERENCE_QUERY):
                    record = dict(row)
                    record["source_file"] = str(path)
                    preferences.append(record)
        except sqlite3.DatabaseError as exc:
            print(f"  ! skipping {path}: {exc}")
        finally:
            connection.close()
    return trials, preferences


# -- Summary -------------------------------------------------------------


def print_summary(trials: list[dict]) -> None:
    print("\nSummary by experiment_condition:")
    for experiment in EXPERIMENTS:
        rows = [t for t in trials if t["experiment_condition"] == experiment]
        if not rows:
            print(f"  {experiment:<15} no data found, skipping")
            continue
        n_files = len({t["source_file"] for t in rows})
        n_participants = len({t["participant_id"] for t in rows})
        n_trials = len(rows)
        print(
            f"  {experiment:<15} files={n_files:<4} participants={n_participants:<4} trials={n_trials}"
        )


# -- Plot helpers ----------------------------------------------------------


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(AXIS)
    ax.tick_params(colors=SECONDARY_INK, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def style_box(bp, color: str) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55)
        box.set_edgecolor(color)
        box.set_linewidth(1.4)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set_color(MUTED)
            artist.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_color(INK)
        median.set_linewidth(1.8)
    for flier in bp["fliers"]:
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_markersize(4)
        flier.set_alpha(0.6)


def save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def present_modalities(rows: list[dict]) -> list[str]:
    found = {r["modality_id"] for r in rows if r["modality_id"]}
    return [m for m in MODALITY_IDS if m in found]


# -- Individual plots --------------------------------------------------


def plot_modality_time(trials: list[dict]) -> None:
    rows = [t for t in trials if t["experiment_condition"] == "modality" and t["elapsed_s"] is not None]
    if not rows:
        print("  modality_time: no data, skipping")
        return
    modalities = present_modalities(rows)
    data = [[r["elapsed_s"] for r in rows if r["modality_id"] == m] for m in modalities]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    style_box(bp, INK)
    for i, m in enumerate(modalities):
        bp["boxes"][i].set_edgecolor(MODALITY_COLORS[m])
        bp["boxes"][i].set_facecolor(MODALITY_COLORS[m])
        bp["boxes"][i].set_alpha(0.6)
    ax.set_xticks(range(1, len(modalities) + 1))
    ax.set_xticklabels([MODALITY_LABELS[m] for m in modalities], fontsize=9)
    ax.set_ylabel("Time (s)")
    ax.set_title("Modality: completion time by modality", fontsize=13)
    style_axes(ax)
    save(fig, "modality_time.png")


def plot_modality_preference(preferences: list[dict]) -> None:
    if not preferences:
        print("  modality_preference: no data, skipping")
        return
    modalities = present_modalities(preferences)
    data = [[p["rating"] for p in preferences if p["modality_id"] == m] for m in modalities]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    style_box(bp, INK)
    for i, m in enumerate(modalities):
        bp["boxes"][i].set_edgecolor(MODALITY_COLORS[m])
        bp["boxes"][i].set_facecolor(MODALITY_COLORS[m])
        bp["boxes"][i].set_alpha(0.6)
    ax.set_xticks(range(1, len(modalities) + 1))
    ax.set_xticklabels([MODALITY_LABELS[m] for m in modalities], fontsize=9)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Preference rating (1-5)")
    ax.set_title("Modality: preference rating by modality", fontsize=13)
    style_axes(ax)
    save(fig, "modality_preference.png")


def plot_modality_success(trials: list[dict]) -> None:
    rows = [t for t in trials if t["experiment_condition"] == "modality" and t["achieved"] is not None]
    if not rows:
        print("  modality_success: no data, skipping")
        return
    modalities = present_modalities(rows)
    fractions = [
        mean(r["achieved"] for r in rows if r["modality_id"] == m) for m in modalities
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (m, frac) in enumerate(zip(modalities, fractions)):
        ax.scatter(i + 1, frac, color=MODALITY_COLORS[m], s=110, zorder=3, edgecolor=INK, linewidth=0.8)
    ax.set_xticks(range(1, len(modalities) + 1))
    ax.set_xticklabels([MODALITY_LABELS[m] for m in modalities], fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Fraction achieved")
    ax.set_title("Modality: success rate by modality", fontsize=13)
    style_axes(ax)
    save(fig, "modality_success.png")


def plot_learning_curve_time(trials: list[dict]) -> None:
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "learning_curve"
        and t["modality_id"] in COMPARE_MODES
        and t["elapsed_s"] is not None
    ]
    if not rows:
        print("  learning_curve_time: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in COMPARE_MODES:
        mode_rows = [r for r in rows if r["modality_id"] == mode]
        if not mode_rows:
            continue
        xs = [r["trial_index"] + 1 for r in mode_rows]
        ys = [r["elapsed_s"] for r in mode_rows]
        ax.scatter(
            xs, ys,
            color=MODALITY_COLORS[mode], label=COMPARE_LABELS[mode],
            alpha=0.65, s=28, edgecolor="none",
        )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Time (s)")
    ax.set_title("Learning curve: time per trial", fontsize=13)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, fontsize=10)
    save(fig, "learning_curve_time.png")


def grouped_boxplot(
    ax,
    groups: list,
    group_labels: list[str],
    series: list[str],
    values: dict,
    series_colors: dict,
    series_labels: dict,
) -> None:
    n_series = len(series)
    box_width = 0.7 / n_series
    for j, s in enumerate(series):
        offset = (j - (n_series - 1) / 2) * box_width
        positions = [i + 1 + offset for i in range(len(groups))]
        data = [values.get((g, s), []) for g in groups]
        bp = ax.boxplot(
            data, positions=positions, widths=box_width * 0.85, patch_artist=True,
        )
        style_box(bp, series_colors[s])
        for box in bp["boxes"]:
            box.set_facecolor(series_colors[s])
            box.set_edgecolor(series_colors[s])
            box.set_alpha(0.6)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(group_labels)
    legend_handles = [
        Patch(facecolor=series_colors[s], edgecolor=series_colors[s], alpha=0.6, label=series_labels[s])
        for s in series
    ]
    ax.legend(handles=legend_handles, frameon=False, labelcolor=INK, fontsize=10)


def plot_noise_time(trials: list[dict]) -> None:
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "noise"
        and t["modality_id"] in COMPARE_MODES
        and t["noise"] is not None
        and t["elapsed_s"] is not None
    ]
    if not rows:
        print("  noise_time: no data, skipping")
        return
    magnitudes = sorted({r["noise"] for r in rows})
    values = defaultdict(list)
    for r in rows:
        values[(r["noise"], r["modality_id"])].append(r["elapsed_s"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_boxplot(
        ax,
        groups=magnitudes,
        group_labels=[f"{m:g}" for m in magnitudes],
        series=COMPARE_MODES,
        values=values,
        series_colors=MODALITY_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Noise magnitude (mm / deg)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Noise: completion time by magnitude", fontsize=13)
    style_axes(ax)
    save(fig, "noise_time.png")


def plot_latency_time(trials: list[dict]) -> None:
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "latency"
        and t["modality_id"] in COMPARE_MODES
        and t["perceived_ms"] is not None
        and t["elapsed_s"] is not None
    ]
    if not rows:
        print("  latency_time: no data, skipping")
        return
    latencies = sorted({r["perceived_ms"] for r in rows})
    values = defaultdict(list)
    for r in rows:
        values[(r["perceived_ms"], r["modality_id"])].append(r["elapsed_s"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_boxplot(
        ax,
        groups=latencies,
        group_labels=[f"{l:g}" for l in latencies],
        series=COMPARE_MODES,
        values=values,
        series_colors=MODALITY_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Perceived latency (ms)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Latency: completion time by perceived latency", fontsize=13)
    style_axes(ax)
    save(fig, "latency_time.png")


def precision_key(row: dict):
    return (row["precision_linear_mm"], row["precision_angular_deg"])


def precision_groups(rows: list[dict]) -> list[tuple[float, float]]:
    keys = {precision_key(r) for r in rows}
    # Easiest -> hardest: largest tolerance first.
    return sorted(keys, key=lambda k: (-k[0], -k[1]))


def precision_label(key: tuple[float, float]) -> str:
    mm, deg = key
    return f"{mm:g}mm/{deg:g}deg"


def plot_precision_time(trials: list[dict]) -> None:
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "precision"
        and t["modality_id"] in COMPARE_MODES
        and t["precision_linear_mm"] is not None
        and t["elapsed_s"] is not None
    ]
    if not rows:
        print("  precision_time: no data, skipping")
        return
    groups = precision_groups(rows)
    values = defaultdict(list)
    for r in rows:
        values[(precision_key(r), r["modality_id"])].append(r["elapsed_s"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_boxplot(
        ax,
        groups=groups,
        group_labels=[precision_label(g) for g in groups],
        series=COMPARE_MODES,
        values=values,
        series_colors=MODALITY_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Match threshold (easiest -> hardest)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Precision: completion time by threshold", fontsize=13)
    style_axes(ax)
    save(fig, "precision_time.png")


def plot_precision_success(trials: list[dict]) -> None:
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "precision"
        and t["modality_id"] in COMPARE_MODES
        and t["precision_linear_mm"] is not None
        and t["achieved"] is not None
    ]
    if not rows:
        print("  precision_success: no data, skipping")
        return
    groups = precision_groups(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode in COMPARE_MODES:
        mode_rows = [r for r in rows if r["modality_id"] == mode]
        if not mode_rows:
            continue
        xs, ys = [], []
        for i, g in enumerate(groups):
            group_rows = [r for r in mode_rows if precision_key(r) == g]
            if not group_rows:
                continue
            xs.append(i + 1)
            ys.append(mean(r["achieved"] for r in group_rows))
        ax.scatter(
            xs, ys, color=MODALITY_COLORS[mode], label=COMPARE_LABELS[mode],
            s=110, zorder=3, edgecolor=INK, linewidth=0.8,
        )
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([precision_label(g) for g in groups])
    ax.set_xlabel("Match threshold (easiest -> hardest)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Fraction achieved")
    ax.set_title("Precision: success rate by threshold", fontsize=13)
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK, fontsize=10)
    save(fig, "precision_success.png")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <folder>")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser()
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    paths = find_sqlite_files(folder)
    print(f"Found {len(paths)} experiment.sqlite file(s) under {folder}")
    if not paths:
        print("Nothing to plot.")
        return

    trials, preferences = load_data(paths)
    print_summary(trials)

    print("\nPlots:")
    plot_modality_time(trials)
    plot_modality_preference(preferences)
    plot_modality_success(trials)
    plot_learning_curve_time(trials)
    plot_noise_time(trials)
    plot_latency_time(trials)
    plot_precision_time(trials)
    plot_precision_success(trials)


if __name__ == "__main__":
    main()
