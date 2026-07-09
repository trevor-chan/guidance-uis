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
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Palette ------------------------------------------------------------
#
# Family = data dimensionality: 1D gray, 2D blue, 3D green. Within the 2D and
# 3D families, shades distinguish reference frame (user / patient /
# transducer). M3 and M5 are the two-mode comparison plots' series, so their
# shade doubles as that family's canonical tone — one color per mode
# everywhere it appears.

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
MODALITY_COLORS = {
    "M1": "#7f7f7f",  # 1D — gray
    "M2": "#9ecae1",  # 2D/User — light blue
    "M3": "#1f77b4",  # 2D/Patient — blue
    "M4": "#08519c",  # 2D/Transducer — dark blue
    "M5": "#2ca02c",  # 3D/User — green
    "M6": "#74c476",  # 3D/Patient — light green
    "M7": "#005a32",  # 3D/Transducer — dark green
}
COMPARE_MODES = ["M3", "M5"]  # 2D/Patient vs 3D/User: the two-series plots
COMPARE_LABELS = {"M3": "M3 (2D/Patient)", "M5": "M5 (3D/User)"}
COMPARE_COLORS = {m: MODALITY_COLORS[m] for m in COMPARE_MODES}

INK = "#1a1a1a"
MUTED = "#8c8c8c"
GRID_COLOR = "#cccccc"
SPINE_COLOR = "#333333"

FIGSIZE = (6, 4.5)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": SPINE_COLOR,
        "ytick.color": SPINE_COLOR,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "figure.figsize": FIGSIZE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "legend.frameon": False,
    }
)

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
    ax.set_axisbelow(True)
    ax.grid(axis="y", zorder=0)


def style_box(bp, color: str) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.55)
        box.set_edgecolor(color)
        box.set_linewidth(1.2)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set_color(MUTED)
            artist.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color(INK)
        median.set_linewidth(1.5)
    for flier in bp["fliers"]:
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_markersize(4)
        flier.set_alpha(0.6)


def save(fig, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def present_modalities(rows: list[dict]) -> list[str]:
    found = {r["modality_id"] for r in rows if r["modality_id"]}
    return [m for m in MODALITY_IDS if m in found]


# -- Confidence intervals ---------------------------------------------------

# Two-tailed 97.5th-percentile Student's t critical values, df 1-30. Beyond
# df=30 a Cornish-Fisher expansion around the normal quantile is accurate to
# the precision these plots need, avoiding a scipy dependency.
_T_TABLE_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z_975 = 1.959964


def t_critical_975(df: int) -> float:
    if df in _T_TABLE_975:
        return _T_TABLE_975[df]
    z = _Z_975
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """t-based 95% CI of the mean. Returns (mean, lower, upper); lower==upper==mean if n==1."""
    n = len(values)
    m = mean(values)
    if n <= 1:
        return m, m, m
    se = stdev(values) / sqrt(n)
    half = t_critical_975(n - 1) * se
    return m, m - half, m + half


def wilson_ci95(successes: float, n: int) -> tuple[float, float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (phat, lower, upper)."""
    z = _Z_975
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return phat, center - half, center + half


# -- Individual plots --------------------------------------------------


def plot_modality_time(trials: list[dict]) -> None:
    rows = [t for t in trials if t["experiment_condition"] == "modality" and t["elapsed_s"] is not None]
    if not rows:
        print("  modality_time: no data, skipping")
        return
    modalities = present_modalities(rows)
    data = [[r["elapsed_s"] for r in rows if r["modality_id"] == m] for m in modalities]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    style_box(bp, INK)
    for i, m in enumerate(modalities):
        bp["boxes"][i].set_edgecolor(MODALITY_COLORS[m])
        bp["boxes"][i].set_facecolor(MODALITY_COLORS[m])
        bp["boxes"][i].set_alpha(0.6)
    ax.set_xticks(range(1, len(modalities) + 1))
    ax.set_xticklabels([MODALITY_LABELS[m] for m in modalities], fontsize=9)
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Modality: time to match by modality")
    style_axes(ax)
    save(fig, "modality_time.png")


def bar_with_ci(ax, modalities: list[str], means: list[float], los: list[float], his: list[float]) -> None:
    xs = list(range(1, len(modalities) + 1))
    colors = [MODALITY_COLORS[m] for m in modalities]
    ax.bar(xs, means, width=0.62, color=colors, alpha=0.85, edgecolor=INK, linewidth=0.8, zorder=2)
    lo_err = [m - lo for m, lo in zip(means, los)]
    hi_err = [hi - m for m, hi in zip(means, his)]
    ax.errorbar(
        xs, means, yerr=[lo_err, hi_err],
        fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([MODALITY_LABELS[m] for m in modalities], fontsize=9)


def plot_modality_preference(preferences: list[dict]) -> None:
    if not preferences:
        print("  modality_preference: no data, skipping")
        return
    modalities = present_modalities(preferences)
    means, los, his = [], [], []
    for m in modalities:
        ratings = [p["rating"] for p in preferences if p["modality_id"] == m]
        mn, lo, hi = mean_ci95(ratings)
        means.append(mn)
        los.append(lo)
        his.append(hi)

    fig, ax = plt.subplots()
    bar_with_ci(ax, modalities, means, los, his)
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Preference rating (1-5)")
    ax.set_title("Modality: preference rating by modality (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "modality_preference.png")


def plot_modality_success(trials: list[dict]) -> None:
    rows = [t for t in trials if t["experiment_condition"] == "modality" and t["achieved"] is not None]
    if not rows:
        print("  modality_success: no data, skipping")
        return
    modalities = present_modalities(rows)
    means, los, his = [], [], []
    for m in modalities:
        outcomes = [r["achieved"] for r in rows if r["modality_id"] == m]
        phat, lo, hi = wilson_ci95(sum(outcomes), len(outcomes))
        means.append(phat)
        los.append(lo)
        his.append(hi)

    fig, ax = plt.subplots()
    bar_with_ci(ax, modalities, means, los, his)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success rate")
    ax.set_title("Modality: success rate by modality (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "modality_success.png")


def point_range_by_group(
    ax,
    groups: list,
    group_labels: list[str],
    series: list[str],
    values: dict,
    series_colors: dict,
    series_labels: dict,
) -> None:
    """Mean line with 95% CI point-range per group. All series share the same
    x position (no dodge) — overlap is expected, color differentiates. A line
    threads through each series' mean markers."""
    for s in series:
        xs, means, lo_err, hi_err = [], [], [], []
        for i, g in enumerate(groups):
            vals = values.get((g, s), [])
            if not vals:
                continue
            m, lo, hi = mean_ci95(vals)
            xs.append(i + 1)
            means.append(m)
            lo_err.append(m - lo)
            hi_err.append(hi - m)
        if not xs:
            continue
        color = series_colors[s]
        ax.plot(xs, means, "-", color=color, linewidth=1.4, zorder=2)
        ax.errorbar(
            xs, means, yerr=[lo_err, hi_err],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.2, capsize=4, capthick=1.2, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6,
            label=series_labels[s], zorder=3,
        )
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(group_labels)
    ax.legend(loc="best")


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
    trial_numbers = sorted({r["trial_index"] + 1 for r in rows})
    values = defaultdict(list)
    for r in rows:
        values[(r["trial_index"] + 1, r["modality_id"])].append(r["elapsed_s"])

    fig, ax = plt.subplots()
    point_range_by_group(
        ax,
        groups=trial_numbers,
        group_labels=[str(n) for n in trial_numbers],
        series=COMPARE_MODES,
        values=values,
        series_colors=COMPARE_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Learning curve: time per trial (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "learning_curve_time.png")


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

    fig, ax = plt.subplots()
    point_range_by_group(
        ax,
        groups=magnitudes,
        group_labels=[f"{m:g}" for m in magnitudes],
        series=COMPARE_MODES,
        values=values,
        series_colors=COMPARE_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Noise magnitude (mm / deg)")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Noise: completion time by magnitude (mean ± 95% CI)")
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

    fig, ax = plt.subplots()
    point_range_by_group(
        ax,
        groups=latencies,
        group_labels=[f"{l:g}" for l in latencies],
        series=COMPARE_MODES,
        values=values,
        series_colors=COMPARE_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Perceived latency (ms)")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Latency: completion time by perceived latency (mean ± 95% CI)")
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

    fig, ax = plt.subplots()
    point_range_by_group(
        ax,
        groups=groups,
        group_labels=[precision_label(g) for g in groups],
        series=COMPARE_MODES,
        values=values,
        series_colors=COMPARE_COLORS,
        series_labels=COMPARE_LABELS,
    )
    ax.set_xlabel("Precision threshold (mm / deg)")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Precision: completion time by threshold (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "precision_time.png")


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


if __name__ == "__main__":
    main()
