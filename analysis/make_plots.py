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

import random
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
# everywhere it appears. Shades are generated from one base hue per family so
# the family relationship stays visually systematic rather than hand-tuned.


def lighten(hex_color: str, factor: float) -> str:
    """Blend a hex color toward white by `factor` in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * factor)
    g = round(g + (255 - g) * factor)
    b = round(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def darken(hex_color: str, factor: float) -> str:
    """Blend a hex color toward black by `factor` in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = round(r * (1 - factor))
    g = round(g * (1 - factor))
    b = round(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


MODALITY_IDS = [f"M{i}" for i in range(1, 8)]
MODALITY_LABELS = {
    "M1": "M1 (1D)",
    "M2": "M2 (2D/User)",
    "M3": "M3 (2D/Patient)",
    "M4": "M4 (2D/Transducer)",
    "M5": "M5 (3D/User)",
    "M6": "M6 (3D/Patient)",
    "M7": "M7 (3D/Transducer)",
}

_GRAY = "#767676"
_BLUE = "#2f6f9e"  # 2D/Patient — canonical "2D" tone in compare plots
_GREEN = "#3f8f57"  # 3D/User — canonical "3D" tone in compare plots

MODALITY_COLORS = {
    "M1": _GRAY,
    "M2": lighten(_BLUE, 0.42),
    "M3": _BLUE,
    "M4": darken(_BLUE, 0.35),
    "M5": _GREEN,
    "M6": lighten(_GREEN, 0.42),
    "M7": darken(_GREEN, 0.35),
}
# Dot clouds sit a shade lighter than their summary marker/bar so the mean +
# CI reads clearly on top of the raw data.
MODALITY_DOT_COLORS = {m: lighten(c, 0.5) for m, c in MODALITY_COLORS.items()}

COMPARE_MODES = ["M3", "M5"]  # 2D/Patient vs 3D/User: the two-series plots
COMPARE_LABELS = {"M3": "M3 (2D/Patient)", "M5": "M5 (3D/User)"}
COMPARE_COLORS = {m: MODALITY_COLORS[m] for m in COMPARE_MODES}
COMPARE_DOT_COLORS = {m: MODALITY_DOT_COLORS[m] for m in COMPARE_MODES}

INK = "#1a1a1a"
GRID_COLOR = "#e3e3e3"
SPINE_COLOR = "#333333"

FIGSIZE = (6.5, 4.5)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": SPINE_COLOR,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": SPINE_COLOR,
        "ytick.color": SPINE_COLOR,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
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

# Deterministic horizontal jitter for dot clouds: a fixed seed keeps repeated
# runs over the same data visually stable instead of reshuffling each time.
_JITTER_RNG = random.Random(20260709)


def jittered_xs(x_center: float, n: int, width: float) -> list[float]:
    return [x_center + _JITTER_RNG.uniform(-width, width) for _ in range(n)]


# -- Data loading ------------------------------------------------------------


def find_sqlite_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("experiment.sqlite"))


# Columns added by idempotent ALTER TABLE migrations (see study/storage.py
# _ensure_schema) may be absent from .sqlite files predating that migration.
# This script opens files read-only and must not mutate study data to add
# them, so missing columns are substituted with NULL at query time instead.
CONDITIONS_MIGRATED_COLUMNS = ("precision_linear_mm", "precision_angular_deg")

# trial_* columns hold the per-trial noise/latency/precision magnitude for
# the scrambled noise/latency/precision blocks (one condition = one whole
# mode's shuffled ramp; see SequenceGenerator.make_block's trial_overrides).
# Older data predating that change only has the per-condition columns above,
# where every trial in a condition shared one fixed magnitude — COALESCE
# prefers the trial-level value and falls back to the condition-level one so
# both eras of data resolve to the same output columns.
TRIALS_MIGRATED_COLUMNS = (
    "trial_noise", "trial_latency_ms", "trial_perceived_ms",
    "trial_precision_linear_mm", "trial_precision_angular_deg",
)

TRIAL_QUERY_TEMPLATE = """
SELECT
    sessions.experiment_condition AS experiment_condition,
    sessions.participant_id AS participant_id,
    conditions.modality_id AS modality_id,
    COALESCE({trial_noise}, conditions.noise) AS noise,
    COALESCE({trial_latency_ms}, conditions.latency_ms) AS latency_ms,
    COALESCE({trial_perceived_ms}, conditions.perceived_ms) AS perceived_ms,
    COALESCE({trial_precision_linear_mm}, {precision_linear_mm}) AS precision_linear_mm,
    COALESCE({trial_precision_angular_deg}, {precision_angular_deg}) AS precision_angular_deg,
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
    existing_conditions = {row[1] for row in connection.execute("PRAGMA table_info(conditions)")}
    existing_trials = {row[1] for row in connection.execute("PRAGMA table_info(trials)")}
    columns = {
        name: f"conditions.{name}" if name in existing_conditions else "NULL"
        for name in CONDITIONS_MIGRATED_COLUMNS
    }
    columns.update({
        name: f"trials.{name}" if name in existing_trials else "NULL"
        for name in TRIALS_MIGRATED_COLUMNS
    })
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


def apply_category_ticklabels(ax, xs: list[float], labels: list[str], rotate_len: int = 8) -> None:
    """Rotate long category labels ~38° right-aligned so they never overlap;
    short labels stay horizontal."""
    ax.set_xticks(xs)
    flat = [str(label).replace("\n", " ") for label in labels]
    if max(len(label) for label in flat) > rotate_len:
        ax.set_xticklabels(flat, rotation=38, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(flat)


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


# -- Summary-with-dots primitives -------------------------------------------
#
# Nature "show the dots" convention: every bar / mean-marker gets the raw,
# per-trial values plotted underneath as a small semi-transparent jittered
# dot cloud in a lighter tone, with the mean ± 95% CI drawn on top at full
# opacity so the summary reads clearly over the cloud.


def draw_dot_cloud(ax, x_center: float, values: list[float], color: str, width: float, zorder: float) -> None:
    if not values:
        return
    xs = jittered_xs(x_center, len(values), width)
    ax.plot(
        xs, values, "o", color=color, markersize=3.5, alpha=0.45,
        markeredgewidth=0, zorder=zorder,
    )


def point_range_by_category(
    ax,
    categories: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    *,
    raw_values: dict[str, list[float]] | None = None,
    means: list[float] | None = None,
    los: list[float] | None = None,
    his: list[float] | None = None,
    dot_colors: dict[str, str] | None = None,
) -> None:
    """Single-series mean ± 95% CI point-range, one point per category (no
    connecting line — categories here are unordered groups, not a swept
    variable). If raw_values is given, an underlying jittered dot cloud is
    drawn per category, in that category's (lighter) color."""
    xs = list(range(1, len(categories) + 1))
    if raw_values is not None:
        for x, cat in zip(xs, categories):
            draw_dot_cloud(ax, x, raw_values[cat], (dot_colors or colors)[cat], width=0.13, zorder=1)
        means, los, his = [], [], []
        for cat in categories:
            m, lo, hi = mean_ci95(raw_values[cat])
            means.append(m)
            los.append(lo)
            his.append(hi)
    lo_err = [m - lo for m, lo in zip(means, los)]
    hi_err = [hi - m for m, hi in zip(means, his)]
    for x, m, loe, hie, cat in zip(xs, means, lo_err, hi_err, categories):
        color = colors[cat]
        ax.errorbar(
            x, m, yerr=[[loe], [hie]],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.3, capsize=4, capthick=1.3, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3,
        )
    apply_category_ticklabels(ax, xs, [labels[c] for c in categories])


def bar_with_ci(
    ax,
    categories: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    *,
    raw_values: dict[str, list[float]] | None = None,
    means: list[float] | None = None,
    los: list[float] | None = None,
    his: list[float] | None = None,
    dot_colors: dict[str, str] | None = None,
) -> None:
    """Bar to the mean, with an optional jittered dot cloud of raw values
    layered on top of the bar, and the 95% CI drawn above both."""
    xs = list(range(1, len(categories) + 1))
    if raw_values is not None:
        means, los, his = [], [], []
        for cat in categories:
            m, lo, hi = mean_ci95(raw_values[cat])
            means.append(m)
            los.append(lo)
            his.append(hi)
    bar_colors = [colors[c] for c in categories]
    ax.bar(xs, means, width=0.6, color=bar_colors, alpha=0.85, edgecolor=INK, linewidth=0.9, zorder=2)
    if raw_values is not None:
        for x, cat in zip(xs, categories):
            draw_dot_cloud(ax, x, raw_values[cat], (dot_colors or colors)[cat], width=0.16, zorder=2.5)
    lo_err = [m - lo for m, lo in zip(means, los)]
    hi_err = [hi - m for m, hi in zip(means, his)]
    ax.errorbar(
        xs, means, yerr=[lo_err, hi_err],
        fmt="none", ecolor=INK, elinewidth=1.3, capsize=4, capthick=1.3, zorder=3,
    )
    apply_category_ticklabels(ax, xs, [labels[c] for c in categories])


def point_range_by_group(
    ax,
    groups: list,
    group_labels: list[str],
    series: list[str],
    values: dict,
    series_colors: dict,
    series_labels: dict,
    dot_colors: dict | None = None,
) -> None:
    """Mean line with 95% CI point-range per group, plus a jittered raw-value
    dot cloud per (group, series). All series share the same x position (no
    dodge) — overlap is expected, color differentiates. A thin line threads
    through each series' mean markers, drawn under the markers."""
    for s in series:
        color = series_colors[s]
        dot_color = (dot_colors or series_colors)[s]
        xs, means, lo_err, hi_err = [], [], [], []
        for i, g in enumerate(groups):
            vals = values.get((g, s), [])
            if not vals:
                continue
            x = i + 1
            draw_dot_cloud(ax, x, vals, dot_color, width=0.15, zorder=1)
            m, lo, hi = mean_ci95(vals)
            xs.append(x)
            means.append(m)
            lo_err.append(m - lo)
            hi_err.append(hi - m)
        if not xs:
            continue
        ax.plot(xs, means, "-", color=color, linewidth=1.5, zorder=2)
        ax.errorbar(
            xs, means, yerr=[lo_err, hi_err],
            fmt="o", color=color, ecolor=color,
            elinewidth=1.3, capsize=4, capthick=1.3, markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.7,
            label=series_labels[s], zorder=3,
        )
    apply_category_ticklabels(ax, list(range(1, len(groups) + 1)), group_labels)
    ax.legend(loc="best")


# -- Individual plots --------------------------------------------------


def plot_modality_time(trials: list[dict]) -> None:
    rows = [t for t in trials if t["experiment_condition"] == "modality" and t["elapsed_s"] is not None]
    if not rows:
        print("  modality_time: no data, skipping")
        return
    modalities = present_modalities(rows)
    raw_values = {m: [r["elapsed_s"] for r in rows if r["modality_id"] == m] for m in modalities}

    fig, ax = plt.subplots()
    point_range_by_category(
        ax, modalities, MODALITY_COLORS, MODALITY_LABELS,
        raw_values=raw_values, dot_colors=MODALITY_DOT_COLORS,
    )
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Modality: time to match by modality")
    style_axes(ax)
    save(fig, "modality_time.png")


def plot_modality_preference(preferences: list[dict]) -> None:
    if not preferences:
        print("  modality_preference: no data, skipping")
        return
    modalities = present_modalities(preferences)
    raw_values = {m: [p["rating"] for p in preferences if p["modality_id"] == m] for m in modalities}

    fig, ax = plt.subplots()
    bar_with_ci(
        ax, modalities, MODALITY_COLORS, MODALITY_LABELS,
        raw_values=raw_values, dot_colors=MODALITY_DOT_COLORS,
    )
    ax.set_ylim(0.7, 5.3)
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
    # Binary outcomes: no dot cloud (a 0/1 jitter carries no information),
    # just a clean point + Wilson 95% CI per modality.
    point_range_by_category(
        ax, modalities, MODALITY_COLORS, MODALITY_LABELS,
        means=means, los=los, his=his,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Modality: success rate by modality (mean ± 95% CI)")
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
        dot_colors=COMPARE_DOT_COLORS,
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
        dot_colors=COMPARE_DOT_COLORS,
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
        dot_colors=COMPARE_DOT_COLORS,
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
        dot_colors=COMPARE_DOT_COLORS,
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
