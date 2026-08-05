#!/usr/bin/env python3
"""Pool experiment.sqlite files under a folder and render summary plots.

Usage:
    python3 analysis/make_plots.py <folder>
    python3 analysis/make_plots.py <folder> --participant P3
    python3 analysis/make_plots.py <folder> --threshold 10 10
    python3 analysis/make_plots.py <folder> --participant P3 --threshold 10 10

<folder> is scanned recursively for files named experiment.sqlite (e.g. point
it at a data-category bin such as .../visualexperiment/real or .../practice).
Every trial is pooled across every file found, grouped by each session's
experiment_condition. Axis values (magnitudes, thresholds, latencies, trial
counts) are always read from the data itself, never hardcoded, so changes to
the study configuration do not silently break these plots.

--participant <ID> restricts the pool to sessions whose participant_id
matches <ID> (case-insensitive, exact match). Plot filenames get a
_<ID> suffix so they don't overwrite the pooled plots.

--threshold <linear_mm> <angular_deg> re-derives each trial's time-to-match
from its recorded trajectory under this threshold instead of the one the
trial actually ran with (see apply_threshold_override / derive_match).
Affects modality_time, noise_time, latency_time, precision_time,
learning_curve_time, and modality_success; modality_preference is unaffected.
Plot filenames get a _thr<linear>x<angular> suffix.

modality_figure.png is a combined publication figure (time / success /
preference stacked panels) and always renders at a hardcoded 10mm/10deg
re-derived threshold (see FIGURE_THRESHOLD), regardless of --threshold.

conditions_figure.png is a companion combined figure (noise / latency /
precision time-to-match panels, M3 vs M5, log-scaled magnitude axes). Unlike
every other plot in this file, it applies NO threshold re-derivation at all --
not FIGURE_THRESHOLD, and not --threshold either -- so it always reflects
elapsed_s/achieved exactly as recorded: noise/latency trials under the live
5mm/5deg rule they actually ran under, precision trials under each trial's
own per-trial threshold.

modality_summary.csv and conditions_summary.csv are self-documented (source
folders/files, threshold used, generation date in leading `#` comment lines)
per-row summary statistics for modality_figure.png / conditions_figure.png,
written alongside them in analysis/plots/. Both are computed by
modality_figure_stats() / conditions_figure_stats() -- the SAME functions the
figures themselves draw from -- so the CSVs and the figures can never
disagree; nothing is recomputed independently for the CSV.

learning_curve_individual_naive.png / learning_curve_individual_censored.png
plot each learning_curve participant's own points and an exponential-decay
fit (mu = c + a*exp(-b*trial)) per participant per panel (M3, M5). naive
treats timeouts as y=90 and fits OLS; censored fits a right-censored Gaussian
MLE instead (timeouts contribute P(time>90), not a y=90 point) -- see
fit_naive_exp_decay / fit_censored_exp_decay and
plot_learning_curve_individual's docstring for why the censored curve comes
out ABOVE naive at heavily-timed-out trials, not below.

learning_curve_averaged_naive.png / learning_curve_averaged_censored.png pool
ALL participants' trials into ONE fit per mode (M3, M5), both overlaid on a
single axes, with a 95% bootstrap confidence band (bootstrap_curve_band,
N_BOOT=500 resamples) around each fitted curve. Same naive-vs-censored
semantics as the individual figures above, just pooled instead of per-
participant -- see plot_learning_curve_averaged. *_exP1P2 variants of these
exclude participant_ids matching "P1"/"P2" (pattern match, not exact string
-- see learning_curve_individual_rows) and cap y_top at 220.

learning_curve_cumulative_naive_exP1P2.png / _censored_exP1P2.png are the
excluded-cohort pooled figures again, but with x = each participant's own
cumulative time-on-task (running sum of elapsed_s, including timeouts)
instead of trial number -- see learning_curve_cumulative_arrays /
plot_learning_curve_cumulative.

learning_curve_success.png: per-trial success rate (line + Wilson 95% CI
whisker, no fit, no band), both modes overlaid, same P1/P2 exclusion as the
other exP1P2 figures -- see plot_learning_curve_success.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import random
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cbook
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm

# mathtext.default="regular" makes $...$ mathtext segments (the learning-
# curve legend equations) render in the surrounding sans-serif font instead
# of mathtext's own default Computer Modern font, while keeping proper
# superscript layout for exponents. No other figure in this file uses
# mathtext, so this has no effect elsewhere.
matplotlib.rcParams["mathtext.default"] = "regular"

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


def darken_hsl(hex_color: str, factor: float) -> str:
    """Darken a hex color by scaling its HSL lightness by (1 - factor).

    Unlike darken() (a linear RGB blend toward black), this holds hue and
    saturation fixed -- a straight RGB blend shifts the hue of saturated
    colors (e.g. modality_figure's fills), an HSL lightness scale doesn't.
    Used for modality_figure's box/bar outlines, which are each fill's own
    darker shade rather than a flat black."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l * (1 - factor), s)
    return f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}"


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

# -- Combined manuscript figure (modality_figure.png) -----------------------
#
# A dedicated palette/ordering fixed by the manuscript figure spec: dimension
# grouped left-to-right (1D, then 2D transducer/user/patient, then 3D
# transducer/user/patient), independent of MODALITY_COLORS/MODALITY_LABELS
# above (those serve the individual per-metric plots and order by M-number).
FIGURE_ORDER = ["M1", "M4", "M7", "M2", "M5", "M3", "M6"]
FIGURE_LABELS = {
    "M1": "1D",
    "M4": "2D Transducer",
    "M2": "2D User",
    "M3": "2D Patient",
    "M7": "3D Transducer",
    "M5": "3D User",
    "M6": "3D Patient",
}
FIGURE_COLORS = {
    "M1": "#C4C4C4",  # 1D, grey
    "M4": "#A87FCE",  # 2D Transducer, purple (mid)
    "M7": "#7A3EB0",  # 3D Transducer, purple (darker)
    "M2": "#72A1E0",  # 2D User, blue (mid)
    "M5": "#2E6FCC",  # 3D User, blue (darker)
    "M3": "#70C598",  # 2D Patient, green (mid)
    "M6": "#2E9E63",  # 3D Patient, green (darker)
}
FIGURE_THRESHOLD = (10.0, 10.0)  # hardcoded for modality_figure.png, see plot_modality_figure
TIMEOUT_COLOR = "#E06666"  # muted red, close in visual weight to the black trial dots

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

# Continuous-hold duration required to register a match, mirroring
# core.HOLD_DURATION / study/activities.py TrialActivity. Not imported
# directly since core.py sits outside analysis/ on a different sys.path
# root; this script already re-derives everything else from the sqlite data
# rather than importing study code, so a mirrored constant fits that pattern.
HOLD_S = 1.0

# Valid ranges for CI clipping (see mean_ci95's bounds param): a t-based CI is
# a theoretical range and can extend past values the underlying quantity can
# never actually take. Trial timeout mirrors the 90s cap in index*.html's
# study UI; preference ratings are the fixed 1-5 scale (see
# experiment-session.js rating buttons).
TIME_BOUNDS = (0.0, 90.0)
PREFERENCE_BOUNDS = (1.0, 5.0)

# Deterministic horizontal jitter for dot clouds: a fixed seed keeps repeated
# runs over the same data visually stable instead of reshuffling each time.
_JITTER_RNG = random.Random(20260709)


def jittered_xs(x_center: float, n: int, width: float) -> list[float]:
    return [x_center + _JITTER_RNG.uniform(-width, width) for _ in range(n)]


# -- Data loading ------------------------------------------------------------


def find_sqlite_files(folder: Path) -> list[Path]:
    return sorted(folder.rglob("experiment.sqlite"))


# Data bins as laid out by study/storage.py's StorageConfig: {root}/{category}
# siblings, category in ("real", "practice", "trash"). This script otherwise
# treats <folder> as an opaque tree to rglob (see find_sqlite_files above),
# but conditions_figure.png specifically pools practice + real (never trash)
# regardless of which single folder the user pointed the CLI at.
DATA_BIN_NAMES = ("real", "practice", "trash")
CONDITIONS_FIGURE_BINS = ("practice", "real")


def conditions_figure_search_roots(folder: Path) -> list[Path]:
    """Folders to pool for conditions_figure.png's practice+real data.

    If `folder` IS itself a bin (its name matches one of DATA_BIN_NAMES,
    e.g. .../visualexperiment/real), its parent is treated as the data root
    -- so passing any one bin still pulls in both practice and real, not
    just the one named. Otherwise `folder` itself is treated as the data
    root. Only bins that actually exist on disk are returned; trash is never
    included."""
    root = folder.parent if folder.name in DATA_BIN_NAMES else folder
    return [p for name in CONDITIONS_FIGURE_BINS if (p := root / name).is_dir()]


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


# -- Threshold override (re-derive time-to-match from trajectories) --------
#
# --threshold re-plays each trial's recorded trajectory_samples against a
# different (linear_mm, angular_deg) pair than the one the trial actually ran
# with, reproducing the live continuous-hold rule (study/activities.py
# TrialActivity.step / trial.py Trial.step): a match is registered once the
# pose stays within tolerance for HOLD_S seconds straight.

TRAJECTORY_QUERY = """
SELECT run_id, trial_index, elapsed_s, linear_m, angular_deg
FROM trajectory_samples
ORDER BY run_id, trial_index, sample_index
"""


def load_trajectories(paths: list[Path]) -> dict[tuple[str, int], list[tuple[float, float, float]]]:
    """One query per file, grouped by (run_id, trial_index) in memory.

    run_id is a uuid4 (study/storage.py start_condition), so keys are unique
    across every file pooled, no source_file disambiguation needed. Rows are
    already ordered by sample_index (capture order) within each trial, which
    is what the derivation needs — elapsed_s spacing isn't perfectly uniform.
    """
    trajectories: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for path in paths:
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            with connection:
                for run_id, trial_index, elapsed_s, linear_m, angular_deg in connection.execute(TRAJECTORY_QUERY):
                    if elapsed_s is None or linear_m is None or angular_deg is None:
                        continue
                    trajectories[(run_id, trial_index)].append((elapsed_s, linear_m, angular_deg))
        except sqlite3.DatabaseError as exc:
            print(f"  ! skipping trajectories in {path}: {exc}")
        finally:
            connection.close()
    return trajectories


def derive_match(
    samples: list[tuple[float, float, float]],
    linear_mm: float,
    angular_deg_thr: float,
    hold_s: float = HOLD_S,
) -> tuple[float | None, bool]:
    """Re-derive time-to-match for one trial under an overridden threshold.

    samples: (elapsed_s, linear_m, angular_deg) triples in capture order.
    Finds the earliest T such that every sample with elapsed_s in
    [T, T+hold_s] is within (linear_mm, angular_deg_thr), i.e. the trajectory
    holds continuously through a full hold_s window starting at T. Returns
    (T + hold_s, True) — the moment the hold completes — for the earliest
    such T, or (None, False) if no window is confirmed by the recorded data
    (a derived timeout; the caller falls back to the trial's recorded
    elapsed_s).

    A window is only confirmed if a sample exists at or past its end — a
    window whose tail runs past the last recorded sample (e.g. because
    recording stopped when the *original*, looser threshold matched first)
    is not counted, since there's no data confirming the hold actually
    continued that far.
    """
    n = len(samples)
    if n == 0:
        return None, False

    ok = [lin_m * 1000.0 <= linear_mm and ang <= angular_deg_thr for _, lin_m, ang in samples]

    # next_bad[i]: index of the first sample at/after i that violates the
    # threshold, or n if it holds through the end of the trajectory.
    next_bad = [n] * (n + 1)
    for i in range(n - 1, -1, -1):
        next_bad[i] = i if not ok[i] else next_bad[i + 1]

    cover = 0  # first index with elapsed_s >= current window end; monotonic
    for i in range(n):
        if not ok[i]:
            continue
        end = samples[i][0] + hold_s
        while cover < n and samples[cover][0] < end:
            cover += 1
        if cover == n:
            # Trajectory doesn't reach this window's end, nor any later one
            # (elapsed_s only grows with i) -- no confirmable window exists.
            break
        if next_bad[i] == n or samples[next_bad[i]][0] > end:
            return end, True
    return None, False


def apply_threshold_override(
    trials: list[dict],
    trajectories: dict[tuple[str, int], list[tuple[float, float, float]]],
    linear_mm: float,
    angular_deg_thr: float,
) -> tuple[list[dict], int, int]:
    """Replace elapsed_s/achieved on every trial with threshold-derived
    values. Returns (derived_trials, n_newly_matched, n_newly_timed_out)
    relative to each trial's recorded outcome."""
    derived: list[dict] = []
    n_newly_matched = 0
    n_newly_timed_out = 0
    for t in trials:
        samples = trajectories.get((t["run_id"], t["trial_index"]), [])
        match_time, achieved = derive_match(samples, linear_mm, angular_deg_thr)
        record = dict(t)
        record["elapsed_s"] = match_time if achieved else t["elapsed_s"]
        record["achieved"] = achieved
        derived.append(record)
        if achieved and not t["achieved"]:
            n_newly_matched += 1
        elif not achieved and t["achieved"]:
            n_newly_timed_out += 1
    return derived, n_newly_matched, n_newly_timed_out


# -- Summary -------------------------------------------------------------


def print_summary(trials: list[dict], participant: str | None = None) -> None:
    if participant:
        print(f"\nSummary by experiment_condition (filtered to participant {participant}):")
    else:
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


def save(fig, name: str, suffix: str = "", tight: bool = True) -> None:
    """tight=False skips fig.tight_layout(), which otherwise recomputes (and
    silently overrides) any explicit fig.subplots_adjust() spacing -- needed
    by callers that hand-tune wspace/hspace themselves."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if suffix:
        stem, ext = name.rsplit(".", 1)
        name = f"{stem}_{suffix}.{ext}"
    out_path = PLOTS_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary_csv(
    name: str,
    meta_lines: list[str],
    header: list[str],
    rows: list[list],
    suffix: str = "",
) -> None:
    """Writes a summary CSV next to the figures in PLOTS_DIR, self-documented
    with `#`-prefixed metadata lines above the header row (source
    folders/files, threshold used, generated date -- see callers). Sits
    alongside save() as the CSV counterpart: same directory, same
    <stem>_<suffix>.<ext> naming."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if suffix:
        stem, ext = name.rsplit(".", 1)
        name = f"{stem}_{suffix}.{ext}"
    out_path = PLOTS_DIR / name
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        for line in meta_lines:
            handle.write(f"{line}\n")
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {out_path}")


def present_modalities(rows: list[dict]) -> list[str]:
    found = {r["modality_id"] for r in rows if r["modality_id"]}
    return [m for m in MODALITY_IDS if m in found]


def figure_modalities(rows: list[dict]) -> list[str]:
    found = {r["modality_id"] for r in rows if r["modality_id"]}
    return [m for m in FIGURE_ORDER if m in found]


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


def mean_ci95(
    values: list[float], bounds: tuple[float, float] | None = None
) -> tuple[float, float, float]:
    """t-based 95% CI of the mean. Returns (mean, lower, upper); lower==upper==mean if n==1.

    bounds, if given, clips (lower, upper) to [bounds[0], bounds[1]]. The
    t-based interval is a theoretical range and can extend past values that
    are physically impossible for the quantity being measured (e.g. negative
    time, a rating below 1) — this clips that away without touching the mean.

    The mean itself is NOT guaranteed to lie within bounds: bounds describe
    the idealized valid range, but individual recorded values can land just
    past it (e.g. a trial's timeout check is elapsed >= 90.0 against an
    unclamped elapsed = time.perf_counter() - start_time, so a timed-out
    trial's elapsed_s is routinely a hair over 90, not exactly 90 — see
    trial.py Trial.step). If every value in a category is such a near-miss
    timeout, the raw mean can itself exceed bounds[1]. Clipping lo/hi to
    bounds alone would then put the clipped bound on the wrong side of the
    mean (e.g. hi < m), producing a negative errorbar distance downstream.
    So lo/hi are also clamped against the mean after the bounds clip,
    guaranteeing lo <= m <= hi always: the CI degenerates to a point on
    whichever side has no room left within the valid range, which is the
    correct behavior rather than a crash.
    """
    n = len(values)
    m = mean(values)
    if n <= 1:
        lo = hi = m
    else:
        se = stdev(values) / sqrt(n)
        half = t_critical_975(n - 1) * se
        lo, hi = m - half, m + half
    if bounds is not None:
        lo = min(max(lo, bounds[0]), m)
        hi = max(min(hi, bounds[1]), m)
    return m, lo, hi


def wilson_ci95(successes: float, n: int) -> tuple[float, float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (phat, lower, upper).

    Unlike the t-based interval, Wilson's bounds are mathematically within
    [0, 1] for any phat in [0, 1] in exact arithmetic (center is a convex
    combination of phat and 0.5, and half <= center - 0 / 1 - center for all
    valid inputs) — but at phat=0 or phat=1, center and half are each derived
    from the same z**2/(2n) term via a different arithmetic path (one through
    sqrt), so floating-point rounding can make their difference land a hair
    outside [0, 1] (e.g. -2.8e-17 for n=7, k=0 — confirmed empirically) or, at
    phat=1 exactly (a 100% success rate, a routine real result for an easy
    condition), can round DOWN to 0.9999999999999999 — below phat itself
    (confirmed empirically: n=4, k=4). Both are clipped, and lo/hi are then
    also clamped against phat so lo <= phat <= hi always holds even after
    clipping — the same reasoning as mean_ci95's bounds clamp, and for the
    same reason: callers subtract (phat - lo) and (hi - phat) to build
    matplotlib errorbar distances, which raise on a negative result."""
    z = _Z_975
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    lo = min(max(0.0, center - half), phat)
    hi = max(min(1.0, center + half), phat)
    return phat, lo, hi


# -- Individual learning-curve exponential-decay fits ------------------------
#
# mu(trial) = c + a * exp(-b * trial): decays from c+a (trial 1) toward
# asymptote c as trial -> infinity, at rate b. Fit independently per
# participant per panel -- no shared/pooled structure across participants.

EXP_DECAY_BOUNDS = ([0.0, 0.0, 1e-6], [np.inf, np.inf, 2.0])  # c>=0, a>=0, 0<b<2


def exp_decay(trial, c, a, b):
    return c + a * np.exp(-b * np.asarray(trial, dtype=float))


def fit_naive_exp_decay(trial: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """OLS fit of exp_decay to (trial, y) -- y already has timeouts baked in
    at the cap (see plot_learning_curve_individual's naive branch), so this
    treats every point as a real observation. Returns None if curve_fit
    can't converge within EXP_DECAY_BOUNDS (e.g. too few/degenerate points)."""
    c0 = float(np.min(y))
    a0 = max(float(np.max(y) - c0), 1e-3)
    b0 = 0.3
    try:
        popt, _ = curve_fit(
            exp_decay, trial, y, p0=[c0, a0, b0],
            bounds=EXP_DECAY_BOUNDS, maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None
    return float(popt[0]), float(popt[1]), float(popt[2])


def censored_neg_log_likelihood(
    params: np.ndarray,
    trial_completed: np.ndarray,
    y_completed: np.ndarray,
    trial_censored: np.ndarray,
    cap: float,
) -> float:
    """-log likelihood under mu(trial)=c+a*exp(-b*trial), Gaussian residuals
    (sd sigma): completed trials contribute their exact density
    (norm.logpdf), right-censored trials (recorded time == cap, true time
    unknown but >= cap) contribute P(time > cap) = norm.logsf(cap, mu, sigma)
    instead of a point value -- so a censored trial pulls the fit toward
    "at least this long," not "exactly this long" the way naive's y=cap
    would."""
    c, a, b, sigma = params
    if sigma <= 0:
        return np.inf
    ll = 0.0
    if len(y_completed):
        mu_completed = exp_decay(trial_completed, c, a, b)
        ll += np.sum(norm.logpdf(y_completed, loc=mu_completed, scale=sigma))
    if len(trial_censored):
        mu_censored = exp_decay(trial_censored, c, a, b)
        ll += np.sum(norm.logsf(cap, loc=mu_censored, scale=sigma))
    return -ll


def fit_censored_exp_decay(
    trial_completed: np.ndarray,
    y_completed: np.ndarray,
    trial_censored: np.ndarray,
    cap: float = 90.0,
    init: tuple[float, float, float] | None = None,
) -> tuple[float, float, float, float] | None:
    """Maximum-likelihood fit of exp_decay under right-censoring at `cap`
    (see censored_neg_log_likelihood). `init`, if given, seeds (c, a, b) --
    e.g. from fit_naive_exp_decay on the completed-only points, a reasonable
    starting point before the censored trials' pull is added in. Returns
    None if the optimizer doesn't report success."""
    if init is not None:
        c0, a0, b0 = init
    elif len(y_completed):
        c0 = float(np.min(y_completed))
        a0 = max(float(np.max(y_completed) - c0), 1e-3)
        b0 = 0.3
    else:
        c0, a0, b0 = cap * 0.5, cap * 0.5, 0.3
    if len(y_completed) >= 2:
        resid0 = y_completed - exp_decay(trial_completed, c0, a0, b0)
        sigma0 = max(float(np.std(resid0)), 0.5)
    else:
        sigma0 = max(cap * 0.1, 0.5)
    result = minimize(
        censored_neg_log_likelihood, x0=[c0, a0, b0, sigma0],
        args=(trial_completed, y_completed, trial_censored, cap),
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None), (1e-6, 2.0), (1e-6, None)],
    )
    if not result.success:
        return None
    c, a, b, sigma = result.x
    return float(c), float(a), float(b), float(sigma)


def trial_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(trial_nums, achieved_flags, elapsed) arrays from a list of trial
    dicts -- shared by every learning-curve fit/plot path (per-participant
    and pooled) so there's one place that defines this mapping."""
    rows = sorted(rows, key=lambda r: r["trial_index"])
    trial_nums = np.array([r["trial_index"] + 1 for r in rows], dtype=float)
    achieved_flags = np.array([bool(r["achieved"]) for r in rows])
    elapsed = np.array([r["elapsed_s"] for r in rows], dtype=float)
    return trial_nums, achieved_flags, elapsed


def fit_curve_for_mode(
    trial_nums: np.ndarray, achieved_flags: np.ndarray, elapsed: np.ndarray, censored: bool
) -> tuple[float, ...] | None:
    """One fit call, naive or censored, shared by every caller (per-
    participant figures, pooled/averaged figures, bootstrap replicates, and
    the y_top pre-passes) so the fitting math lives in exactly one place.
    Returns (c,a,b) for naive, (c,a,b,sigma) for censored, or None if there
    isn't enough data / the fit doesn't converge. No printing here -- each
    caller knows its own context (which participant, which mode, which
    bootstrap replicate) and prints accordingly."""
    completed_mask = achieved_flags
    if not censored:
        if len(trial_nums) < 4:
            return None
        y_all = np.where(achieved_flags, elapsed, 90.0)
        return fit_naive_exp_decay(trial_nums, y_all)
    n_completed = int(completed_mask.sum())
    if n_completed < 4:
        return None
    naive_init = fit_naive_exp_decay(trial_nums[completed_mask], elapsed[completed_mask])
    return fit_censored_exp_decay(
        trial_nums[completed_mask], elapsed[completed_mask],
        trial_nums[~completed_mask], cap=90.0,
        init=naive_init if naive_init else None,
    )


N_BOOT = 500  # bootstrap replicates for the averaged learning-curve figures' 95% band
LEARNING_CURVE_BOOTSTRAP_SEED = 20260810  # fixed seed: reruns over the same data reproduce the same band


def bootstrap_curve_band(
    trial_nums: np.ndarray,
    achieved_flags: np.ndarray,
    elapsed: np.ndarray,
    censored: bool,
    smooth_trials: np.ndarray,
    n_boot: int = N_BOOT,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """95% bootstrap band for a POOLED exp_decay fit: resample the (trial,
    achieved, elapsed) triples WITH REPLACEMENT n_boot times, refit the SAME
    model (fit_curve_for_mode, naive or censored per `censored`) each time,
    evaluate over smooth_trials, and take the 2.5th/97.5th percentile across
    converged replicates at each point.

    Replicates that fail to converge (fit_curve_for_mode returns None -- can
    happen if an unlucky resample drops below the 4-point minimum, or the
    optimizer just doesn't converge on that resample) are skipped, not
    counted toward the band. Returns (None, None, n_converged) if fewer than
    half the replicates converge -- the caller decides whether to still show
    the point-estimate curve without a band in that case.

    Returns (lo, hi, n_converged)."""
    n = len(trial_nums)
    rng = np.random.default_rng(LEARNING_CURVE_BOOTSTRAP_SEED)
    curves = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_curve_for_mode(trial_nums[idx], achieved_flags[idx], elapsed[idx], censored)
        if fit is None:
            continue
        curves.append(exp_decay(smooth_trials, *fit[:3]))
    if len(curves) < n_boot * 0.5:
        return None, None, len(curves)
    arr = np.array(curves)
    return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0), len(curves)


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
    bounds: tuple[float, float] | None = None,
) -> None:
    """Single-series mean ± 95% CI point-range, one point per category (no
    connecting line — categories here are unordered groups, not a swept
    variable). If raw_values is given, an underlying jittered dot cloud is
    drawn per category, in that category's (lighter) color. bounds, when
    raw_values is given, clips each category's CI to a physically valid range
    (see mean_ci95); ignored when means/los/his are passed in precomputed
    (e.g. Wilson CIs, already bounded)."""
    xs = list(range(1, len(categories) + 1))
    if raw_values is not None:
        for x, cat in zip(xs, categories):
            draw_dot_cloud(ax, x, raw_values[cat], (dot_colors or colors)[cat], width=0.13, zorder=1)
        means, los, his = [], [], []
        for cat in categories:
            m, lo, hi = mean_ci95(raw_values[cat], bounds=bounds)
            means.append(m)
            los.append(lo)
            his.append(hi)
    lo_err = [m - lo for m, lo in zip(means, los)]
    hi_err = [hi - m for m, hi in zip(means, his)]
    for x, m, loe, hie, cat in zip(xs, means, lo_err, hi_err, categories):
        color = colors[cat]
        outline = darken_hsl(color, 0.40)
        ax.errorbar(
            x, m, yerr=[[loe], [hie]],
            fmt="o", color=color, ecolor=outline,
            elinewidth=1.3, capsize=4, capthick=1.3, markersize=7,
            markeredgecolor=outline, markeredgewidth=0.8, zorder=3,
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
    bounds: tuple[float, float] | None = None,
) -> None:
    """Bar to the mean, with an optional jittered dot cloud of raw values
    layered on top of the bar, and the 95% CI drawn above both. bounds clips
    each category's CI to a physically valid range (see mean_ci95)."""
    xs = list(range(1, len(categories) + 1))
    if raw_values is not None:
        means, los, his = [], [], []
        for cat in categories:
            m, lo, hi = mean_ci95(raw_values[cat], bounds=bounds)
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
    # One errorbar call per category (not one call across all xs): ecolor
    # can't take a per-point color list once capsize>0 -- the caps are drawn
    # as markers sharing a single markeredgecolor, so a list raises.
    for x, m, loe, hie, cat in zip(xs, means, lo_err, hi_err, categories):
        outline = darken_hsl(colors[cat], 0.40)
        ax.errorbar(
            x, m, yerr=[[loe], [hie]],
            fmt="none", ecolor=outline, elinewidth=1.3, capsize=4, capthick=1.3, zorder=3,
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
    bounds: tuple[float, float] | None = None,
    show_ci: bool = True,
) -> None:
    """Mean line with (by default) a 95% CI point-range per group, plus a
    jittered raw-value dot cloud per (group, series). All series share the
    same x position (no dodge) — overlap is expected, color differentiates. A
    thin line threads through each series' mean markers, drawn under the
    markers. bounds clips each CI to a physically valid range (see
    mean_ci95); ignored when show_ci=False. show_ci=False draws just the mean
    markers (no whiskers) — same line/dot-cloud otherwise."""
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
            xs.append(x)
            if show_ci:
                m, lo, hi = mean_ci95(vals, bounds=bounds)
                lo_err.append(m - lo)
                hi_err.append(hi - m)
            else:
                m = mean(vals)
            means.append(m)
        if not xs:
            continue
        ax.plot(xs, means, "-", color=color, linewidth=1.5, zorder=2)
        if show_ci:
            outline = darken_hsl(color, 0.40)
            ax.errorbar(
                xs, means, yerr=[lo_err, hi_err],
                fmt="o", color=color, ecolor=outline,
                elinewidth=1.3, capsize=4, capthick=1.3, markersize=6.5,
                markeredgecolor=outline, markeredgewidth=0.7,
                label=series_labels[s], zorder=3,
            )
        else:
            ax.plot(
                xs, means, "o", color=color, markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.7,
                label=series_labels[s], zorder=3,
            )
    apply_category_ticklabels(ax, list(range(1, len(groups) + 1)), group_labels)
    ax.legend(loc="best")


# -- Individual plots --------------------------------------------------


def plot_modality_time(trials: list[dict], suffix: str = "") -> None:
    rows = [t for t in trials if t["experiment_condition"] == "modality" and t["elapsed_s"] is not None]
    if not rows:
        print("  modality_time: no data, skipping")
        return
    modalities = present_modalities(rows)
    raw_values = {m: [r["elapsed_s"] for r in rows if r["modality_id"] == m] for m in modalities}

    fig, ax = plt.subplots()
    point_range_by_category(
        ax, modalities, MODALITY_COLORS, MODALITY_LABELS,
        raw_values=raw_values, dot_colors=MODALITY_DOT_COLORS, bounds=TIME_BOUNDS,
    )
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Modality: time to match by modality")
    style_axes(ax)
    save(fig, "modality_time.png", suffix)


def plot_modality_preference(preferences: list[dict], suffix: str = "") -> None:
    if not preferences:
        print("  modality_preference: no data, skipping")
        return
    modalities = present_modalities(preferences)
    raw_values = {m: [p["rating"] for p in preferences if p["modality_id"] == m] for m in modalities}

    fig, ax = plt.subplots()
    bar_with_ci(
        ax, modalities, MODALITY_COLORS, MODALITY_LABELS,
        raw_values=raw_values, dot_colors=MODALITY_DOT_COLORS, bounds=PREFERENCE_BOUNDS,
    )
    ax.set_ylim(0.7, 5.3)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Preference rating (1-5)")
    ax.set_title("Modality: preference rating by modality (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "modality_preference.png", suffix)


def plot_modality_success(trials: list[dict], suffix: str = "") -> None:
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
    save(fig, "modality_success.png", suffix)


def modality_figure_stats(trials: list[dict], preferences: list[dict]) -> list[dict]:
    """Per-modality summary stats for modality_figure.png, in FIGURE_ORDER.

    This is the SINGLE source of the numbers plot_modality_figure draws (box
    quartiles, success bar+CI, preference bar+CI) -- it also backs
    modality_summary.csv, so the figure and the CSV can never drift: both
    read the same dict this function returns, nothing is recomputed
    independently. `trials` must already be re-derived at FIGURE_THRESHOLD by
    the caller (see main()); this function does not re-derive anything.

    time_median/time_q1/time_q3 come from matplotlib.cbook.boxplot_stats
    (whis=1.5) -- the exact function ax.boxplot() itself calls internally,
    so these are guaranteed to equal the rendered box edges, not a
    similar-but-different percentile method. time_min/time_max are each
    modality's true data extremes (every trial is plotted as a dot
    regardless of whisker range, so those extremes are visible in the
    figure even though showfliers=False suppresses matplotlib's own flier
    markers for anything past the whiskers).

    pref_mean/pref_ci95_lo/pref_ci95_hi fall back to 1.0 when a modality has
    zero ratings, matching plot_modality_figure's own bar-height fallback
    for that case (a rendering placeholder, not a real statistic) --
    pref_median/pref_min/pref_max stay None since there's nothing to compute
    them from."""
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "modality"
        and t["elapsed_s"] is not None
        and t["achieved"] is not None
    ]
    if not rows:
        return []
    modalities = figure_modalities(rows)
    stats = []
    for m in modalities:
        mrows = [r for r in rows if r["modality_id"] == m]
        times = [r["elapsed_s"] for r in mrows]
        achieved_flags = [bool(r["achieved"]) for r in mrows]
        participants = {r["participant_id"] for r in mrows if r["participant_id"]}

        time_mean, time_lo, time_hi = mean_ci95(times, bounds=TIME_BOUNDS)
        time_sd = stdev(times) if len(times) > 1 else 0.0
        box = matplotlib.cbook.boxplot_stats(times, whis=1.5)[0]

        successes = sum(achieved_flags)
        n_outcomes = len(achieved_flags)
        phat, succ_lo, succ_hi = wilson_ci95(successes, n_outcomes)

        ratings = [p["rating"] for p in preferences if p["modality_id"] == m]
        if ratings:
            pref_mean, pref_lo, pref_hi = mean_ci95(ratings, bounds=PREFERENCE_BOUNDS)
            pref_sd = stdev(ratings) if len(ratings) > 1 else 0.0
            pref_median, pref_min, pref_max = median(ratings), min(ratings), max(ratings)
        else:
            pref_mean = pref_lo = pref_hi = 1.0  # mirrors the figure's zero-data bar-height fallback
            pref_sd = pref_median = pref_min = pref_max = None

        stats.append(dict(
            modality_id=m,
            modality_label=FIGURE_LABELS[m],
            times=times,
            achieved_flags=achieved_flags,
            n_trials=len(times),
            n_participants=len(participants),
            time_mean=time_mean, time_sd=time_sd,
            time_ci95_lo=time_lo, time_ci95_hi=time_hi,
            time_median=box["med"], time_q1=box["q1"], time_q3=box["q3"],
            time_min=min(times), time_max=max(times),
            n_timeouts=n_outcomes - successes,
            success_k=successes, success_n=n_outcomes, success_rate=phat,
            success_ci95_lo=succ_lo, success_ci95_hi=succ_hi,
            pref_n=len(ratings), pref_mean=pref_mean, pref_sd=pref_sd,
            pref_ci95_lo=pref_lo, pref_ci95_hi=pref_hi,
            pref_median=pref_median, pref_min=pref_min, pref_max=pref_max,
        ))
    return stats


MODALITY_SUMMARY_HEADER = [
    "modality_label", "modality_id", "n_trials", "n_participants",
    "time_mean", "time_sd", "time_ci95_lo", "time_ci95_hi",
    "time_median", "time_q1", "time_q3", "time_min", "time_max", "n_timeouts",
    "success_k", "success_n", "success_rate", "success_ci95_lo", "success_ci95_hi",
    "pref_n", "pref_mean", "pref_sd", "pref_ci95_lo", "pref_ci95_hi",
    "pref_median", "pref_min", "pref_max",
]


def write_modality_summary_csv(stats: list[dict], meta: dict, suffix: str = "") -> None:
    if not stats:
        print("  modality_summary: no data, skipping")
        return
    paths = meta["paths"]
    meta_lines = [
        "# modality_summary.csv -- per-modality summary statistics backing modality_figure.png",
        "# every value here is read from the exact same computation modality_figure.png draws from",
        "# (see modality_figure_stats in analysis/make_plots.py); the two can never disagree.",
        f"# generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"# source folder: {meta['folder']}",
        f"# source .sqlite files ({len(paths)}):",
        *(f"#   {p}" for p in paths),
        f"# participant filter: {meta.get('participant') or 'all'}",
        "# threshold: re-derived @ 10mm/10deg (FIGURE_THRESHOLD), matching modality_figure.png",
        "# time_median/time_q1/time_q3 are the exact box-plot quartiles rendered in the figure "
        "(matplotlib boxplot_stats, whis=1.5); time_min/time_max are each modality's raw per-trial extremes",
    ]
    rows = [[row[col] for col in MODALITY_SUMMARY_HEADER] for row in stats]
    write_summary_csv("modality_summary.csv", meta_lines, MODALITY_SUMMARY_HEADER, rows, suffix)


def plot_modality_figure(
    trials: list[dict], preferences: list[dict], suffix: str = "", meta: dict | None = None
) -> None:
    """Combined publication figure: time (box+dots) / success / preference,
    three panels side by side on a dimension-grouped x-axis (each panel keeps
    its own x tick labels since they no longer share an axis). `trials` must
    already be re-derived at FIGURE_THRESHOLD by the caller (see main()) —
    this function does not re-derive anything itself, it only renders.

    `meta`, if given, also writes modality_summary.csv from the exact same
    stats used to draw (see modality_figure_stats) -- pass
    {"paths": [...], "folder": Path(...), "participant": str | None}."""
    stats = modality_figure_stats(trials, preferences)
    if not stats:
        print("  modality_figure: no data, skipping")
        if meta is not None:
            print("  modality_summary: no data, skipping")
        return
    modalities = [row["modality_id"] for row in stats]
    xs = list(range(1, len(modalities) + 1))
    colors = [FIGURE_COLORS[m] for m in modalities]
    outline_colors = [darken_hsl(FIGURE_COLORS[m], 0.40) for m in modalities]
    tick_labels = [FIGURE_LABELS[m] for m in modalities]

    # figsize widened so the extra wspace set below (see the tight_layout
    # call near the end) comes out of new space, not the panels themselves
    # getting narrower.
    fig, (ax_time, ax_success, ax_pref) = plt.subplots(1, 3, figsize=(15, 5.5))

    # -- Panel A: time to match (conventional box-and-whisker + raw dots) --
    time_data = [row["times"] for row in stats]
    bp = ax_time.boxplot(
        time_data, positions=xs, widths=0.6, whis=1.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(linewidth=1.6),
        boxprops=dict(linewidth=1.6),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
        zorder=2,
    )
    # Each box's outline/whiskers/caps/median take a darker shade of that
    # category's own fill instead of a flat black -- one box per category,
    # but 2 whiskers and 2 caps per box (both ends), 1 median each.
    for i, (patch, m) in enumerate(zip(bp["boxes"], modalities)):
        patch.set_facecolor(FIGURE_COLORS[m])
        patch.set_alpha(1.0)
        patch.set_edgecolor(outline_colors[i])
        bp["medians"][i].set_color(outline_colors[i])
        for whisker in bp["whiskers"][2 * i : 2 * i + 2]:
            whisker.set_color(outline_colors[i])
        for cap in bp["caps"][2 * i : 2 * i + 2]:
            cap.set_color(outline_colors[i])

    any_timeout = False
    for x, row in zip(xs, stats):
        matched = [t for t, ok in zip(row["times"], row["achieved_flags"]) if ok]
        timed_out = [t for t, ok in zip(row["times"], row["achieved_flags"]) if not ok]
        if matched:
            jx = jittered_xs(x, len(matched), 0.12)
            ax_time.plot(jx, matched, "o", color="black", markersize=5, alpha=0.45, markeredgewidth=0, zorder=3)
        if timed_out:
            any_timeout = True
            jx = jittered_xs(x, len(timed_out), 0.12)
            # Timed-out trials sit right at (or a hair past) the 90s cap, i.e.
            # the axis top below -- clip_on=False keeps them from being
            # half-cut by the axis edge instead of drawn in full.
            ax_time.plot(
                jx, timed_out, "o", color=TIMEOUT_COLOR, markersize=5, alpha=0.35, markeredgewidth=0,
                zorder=3, clip_on=False,
            )
    ax_time.axhline(90, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax_time.set_ylim(0, 94.5)  # 5% headroom above the 90s cap, nothing clipped
    time_handle = Line2D(
        [], [], marker="o", color="black", linestyle="None",
        markersize=7, alpha=0.45, label="Time to match",
    )
    handles = [time_handle]
    if any_timeout:
        handles.append(Line2D(
            [], [], marker="o", color=TIMEOUT_COLOR, linestyle="None",
            markersize=7, alpha=0.35, label="Timeout",
        ))
    # Bottom-left of this panel always has box/dot data near it (every
    # category's bulk sits low), so the legend needs its own opaque backing
    # to stay legible instead of blending into that clutter.
    ax_time.legend(
        handles=handles, loc="lower left", fontsize=12, numpoints=1, markerscale=1.0,
        frameon=True, framealpha=0.9, facecolor="white", edgecolor="none",
    )
    ax_time.set_ylabel("Time to match (s)", fontsize=15)
    ax_time.set_xticks(xs)
    ax_time.set_xticklabels(tick_labels, rotation=90, fontsize=13)
    style_axes(ax_time)

    # -- Panel B: success rate (bar to mean + Wilson 95% CI) --
    means = [row["success_rate"] for row in stats]
    los = [row["success_ci95_lo"] for row in stats]
    his = [row["success_ci95_hi"] for row in stats]
    ax_success.bar(xs, means, width=0.6, color=colors, alpha=1.0, edgecolor=outline_colors, linewidth=1.6, zorder=2)
    # One errorbar call per category (not one call across all xs): ecolor
    # can't take a per-point color list once capsize>0 -- the caps are drawn
    # as markers sharing a single markeredgecolor, so a list raises.
    for x, m, lo, hi, outline in zip(xs, means, los, his, outline_colors):
        ax_success.errorbar(
            x, m, yerr=[[m - lo], [hi - m]],
            fmt="none", ecolor=outline, elinewidth=1.8, capsize=5, capthick=1.8, zorder=4,
        )
    ax_success.set_ylim(0, 1.05)  # 5% headroom above 1.0 so no CI cap gets clipped
    ax_success.set_ylabel("Success rate", fontsize=15)
    ax_success.set_xticks(xs)
    ax_success.set_xticklabels(tick_labels, rotation=90, fontsize=13)
    style_axes(ax_success)

    # -- Panel C: preference (bar to mean + 95% CI) --
    means = [row["pref_mean"] for row in stats]
    los = [row["pref_ci95_lo"] for row in stats]
    his = [row["pref_ci95_hi"] for row in stats]
    ax_pref.bar(xs, means, width=0.6, color=colors, alpha=1.0, edgecolor=outline_colors, linewidth=1.6, zorder=2)
    for x, m, lo, hi, outline in zip(xs, means, los, his, outline_colors):
        ax_pref.errorbar(
            x, m, yerr=[[m - lo], [hi - m]],
            fmt="none", ecolor=outline, elinewidth=1.8, capsize=5, capthick=1.8, zorder=4,
        )
    ax_pref.set_ylim(0, 5.25)  # 5% headroom above 5 so no CI cap gets clipped
    ax_pref.set_yticks([1, 2, 3, 4, 5])
    ax_pref.set_ylabel("Preference (1 = worst, 5 = best)", fontsize=15)
    ax_pref.set_xticks(xs)
    ax_pref.set_xticklabels(tick_labels, rotation=90, fontsize=13)
    style_axes(ax_pref)

    for ax in (ax_time, ax_success, ax_pref):
        ax.tick_params(axis="y", labelsize=13)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)

    # tight_layout first, to size the outer margins around the rotated tick
    # labels; subplots_adjust afterward only widens wspace (roughly double
    # matplotlib's default ~0.2) without disturbing those margins -- calling
    # save()'s own tight_layout (tight=True) would recompute wspace itself
    # and silently undo this.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    save(fig, "modality_figure.png", suffix, tight=False)

    if meta is not None:
        write_modality_summary_csv(stats, meta, suffix)


CONDITIONS_MAGNITUDE_KEYS = {
    "noise": "noise",
    "latency": "perceived_ms",
    "precision": "precision_linear_mm",
}


def conditions_figure_rows(trials: list[dict], experiment: str, magnitude_key: str) -> list[dict]:
    return [
        t for t in trials
        if t["experiment_condition"] == experiment
        and t["modality_id"] in COMPARE_MODES
        and t[magnitude_key] is not None
        and t["elapsed_s"] is not None
    ]


def conditions_figure_stats(trials: list[dict]) -> dict[str, list[dict]]:
    """Per-(experiment, series, x-value) summary stats for
    conditions_figure.png, in the figure's own x-order (ascending magnitude;
    for precision that's tightest-to-loosest threshold since it's sorted by
    precision_linear_mm ascending).

    This is the SINGLE source of the mean+CI numbers plot_conditions_figure
    draws -- it also backs conditions_summary.csv, so the two can never
    drift: both read the same list of dicts this function returns. `trials`
    must be the RAW, non-threshold-derived, practice+real-pooled trial list
    (see main() / conditions_figure_search_roots) -- this function does not
    re-derive or re-pool anything.

    Returns {"noise": [...], "latency": [...], "precision": [...]}; an
    experiment's list is empty if it has no data (caller decides how to
    report that)."""
    result: dict[str, list[dict]] = {}
    for experiment, magnitude_key in CONDITIONS_MAGNITUDE_KEYS.items():
        rows = conditions_figure_rows(trials, experiment, magnitude_key)
        if not rows:
            result[experiment] = []
            continue
        magnitude_values = sorted({r[magnitude_key] for r in rows})
        entries = []
        for s in COMPARE_MODES:
            for xv in magnitude_values:
                srows = [r for r in rows if r[magnitude_key] == xv and r["modality_id"] == s]
                if not srows:
                    continue
                times = [r["elapsed_s"] for r in srows]
                achieved_flags = [r["achieved"] for r in srows if r["achieved"] is not None]
                participants = {r["participant_id"] for r in srows if r["participant_id"]}
                time_mean, time_lo, time_hi = mean_ci95(times, bounds=TIME_BOUNDS)
                time_sd = stdev(times) if len(times) > 1 else 0.0
                box = matplotlib.cbook.boxplot_stats(times, whis=1.5)[0]
                entries.append(dict(
                    experiment=experiment,
                    modality_id=s,
                    series_label=FIGURE_LABELS[s],
                    x_value=xv,
                    x_label=f"{xv:g}",
                    n_trials=len(times),
                    n_participants=len(participants),
                    time_mean=time_mean, time_sd=time_sd,
                    time_ci95_lo=time_lo, time_ci95_hi=time_hi,
                    time_median=box["med"], time_q1=box["q1"], time_q3=box["q3"],
                    time_min=min(times), time_max=max(times),
                    n_timeouts=sum(1 for ok in achieved_flags if not ok),
                ))
        result[experiment] = entries
    return result


CONDITIONS_SUMMARY_HEADER = [
    "experiment", "series_label", "modality_id", "x_value", "x_label",
    "n_trials", "n_participants",
    "time_mean", "time_sd", "time_ci95_lo", "time_ci95_hi",
    "time_median", "time_q1", "time_q3", "time_min", "time_max", "n_timeouts",
]


def write_conditions_summary_csv(stats: dict[str, list[dict]], meta: dict, suffix: str = "") -> None:
    missing = [experiment for experiment, entries in stats.items() if not entries]
    present = [experiment for experiment in CONDITIONS_MAGNITUDE_KEYS if stats.get(experiment)]
    if not present:
        print(f"  conditions_summary: no data for {', '.join(missing)}, skipping")
        return
    if missing:
        print(f"  conditions_summary: no data for {', '.join(missing)}, writing remaining experiments only")
    roots = meta["roots"]
    paths = meta["paths"]
    meta_lines = [
        "# conditions_summary.csv -- per (experiment, series, x-value) summary statistics backing conditions_figure.png",
        "# every value here is read from the exact same computation conditions_figure.png draws from",
        "# (see conditions_figure_stats in analysis/make_plots.py); the two can never disagree.",
        f"# generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "# data pooled from practice + real bins (never trash):",
        *([f"#   root: {r}" for r in roots] if roots else ["#   (no practice/real bin found)"]),
        f"# source .sqlite files ({len(paths)}):",
        *(f"#   {p}" for p in paths),
        f"# participant filter: {meta.get('participant') or 'all'}",
        "# threshold: recorded as-run, NOT re-derived -- noise/latency reflect the live 5mm/5deg matching",
        "# rule those trials actually ran under; precision reflects each trial's own per-trial threshold",
    ]
    rows = [
        [row[col] for col in CONDITIONS_SUMMARY_HEADER]
        for experiment in CONDITIONS_MAGNITUDE_KEYS
        for row in stats.get(experiment, [])
    ]
    write_summary_csv("conditions_summary.csv", meta_lines, CONDITIONS_SUMMARY_HEADER, rows, suffix)


def plot_conditions_figure(trials: list[dict], suffix: str = "", meta: dict | None = None) -> None:
    """Companion to plot_modality_figure: noise / latency / precision
    time-to-match, each panel comparing M3 (2D/Patient) vs M5 (3D/User) on a
    log-scaled x-axis of the condition's own magnitude. Same size, rcParams,
    line weights, text sizes, headroom convention, and no-title style as
    plot_modality_figure -- see that function's comments for why tight_layout
    and subplots_adjust are sequenced the way they are below.

    `trials` must be the RAW, non-threshold-derived trial list, ALREADY
    pooled from the practice + real data bins by main() (see
    conditions_figure_search_roots) -- this is the one plot in the file that
    (a) draws from both bins regardless of which single folder the CLI was
    pointed at, and (b) intentionally skips both FIGURE_THRESHOLD and any
    --threshold override, so noise/latency reflect the live 5mm/5deg rule
    they actually ran under and precision reflects each trial's own
    per-trial threshold, exactly as recorded.

    `meta`, if given, also writes conditions_summary.csv from the exact same
    stats used to draw (see conditions_figure_stats) -- pass
    {"roots": [...], "paths": [...], "participant": str | None}."""
    stats = conditions_figure_stats(trials)
    missing = [experiment for experiment, entries in stats.items() if not entries]
    if missing:
        print(f"  conditions_figure: no data for {', '.join(missing)}, skipping")
        if meta is not None:
            write_conditions_summary_csv(stats, meta, suffix)
        return

    # M3 (2D/Patient) is a complete layer below M5 (3D/User) -- explicit,
    # distinct zorder per series on EVERY artist (connecting line, marker,
    # CI line, caps), not reliance on draw-call order. ax.errorbar's zorder
    # kwarg applies uniformly to its dataline/caplines/barlinecollection (all
    # get the same zorder; the dataline gets +0.1 internally so the marker
    # itself sits a hair above its own whiskers) -- confirmed by inspecting
    # the returned ErrorbarContainer.
    series_markers = {"M3": "^", "M5": "s"}
    series_zorder = {"M3": 2, "M5": 3}

    def draw_panel(ax, entries):
        for s in COMPARE_MODES:
            series_entries = [e for e in entries if e["modality_id"] == s]
            if not series_entries:
                continue
            color = FIGURE_COLORS[s]
            z = series_zorder[s]
            xs_ser = [e["x_value"] for e in series_entries]
            means = [e["time_mean"] for e in series_entries]
            lo_err = [e["time_mean"] - e["time_ci95_lo"] for e in series_entries]
            hi_err = [e["time_ci95_hi"] - e["time_mean"] for e in series_entries]
            ax.plot(xs_ser, means, "-", color=color, linewidth=1.5, zorder=z)
            # Single light fill color everywhere for this series: no
            # darkened outline (markeredgecolor=color, matching the fill) and
            # no darkened CI (ecolor=color too). markeredgewidth must still
            # be kept non-zero -- errorbar() reuses this ONE property for
            # both the marker's own edge thickness AND the cap tick
            # thickness (confirmed by inspection: an explicit
            # markeredgewidth always wins over capthick when both are
            # given), so markeredgewidth=0 would silently kill the caps
            # even though the marker itself would look fine (a
            # same-color edge is invisible either way).
            ax.errorbar(
                xs_ser, means, yerr=[lo_err, hi_err],
                fmt=series_markers[s], color=color, ecolor=color,
                elinewidth=1.8, capsize=5, capthick=1.8, markersize=10,
                markeredgecolor=color, markeredgewidth=1.8, zorder=z,
            )
        ax.set_ylim(0, 80)  # y=90 timeout line no longer fits this range, so it's dropped below
        ax.set_ylabel("Time to match (s)", fontsize=15)

    fig, (ax_noise, ax_latency, ax_precision) = plt.subplots(1, 3, figsize=(15, 5.5))

    noise_values = sorted({e["x_value"] for e in stats["noise"]})
    draw_panel(ax_noise, stats["noise"])
    ax_noise.set_xticks(noise_values)
    ax_noise.set_xticklabels([f"{v:g}" for v in noise_values])
    ax_noise.set_xlabel("Noise (mm & deg)", fontsize=15)

    latency_values = sorted({e["x_value"] for e in stats["latency"]})
    draw_panel(ax_latency, stats["latency"])
    ax_latency.set_xticks(latency_values)
    ax_latency.set_xticklabels([f"{v:g}" for v in latency_values])
    ax_latency.set_xlabel("Latency (ms)", fontsize=15)

    # Ascending sort already puts the tightest threshold (smallest mm) on the
    # left and loosest (largest mm) on the right -- no reversal needed.
    precision_values = sorted({e["x_value"] for e in stats["precision"]})
    draw_panel(ax_precision, stats["precision"])
    ax_precision.set_xticks(precision_values)
    ax_precision.set_xticklabels([f"{v:g}" for v in precision_values])
    ax_precision.set_xlabel("Precision threshold (mm & deg)", fontsize=15)

    for ax in (ax_noise, ax_latency, ax_precision):
        ax.tick_params(axis="both", labelsize=13)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)
        style_axes(ax)

    # Explicit proxy handles, not the errorbar artists themselves -- an
    # errorbar's legend entry otherwise reuses its connecting line, drawing a
    # line through the marker. linestyle="none" gives a clean marker-only
    # legend. Figure-level legend (not tied to one axes) anchored at the
    # figure's own top-right corner, clear of every panel's data.
    legend_handles = [
        Line2D(
            [], [], marker=series_markers[s], color=FIGURE_COLORS[s], linestyle="none",
            markersize=10, markeredgecolor=FIGURE_COLORS[s], markeredgewidth=1.8,
            label=FIGURE_LABELS[s],
        )
        for s in COMPARE_MODES
    ]
    fig.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.97),
        fontsize=12, numpoints=1,
    )

    # Same tight_layout-then-subplots_adjust sequencing and wspace as
    # plot_modality_figure, so the two figures share consistent proportions.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)
    save(fig, "conditions_figure.png", suffix, tight=False)

    if meta is not None:
        write_conditions_summary_csv(stats, meta, suffix)


def plot_learning_curve_time(trials: list[dict], suffix: str = "") -> None:
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
        show_ci=False,
    )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Learning curve: time per trial (mean)")
    style_axes(ax)
    save(fig, "learning_curve_time.png", suffix)


def learning_curve_individual_rows(
    trials: list[dict], modality_id: str, exclude_participants: list[str] | None = None
) -> list[dict]:
    """exclude_participants tokens are matched as a PREFIX pattern, not an
    exact string: a token T excludes any participant_id that, case-
    insensitively, equals T or starts with T followed by a non-digit
    separator -- regex ^T($|[^0-9]) per token. This lets a short token like
    "P1" exclude real ids such as "P1-REAL" or "P1_pilot" without also
    matching "P10"/"P11"/... (the character right after "P1" in those is
    itself a digit, which [^0-9] rejects). See plot_learning_curve_averaged's
    exP1P2 variants, which report the literal ids that actually matched so
    it's obvious whether the exclusion landed on real data."""
    exclude_patterns = [
        re.compile(rf"^{re.escape(token)}($|[^0-9])", re.IGNORECASE)
        for token in (exclude_participants or [])
    ]
    return [
        t for t in trials
        if t["experiment_condition"] == "learning_curve"
        and t["modality_id"] == modality_id
        and t["elapsed_s"] is not None
        and t["achieved"] is not None
        and t["participant_id"]
        and not any(p.match(t["participant_id"]) for p in exclude_patterns)
    ]


def learning_curve_censored_y_top(
    trials: list[dict], headroom: float = 1.05, fallback: float = 94.5, cap: float = 180.0
) -> float:
    """Shared y-axis top for BOTH learning-curve figures: 5% headroom above
    the highest point any participant's CENSORED fitted curve reaches (over
    trials 1..that panel's max), across both panels -- the censored curves
    are the ones that can exceed the raw data / the 90s cap, so they set the
    scale for both plots (see plot_learning_curve_individual, which is
    passed this value for its y_top so naive and censored render on an
    identical scale). Capped at `cap` so one extreme participant's fit can't
    squash everyone else's curves down to near-flat lines.

    This re-runs the same fit_censored_exp_decay call plot_learning_curve_
    individual(censored=True) makes per participant, but silently (no skip/
    no-converge prints) -- that render prints those notes itself when it
    runs for real, so this pre-pass doesn't double them up. Falls back to
    `fallback` if there's no data or no participant fit converges (matching
    the figure's previous hardcoded ceiling)."""
    peak = 0.0
    any_curve = False
    for m in COMPARE_MODES:
        rows = learning_curve_individual_rows(trials, m)
        if not rows:
            continue
        trial_max = max(r["trial_index"] + 1 for r in rows)
        smooth_trials = np.linspace(1, trial_max, 200)
        by_participant = defaultdict(list)
        for r in rows:
            by_participant[r["participant_id"]].append(r)
        for prows in by_participant.values():
            trial_nums, achieved_flags, elapsed = trial_arrays(prows)
            fit = fit_curve_for_mode(trial_nums, achieved_flags, elapsed, censored=True)
            if fit is None:
                continue
            any_curve = True
            peak = max(peak, float(np.max(exp_decay(smooth_trials, *fit[:3]))))
    return min(peak * headroom, cap) if any_curve else fallback


def plot_learning_curve_individual(
    trials: list[dict], suffix: str = "", censored: bool = False, y_top: float = 94.5
) -> None:
    """Per-participant learning-curve figure: mu(trial) = c + a*exp(-b*trial),
    fit independently per participant per panel (M3, M5) -- no shared/pooled
    structure across participants. Same size/style/no-title conventions as
    plot_modality_figure / plot_conditions_figure.

    censored=False ("naive"): timeouts counted as y=90, ordinary least
    squares (fit_naive_exp_decay) on every point including those.

    censored=True: timeouts contribute P(time>90) to a right-censored
    Gaussian likelihood (fit_censored_exp_decay) instead of a y=90 point --
    see that function's docstring. Per a derivation + simulation check (see
    conversation / commit history), the censored curve is mathematically
    expected to sit ABOVE the naive curve at heavily-timed-out trials, not
    below: naive's y=90 substitution is a classic Tobit-style downward-biased
    estimator (a timeout means "at least 90", never "exactly 90", so naive
    systematically underestimates by discarding the "could be much higher"
    information censoring carries).

    `y_top` should be the SAME value (see learning_curve_censored_y_top) for
    both the naive and censored calls covering the same data, so the two
    figures share an identical y-scale and are directly comparable -- the
    dashed y=90 timeout line stays put either way, it just sits lower in the
    frame when y_top is pulled up by high censored-curve peaks."""
    name = "learning_curve_individual_censored.png" if censored else "learning_curve_individual_naive.png"
    label = "censored" if censored else "naive"
    panel_rows = {m: learning_curve_individual_rows(trials, m) for m in COMPARE_MODES}
    if not any(panel_rows.values()):
        print(f"  learning_curve_individual_{label}: no data, skipping")
        return

    participants = sorted({r["participant_id"] for rows in panel_rows.values() for r in rows})
    cmap = plt.get_cmap("tab20")
    participant_colors = {p: cmap(i % 20) for i, p in enumerate(participants)}
    if len(participants) > 12:
        print(
            f"  learning_curve_individual_{label}: {len(participants)} participants overlaid on "
            "one axes per panel -- likely too crowded to read individually; consider switching to "
            "small-multiples (one subplot per participant) if so."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    any_drawn = False
    for ax, m in zip(axes, COMPARE_MODES):
        rows = panel_rows[m]
        ax.set_title(FIGURE_LABELS[m], fontsize=15)  # only per-panel identification; no figure title
        ax.axhline(90, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
        ax.set_ylim(0, y_top)
        if not rows:
            continue
        trial_max = max(r["trial_index"] + 1 for r in rows)
        ax.set_xlim(1, trial_max)
        smooth_trials = np.linspace(1, trial_max, 200)

        by_participant = defaultdict(list)
        for r in rows:
            by_participant[r["participant_id"]].append(r)

        for p in sorted(by_participant):
            trial_nums, achieved_flags, elapsed = trial_arrays(by_participant[p])
            color = participant_colors[p]
            completed_mask = achieved_flags
            censored_mask = ~achieved_flags

            if completed_mask.any():
                ax.plot(
                    trial_nums[completed_mask], elapsed[completed_mask], "o", color=color,
                    markersize=6, alpha=0.85, markeredgewidth=0, zorder=3,
                )
            if censored_mask.any():
                ax.plot(
                    trial_nums[censored_mask], np.full(int(censored_mask.sum()), 90.0), "o",
                    color=TIMEOUT_COLOR, markersize=6, alpha=0.85, markeredgewidth=0,
                    zorder=3, clip_on=False,
                )
            any_drawn = True

            fit = fit_curve_for_mode(trial_nums, achieved_flags, elapsed, censored)
            if fit is None:
                if not censored:
                    if len(trial_nums) < 4:
                        print(f"  learning_curve_individual_naive: {p}/{m}: only {len(trial_nums)} trials, skipping fit")
                    else:
                        print(f"  learning_curve_individual_naive: {p}/{m}: fit did not converge, skipping curve")
                else:
                    n_completed = int(completed_mask.sum())
                    if n_completed < 4:
                        print(
                            f"  learning_curve_individual_censored: {p}/{m}: only {n_completed} completed "
                            "trials, skipping fit"
                        )
                    else:
                        print(f"  learning_curve_individual_censored: {p}/{m}: fit did not converge, skipping curve")
                continue
            curve_y = exp_decay(smooth_trials, *fit[:3])
            ax.plot(smooth_trials, curve_y, "-", color=color, linewidth=1.8, zorder=2)

        ax.set_xlabel("Trial", fontsize=15)
        ax.set_ylabel("Time to match (s)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)
        style_axes(ax)

    if not any_drawn:
        plt.close(fig)
        print(f"  learning_curve_individual_{label}: no data, skipping")
        return

    legend_handles = [
        Line2D([], [], marker="o", color=participant_colors[p], linestyle="none", markersize=7, label=p)
        for p in participants
    ]
    legend_handles.append(
        Line2D([], [], marker="o", color=TIMEOUT_COLOR, linestyle="none", markersize=7, label="Timeout")
    )
    fig.legend(
        handles=legend_handles, loc="upper right", bbox_to_anchor=(0.995, 0.98),
        fontsize=9, numpoints=1,
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.35, right=0.86)  # right margin reserved for the participant legend
    save(fig, name, suffix, tight=False)


def learning_curve_averaged_y_top(
    trials: list[dict],
    headroom: float = 1.05,
    exclude_participants: list[str] | None = None,
    cap: float | None = None,
) -> float:
    """Shared y-axis top for BOTH averaged learning-curve figures: 5%
    headroom above the single highest point reached across both modes'
    POOLED censored fit -- including the bootstrap band's upper bound, not
    just the point-estimate curve, so the band itself is never clipped. No
    ceiling by default (unlike learning_curve_censored_y_top's per-
    participant 180 cap -- there's only two pooled curves here, not one per
    participant, so there's usually nothing for a single extreme fit to
    squash); pass `cap` to impose one anyway (used for the exP1P2 variants,
    capped at 220, since removing two participants can otherwise leave a
    small remaining cohort whose bootstrap band blows up).

    exclude_participants, if given, is passed straight through to
    learning_curve_individual_rows -- pass the SAME list used for the actual
    figure render so the y_top matches what's being plotted (see the
    exP1P2 variants of plot_learning_curve_averaged).

    Quiet pre-pass (band computed but not printed) -- plot_learning_curve_
    averaged(censored=True) reruns and prints its own band computation when
    it renders for real, so this doesn't double up notes. Falls back to 94.5
    if there's no data or the censored fit doesn't converge for either
    mode."""
    peak = 0.0
    any_curve = False
    for m in COMPARE_MODES:
        rows = learning_curve_individual_rows(trials, m, exclude_participants=exclude_participants)
        if len(rows) < 4:
            continue
        trial_nums, achieved_flags, elapsed = trial_arrays(rows)
        fit = fit_curve_for_mode(trial_nums, achieved_flags, elapsed, censored=True)
        if fit is None:
            continue
        any_curve = True
        trial_max = int(trial_nums.max())
        smooth_trials = np.linspace(1, trial_max, 200)
        peak = max(peak, float(np.max(exp_decay(smooth_trials, *fit[:3]))))
        _, hi, _ = bootstrap_curve_band(trial_nums, achieved_flags, elapsed, True, smooth_trials)
        if hi is not None:
            peak = max(peak, float(np.max(hi)))
    if not any_curve:
        return 94.5
    y_top = peak * headroom
    return min(y_top, cap) if cap is not None else y_top


def draw_learning_curve_averaged_panel(
    ax,
    trials: list[dict],
    censored: bool,
    y_top: float,
    exclude_participants: list[str] | None = None,
    label: str = "averaged",
) -> bool:
    """Draws the pooled-fit-per-mode time-to-match panel (points, fitted
    curve, bootstrap band, axis styling, its own legend) into a caller-
    supplied `ax` -- the actual rendering logic behind plot_learning_curve_
    averaged, factored out so plot_learning_curve_figure can compose it into
    a multi-panel figure without reimplementing the fit/bootstrap math.
    `label` prefixes this panel's own per-mode skip/no-converge prints (the
    caller decides what an overall "nothing drawn" verdict means for its own
    figure/file, so this does NOT print an overall no-data message).
    Returns True if at least one mode's curve was drawn."""
    mode_rows = {
        m: learning_curve_individual_rows(trials, m, exclude_participants=exclude_participants)
        for m in COMPARE_MODES
    }
    ax.axhline(90, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_ylim(0, y_top)

    # M5 (3D User, blue) scatter points are squares, M3 (2D Patient, green)
    # scatter points are triangles -- the same shape convention conditions_
    # figure already uses to tell the two modes apart independent of color.
    # Triangles render visually smaller than squares at equal markersize, so
    # M3's size is bumped 1.5x to read as similarly weighted on the panel.
    series_markers = {"M3": "^", "M5": "s"}
    series_markersize = {"M3": 3.5 * 1.5, "M5": 3.5}

    any_drawn = False
    trial_max_global = 1
    eq_by_mode = {}  # mode -> "y = c + a*e^(-bx)", one entry per mode whose fit converged
    for m in COMPARE_MODES:
        rows = mode_rows[m]
        color = FIGURE_COLORS[m]
        if not rows:
            continue
        trial_nums, achieved_flags, elapsed = trial_arrays(rows)
        trial_max = int(trial_nums.max())
        trial_max_global = max(trial_max_global, trial_max)
        smooth_trials = np.linspace(1, trial_max, 200)
        completed_mask = achieved_flags
        censored_mask = ~achieved_flags

        # Pooled points shown lightly (low alpha) at their EXACT trial
        # number -- no horizontal jitter. Many participants' trials stack at
        # each integer trial number, but overplotting there is truthful
        # (that's really where the data is); shifting points off their real
        # x to reduce visual overlap would misalign them against the fitted
        # curve and axis ticks, which stay at the exact integer trial.
        if completed_mask.any():
            ax.plot(
                trial_nums[completed_mask], elapsed[completed_mask], series_markers[m], color=color,
                markersize=series_markersize[m], alpha=0.25, markeredgewidth=0, zorder=2,
            )
        if censored_mask.any():
            # Same marker shape AND color as this mode's matched points --
            # timeouts are distinguished only by sitting at y=90, not by a
            # separate color.
            ax.plot(
                trial_nums[censored_mask], np.full(int(censored_mask.sum()), 90.0), series_markers[m],
                color=color, markersize=series_markersize[m], alpha=0.35, markeredgewidth=0,
                zorder=2, clip_on=False,
            )

        fit = fit_curve_for_mode(trial_nums, achieved_flags, elapsed, censored)
        if fit is None:
            n_completed = int(completed_mask.sum())
            reason = (
                f"only {len(trial_nums)} trials" if not censored and len(trial_nums) < 4
                else f"only {n_completed} completed trials" if censored and n_completed < 4
                else "fit did not converge"
            )
            print(f"  learning_curve_{label}: {m}: {reason}, skipping curve")
            continue
        any_drawn = True
        curve_y = exp_decay(smooth_trials, *fit[:3])
        c, a, b = fit[0], fit[1], fit[2]
        # mathtext (enclosing $...$) so "-b*x" renders as a true superscript
        # on e instead of literal "^(-...)" characters.
        eq = rf"$y = {c:.3g} + {a:.3g}\,e^{{-{b:.3g}x}}$"

        lo, hi, n_converged = bootstrap_curve_band(trial_nums, achieved_flags, elapsed, censored, smooth_trials)
        if lo is None:
            print(
                f"  learning_curve_{label}: {m}: only {n_converged}/{N_BOOT} bootstrap "
                "replicates converged (<50%), skipping band"
            )
        else:
            ax.fill_between(smooth_trials, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)

        ax.plot(smooth_trials, curve_y, "-", color=color, linewidth=2.4, zorder=3)
        eq_by_mode[m] = eq

    if not any_drawn:
        return False

    # set_xticks() BEFORE set_xlim(): matplotlib silently expands the view
    # to include any tick location outside the current limits, so calling
    # set_xticks() second would undo the xlim(1, ...) below (confirmed --
    # this previously left the axis starting at 0 despite the explicit
    # xlim(1, ...) call that used to come after it). Ticks start at 1, not
    # 0, for the same reason: the axis must start at trial 1.
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(1, trial_max_global + 1)  # a little breathing room past the last trial; no tick added there
    ax.set_xlabel("Trial", fontsize=15)
    ax.set_ylabel("Time to match (s)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    style_axes(ax)

    # Each legend row IS the mode label + its fitted equation, e.g.
    # "3D User      $y = 42.3 + 120\,e^{-0.648x}$" -- no separate "Timeout"
    # entry and no standalone equation text elsewhere on the panel. Labels
    # are space-padded to a common width so the "$y = ...$" column roughly
    # lines up between rows. No per-Text "family" override here -- that
    # would win over mathtext.default="regular"'s "match the surrounding
    # text" behavior (confirmed: an earlier "family": "monospace" override
    # made the mathtext render in monospace too, not the figure's sans-
    # serif). Leaving font selection to the figure-wide rcParams
    # (font.family="sans-serif") is what makes the equations match the
    # "2D Patient"/"3D User" labels' typeface.
    present_modes = [m for m in COMPARE_MODES if mode_rows[m]]
    label_width = max((len(FIGURE_LABELS[m]) for m in present_modes), default=0)
    legend_handles = [
        Line2D(
            [], [], color=FIGURE_COLORS[m], linewidth=2.4,
            label=f"{FIGURE_LABELS[m].ljust(label_width)}   {eq_by_mode[m]}",
        )
        for m in present_modes
    ]
    ax.legend(handles=legend_handles, loc="upper right", numpoints=1, fontsize=9)
    return True


def plot_learning_curve_averaged(
    trials: list[dict],
    suffix: str = "",
    censored: bool = False,
    y_top: float = 94.5,
    exclude_participants: list[str] | None = None,
    name_suffix: str = "",
) -> None:
    """Pooled learning-curve figure: ONE exp_decay fit per mode (M3, M5),
    pooling every participant's trials together (no per-participant
    structure -- contrast with plot_learning_curve_individual), both modes
    overlaid on a single axes with a 95% bootstrap confidence band
    (bootstrap_curve_band) around each fitted curve. Same fit/model
    semantics as plot_learning_curve_individual: censored=False ("naive")
    treats timeouts as y=90 and fits OLS; censored=True fits a right-
    censored Gaussian MLE (see fit_censored_exp_decay's docstring for why
    its curve sits ABOVE naive at heavily-timed-out trials).

    Not called by main() by default (see plot_learning_curve_figure, which
    composes this same rendering -- via draw_learning_curve_averaged_panel
    -- into one of its two panels instead of its own standalone file); kept
    as a standalone entry point for re-enabling this exact output later.

    `y_top` should be the SAME value (see learning_curve_averaged_y_top) for
    both the naive and censored calls, so the two figures share an
    identical y-scale.

    exclude_participants drops those participant_ids (exact match, see
    learning_curve_individual_rows) before pooling -- everything downstream
    (fit, band, y_top the caller should have computed to match) uses only
    the remaining participants. `name_suffix` (e.g. "_exP1P2") is appended
    to the output filename so an excluded-cohort run doesn't overwrite the
    full-cohort one."""
    name = f"learning_curve_averaged_{'censored' if censored else 'naive'}{name_suffix}.png"
    label = f"averaged_{'censored' if censored else 'naive'}{name_suffix}"
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if not draw_learning_curve_averaged_panel(ax, trials, censored, y_top, exclude_participants, label):
        plt.close(fig)
        print(f"  learning_curve_{label}: no data, skipping")
        return
    save(fig, name, suffix)


def learning_curve_cumulative_arrays(
    trials: list[dict], modality_id: str, exclude_participants: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled (x_cumulative, achieved_flags, elapsed) for ONE mode, where
    x_cumulative is each participant's own running total of elapsed_s
    (including timeouts, ~90s each) through and including that trial --
    ordered by trial_index, each participant's clock starting at 0. E.g. a
    participant whose first three trials took 60, 65, 25s lands points at
    x=60, x=125, x=150 (y = 65 and y = 25 for the 2nd/3rd, respectively).

    Row filtering/exclusion is identical to the trial-number figures (same
    learning_curve_individual_rows call) -- only the x-axis construction
    differs; fit_curve_for_mode / bootstrap_curve_band are agnostic to
    what the x-values mean, so no fitting code changes for this variant."""
    rows = learning_curve_individual_rows(trials, modality_id, exclude_participants=exclude_participants)
    by_participant = defaultdict(list)
    for r in rows:
        by_participant[r["participant_id"]].append(r)
    x_list, achieved_list, elapsed_list = [], [], []
    for prows in by_participant.values():
        prows = sorted(prows, key=lambda r: r["trial_index"])
        cumulative = 0.0
        for r in prows:
            cumulative += r["elapsed_s"]
            x_list.append(cumulative)
            achieved_list.append(bool(r["achieved"]))
            elapsed_list.append(r["elapsed_s"])
    return (
        np.array(x_list, dtype=float),
        np.array(achieved_list, dtype=bool),
        np.array(elapsed_list, dtype=float),
    )


def learning_curve_cumulative_y_top(
    trials: list[dict],
    headroom: float = 1.05,
    exclude_participants: list[str] | None = None,
    cap: float | None = None,
) -> float:
    """Shared y-axis top for the cumulative-time excluded-cohort figure
    pair -- same construction as learning_curve_averaged_y_top (5% headroom
    above the censored fit + bootstrap band peak across both modes), just
    over the cumulative-time x-axis instead of trial number. See that
    function's docstring for the cap/fallback semantics."""
    peak = 0.0
    any_curve = False
    for m in COMPARE_MODES:
        x_cum, achieved_flags, elapsed = learning_curve_cumulative_arrays(trials, m, exclude_participants)
        if len(x_cum) < 4:
            continue
        fit = fit_curve_for_mode(x_cum, achieved_flags, elapsed, censored=True)
        if fit is None:
            continue
        any_curve = True
        smooth_x = np.linspace(0, float(x_cum.max()), 200)
        peak = max(peak, float(np.max(exp_decay(smooth_x, *fit[:3]))))
        _, hi, _ = bootstrap_curve_band(x_cum, achieved_flags, elapsed, True, smooth_x)
        if hi is not None:
            peak = max(peak, float(np.max(hi)))
    if not any_curve:
        return 94.5
    y_top = peak * headroom
    return min(y_top, cap) if cap is not None else y_top


def plot_learning_curve_cumulative(
    trials: list[dict],
    suffix: str = "",
    censored: bool = False,
    y_top: float = 94.5,
    exclude_participants: list[str] | None = None,
    name_suffix: str = "",
) -> None:
    """Cumulative-time variant of plot_learning_curve_averaged: x is each
    participant's own running total of time-on-task (see
    learning_curve_cumulative_arrays) instead of trial number 1..20. Same
    model, same naive-vs-censored semantics, same bootstrap band -- only the
    x-axis meaning and its 0-anchored range differ, so this mirrors that
    function's structure closely rather than sharing code directly (the
    per-panel bookkeeping -- x_max, smooth grid, legend -- reads more
    clearly kept separate than parameterized through a shared helper for
    just two callers)."""
    name = f"learning_curve_cumulative_{'censored' if censored else 'naive'}{name_suffix}.png"
    label = f"cumulative_{'censored' if censored else 'naive'}{name_suffix}"

    mode_data = {m: learning_curve_cumulative_arrays(trials, m, exclude_participants) for m in COMPARE_MODES}
    if all(len(d[0]) == 0 for d in mode_data.values()):
        print(f"  learning_curve_{label}: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(90, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_ylim(0, y_top)

    any_drawn = False
    any_timeout = False
    x_max_global = 1.0
    for m in COMPARE_MODES:
        x_cum, achieved_flags, elapsed = mode_data[m]
        color = FIGURE_COLORS[m]
        if len(x_cum) == 0:
            continue
        x_max = float(x_cum.max())
        x_max_global = max(x_max_global, x_max)
        smooth_x = np.linspace(0, x_max, 200)
        completed_mask = achieved_flags
        censored_mask = ~achieved_flags

        # No jitter here (unlike the trial-number figure): x is continuous
        # cumulative time, points aren't stacked at shared integer x-values.
        if completed_mask.any():
            ax.plot(
                x_cum[completed_mask], elapsed[completed_mask], "o", color=color,
                markersize=3.5, alpha=0.25, markeredgewidth=0, zorder=2,
            )
        if censored_mask.any():
            any_timeout = True
            ax.plot(
                x_cum[censored_mask], np.full(int(censored_mask.sum()), 90.0), "o",
                color=TIMEOUT_COLOR, markersize=3.5, alpha=0.35, markeredgewidth=0,
                zorder=2, clip_on=False,
            )

        fit = fit_curve_for_mode(x_cum, achieved_flags, elapsed, censored)
        if fit is None:
            n_completed = int(completed_mask.sum())
            reason = (
                f"only {len(x_cum)} points" if not censored and len(x_cum) < 4
                else f"only {n_completed} completed points" if censored and n_completed < 4
                else "fit did not converge"
            )
            print(f"  learning_curve_{label}: {m}: {reason}, skipping curve")
            continue
        any_drawn = True
        curve_y = exp_decay(smooth_x, *fit[:3])

        lo, hi, n_converged = bootstrap_curve_band(x_cum, achieved_flags, elapsed, censored, smooth_x)
        if lo is None:
            print(
                f"  learning_curve_{label}: {m}: only {n_converged}/{N_BOOT} bootstrap "
                "replicates converged (<50%), skipping band"
            )
        else:
            ax.fill_between(smooth_x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=1)

        ax.plot(smooth_x, curve_y, "-", color=color, linewidth=2.4, zorder=3)

    if not any_drawn:
        plt.close(fig)
        print(f"  learning_curve_{label}: no data, skipping")
        return

    ax.set_xlim(0, x_max_global)
    ax.set_xlabel("Cumulative time on task (s)", fontsize=15)
    ax.set_ylabel("Time to match (s)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    style_axes(ax)

    legend_handles = [
        Line2D([], [], color=FIGURE_COLORS[m], linewidth=2.4, label=FIGURE_LABELS[m])
        for m in COMPARE_MODES if len(mode_data[m][0])
    ]
    if any_timeout:
        legend_handles.append(
            Line2D([], [], marker="o", color=TIMEOUT_COLOR, linestyle="none", markersize=7, label="Timeout")
        )
    ax.legend(handles=legend_handles, loc="best", fontsize=12, numpoints=1)

    save(fig, name, suffix)


def draw_learning_curve_success_panel(
    ax, trials: list[dict], exclude_participants: list[str] | None = None
) -> bool:
    """Draws the per-trial success-rate panel (line + Wilson 95% CI whisker
    per mode, axis styling, its own legend) into a caller-supplied `ax` --
    the actual rendering logic behind plot_learning_curve_success, factored
    out so plot_learning_curve_figure can compose it into a multi-panel
    figure without reimplementing the Wilson-interval math. Per (mode,
    trial) success rate = matched / n_participants who did that trial --
    no curve fit, so (unlike draw_learning_curve_averaged_panel) there's no
    minimum-points threshold to skip a trial: Wilson's interval is
    well-defined even at n=1. Returns True if at least one mode had data.

    Same darkened-outline treatment as the other figures (marker edge AND
    CI line/caps take darken_hsl(fill, 0.40)) using the SAME single
    errorbar() call for both marker and whisker -- ecolor drives the CI
    color, markeredgecolor drives the marker edge, and markeredgewidth must
    stay non-zero or the caps silently disappear (see plot_conditions_
    figure's comments for the full explanation of that coupling)."""
    mode_rows = {
        m: learning_curve_individual_rows(trials, m, exclude_participants=exclude_participants)
        for m in COMPARE_MODES
    }
    any_drawn = False
    trial_max_global = 1
    for m in COMPARE_MODES:
        rows = mode_rows[m]
        if not rows:
            continue
        color = FIGURE_COLORS[m]
        outline = darken_hsl(color, 0.40)
        by_trial = defaultdict(list)
        for r in rows:
            by_trial[r["trial_index"] + 1].append(bool(r["achieved"]))
        trial_nums = sorted(by_trial)
        if not trial_nums:
            continue
        trial_max_global = max(trial_max_global, max(trial_nums))

        means, los, his = [], [], []
        for t in trial_nums:
            outcomes = by_trial[t]
            phat, lo, hi = wilson_ci95(sum(outcomes), len(outcomes))
            means.append(phat)
            los.append(lo)
            his.append(hi)
        any_drawn = True

        ax.plot(trial_nums, means, "-", color=color, linewidth=1.8, zorder=2)
        lo_err = [mn - lo for mn, lo in zip(means, los)]
        hi_err = [hi - mn for mn, hi in zip(means, his)]
        ax.errorbar(
            trial_nums, means, yerr=[lo_err, hi_err],
            fmt="o", color=color, ecolor=outline,
            elinewidth=1.8, capsize=5, capthick=1.8, markersize=7,
            markeredgecolor=outline, markeredgewidth=1.8, zorder=3,
        )

    if not any_drawn:
        return False

    ax.set_ylim(0, 1.05)
    # set_xticks() before set_xlim() -- see draw_learning_curve_averaged_
    # panel's comment; same fix, same reason (ticks including 0 were
    # silently pulling the left edge back to 0 despite this xlim(1, ...)
    # call previously coming after it).
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(1, trial_max_global)
    ax.set_xlabel("Trial", fontsize=15)
    ax.set_ylabel("Success rate", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)
    style_axes(ax)

    # Explicit proxy handles (not the errorbar artists) -- clean marker-only
    # legend, no connecting line through the markers.
    legend_handles = [
        Line2D(
            [], [], marker="o", color=FIGURE_COLORS[m], linestyle="none", markersize=8,
            markeredgecolor=darken_hsl(FIGURE_COLORS[m], 0.40), markeredgewidth=1.8,
            label=FIGURE_LABELS[m],
        )
        for m in COMPARE_MODES if mode_rows[m]
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=12, numpoints=1)
    return True


def plot_learning_curve_success(
    trials: list[dict], suffix: str = "", exclude_participants: list[str] | None = None
) -> None:
    """learning_curve_success.png: per-trial success rate, both modes (M3,
    M5) overlaid. Not called by main() by default (see
    plot_learning_curve_figure, which composes this same rendering -- via
    draw_learning_curve_success_panel -- into one of its two panels instead
    of its own standalone file); kept as a standalone entry point for
    re-enabling this exact output later."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if not draw_learning_curve_success_panel(ax, trials, exclude_participants):
        plt.close(fig)
        print("  learning_curve_success: no data, skipping")
        return
    save(fig, "learning_curve_success.png", suffix)


def plot_learning_curve_figure(
    trials: list[dict],
    suffix: str = "",
    exclude_participants: list[str] | None = None,
    y_top: float = 94.5,
) -> None:
    """learning_curve_figure.png: single panel -- the pooled CENSORED
    time-to-match fit + 95% bootstrap band (draw_learning_curve_averaged_
    panel, censored=True), always on the excluded cohort passed in via
    `exclude_participants` (main() passes the P1/P2 pattern-match exclusion
    -- see learning_curve_individual_rows).

    Composition only: no fit/bootstrap math is reimplemented here, the
    panel is drawn by calling the exact same function plot_learning_curve_
    averaged calls internally. Was previously a two-panel figure (this
    panel plus a success-rate panel via draw_learning_curve_success_panel);
    that draw function -- and plot_learning_curve_success, which still
    calls it for its own standalone figure -- are both kept in the file
    unchanged, just no longer composed in here.

    `y_top` should be learning_curve_averaged_y_top(trials,
    exclude_participants=..., cap=120.0) -- the same value that would be
    used for the standalone censored_exP1P2 figure -- so this panel matches
    that output exactly."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    any_drawn = draw_learning_curve_averaged_panel(
        ax, trials, censored=True, y_top=y_top,
        exclude_participants=exclude_participants, label="figure_time",
    )
    if not any_drawn:
        plt.close(fig)
        print("  learning_curve_figure: no data, skipping")
        return
    save(fig, "learning_curve_figure.png", suffix)


def plot_noise_time(trials: list[dict], suffix: str = "") -> None:
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
        bounds=TIME_BOUNDS,
    )
    ax.set_xlabel("Noise magnitude (mm & deg)")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Noise: completion time by magnitude (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "noise_time.png", suffix)


def plot_latency_time(trials: list[dict], suffix: str = "") -> None:
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
        bounds=TIME_BOUNDS,
    )
    ax.set_xlabel("Perceived latency (ms)")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Latency: completion time by perceived latency (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "latency_time.png", suffix)


def precision_key(row: dict):
    return (row["precision_linear_mm"], row["precision_angular_deg"])


def precision_groups(rows: list[dict]) -> list[tuple[float, float]]:
    keys = {precision_key(r) for r in rows}
    # Easiest -> hardest: largest tolerance first.
    return sorted(keys, key=lambda k: (-k[0], -k[1]))


def precision_label(key: tuple[float, float]) -> str:
    mm, deg = key
    return f"{mm:g}mm/{deg:g}deg"


def plot_precision_time(trials: list[dict], suffix: str = "") -> None:
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
        bounds=TIME_BOUNDS,
    )
    ax.set_xlabel("Precision threshold (mm & deg)")
    ax.set_ylabel("Time to match (s)")
    ax.set_title("Precision: completion time by threshold (mean ± 95% CI)")
    style_axes(ax)
    save(fig, "precision_time.png", suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pool experiment.sqlite files under a folder and render summary plots."
    )
    parser.add_argument("folder", help="Folder to scan recursively for experiment.sqlite files")
    parser.add_argument(
        "--participant",
        help="Only include sessions whose participant_id matches this value (case-insensitive)",
    )
    parser.add_argument(
        "--threshold",
        nargs=2,
        type=float,
        metavar=("LINEAR_MM", "ANGULAR_DEG"),
        help=(
            "Re-derive time-to-match from recorded trajectories under this "
            "(linear_mm, angular_deg) threshold instead of the threshold each "
            "trial actually ran with, using the live 1.0s continuous-hold "
            "rule. Affects all time plots and modality_success; "
            "modality_preference is unaffected. Output filenames get a "
            "_thrLINEARxANGULAR suffix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    paths = find_sqlite_files(folder)
    print(f"Found {len(paths)} experiment.sqlite file(s) under {folder}")
    if not paths:
        print("Nothing to plot.")
        return

    trials, preferences = load_data(paths)

    suffix = ""
    if args.participant:
        available = sorted({t["participant_id"] for t in trials if t["participant_id"]})
        matches = {p.lower() for p in available if p.lower() == args.participant.lower()}
        if not matches:
            print(f"\nNo sessions found for participant '{args.participant}'.")
            if available:
                print(f"Available participant IDs: {', '.join(available)}")
            else:
                print("No participant IDs found in this folder.")
            return
        trials = [t for t in trials if t["participant_id"] and t["participant_id"].lower() in matches]
        preferences = [p for p in preferences if p["participant_id"] and p["participant_id"].lower() in matches]
        suffix = args.participant

    participant_suffix = suffix

    # conditions_figure.png pools practice + real bins (never trash) instead
    # of just the folder passed on the command line -- every other plot
    # below still only uses `trials` from that folder. It also uses the
    # trials exactly as recorded: no FIGURE_THRESHOLD re-derivation (that's
    # modality_figure-only) and no --threshold override either, so this is a
    # wholly separate load_data() call, never touched by
    # apply_threshold_override.
    conditions_roots = conditions_figure_search_roots(folder)
    conditions_paths = sorted({p for root in conditions_roots for p in find_sqlite_files(root)})
    conditions_trials, _ = load_data(conditions_paths) if conditions_paths else ([], [])
    if args.participant:
        conditions_trials = [
            t for t in conditions_trials
            if t["participant_id"] and t["participant_id"].lower() == args.participant.lower()
        ]
    print("\nconditions_figure data pool (practice + real bins, never trash):")
    if conditions_roots:
        for root in conditions_roots:
            print(f"  root: {root}")
        for p in conditions_paths:
            print(f"    file: {p}")
    else:
        print(f"  no practice/real bin found as a sibling of, or under, {folder}")
    for experiment in ("noise", "latency", "precision"):
        rows = [t for t in conditions_trials if t["experiment_condition"] == experiment]
        if not rows:
            print(f"  {experiment:<10} no data found")
            continue
        n_files = len({t["source_file"] for t in rows})
        n_participants = len({t["participant_id"] for t in rows})
        print(f"  {experiment:<10} files={n_files:<4} participants={n_participants:<4} trials={len(rows)}")

    trajectories = load_trajectories(paths)

    # modality_figure.png always renders at FIGURE_THRESHOLD (10mm/10deg),
    # independent of any --threshold passed for the other plots below.
    figure_trials, n_matched_fig, n_timed_out_fig = apply_threshold_override(
        trials, trajectories, *FIGURE_THRESHOLD
    )
    print(
        f"\nRe-derived modality_figure at {FIGURE_THRESHOLD[0]:g}mm/{FIGURE_THRESHOLD[1]:g}deg, "
        f"hold {HOLD_S:g}s: {n_matched_fig} newly matched, {n_timed_out_fig} newly "
        f"timed out (of {len(figure_trials)} trials)."
    )

    if args.threshold:
        linear_mm, angular_deg_thr = args.threshold
        trials, n_matched, n_timed_out = apply_threshold_override(
            trials, trajectories, linear_mm, angular_deg_thr
        )
        print(
            f"\nRe-derived at threshold {linear_mm:g}mm/{angular_deg_thr:g}deg, "
            f"hold {HOLD_S:g}s: {n_matched} newly matched, {n_timed_out} newly "
            f"timed out (of {len(trials)} trials)."
        )
        threshold_suffix = f"thr{linear_mm:g}x{angular_deg_thr:g}"
        suffix = "_".join(s for s in (suffix, threshold_suffix) if s)

    print_summary(trials, participant=args.participant)

    # main() ships exactly three figures: modality_figure.png,
    # conditions_figure.png, learning_curve_figure.png (+ the first two's
    # summary CSVs). Every other plot_* function in this file is still
    # defined and fully working -- see e.g. plot_modality_time,
    # plot_learning_curve_individual, plot_noise_time -- just not called
    # here, so any of them can be re-enabled later without reimplementing
    # anything.
    print("\nPlots:")
    plot_modality_figure(
        figure_trials, preferences, participant_suffix,
        meta={"paths": paths, "folder": folder, "participant": args.participant},
    )

    # learning_curve_figure.png always uses the P1/P2 pattern-matched
    # exclusion (see learning_curve_individual_rows) -- P1-style tokens
    # match e.g. "P1-REAL" but not "P10"/"P12".
    lc_ids_present = sorted({
        t["participant_id"] for t in trials
        if t["experiment_condition"] == "learning_curve" and t["participant_id"]
    })
    lc_exclude_requested = ["P1", "P2"]
    lc_exclude_patterns = [
        re.compile(rf"^{re.escape(token)}($|[^0-9])", re.IGNORECASE) for token in lc_exclude_requested
    ]
    lc_exclude_matched = [pid for pid in lc_ids_present if any(p.match(pid) for p in lc_exclude_patterns)]
    print(f"  learning_curve participant_ids present: {lc_ids_present or '(none)'}")
    print(f"  learning_curve exclusion tokens={lc_exclude_requested}, matched ids={lc_exclude_matched or '(none)'}")
    lc_avg_ex_y_top = learning_curve_averaged_y_top(
        trials, exclude_participants=lc_exclude_requested, cap=120.0
    )
    print(f"  learning_curve_figure: shared y_top (left panel) = {lc_avg_ex_y_top:.2f}")
    plot_learning_curve_figure(
        trials, suffix, exclude_participants=lc_exclude_requested, y_top=lc_avg_ex_y_top,
    )

    plot_conditions_figure(
        conditions_trials, participant_suffix,
        meta={"roots": conditions_roots, "paths": conditions_paths, "participant": args.participant},
    )


if __name__ == "__main__":
    main()
