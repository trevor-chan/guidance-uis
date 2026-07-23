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
"""

from __future__ import annotations

import argparse
import colorsys
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
from matplotlib.lines import Line2D

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
FIGURE_ORDER = ["M1", "M4", "M2", "M3", "M7", "M5", "M6"]
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
    "M1": "#C4C4C4",
    "M4": "#C1F5DF",
    "M2": "#89EEBF",
    "M3": "#5CC588",
    "M7": "#C8DCF5",
    "M5": "#6097F0",
    "M6": "#2D65D2",
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


def plot_modality_figure(trials: list[dict], preferences: list[dict], suffix: str = "") -> None:
    """Combined publication figure: time (box+dots) / success / preference,
    three panels side by side on a dimension-grouped x-axis (each panel keeps
    its own x tick labels since they no longer share an axis). `trials` must
    already be re-derived at FIGURE_THRESHOLD by the caller (see main()) —
    this function does not re-derive anything itself, it only renders."""
    rows = [
        t
        for t in trials
        if t["experiment_condition"] == "modality"
        and t["elapsed_s"] is not None
        and t["achieved"] is not None
    ]
    if not rows:
        print("  modality_figure: no data, skipping")
        return
    modalities = figure_modalities(rows)
    xs = list(range(1, len(modalities) + 1))
    colors = [FIGURE_COLORS[m] for m in modalities]
    outline_colors = [darken_hsl(FIGURE_COLORS[m], 0.40) for m in modalities]
    tick_labels = [FIGURE_LABELS[m] for m in modalities]

    # figsize widened so the extra wspace set below (see the tight_layout
    # call near the end) comes out of new space, not the panels themselves
    # getting narrower.
    fig, (ax_time, ax_success, ax_pref) = plt.subplots(1, 3, figsize=(15, 5.5))

    # -- Panel A: time to match (conventional box-and-whisker + raw dots) --
    time_data = [[r["elapsed_s"] for r in rows if r["modality_id"] == m] for m in modalities]
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
    for x, m in zip(xs, modalities):
        matched = [r["elapsed_s"] for r in rows if r["modality_id"] == m and r["achieved"]]
        timed_out = [r["elapsed_s"] for r in rows if r["modality_id"] == m and not r["achieved"]]
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
    means, los, his = [], [], []
    for m in modalities:
        outcomes = [r["achieved"] for r in rows if r["modality_id"] == m]
        phat, lo, hi = wilson_ci95(sum(outcomes), len(outcomes))
        means.append(phat)
        los.append(lo)
        his.append(hi)
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
    means, los, his = [], [], []
    for m in modalities:
        ratings = [p["rating"] for p in preferences if p["modality_id"] == m]
        if ratings:
            mn, lo, hi = mean_ci95(ratings, bounds=PREFERENCE_BOUNDS)
        else:
            mn = lo = hi = 1.0
        means.append(mn)
        los.append(lo)
        his.append(hi)
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


def plot_conditions_figure(trials: list[dict], suffix: str = "") -> None:
    """Companion to plot_modality_figure: noise / latency / precision
    time-to-match, each panel comparing M3 (2D/Patient) vs M5 (3D/User) on a
    log-scaled x-axis of the condition's own magnitude. Same size, rcParams,
    line weights, text sizes, headroom convention, and no-title style as
    plot_modality_figure -- see that function's comments for why tight_layout
    and subplots_adjust are sequenced the way they are below.

    `trials` must be the RAW, non-threshold-derived trial list (see main()) --
    this is the one plot in the file that intentionally skips both
    FIGURE_THRESHOLD and any --threshold override, so noise/latency reflect
    the live 5mm/5deg rule they actually ran under and precision reflects
    each trial's own per-trial threshold, exactly as recorded."""
    noise_rows = [
        t for t in trials
        if t["experiment_condition"] == "noise"
        and t["modality_id"] in COMPARE_MODES
        and t["noise"] is not None
        and t["elapsed_s"] is not None
    ]
    latency_rows = [
        t for t in trials
        if t["experiment_condition"] == "latency"
        and t["modality_id"] in COMPARE_MODES
        and t["perceived_ms"] is not None
        and t["elapsed_s"] is not None
    ]
    precision_rows = [
        t for t in trials
        if t["experiment_condition"] == "precision"
        and t["modality_id"] in COMPARE_MODES
        and t["precision_linear_mm"] is not None
        and t["elapsed_s"] is not None
    ]
    missing = [
        name for name, rows in
        [("noise", noise_rows), ("latency", latency_rows), ("precision", precision_rows)]
        if not rows
    ]
    if missing:
        print(f"  conditions_figure: no data for {', '.join(missing)}, skipping")
        return

    # M3 (2D/Patient) is always drawn before M5 (3D/User) -- COMPARE_MODES is
    # already ["M3", "M5"] -- so M5's artists land on top wherever the two
    # series overlap, consistently across all three panels.
    series_markers = {"M3": "^", "M5": "s"}

    def draw_panel(ax, rows, magnitude_key, magnitude_values, show_legend=False, show_ylabel=False):
        for s in COMPARE_MODES:
            color = FIGURE_COLORS[s]
            xs_ser, means, lo_err, hi_err = [], [], [], []
            for xv in magnitude_values:
                vals = [r["elapsed_s"] for r in rows if r[magnitude_key] == xv and r["modality_id"] == s]
                if not vals:
                    continue
                m, lo, hi = mean_ci95(vals, bounds=TIME_BOUNDS)
                xs_ser.append(xv)
                means.append(m)
                lo_err.append(m - lo)
                hi_err.append(hi - m)
            if not xs_ser:
                continue
            ax.plot(xs_ser, means, "-", color=color, linewidth=1.5, zorder=2)
            # No outlines anywhere here (unlike modality_figure): fill color
            # only, on both the marker and its CI line/caps.
            ax.errorbar(
                xs_ser, means, yerr=[lo_err, hi_err],
                fmt=series_markers[s], color=color, ecolor=color,
                elinewidth=1.8, capsize=5, capthick=1.8, markersize=9,
                markeredgewidth=0, zorder=3, label=FIGURE_LABELS[s],
            )
        ax.axhline(90, color="black", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
        ax.set_ylim(0, 94.5)  # 5% headroom above the 90s cap, matching plot_modality_figure
        if show_ylabel:
            ax.set_ylabel("Time to match (s)", fontsize=15)
        else:
            ax.tick_params(axis="y", labelleft=False)
        if show_legend:
            ax.legend(loc="best", fontsize=12, numpoints=1)

    fig, (ax_noise, ax_latency, ax_precision) = plt.subplots(1, 3, figsize=(15, 5.5))

    noise_values = sorted({r["noise"] for r in noise_rows})
    draw_panel(ax_noise, noise_rows, "noise", noise_values, show_legend=True, show_ylabel=True)
    ax_noise.set_xticks(noise_values)
    ax_noise.set_xticklabels([f"{v:g}" for v in noise_values])
    ax_noise.set_xlabel("Noise (mm / deg)", fontsize=15)

    latency_values = sorted({r["perceived_ms"] for r in latency_rows})
    draw_panel(ax_latency, latency_rows, "perceived_ms", latency_values)
    ax_latency.set_xticks(latency_values)
    ax_latency.set_xticklabels([f"{v:g}" for v in latency_values])
    ax_latency.set_xlabel("Perceived latency (ms)", fontsize=15)

    # Ascending sort already puts the tightest threshold (smallest mm) on the
    # left and loosest (largest mm) on the right -- no reversal needed.
    precision_values = sorted({r["precision_linear_mm"] for r in precision_rows})
    draw_panel(ax_precision, precision_rows, "precision_linear_mm", precision_values)
    ax_precision.set_xticks(precision_values)
    ax_precision.set_xticklabels([f"{v:g}" for v in precision_values])
    ax_precision.set_xlabel("Precision threshold (mm / deg)", fontsize=15)

    for ax in (ax_noise, ax_latency, ax_precision):
        ax.tick_params(axis="both", labelsize=13)
        ax.spines["left"].set_linewidth(1.6)
        ax.spines["bottom"].set_linewidth(1.6)
        style_axes(ax)

    # Middle/right panels carry no y tick labels, so they need less breathing
    # room than plot_modality_figure's wspace=0.4 -- panels sit closer together.
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    save(fig, "conditions_figure.png", suffix, tight=False)


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
    ax.set_xlabel("Noise magnitude (mm / deg)")
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
    ax.set_xlabel("Precision threshold (mm / deg)")
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
    # conditions_figure.png uses the trials exactly as recorded -- no
    # FIGURE_THRESHOLD re-derivation (that's modality_figure-only) and no
    # --threshold override either (apply_threshold_override always returns a
    # new list, so rebinding `trials` below doesn't touch this reference).
    raw_trials = trials
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

    print("\nPlots:")
    plot_modality_time(trials, suffix)
    plot_modality_preference(preferences, suffix)
    plot_modality_success(trials, suffix)
    plot_modality_figure(figure_trials, preferences, participant_suffix)
    plot_learning_curve_time(trials, suffix)
    plot_noise_time(trials, suffix)
    plot_latency_time(trials, suffix)
    plot_precision_time(trials, suffix)
    plot_conditions_figure(raw_trials, participant_suffix)


if __name__ == "__main__":
    main()
