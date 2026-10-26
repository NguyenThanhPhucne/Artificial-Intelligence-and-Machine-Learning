"""Shared figure style and export for the report.

I fix one colour per class and reuse it in every figure, so a reader who learns "orange = FTP" in
one chart can rely on it in all of them. The three hues come from a validated categorical palette
(blue, orange, aqua) that stays distinguishable under the common forms of colour-vision deficiency;
because aqua has low contrast on white, I always add direct value labels instead of relying on
colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from . import config

CLASS_ORDER = ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest"]
# Models are a second categorical dimension; I give them neutral inks so they never borrow a class colour.
BASELINE_INK = "#898781"
TUNED_INK = "#0b0b0b"
CLASS_COLORS = {
    "Benign": "#2a78d6",
    "FTP-BruteForce": "#eb6834",
    "SSH-Bruteforce": "#1baf7a",
}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Diverging map for correlations: blue (negative) -> neutral grey (zero) -> red (positive).
# A grey midpoint makes "no correlation" read as absence rather than as a third colour.
CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1d1d"]
)


def apply_style() -> None:
    """Set a quiet, print-friendly matplotlib style: light grid, muted axes, readable fonts."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def save_figure(fig: Figure, name: str, folder: Path = config.FIGURES_DIR) -> Path:
    """Save a figure as a 300-dpi PNG (the resolution I insert into the Word report)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    return path


def plot_class_distribution(raw_counts: pd.Series, train_counts: pd.Series) -> Figure:
    """Two panels of horizontal bars: class sizes in the raw file and in my training set.

    I use a log axis because FTP-BruteForce has 37 training flows next to ~51,000 benign ones; on a
    linear axis its bar would be invisible. Every bar carries its count and share as a direct label,
    so the chart can be read without the colours.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    panels = [
        (axes[0], raw_counts, f"Raw file ({raw_counts.sum():,} flows)"),
        (axes[1], train_counts, f"Training set after de-duplication ({train_counts.sum():,} flows)"),
    ]
    for ax, counts, title in panels:
        counts = counts.reindex(CLASS_ORDER)
        positions = np.arange(len(CLASS_ORDER))
        ax.barh(positions, counts, height=0.6, color=[CLASS_COLORS[c] for c in CLASS_ORDER])
        for pos, value in zip(positions, counts):
            ax.text(value * 1.25, pos, f"{value:,} ({value / counts.sum():.2%})",
                    va="center", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(10, counts.max() * 40)
        ax.set_yticks(positions, CLASS_ORDER)
        ax.set_xlabel("Flows (log scale)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    # The panels share the y-axis, so inverting it once puts Benign on top in both.
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, annotate_above: float = 0.9) -> Figure:
    """Lower-triangle heatmap of Pearson correlations on a fixed -1..1 diverging scale.

    I only print the value in cells with |r| >= `annotate_above`: those are the redundant pairs the
    text discusses, and labelling all 190 cells would bury them.
    """
    # Dropping the first row and last column removes the empty diagonal edge of the triangle.
    values = corr.to_numpy()[1:, :-1]
    rows, cols = corr.index[1:], corr.columns[:-1]
    masked = np.ma.masked_where(np.triu(np.ones_like(values, dtype=bool), k=1), values)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(masked, cmap=CORRELATION_CMAP, vmin=-1, vmax=1)
    for i in range(len(rows)):
        for j in range(i + 1):
            r = values[i, j]
            if abs(r) >= annotate_above:
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if abs(r) > 0.75 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02)
    bar.set_label("Pearson r on symlog-transformed values")
    bar.outline.set_visible(False)
    ax.set_title("Correlation of the 20 highest-variance features (training set)")
    fig.tight_layout()
    return fig


def plot_class_boxplots(X: pd.DataFrame, y: pd.Series, columns: list[str], ncols: int = 3) -> Figure:
    """Small multiples: one box plot per feature, one box per class, on a symlog y-axis.

    A symlog axis is linear near zero and logarithmic beyond, so it shows the zeros of FTP flows,
    the -1 window marker and multi-kilobyte values on one readable scale. Whiskers follow Tukey's
    1.5 x IQR rule, so the dots are exactly the points that rule calls outliers.
    """
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.3 * nrows), squeeze=False)
    sizes = y.value_counts()
    tick_labels = [f"{c.split('-')[0]}\n(n={sizes.get(c, 0):,})" for c in CLASS_ORDER]
    for ax, column in zip(axes.flat, columns):
        groups = [X.loc[y == c, column].to_numpy() for c in CLASS_ORDER]
        boxes = ax.boxplot(
            groups, widths=0.5, patch_artist=True, showfliers=True,
            medianprops={"linewidth": 2.2},
            flierprops={"marker": "o", "markersize": 2, "markeredgewidth": 0, "alpha": 0.35},
        )
        # The attack classes are so uniform that their boxes often collapse to a single line, so I
        # draw the median in the class colour: the class stays identifiable even with zero IQR.
        for patch, median, flier, cls in zip(boxes["boxes"], boxes["medians"], boxes["fliers"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[cls] + "59")  # ~35% opacity fill
            patch.set_edgecolor(CLASS_COLORS[cls])
            median.set_color(CLASS_COLORS[cls])
            flier.set_markerfacecolor(CLASS_COLORS[cls])
        for whisker in boxes["whiskers"] + boxes["caps"]:
            whisker.set_color(INK_MUTED)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(1, len(CLASS_ORDER) + 1), tick_labels, fontsize=8)
        ax.set_title(column)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Value (symlog scale)")
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_cv_comparison(scores: pd.DataFrame, metrics: dict[str, str]) -> Figure:
    """Dot plot of baseline versus tuned cross-validation scores, one panel per metric.

    `scores` has columns model, stage ("Baseline" or "Tuned"), metric, mean, std. Every score here
    is close to 1, so a bar chart starting at zero would hide all differences; I use dots with
    +/- 1 standard-deviation bars on a zoomed axis. Exact values are in the accompanying table, so I
    do not print them on the chart, where they would collide.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    positions = {model: i for i, model in enumerate(MODEL_ORDER)}
    for ax, (metric, label) in zip(axes, metrics.items()):
        subset = scores[scores["metric"] == metric]
        lows = (subset["mean"] - subset["std"]).min()
        # A score cannot exceed 1, so I clip the upper error bar there instead of drawing past it.
        highs = np.minimum(subset["mean"] + subset["std"], 1.0).max()
        for stage, ink, marker, offset in (("Baseline", BASELINE_INK, "o", -0.15), ("Tuned", TUNED_INK, "D", 0.15)):
            rows = subset[subset["stage"] == stage]
            ys = rows["model"].map(positions) + offset
            upper = np.minimum(rows["std"], 1.0 - rows["mean"])
            ax.errorbar(rows["mean"], ys, xerr=[rows["std"], upper], fmt=marker, color=ink,
                        markersize=6, capsize=3, linewidth=1.2, label=stage)
        span = max(highs - lows, 1e-4)
        ax.set_xlim(lows - span * 0.1, highs + span * 0.1)
        # Without this, matplotlib writes ticks as offsets from 1 ("-0.0002 +1"), which is unreadable.
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_title(label)
        ax.set_xlabel("Cross-validation score (mean ± 1 SD)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    axes[0].set_ylim(len(MODEL_ORDER) - 0.5, -0.6)
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_timing(timing: pd.DataFrame) -> Figure:
    """Two bar panels: training time (refit on 60,000 flows) and inference time per 1,000 flows.

    They are different quantities with different units, so I give each its own axis rather than a
    dual-axis chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), sharey=True)
    panels = [
        ("fit_seconds", "Training time on 60,000 flows (s)", "{:.2f} s"),
        ("predict_ms_per_1000", "Inference time per 1,000 flows (ms)", "{:.2f} ms"),
    ]
    timing = timing.set_index("model").reindex(MODEL_ORDER)
    positions = np.arange(len(MODEL_ORDER))
    for ax, (column, title, fmt) in zip(axes, panels):
        values = timing[column]
        ax.barh(positions, values, height=0.55, color=INK_SECONDARY)
        for pos, value in zip(positions, values):
            ax.text(value + values.max() * 0.02, pos, fmt.format(value), va="center", fontsize=9)
        ax.set_xlim(0, values.max() * 1.25)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(positions, MODEL_ORDER)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def _share_label(share: float) -> str:
    """Format a share with two decimals without rounding 1 in 50,000 to 0% or 50,856/50,857 to 100%."""
    if 0 < share < 0.0001:
        return "<0.01%"
    if 0.9999 < share < 1:
        return ">99.99%"
    return f"{share:.2%}"


def plot_confusion_matrices(results: dict[str, tuple[np.ndarray, np.ndarray]], title_suffix: str) -> Figure:
    """One confusion matrix per model: colour = share of the true class, text = count and share.

    I colour by the row-normalised share (recall) so the 37-flow FTP row is as visible as the
    50,000-flow Benign row; raw counts are printed so no information is lost.
    """
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, len(results), figsize=(4.3 * len(results), 4.1))
    axes = np.atleast_1d(axes)
    short = [c.split("-")[0] for c in CLASS_ORDER]
    cmap = LinearSegmentedColormap.from_list("recall", ["#ffffff", "#cde2fb", "#2a78d6", "#104281"])
    for ax, (name, (y_true, y_pred)) in zip(axes, results.items()):
        counts = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                # Two decimals, because 5 of 50,857 flows is 0.01%, which one decimal would print as 0.0%.
                label = "0" if counts[i, j] == 0 else f"{counts[i, j]:,}\n({_share_label(shares[i, j])})"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8, color="#ffffff" if shares[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(short)), short)
        ax.set_yticks(range(len(short)), short)
        ax.set_xlabel("Predicted class")
        ax.set_title(name, fontsize=10)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("True class")
    fig.suptitle(title_suffix, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cv_vs_test(table: pd.DataFrame, metrics: dict[str, str]) -> Figure:
    """Per metric: cross-validation mean ± SD (training set) next to the test score with its 95% CI.

    `table` has columns model, metric, cv_mean, cv_std, test, test_low, test_high. Showing both on
    one axis lets the reader see whether the test score falls inside the range the cross-validation
    predicted, which is the evidence that the model generalises rather than memorises.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    positions = {model: i for i, model in enumerate(MODEL_ORDER)}
    for ax, (metric, label) in zip(axes, metrics.items()):
        rows = table[table["metric"] == metric]
        ys = rows["model"].map(positions)
        cv_upper = np.minimum(rows["cv_std"], 1.0 - rows["cv_mean"])
        ax.errorbar(rows["cv_mean"], ys - 0.15, xerr=[rows["cv_std"], cv_upper], fmt="o",
                    color=BASELINE_INK, markersize=6, capsize=3, linewidth=1.2,
                    label="Cross-validation (mean ± 1 SD)")
        ax.errorbar(rows["test"], ys + 0.15,
                    xerr=[rows["test"] - rows["test_low"], rows["test_high"] - rows["test"]],
                    fmt="D", color=TUNED_INK, markersize=6, capsize=3, linewidth=1.2,
                    label="Test set (95% bootstrap CI)")
        low = min((rows["cv_mean"] - rows["cv_std"]).min(), rows["test_low"].min())
        span = max(1.0 - low, 1e-4)
        ax.set_xlim(low - span * 0.1, 1.0 + span * 0.05)
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_title(label)
        ax.set_xlabel("Score")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    axes[0].set_ylim(len(MODEL_ORDER) - 0.5, -0.6)
    # The legend sits below the panels: inside a panel it would cover an error bar.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    return fig


def plot_ablation(table: pd.DataFrame, variant_labels: dict[str, str]) -> Figure:
    """One panel per model; rows are pipeline variants, dots are CV macro F1 (mean ± 1 SD).

    The main pipeline is drawn in black and every variant in grey, with a vertical reference line at
    the main pipeline's score, so each change reads as a shift left or right of that line.
    """
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4.2 * len(MODEL_ORDER), 2.9), sharey=True)
    variants = list(variant_labels)
    low = (table["f1_macro_mean"] - table["f1_macro_std"]).min()
    for ax, model in zip(axes, MODEL_ORDER):
        rows = table[table["model"] == model].set_index("variant").reindex(variants)
        main = rows.loc["main", "f1_macro_mean"]
        ax.axvline(main, color=GRID, linewidth=1.5, zorder=0)
        for i, variant in enumerate(variants):
            mean, std = rows.loc[variant, "f1_macro_mean"], rows.loc[variant, "f1_macro_std"]
            ink = TUNED_INK if variant == "main" else BASELINE_INK
            ax.errorbar(mean, i, xerr=[[std], [min(std, 1.0 - mean)]], fmt="D" if variant == "main" else "o",
                        color=ink, markersize=6, capsize=3, linewidth=1.2)
        ax.set_xlim(low - (1 - low) * 0.1, 1.0 + (1 - low) * 0.05)
        ax.set_title(model)
        ax.set_xlabel("CV macro F1 (mean ± 1 SD)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(variants)), [variant_labels[v] for v in variants])
    axes[0].set_ylim(len(variants) - 0.5, -0.5)
    fig.tight_layout()
    return fig


def plot_importance(importance: pd.Series, title: str, xlabel: str) -> Figure:
    """Horizontal bars of the top features, largest at the top, values printed at the bar ends."""
    importance = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(importance) + 1))
    positions = np.arange(len(importance))
    ax.barh(positions, importance.to_numpy(), height=0.6, color=INK_SECONDARY)
    for pos, value in zip(positions, importance.to_numpy()):
        ax.text(value + importance.max() * 0.01, pos, f"{value:.3f}", va="center", fontsize=8)
    ax.set_yticks(positions, importance.index, fontsize=8.5)
    ax.set_xlim(0, importance.max() * 1.15)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
