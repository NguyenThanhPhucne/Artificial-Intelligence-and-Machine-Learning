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
