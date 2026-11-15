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
from matplotlib.ticker import MaxNLocator

from . import config
from .models import MODEL_NAMES

CLASS_ORDER = ["Benign", "FTP-BruteForce", "SSH-Bruteforce"]
MODEL_ORDER = list(MODEL_NAMES)
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
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.42 * len(MODEL_ORDER) + 1.3),
                             sharey=True)
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
        # Scores differ only in the fourth decimal, so long tick labels would overlap; four ticks suffice.
        ax.xaxis.set_major_locator(MaxNLocator(4))
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(MODEL_ORDER) + 1.2), sharey=True)
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

    ncols = min(4, len(results))
    nrows = int(np.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.0 * nrows), squeeze=False)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    axes = axes.flat
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
    for ax in list(axes)[::ncols]:
        ax.set_ylabel("True class")
    fig.suptitle(title_suffix, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cv_vs_test(table: pd.DataFrame, metrics: dict[str, str]) -> Figure:
    """Per metric: cross-validation mean ± SD (training set) next to the test score with its 95% CI.

    `table` has columns model, metric, cv_mean, cv_std, test, test_low, test_high. Showing both on
    one axis lets the reader see whether the test score falls inside the range the cross-validation
    predicted, which is the evidence that the model generalises rather than memorises.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 0.42 * len(MODEL_ORDER) + 1.3),
                             sharey=True)
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
        ax.xaxis.set_major_locator(MaxNLocator(4))
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
    ncols = 4
    nrows = int(np.ceil(len(MODEL_ORDER) / ncols))
    variants = list(variant_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, (0.3 * len(variants) + 1.0) * nrows),
                             sharey=True, squeeze=False)
    low = (table["f1_macro_mean"] - table["f1_macro_std"]).min()
    for ax, model in zip(axes.flat, MODEL_ORDER):
        rows = table[table["model"] == model].set_index("variant").reindex(variants)
        main = rows.loc["main", "f1_macro_mean"]
        ax.axvline(main, color=GRID, linewidth=1.5, zorder=0)
        for i, variant in enumerate(variants):
            mean, std = rows.loc[variant, "f1_macro_mean"], rows.loc[variant, "f1_macro_std"]
            if np.isnan(mean):  # variant not applicable to this model (no class_weight parameter)
                ax.text(low, i, "  n/a", va="center", fontsize=8, color=INK_MUTED)
                continue
            ink = TUNED_INK if variant == "main" else BASELINE_INK
            ax.errorbar(mean, i, xerr=[[std], [min(std, 1.0 - mean)]], fmt="D" if variant == "main" else "o",
                        color=ink, markersize=6, capsize=3, linewidth=1.2)
        ax.set_xlim(low - (1 - low) * 0.1, 1.0 + (1 - low) * 0.05)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("CV macro F1 (mean ± 1 SD)", fontsize=8)
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(variants)), [variant_labels[v] for v in variants], fontsize=8)
        ax.set_ylim(len(variants) - 0.5, -0.5)
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


# --- Figures for techniques taught in the module (EDA, scaling, comparison, learning curves) -------

def plot_histograms(X: pd.DataFrame, columns: list[str], stats: pd.DataFrame, ncols: int = 3) -> Figure:
    """Univariate histograms of the training set, one panel per feature, skewness in the title.

    On a linear axis these heavy-tailed features put almost every flow in the first bin, so the
    x-axis uses symlog (linear near zero, logarithmic beyond) with bins spaced on the same scale; the
    title keeps the skewness and kurtosis computed on the raw values, which is what the shape means.
    """
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.9 * nrows), squeeze=False)
    for ax, column in zip(axes.flat, columns):
        values = X[column].to_numpy(dtype=float)
        transformed = np.sign(values) * np.log1p(np.abs(values))
        edges = np.linspace(transformed.min(), transformed.max(), 41)
        bins = np.sign(edges) * np.expm1(np.abs(edges))
        ax.hist(values, bins=np.unique(bins), color=INK_SECONDARY, alpha=0.85)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_title(f"{column}\nskew {stats.loc[column, 'skewness']:.1f}, kurtosis {stats.loc[column, 'kurtosis']:.0f}",
                     fontsize=9.5)
        ax.set_xlabel("Value (symlog scale)", fontsize=8)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Flows (log scale)")
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_scatter(X: pd.DataFrame, y: pd.Series, x_col: str, y_col: str, max_per_class: int = 3_000,
                 seed: int = config.SEED) -> Figure:
    """Bivariate scatter plot of two features coloured by class, on symlog axes.

    Benign has ~51,000 flows, so I draw at most `max_per_class` random flows per class (all 37 FTP
    flows are kept) and draw the rare classes last so they are not hidden under benign points.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    rng = np.random.default_rng(seed)
    # Attack flows are so uniform that thousands of them share one (x, y) pair and are drawn as a
    # single dot; the legend states how many distinct points each class really has.
    distinct = X[[x_col, y_col]].groupby(y, observed=True).apply(lambda g: len(g.drop_duplicates())).to_dict()
    for cls in CLASS_ORDER:
        rows = np.flatnonzero(y.to_numpy() == cls)
        if len(rows) > max_per_class:
            rows = rng.choice(rows, max_per_class, replace=False)
        ax.scatter(X[x_col].iloc[rows], X[y_col].iloc[rows], s=10 if cls == "Benign" else 22,
                   alpha=0.35 if cls == "Benign" else 0.8, color=CLASS_COLORS[cls], edgecolors="none",
                   label=f"{cls} (n={(y == cls).sum():,}; {distinct[cls]:,} distinct points)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(f"{x_col} (symlog scale)")
    ax.set_ylabel(f"{y_col} (symlog scale)")
    ax.set_title(f"{y_col} against {x_col}, training set")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    return fig


def plot_scaling_comparison(raw: pd.Series, standard: np.ndarray, minmax: np.ndarray) -> Figure:
    """One feature before scaling, after StandardScaler and after MinMaxScaler (fitted on train).

    Both scalers are linear, so the shape of the distribution never changes; only the axis does.
    I print the resulting range under each panel because that is the difference the lecture stresses:
    standardisation keeps outliers far from the centre, min-max squeezes the bulk of the data into a
    tiny part of [0, 1] when one extreme value exists.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    panels = [(raw.to_numpy(), "Raw values"), (standard, "StandardScaler (mean 0, SD 1)"),
              (minmax, "MinMaxScaler (range 0 to 1)")]
    for ax, (values, title) in zip(axes, panels):
        ax.hist(values, bins=60, color=INK_SECONDARY)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"min {values.min():,.3g}, median {np.median(values):,.3g}, max {values.max():,.3g}",
                      fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Flows (log scale)")
    fig.suptitle(raw.name, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_metric_bars(table: pd.DataFrame, metrics: dict[str, str]) -> Figure:
    """Bar chart per metric comparing every model on the same test-set scores (template step 7).

    All scores are above 0.98, so bars drawn from zero would look identical; each panel's axis
    starts just below the lowest score, which I state in the axis label, and every bar carries its
    exact value so the chart cannot exaggerate a difference.
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 0.36 * len(MODEL_ORDER) + 1.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    positions = np.arange(len(MODEL_ORDER))
    for ax, (metric, label) in zip(axes, metrics.items()):
        values = table.reindex(MODEL_ORDER)[metric]
        start = np.floor((values.min() - (1 - values.min()) * 0.3) * 1000) / 1000
        ax.barh(positions, values - start, left=start, height=0.6, color=INK_SECONDARY)
        for pos, value in zip(positions, values):
            ax.text(value, pos, f" {value:.4f}", va="center", fontsize=8)
        ax.set_xlim(start, 1 + (1 - start) * 0.35)
        # Three ticks from the axis start to 1.0: a score cannot exceed 1, so no tick is drawn beyond it.
        ax.set_xticks(np.linspace(start, 1.0, 3))
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_title(label)
        ax.set_xlabel(f"Test score (axis starts at {start:.3f})", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(positions, MODEL_ORDER)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def plot_learning_curves(curves: dict[str, pd.DataFrame], ncols: int = 4) -> Figure:
    """Learning curves: training- and validation-fold macro F1 against training-set size.

    A large, persistent gap means over-fitting; two low curves mean under-fitting; two curves that
    meet near 1.0 mean more data would not help. All panels share the y-axis so models compare
    directly.
    """
    nrows = int(np.ceil(len(curves) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), sharey=True, squeeze=False)
    low = min(min((c["valid_mean"] - c["valid_std"]).min(), c["train_mean"].min()) for c in curves.values())
    for ax, (name, curve) in zip(axes.flat, curves.items()):
        for column, ink, marker, label in (("train", BASELINE_INK, "o", "Training folds"),
                                           ("valid", TUNED_INK, "D", "Validation folds")):
            ax.plot(curve["train_flows"], curve[f"{column}_mean"], marker=marker, color=ink,
                    markersize=4, linewidth=1.2, label=label)
            ax.fill_between(curve["train_flows"], curve[f"{column}_mean"] - curve[f"{column}_std"],
                            np.minimum(curve[f"{column}_mean"] + curve[f"{column}_std"], 1.0),
                            color=ink, alpha=0.12, linewidth=0)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(max(low - (1 - low) * 0.1, 0), 1.0 + (1 - low) * 0.05)
        ax.set_xlabel("Training flows", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro F1")
    for ax in list(axes.flat)[len(curves):]:
        ax.set_visible(False)
    axes[0, 0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_decision_tree(pipeline, max_depth: int = 3) -> Figure:
    """Draw the fitted Decision Tree with scikit-learn's plot_tree (as in the Week 4 lab).

    The tree splits standardised inputs, so thresholds on the drawing are z-scores; the raw-unit
    rules are printed separately by `evaluation.tree_rules_in_original_units`.
    """
    from sklearn.tree import plot_tree

    model = pipeline.named_steps["model"]
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    fig, ax = plt.subplots(figsize=(13, 5.5))
    boxes = plot_tree(model, max_depth=max_depth, feature_names=names, class_names=list(model.classes_),
                      filled=False, impurity=False, proportion=False, rounded=True, fontsize=7, ax=ax)
    # plot_tree's own palette would give Benign, FTP and SSH different colours from every other figure,
    # so I recolour each node with its majority class colour, paler when the node is less pure.
    for box in boxes:
        text = box.get_text()
        if "class = " not in text:
            continue
        cls = text.rsplit("class = ", 1)[1].strip()
        counts = [float(v) for v in text.split("value = [", 1)[1].split("]", 1)[0].split(",")]
        purity = max(counts) / sum(counts)
        # Blend with white instead of using transparency, so the arrows behind a node stay hidden.
        weight = 0.25 + 0.5 * purity
        rgb = np.array(mpl.colors.to_rgb(CLASS_COLORS.get(cls, SURFACE)))
        box.get_bbox_patch().set_facecolor(tuple(1 - weight * (1 - rgb)))
    ax.set_title(f"Decision Tree (depth {model.get_depth()}, {model.get_n_leaves()} leaves): first {max_depth} levels; "
                 f"thresholds in standardised units; samples include the synthetic SMOTE flows")
    fig.tight_layout()
    return fig
