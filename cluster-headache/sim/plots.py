"""Figures.

One palette, one style, applied everywhere.  Categorical hues are assigned in fixed
order and never cycled; magnitude uses a single-hue sequential ramp; every multi-series
plot carries a legend and a distinct marker, so identity is never colour alone.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a84"
GRID = "#e4e3de"

# validated categorical order (adjacent-pair CVD dE 9.1, normal-vision 19.6, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# single-hue sequential ramp for magnitude
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#a9c7ec", "#5b95dd", "#2a78d6", "#134a8c"])
# two hues + neutral midpoint for polarity
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#c9c8c2", "#eb6834"])

_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
}


def use_style() -> None:
    plt.rcParams.update(_STYLE)


def new_fig(*args, **kwargs):
    use_style()
    return plt.subplots(*args, **kwargs)


def suptitle(fig, text: str) -> None:
    """Figure title placed above the axes rather than on top of the first subplot title."""
    fig.tight_layout()
    fig.suptitle(text, x=0.005, y=1.005, ha="left", va="bottom", fontsize=12.5, color=INK)


def finish(fig, path: str, caption: Optional[str] = None) -> str:
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
        fig.subplots_adjust(bottom=0.16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Reusable panels
# --------------------------------------------------------------------------------------


def series_plot(
    ax,
    x: Sequence[float],
    ys: Dict[str, Sequence[float]],
    errs: Optional[Dict[str, Sequence[float]]] = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    label_last: bool = True,
) -> None:
    for i, (name, y) in enumerate(ys.items()):
        colour = SERIES[i % len(SERIES)]
        marker = MARKERS[i % len(MARKERS)]
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, color=colour, marker=marker, markersize=5.5, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3 + i)
        if errs and name in errs:
            e = np.asarray(errs[name], dtype=float)
            ax.fill_between(x, y - e, y + e, color=colour, alpha=0.13, linewidth=0, zorder=2)
        if label_last and len(x):
            ax.annotate(name, (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                        fontsize=8, color=INK_2, va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=10)


def node_map(ax, positions: np.ndarray, values: np.ndarray, title: str = "",
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             size: float = 7.0, cmap=SEQ, cbar_label: str = ""):
    sc = ax.scatter(positions[:, 0], positions[:, 1], c=values, s=size, cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=8)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK_MUTED)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8, color=INK_2)
    return sc


def heat(ax, mat: np.ndarray, title: str = "", cmap=SEQ, cbar_label: str = "", log: bool = False):
    m = np.log10(np.maximum(mat, 1e-12)) if log else mat
    im = ax.imshow(m, cmap=cmap, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=8, fontsize=9.5)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK_MUTED)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8, color=INK_2)
    return im


def grid_phase_map(ax, theta: np.ndarray, grid_shape: Tuple[int, int], title: str = ""):
    """Phase field on a lattice, drawn as an image rather than a scatter of dots."""
    im = ax.imshow(np.mod(theta, 2 * np.pi).reshape(grid_shape), cmap="twilight",
                   vmin=0, vmax=2 * np.pi, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=6, fontsize=9.5)
    return im


def phase_map(ax, positions: np.ndarray, theta: np.ndarray, title: str = "", size: float = 7.0):
    """Phase is circular, so it gets a cyclic map - the one place a non-monotone ramp is
    correct."""
    sc = ax.scatter(positions[:, 0], positions[:, 1], c=np.mod(theta, 2 * np.pi),
                    s=size, cmap="twilight", vmin=0, vmax=2 * np.pi, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=8, fontsize=9.5)
    return sc
