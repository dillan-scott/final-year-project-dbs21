from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

_TQDM_ORIG_INIT: dict = {}


def set_progress(enabled: bool = True) -> None:
    """
    Globally toggle every tqdm progress bar on or off.
    Forces the ``disable`` flag on every bar at construction time.
    """
    import importlib

    classes = []
    for modname in ("tqdm", "tqdm.std", "tqdm.notebook", "tqdm.auto"):
        try:
            classes.append(importlib.import_module(modname).tqdm)
        except Exception:
            pass

    for cls in set(classes):
        if cls not in _TQDM_ORIG_INIT:
            _TQDM_ORIG_INIT[cls] = cls.__init__
        orig_init = _TQDM_ORIG_INIT[cls]

        def make(orig):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = not enabled
                orig(self, *args, **kwargs)

            return __init__

        cls.__init__ = make(orig_init)


def apply_paper_style() -> None:
    """Set global matplotlib rcParams for thesis/paper-ready figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, name: str, figure_dir: str) -> None:
    """Save a figure as both vector PDF (for LaTeX) and PNG (300 dpi fallback)"""
    os.makedirs(figure_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figure_dir, f"{name}.{ext}"))


def save_latex(
    df,
    filename: str,
    table_dir: str,
    caption: str = "",
    label: str = "",
    float_fmt: str = "%.3f",
) -> None:
    """Export a DataFrame as a LaTeX table with consistent formatting"""
    os.makedirs(table_dir, exist_ok=True)
    latex = df.to_latex(
        float_format=float_fmt,
        escape=False,
        caption=caption,
        label=label,
        na_rep="\\textendash",
    )
    # LaTeX-safe: escape literal percents and the unicode +/- from agg()
    latex = latex.replace("%", r"\%").replace("±", r"$\pm$").replace("_", r"\_")
    with open(os.path.join(table_dir, filename), "w", encoding="utf-8") as f:
        f.write(latex)


def agg(values: Sequence[float]) -> str:
    """Format a sequence as 'mean ± std' for tables (siunitx-compatible)"""
    vals = np.asarray(values, dtype=float)
    return f"{vals.mean():.3f} ± {vals.std(ddof=0):.3f}"
