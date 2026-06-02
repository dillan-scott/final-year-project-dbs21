from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def apply_paper_style() -> None:
    """Set global matplotlib rcParams for thesis/paper-ready figures."""
    plt.rcParams.update({
        'font.family':     'serif',
        'font.serif':      ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
        'font.size':       10,
        'axes.titlesize':  10,
        'axes.labelsize':  10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'savefig.dpi':     300,
        'savefig.bbox':    'tight',
        'lines.linewidth': 1.6,
        'axes.grid':       True,
        'grid.alpha':      0.3,
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
    })


def save_fig(fig, name: str, figure_dir: str) -> None:
    """Save a figure as both vector PDF (for LaTeX) and PNG (300 dpi fallback)"""
    os.makedirs(figure_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(figure_dir, f'{name}.{ext}'))


def save_latex(
    df,
    filename: str,
    table_dir: str,
    caption: str = '',
    label: str = '',
    float_fmt: str = '%.3f',
) -> None:
    """Export a DataFrame as a LaTeX table with consistent formatting"""
    os.makedirs(table_dir, exist_ok=True)
    df.to_latex(
        os.path.join(table_dir, filename),
        float_format=float_fmt,
        escape=False,
        caption=caption,
        label=label,
        na_rep='\\textendash',
    )


def agg(values: Sequence[float]) -> str:
    """Format a sequence as 'mean ± std' for tables (siunitx-compatible)"""
    vals = np.asarray(values, dtype=float)
    return f'{vals.mean():.3f} ± {vals.std(ddof=0):.3f}'
