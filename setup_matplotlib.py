"""Project-wide matplotlib configuration helper.

Call ``setup_matplotlib()`` at the start of scripts to apply a
consistent and visually pleasing style to all figures.
"""

from __future__ import annotations

from matplotlib import pyplot as plt

# Default style parameters used across the project
_DEFAULT_STYLE = {
    "figure.figsize": (6, 4),
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
}


def setup_matplotlib() -> None:
    """Apply the default matplotlib style for this project."""
    plt.rcParams.update(_DEFAULT_STYLE)

