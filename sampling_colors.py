"""Color mapping for sampling methods used in plots."""

from __future__ import annotations

# Pastel palette from colorhunt.co
_COLOR_MAP = {
    "ibs": "#A0C4FF",
    "naive": "#FFADAD",
    "fixed": "#FFD6A5",
    "laplace": "#FDFFB6",
    "jeffreys": "#CAFFBF",
}


def get_color(name: str) -> str:
    """Return the color for the given sampling method name.

    Parameters
    ----------
    name : str
        Name of the sampling method. For fixed sampling methods ``name`` may be
        suffixed with the number of samples (e.g. ``"fixed_2"``).  Such names
        are mapped to the base ``"fixed"`` color.
    """

    base_name = name.lower()
    if base_name.startswith("fixed_"):
        base_name = "fixed"

    try:
        return _COLOR_MAP[base_name]
    except KeyError as exc:
        raise ValueError(f"Unknown sampling method '{name}'") from exc
