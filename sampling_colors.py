"""Color mapping for sampling methods used in plots."""

from __future__ import annotations

# More saturated palette for better contrast
_COLOR_MAP = {
    "ibs": "#4D8CFF",      # vibrant blue
    "naive": "#FF6B6B",    # strong red
    "fixed": "#FFA45B",    # saturated orange
    "laplace": "#FFEB3B",  # bright yellow
    "jeffreys": "#75FF63", # vivid green
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
