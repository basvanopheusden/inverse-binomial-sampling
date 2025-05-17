"""Color mapping for sampling methods used in plots."""

from __future__ import annotations

_COLOR_MAP = {
    "ibs": "tab:blue",
    "naive": "tab:orange",
    "fixed": "tab:green",
    "laplace": "tab:red",
    "jeffreys": "tab:purple",
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
