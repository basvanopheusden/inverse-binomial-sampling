"""Color mapping for sampling methods used in plots."""

from __future__ import annotations

_COLOR_MAP = {
    "ibs": "tab:blue",
    "fixed_1": "tab:orange",
    "fixed_2": "tab:green",
    "fixed_3": "tab:red",
    "fixed_5": "tab:purple",
    "fixed_8": "tab:brown",
    "fixed_10": "tab:pink",
}


def get_color(name: str) -> str:
    """Return the color for the given sampling method name."""
    try:
        return _COLOR_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown sampling method '{name}'") from exc
