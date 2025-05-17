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
    """Return the color for the given sampling method name."""
    try:
        return _COLOR_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unknown sampling method '{name}'") from exc
