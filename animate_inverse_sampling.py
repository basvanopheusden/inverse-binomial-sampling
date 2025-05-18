from __future__ import annotations

"""Simple animation illustrating inverse sampling efficiency."""

import math
import random

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import digamma

from setup_matplotlib import setup_matplotlib
from sampling_colors import get_color


def bernoulli_model(rng: random.Random):
    """Return a simulator for a Bernoulli trial using ``rng``."""

    def model(p: float, _: object) -> int:
        return int(rng.random() < p)

    return model


def ibs_sample_path(model, theta: float, observed: int) -> list[int]:
    """Generate samples until ``model`` matches ``observed``."""
    path: list[int] = []
    while True:
        s = model(theta, None)
        path.append(s)
        if s == observed:
            break
    return path


def fixed_sample_path(model, theta: float, M: int) -> list[int]:
    """Generate ``M`` samples from ``model``."""
    return [model(theta, None) for _ in range(M)]


def main() -> None:
    setup_matplotlib()
    rng = random.Random(42)
    p = 0.3
    observed = 1
    M = 5

    model = bernoulli_model(rng)
    ibs_path = ibs_sample_path(model, p, observed)
    fixed_path = fixed_sample_path(model, p, M)

    k = len(ibs_path)
    ibs_ll = digamma(1) - digamma(k)
    fixed_ll = math.log(sum(fixed_path) / M)

    frames = max(k, M)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_title("Inverse sampling")
    axes[1].set_title(f"Fixed sampling (M={M})")
    for ax in axes:
        ax.set_xlim(0, frames + 1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Sample #")
        ax.set_yticks([0, 1])
        ax.axhline(observed, color="gray", linestyle="--")

    ibs_scatter = axes[0].scatter([], [], color=get_color("ibs"))
    fixed_scatter = axes[1].scatter([], [], color=get_color("fixed"))
    ibs_text = axes[0].text(0.05, 0.9, "", transform=axes[0].transAxes)
    fixed_text = axes[1].text(0.05, 0.9, "", transform=axes[1].transAxes)

    def init() -> tuple:
        ibs_scatter.set_offsets([])
        fixed_scatter.set_offsets([])
        ibs_text.set_text("")
        fixed_text.set_text("")
        return ibs_scatter, fixed_scatter, ibs_text, fixed_text

    def update(frame: int) -> tuple:
        if frame < k:
            offsets = ibs_scatter.get_offsets()
            new = list(offsets) + [[frame + 1, ibs_path[frame]]]
            ibs_scatter.set_offsets(new)
        if frame < M:
            offsets = fixed_scatter.get_offsets()
            new = list(offsets) + [[frame + 1, fixed_path[frame]]]
            fixed_scatter.set_offsets(new)
        if frame == frames - 1:
            ibs_text.set_text(f"Samples: {k}\nLL = {ibs_ll:.2f}")
            fixed_text.set_text(f"Samples: {M}\nLL = {fixed_ll:.2f}")
        return ibs_scatter, fixed_scatter, ibs_text, fixed_text

    # Keep a reference to the animation object to prevent it from
    # being garbage collected before ``plt.show`` renders it.
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=500,
        repeat=False,
    )
    plt.show()
    return anim


if __name__ == "__main__":
    _anim = main()
