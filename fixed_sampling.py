# -*- coding: utf-8 -*-
"""Utility helpers for fixed-sample log-likelihood estimation.

This module collects different ways of estimating the log-probability
from a fixed number of simulator samples.  Each method expects the
number of matching simulator outputs (``matches``) and the total number
of samples ``M``.  The methods are exposed via :data:`FIXED_METHODS`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict


def _naive(matches: int, M: int) -> float:
    """Return ``log(matches / M)`` or ``-inf`` if ``matches`` is zero."""
    if M <= 0:
        raise ValueError("M must be positive")
    return math.log(matches / M) if matches else float("-inf")


def _laplace(matches: int, M: int) -> float:
    """Return Laplace-smoothed log probability ``log((m + 1)/(M + 1))``."""
    if M < 0:
        raise ValueError("M must be non-negative")
    return math.log((matches + 1) / (M + 1))


def _jeffreys(matches: int, M: int) -> float:
    """Jeffreys prior ``log((m + 0.5)/(M + 1))``."""
    if M < 0:
        raise ValueError("M must be non-negative")
    return math.log((matches + 0.5) / (M + 1))


def _clipped(matches: int, M: int, eps: float = 1e-12) -> float:
    """Naive estimate with clipping to avoid log(0)."""
    if M <= 0:
        raise ValueError("M must be positive")
    p = matches / M
    if p <= 0.0:
        p = eps
    elif p >= 1.0:
        p = 1.0 - eps
    return math.log(p)


# Mapping of method name to log-probability estimator
FIXED_METHODS: Dict[str, Callable[[int, int], float]] = {
    "naive": _naive,
    "laplace": _laplace,
    "jeffreys": _jeffreys,
    "clipped": _clipped,
}


def fixed_loglikelihood(
    stimuli,
    responses,
    model: Callable[[object, object], object],
    theta: object,
    M: int,
    method: str = "naive",
) -> tuple[float, int]:
    """Estimate log-likelihood with ``M`` samples per trial.

    Parameters
    ----------
    stimuli, responses
        Sequences describing the observations.
    model
        Simulator returning a response for ``(theta, stimulus)``.
    theta
        Parameters passed to ``model``.
    M : int
        Number of simulator samples per trial.
    method : str, optional
        Name of the log-probability estimator from :data:`FIXED_METHODS`.

    Returns
    -------
    log_lik : float
        Estimated log-likelihood.
    total_samples : int
        Total number of simulator samples used.
    """
    if method not in FIXED_METHODS:
        raise KeyError(f"Unknown method '{method}'")
    log_fn = FIXED_METHODS[method]

    log_lik = 0.0
    for s, r_obs in zip(stimuli, responses):
        matches = 0
        for _ in range(M):
            if model(theta, s) == r_obs:
                matches += 1
        log_lik += log_fn(matches, M)
    total_samples = M * len(stimuli)
    return log_lik, total_samples
