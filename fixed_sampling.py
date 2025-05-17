"""Utility functions for fixed sampling log-probability estimates."""

from __future__ import annotations

from typing import Iterable, Callable, Dict

import numpy as np


def _log_fixed_smoothing(observations: Iterable[float], alpha: float, beta: float) -> float:
    """Return log probability estimate with additive smoothing.

    Parameters
    ----------
    observations : iterable
        Sequence of Bernoulli observations (0 or 1 values).
    alpha : float
        Prior success count added to the numerator.
    beta : float
        Prior total count added to the denominator.

    Returns
    -------
    float
        The log probability estimate.
    """
    x = np.asarray(list(observations), dtype=float)
    M = x.size
    if M == 0:
        raise ValueError("observations must be non-empty")
    m = np.sum(x)
    return float(np.log((m + alpha) / (M + beta)))


def naive_log_prob(observations: Iterable[float]) -> float:
    """Naive log probability using the empirical mean."""
    return _log_fixed_smoothing(observations, 0.0, 0.0)


def fixed_plus1_log_prob(observations: Iterable[float]) -> float:
    """Log probability with +1 smoothing on numerator and denominator."""
    return _log_fixed_smoothing(observations, 1.0, 1.0)


def laplace_log_prob(observations: Iterable[float]) -> float:
    """Standard Laplace smoothing (+1 on counts and totals)."""
    return _log_fixed_smoothing(observations, 1.0, 2.0)


def jeffreys_log_prob(observations: Iterable[float]) -> float:
    """Jeffreys prior smoothing using a Beta(0.5, 0.5) prior."""
    return _log_fixed_smoothing(observations, 0.5, 1.0)


# Dictionary of available fixed-sampling estimators
FIXED_SAMPLING_METHODS: Dict[str, Callable[[Iterable[float]], float]] = {
    "naive": naive_log_prob,
    "fixed": fixed_plus1_log_prob,
    "laplace": laplace_log_prob,
    "jeffreys": jeffreys_log_prob,
}
