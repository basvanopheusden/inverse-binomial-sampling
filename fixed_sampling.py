"""Utility functions for fixed sampling log-probability estimates."""

from __future__ import annotations

from typing import Iterable, Callable, Dict, Tuple

import math

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

# Mapping from method name to the corresponding prior parameters
FIXED_SAMPLING_PARAMS: Dict[str, Tuple[float, float]] = {
    "fixed": (1.0, 1.0),
    "laplace": (1.0, 2.0),
    "jeffreys": (0.5, 1.0),
}


def _single_fixed_moments(p: float, M: int, alpha: float, beta: float) -> Tuple[float, float]:
    """Return mean and variance of the log-prob estimate for one trial."""
    log_vals = []
    probs = []
    for k in range(M + 1):
        prob = math.comb(M, k) * (p ** k) * ((1.0 - p) ** (M - k))
        val = math.log((k + alpha) / (M + beta))
        log_vals.append(val)
        probs.append(prob)

    mean = sum(pv * lv for pv, lv in zip(probs, log_vals))
    var = sum(pv * (lv - mean) ** 2 for pv, lv in zip(probs, log_vals))
    return mean, var


def fixed_analytical_mean(probabilities: Iterable[float], M: int, method: str) -> float:
    """Analytical mean of a fixed-sample log-likelihood estimator."""
    if method not in FIXED_SAMPLING_PARAMS:
        raise ValueError(f"Unknown method '{method}'")
    alpha, beta = FIXED_SAMPLING_PARAMS[method]
    return float(
        sum(_single_fixed_moments(p, M, alpha, beta)[0] for p in probabilities)
    )


def fixed_analytical_variance(probabilities: Iterable[float], M: int, method: str) -> float:
    """Analytical variance of a fixed-sample log-likelihood estimator."""
    if method not in FIXED_SAMPLING_PARAMS:
        raise ValueError(f"Unknown method '{method}'")
    alpha, beta = FIXED_SAMPLING_PARAMS[method]
    return float(
        sum(_single_fixed_moments(p, M, alpha, beta)[1] for p in probabilities)
    )
