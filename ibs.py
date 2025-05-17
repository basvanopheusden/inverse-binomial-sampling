"""Utility functions for inverse binomial sampling.

This module implements the core routine for estimating the log-likelihood of
observed discrete data with a simulator model using inverse binomial sampling
as described in the paper:
"Unbiased and Efficient Log-Likelihood Estimation with Inverse Binomial Sampling"
(arXiv:2001.03985v3).
"""

from typing import Iterable, Callable, Tuple

import numpy as np
from scipy.special import digamma, polygamma


def _ibs_single(
    stimuli: Iterable,
    responses: Iterable,
    model: Callable[[object, object], object],
    theta: object,
) -> Tuple[float, float, int]:
    """Run a single IBS estimate over all trials.

    Parameters
    ----------
    stimuli : iterable
        Sequence of stimuli presented in each trial.
    responses : iterable
        Sequence of observed discrete responses.
    model : callable
        Function implementing the simulator. It should return a simulated
        response when called as ``model(theta, stimulus)``.
    theta : object
        Parameters passed to the simulator.

    Returns
    -------
    log_lik : float
        Estimated log-likelihood for the dataset.
    variance : float
        Estimated variance of the log-likelihood estimate.
    samples : int
        Total number of model samples drawn.
    """
    log_lik = 0.0
    variance = 0.0
    samples = 0

    for s, r_obs in zip(stimuli, responses):
        k = 1
        while True:
            r = model(theta, s)
            if r == r_obs:
                break
            k += 1
        log_lik += digamma(1) - digamma(k)
        variance += polygamma(1, 1) - polygamma(1, k)
        samples += k

    return log_lik, variance, samples


def ibs_loglikelihood(
    stimuli: Iterable,
    responses: Iterable,
    model: Callable[[object, object], object],
    theta: object,
    repeats: int = 1,
) -> Tuple[float, float, int]:
    """Estimate log-likelihood via inverse binomial sampling.

    Parameters
    ----------
    stimuli : iterable
        Sequence of stimuli for all trials.
    responses : iterable
        Observed discrete responses for all trials.
    model : callable
        Simulator function ``model(theta, stimulus) -> response``.
    theta : object
        Parameters passed to the simulator.
    repeats : int, optional
        Number of independent repeats of IBS to average. Defaults to 1.

    Returns
    -------
    log_like : float
        The averaged log-likelihood estimate.
    variance : float
        Estimated variance of the averaged log-likelihood.
    total_samples : int
        Total number of simulator samples used across all repeats.
    """
    if repeats < 1:
        raise ValueError("repeats must be a positive integer")

    log_vals = []
    var_vals = []
    total_samples = 0

    for _ in range(repeats):
        ll, var, n = _ibs_single(stimuli, responses, model, theta)
        log_vals.append(ll)
        var_vals.append(var)
        total_samples += n

    log_like = float(np.mean(log_vals))
    variance = float(np.sum(var_vals)) / (repeats ** 2)
    return log_like, variance, total_samples


def ibs_analytical_mean(probabilities: Iterable[float]) -> float:
    """Analytical mean of the IBS log-likelihood estimate.

    Parameters
    ----------
    probabilities : iterable of float
        Probability that the simulator reproduces the observed response for each
        trial.

    Returns
    -------
    float
        Expected value of the IBS log-likelihood estimator.
    """
    return float(np.sum(np.log(probabilities)))


def _dilogarithm(x: float, tol: float = 1e-12, max_iter: int = 100000) -> float:
    """Compute ``Li_2(x)`` for ``0 \u2264 x < 1`` using its series expansion."""
    result = 0.0
    term = x
    k = 1
    while abs(term) > tol and k < max_iter:
        result += term / (k * k)
        k += 1
        term *= x
    return result


def ibs_analytical_variance(probabilities: Iterable[float]) -> float:
    """Analytical variance of the IBS log-likelihood estimate.

    Parameters
    ----------
    probabilities : iterable of float
        Probability that the simulator reproduces the observed response for each
        trial.

    Returns
    -------
    float
        Variance of the IBS log-likelihood estimator.
    """
    return float(np.sum(_dilogarithm(1.0 - p) for p in probabilities))
