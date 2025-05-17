# Script to compare inverse binomial sampling with a fixed-sample approach
# for estimating log-likelihood in a Bernoulli model.

import math
import random
import statistics
from typing import Iterable

from ibs import ibs_loglikelihood
from fixed_sampling import FIXED_SAMPLING_METHODS
from setup_matplotlib import setup_matplotlib
from sampling_colors import get_color

try:
    from matplotlib import pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit('matplotlib is required to run this script') from e


def bernoulli_model(rng):
    """Return a simulator for a Bernoulli trial using ``rng``."""

    def model(p, _):
        return rng.random() < p

    return model


def _mean_std_ignore_inf(values: Iterable[float]) -> tuple[float, float]:
    """Return mean and stdev ignoring infinite or NaN values."""
    values = list(values)
    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        if values and all(v == float("-inf") for v in values):
            return float("-inf"), 0.0
        if values and all(v == float("inf") for v in values):
            return float("inf"), 0.0
        return float("nan"), float("nan")
    mean_val = statistics.mean(finite_vals)
    std_val = statistics.stdev(finite_vals) if len(finite_vals) > 1 else 0.0
    return mean_val, std_val

def fixed_loglikelihood(stimuli, responses, model, theta, M, method):
    """Estimate log-likelihood using a fixed number of samples per trial.

    Parameters
    ----------
    stimuli : iterable
        Input stimuli for each trial.
    responses : iterable
        Observed binary responses for each trial.
    model : callable
        Simulator function ``model(theta, stimulus) -> response``.
    theta : object
        Parameters passed to the simulator.
    M : int
        Number of model samples per trial.
    method : str
        Name of the fixed sampling method to use (see
        ``FIXED_SAMPLING_METHODS``).
    """

    try:
        log_prob_fn = FIXED_SAMPLING_METHODS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown fixed sampling method '{method}'") from exc

    log_lik = 0.0
    for s, r_obs in zip(stimuli, responses):
        obs = []
        for _ in range(M):
            obs.append(1.0 if model(theta, s) == r_obs else 0.0)
        log_lik += log_prob_fn(obs)

    total_samples = M * len(responses)
    return log_lik, total_samples


def run_experiment(p=0.3, n_trials=20, repetitions=100, seed=123):
    rng = random.Random(seed)
    stimuli = [None] * n_trials
    responses = [rng.random() < p for _ in range(n_trials)]
    true_ll = sum(math.log(p if r else 1.0 - p) for r in responses)

    # Use a smaller range for the IBS repeat counts so that the
    # resulting average sample counts for the two methods are
    # roughly comparable.  The fixed sampler draws exactly
    # ``M`` samples per trial while a single IBS repeat requires
    # about twice as many samples on average.  Limiting the number
    # of repeats keeps both curves on a similar scale.
    fixed_samples = [1, 2, 3, 5, 8, 10]
    ibs_repeats = [1, 2, 3, 4, 5]

    results = {"ibs": []}
    for method in FIXED_SAMPLING_METHODS:
        results[method] = []

    for method in FIXED_SAMPLING_METHODS:
        for M in fixed_samples:
            estimates = []
            samples = []
            for rep in range(repetitions):
                rng_run = random.Random(seed + rep)
                ll, n = fixed_loglikelihood(
                    stimuli,
                    responses,
                    bernoulli_model(rng_run),
                    p,
                    M,
                    method,
                )
                estimates.append(ll)
                samples.append(n)
            mean_ll, std_ll = _mean_std_ignore_inf(estimates)
            avg_samples = statistics.mean(samples)
            results[method].append((avg_samples, mean_ll, std_ll))

    for R in ibs_repeats:
        estimates = []
        samples = []
        for rep in range(repetitions):
            rng_run = random.Random(seed + rep)
            ll, _, n = ibs_loglikelihood(stimuli, responses,
                                         bernoulli_model(rng_run), p,
                                         repeats=R)
            estimates.append(ll)
            samples.append(n)
        mean_ll, std_ll = _mean_std_ignore_inf(estimates)
        avg_samples = statistics.mean(samples)
        results["ibs"].append((avg_samples, mean_ll, std_ll))

    return results, true_ll


def main():
    setup_matplotlib()
    results, true_ll = run_experiment()

    ibs_samp = [r[0] for r in results["ibs"]]
    ibs_bias = [r[1] - true_ll for r in results["ibs"]]
    ibs_std = [r[2] for r in results["ibs"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for method in FIXED_SAMPLING_METHODS:
        samp = [r[0] for r in results[method]]
        bias = [r[1] - true_ll for r in results[method]]
        std = [r[2] for r in results[method]]
        label = method.capitalize() if method != "fixed" else "Fixed +1"
        color = get_color(method)
        axes[0].plot(samp, bias, "o-", label=label, color=color)
        axes[1].plot(samp, std, "o-", label=label, color=color)

    axes[0].plot(ibs_samp, ibs_bias, "o-", label="Inverse sampling",
                 color=get_color("ibs"))
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_xlabel("Average model samples")
    axes[0].set_ylabel("Bias")
    axes[0].legend(frameon=False)

    axes[1].plot(ibs_samp, ibs_std, "o-", label="Inverse sampling",
                 color=get_color("ibs"))
    axes[1].set_xlabel("Average model samples")
    axes[1].set_ylabel("Standard deviation")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
