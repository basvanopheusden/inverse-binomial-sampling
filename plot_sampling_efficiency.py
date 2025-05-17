# Script to compare inverse binomial sampling with a fixed-sample approach
# for estimating log-likelihood in a Bernoulli model.

import math
import random
import statistics

from ibs import ibs_loglikelihood
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


def fixed_loglikelihood(stimuli, responses, model, theta, M):
    """Estimate log-likelihood with a fixed number of samples per trial."""
    log_lik = 0.0
    for s, r_obs in zip(stimuli, responses):
        matches = 0
        for _ in range(M):
            if model(theta, s) == r_obs:
                matches += 1
        prob = matches / M if matches else 1e-12
        log_lik += math.log(prob)
    total_samples = M * len(responses)
    return log_lik, total_samples


def run_experiment(p=0.3, n_trials=20, repetitions=100, seed=123):
    rng = random.Random(seed)
    stimuli = [None] * n_trials
    responses = [rng.random() < p for _ in range(n_trials)]

    # Use a smaller range for the IBS repeat counts so that the
    # resulting average sample counts for the two methods are
    # roughly comparable.  The fixed sampler draws exactly
    # ``M`` samples per trial while a single IBS repeat requires
    # about twice as many samples on average.  Limiting the number
    # of repeats keeps both curves on a similar scale.
    fixed_samples = [1, 2, 3, 5, 8, 10]
    ibs_repeats = [1, 2, 3, 4, 5]

    results = {"fixed": [], "ibs": []}

    for M in fixed_samples:
        estimates = []
        samples = []
        for rep in range(repetitions):
            rng_run = random.Random(seed + rep)
            ll, n = fixed_loglikelihood(stimuli, responses,
                                        bernoulli_model(rng_run), p, M)
            estimates.append(ll)
            samples.append(n)
        mean_ll = statistics.mean(estimates)
        std_ll = statistics.stdev(estimates) if len(estimates) > 1 else 0.0
        avg_samples = statistics.mean(samples)
        results["fixed"].append((avg_samples, mean_ll, std_ll))

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
        mean_ll = statistics.mean(estimates)
        std_ll = statistics.stdev(estimates) if len(estimates) > 1 else 0.0
        avg_samples = statistics.mean(samples)
        results["ibs"].append((avg_samples, mean_ll, std_ll))

    return results


def main():
    setup_matplotlib()
    results = run_experiment()

    fixed_samples = [1, 2, 3, 5, 8, 10]

    ibs_samp = [r[0] for r in results["ibs"]]
    ibs_mean = [r[1] for r in results["ibs"]]
    ibs_std = [r[2] for r in results["ibs"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for M, (samp, mean, std) in zip(fixed_samples, results["fixed"]):
        label = f"Fixed {M}"
        color = get_color(f"fixed_{M}")
        axes[0].plot([samp], [mean], "o-", label=label, color=color)
        axes[1].plot([samp], [std], "o-", label=label, color=color)

    axes[0].plot(ibs_samp, ibs_mean, "o-", label="Inverse sampling",
                 color=get_color("ibs"))
    axes[0].set_xlabel("Average model samples")
    axes[0].set_ylabel("Mean log-likelihood")
    axes[0].legend()

    axes[1].plot(ibs_samp, ibs_std, "o-", label="Inverse sampling",
                 color=get_color("ibs"))
    axes[1].set_xlabel("Average model samples")
    axes[1].set_ylabel("Standard deviation")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
