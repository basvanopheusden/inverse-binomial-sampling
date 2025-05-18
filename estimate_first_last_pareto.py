from __future__ import annotations

"""Compare IBS, Jeffreys and Laplace for estimating P(first|last)."""

import math
import random
import statistics
from typing import Dict, List, Tuple

from bigram_model import BigramLanguageModel
from ibs import ibs_loglikelihood
from setup_matplotlib import setup_matplotlib
from sampling_colors import get_color
from fixed_sampling import FIXED_SAMPLING_METHODS

try:
    from matplotlib import pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit("matplotlib is required to run this script") from e


# Helper to sample a pair (first, last) from the bigram model

def sample_pair(model: BigramLanguageModel, rng: random.Random) -> Tuple[str, str]:
    first = rng.choices(
        model.vocabulary,
        weights=[math.exp(model.prior[w]) for w in model.vocabulary],
    )[0]
    second = rng.choices(
        model.vocabulary,
        weights=[math.exp(model.conditional[first][w]) for w in model.vocabulary],
    )[0]
    return first, second


def true_conditional(model: BigramLanguageModel) -> Dict[Tuple[str, str], float]:
    """Return the true P(first|last) for sequences of length 2."""
    probs: Dict[Tuple[str, str], float] = {}
    for last in model.vocabulary:
        total = 0.0
        for first in model.vocabulary:
            p = math.exp(model.prior[first]) * math.exp(model.conditional[first][last])
            probs[(first, last)] = p
            total += p
        for first in model.vocabulary:
            probs[(first, last)] /= total
    return probs


# Jeffreys and Laplace estimates from sampled pairs

def _smoothing_counts(pairs: List[Tuple[str, str]], vocab: List[str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    counts = {last: {first: 0 for first in vocab} for last in vocab}
    totals = {last: 0 for last in vocab}
    for first, last in pairs:
        counts[last][first] += 1
        totals[last] += 1
    return counts, totals


def estimate_smoothing(pairs: List[Tuple[str, str]], vocab: List[str], alpha: float) -> Dict[Tuple[str, str], float]:
    counts, totals = _smoothing_counts(pairs, vocab)
    V = len(vocab)
    result: Dict[Tuple[str, str], float] = {}
    for last in vocab:
        total = totals[last]
        for first in vocab:
            count = counts[last][first]
            result[(first, last)] = (count + alpha) / (total + alpha * V)
    return result


# IBS estimation of joint probabilities followed by normalization

def _pair_model(theta: Tuple[BigramLanguageModel, random.Random], _stimulus: object) -> Tuple[str, str]:
    model, rng = theta
    return sample_pair(model, rng)


def estimate_ibs(model: BigramLanguageModel, repeats: int, rng: random.Random) -> Tuple[Dict[Tuple[str, str], float], int]:
    pairs_prob: Dict[Tuple[str, str], float] = {}
    total_samples = 0
    for last in model.vocabulary:
        for first in model.vocabulary:
            ll, _, n = ibs_loglikelihood(
                [None],
                [(first, last)],
                _pair_model,
                (model, rng),
                repeats=repeats,
            )
            pairs_prob[(first, last)] = math.exp(ll)
            total_samples += n
    # Normalize to conditional probabilities
    for last in model.vocabulary:
        denom = sum(pairs_prob[(f, last)] for f in model.vocabulary)
        for first in model.vocabulary:
            pairs_prob[(first, last)] /= denom
    return pairs_prob, total_samples


def _rmse(est: Dict[Tuple[str, str], float], truth: Dict[Tuple[str, str], float], vocab: List[str]) -> float:
    diffs = [
        (est[(f, l)] - truth[(f, l)]) ** 2
        for l in vocab
        for f in vocab
    ]
    return math.sqrt(statistics.mean(diffs))


def run_experiment(sample_sizes: List[int], ibs_repeats: List[int], repetitions: int = 20, seed: int = 123) -> Dict[str, List[Tuple[float, float]]]:
    rng = random.Random(seed)
    model = BigramLanguageModel()
    truth = true_conditional(model)

    results = {"jeffreys": [], "laplace": [], "ibs": []}

    for n in sample_sizes:
        rmse_j = []
        rmse_l = []
        for rep in range(repetitions):
            rng_run = random.Random(rng.random())
            pairs = [sample_pair(model, rng_run) for _ in range(n)]
            est_j = estimate_smoothing(pairs, model.vocabulary, 0.5)
            est_l = estimate_smoothing(pairs, model.vocabulary, 1.0)
            rmse_j.append(_rmse(est_j, truth, model.vocabulary))
            rmse_l.append(_rmse(est_l, truth, model.vocabulary))
        results["jeffreys"].append((float(n), statistics.mean(rmse_j)))
        results["laplace"].append((float(n), statistics.mean(rmse_l)))

    for R in ibs_repeats:
        rmse_i = []
        samp_i = []
        for rep in range(repetitions):
            rng_run = random.Random(rng.random())
            est_i, samples = estimate_ibs(model, R, rng_run)
            rmse_i.append(_rmse(est_i, truth, model.vocabulary))
            samp_i.append(samples)
        results["ibs"].append((statistics.mean(samp_i), statistics.mean(rmse_i)))

    return results


def main() -> None:
    setup_matplotlib()
    sample_sizes = [5, 10, 20, 50, 100]
    ibs_repeats = [1, 2, 3, 5, 8]
    results = run_experiment(sample_sizes, ibs_repeats)

    fig, ax = plt.subplots(figsize=(6, 4))

    for method in ("jeffreys", "laplace", "ibs"):
        x = [r[0] for r in results[method]]
        y = [r[1] for r in results[method]]
        label = method.capitalize()
        ax.plot(x, y, "o-", label=label, color=get_color(method))

    ax.set_xlabel("Model samples")
    ax.set_ylabel("RMSE of P(first|last)")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
