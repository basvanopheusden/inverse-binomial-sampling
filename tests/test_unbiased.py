import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import random

from ibs import ibs_loglikelihood


def bernoulli_model(rng):
    def model(p, _):
        return rng.random() < p
    return model


# Seed for determinism
SEED = 12345

# Probabilities to test
P_VALUES = [0.1, 0.2, 0.5, 0.8, 0.9]


def test_ibs_unbiased_logp():
    for idx, p in enumerate(P_VALUES):
        rng = random.Random(SEED + idx)
        repeats = int(1000 / p)
        mean, variance, _ = ibs_loglikelihood([
            None
        ], [
            1
        ], bernoulli_model(rng), p, repeats=repeats)
        # Check that log(p) lies within 4 standard deviations of the estimate
        std = math.sqrt(variance)
        assert abs(mean - math.log(p)) < 4 * std



if __name__ == "__main__":
    test_ibs_unbiased_logp()
