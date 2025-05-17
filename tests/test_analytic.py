import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
from ibs import ibs_analytical_mean, ibs_analytical_variance
from fixed_sampling import (
    fixed_analytical_mean,
    fixed_analytical_variance,
    FIXED_SAMPLING_PARAMS,
)


def test_analytic_mean_variance_single():
    p = 0.3
    mean = ibs_analytical_mean([p])
    var = ibs_analytical_variance([p])
    # Expected mean is log(p)
    assert abs(mean - math.log(p)) < 1e-12
    # variance should be less than 1 and positive
    assert 0.0 <= var < 1.0


def _expected_fixed_moments(p, M, alpha, beta):
    mean = 0.0
    for k in range(M + 1):
        prob = math.comb(M, k) * (p ** k) * ((1 - p) ** (M - k))
        mean += prob * math.log((k + alpha) / (M + beta))
    var = 0.0
    for k in range(M + 1):
        prob = math.comb(M, k) * (p ** k) * ((1 - p) ** (M - k))
        val = math.log((k + alpha) / (M + beta))
        var += prob * (val - mean) ** 2
    return mean, var


def test_fixed_analytic_methods():
    p = 0.25
    M = 4
    for method, (alpha, beta) in FIXED_SAMPLING_PARAMS.items():
        mean = fixed_analytical_mean([p], M, method)
        var = fixed_analytical_variance([p], M, method)
        exp_mean, exp_var = _expected_fixed_moments(p, M, alpha, beta)
        assert abs(mean - exp_mean) < 1e-12
        assert abs(var - exp_var) < 1e-12
