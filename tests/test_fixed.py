import math
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fixed_sampling import FIXED_METHODS


def test_fixed_methods_basic():
    m = 5
    M = 10
    expected = {
        "naive": math.log(0.5),
        "laplace": math.log((m + 1) / (M + 1)),
        "jeffreys": math.log((m + 0.5) / (M + 1)),
        "clipped": math.log(0.5),
    }
    for name, func in FIXED_METHODS.items():
        val = func(m, M)
        assert abs(val - expected[name]) < 1e-12


def test_fixed_clipped_zero():
    val = FIXED_METHODS["clipped"](0, 10)
    assert math.isfinite(val)
