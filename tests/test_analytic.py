import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
from ibs import ibs_analytical_mean, ibs_analytical_variance


def test_analytic_mean_variance_single():
    p = 0.3
    mean = ibs_analytical_mean([p])
    var = ibs_analytical_variance([p])
    # Expected mean is log(p)
    assert abs(mean - math.log(p)) < 1e-12
    # variance should be less than 1 and positive
    assert 0.0 <= var < 1.0
