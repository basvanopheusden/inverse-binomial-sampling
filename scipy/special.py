"""Tiny stub of ``scipy.special`` for environments without SciPy.

Only the specific ``digamma`` and ``polygamma`` functions used in this
repository are implemented, and only for the integer arguments required by the
tests.  If SciPy is available, prefer importing it instead of this fallback.
"""

import math

_EULER_MASCHERONI = 0.5772156649015328606

def digamma(n):
    if n <= 0 or int(n) != n:
        raise NotImplementedError("digamma only implemented for positive integers")
    n = int(n)
    result = -_EULER_MASCHERONI
    for k in range(1, n):
        result += 1.0 / k
    return result


def polygamma(m, n):
    if m != 1:
        raise NotImplementedError("polygamma only implemented for m=1")
    if n <= 0 or int(n) != n:
        raise NotImplementedError("polygamma only implemented for positive integers")
    n = int(n)
    result = math.pi ** 2 / 6.0
    for k in range(1, n):
        result -= 1.0 / (k ** 2)
    return result
