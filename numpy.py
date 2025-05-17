"""Small stub of minimal ``numpy`` functionality.

This module defines only the parts of NumPy that are required by the tests in
this repository.  It should only be imported if the real ``numpy`` package is
not available.  The implementations here are very limited and exist solely so
the repository can run in restricted environments.
"""

import builtins

def mean(seq):
    seq = list(seq)
    return builtins.sum(seq) / len(seq) if seq else 0.0

def sum(seq):
    return builtins.sum(seq)
