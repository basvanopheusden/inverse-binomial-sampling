"""Small stub of minimal ``numpy`` functionality.

This module defines only the parts of NumPy that are required by the tests in
this repository.  It should only be imported if the real ``numpy`` package is
not available.  The implementations here are very limited and exist solely so
the repository can run in restricted environments.

The real SciPy package expects ``numpy.show_config`` to exist when it is
imported.  A user hit an ``ImportError`` because this stub did not implement
that function.  To keep the fallback working in environments where SciPy might
be installed, ``show_config`` and a minimal ``numpy.version`` submodule are
provided.
"""

import builtins
import sys
import types

def mean(seq):
    seq = list(seq)
    return builtins.sum(seq) / len(seq) if seq else 0.0

def sum(seq):
    return builtins.sum(seq)


def show_config():
    """Minimal stub for :func:`numpy.show_config`.

    The real function prints NumPy's build configuration.  Here we simply
    output a short message so that callers expecting the function do not fail.
    """
    print("NumPy stub: no configuration available")


# Provide a bare-bones ``numpy.version`` submodule so that packages importing
# ``from numpy.version import version`` succeed.
version = types.ModuleType("numpy.version")
version.version = "0.0.0"
sys.modules[__name__ + ".version"] = version

# Also expose ``__version__`` at the top level for compatibility.
__version__ = version.version
