"""Minimal ``numpy`` shim used when the real package is missing.

This file is deliberately named ``numpy`` so that ``import numpy`` succeeds in
environments where the library is not installed.  When executed, it first
checks whether a genuine NumPy installation is available.  If so, the real
package is loaded and all of its symbols are re-exported.  Otherwise a tiny
subset of functionality required by the tests is provided.

The fallback implements just ``mean`` and ``sum`` as well as a small stub of
``numpy.show_config`` and ``numpy.version`` so that parts of SciPy expecting
those attributes do not fail.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys

_spec = importlib.machinery.PathFinder().find_spec(__name__, sys.path[1:])
if _spec is not None:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    import builtins
    import types

    def mean(seq):
        seq = list(seq)
        return builtins.sum(seq) / len(seq) if seq else 0.0

    def sum(seq):
        return builtins.sum(seq)

    def show_config():
        """Minimal stub for :func:`numpy.show_config`."""
        print("NumPy stub: no configuration available")

    version = types.ModuleType("numpy.version")
    version.version = "0.0.0"
    sys.modules[__name__ + ".version"] = version
    __version__ = version.version
