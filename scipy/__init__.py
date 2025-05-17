"""Provide :mod:`scipy` if the real library is missing.

When imported, this module first checks whether a genuine SciPy installation is
available.  If so, it loads that package and re-exports all of its symbols.
Otherwise a very small subset of functionality implemented in ``scipy.special``
is provided so that the rest of the codebase can run in restricted
environments.
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
    from . import special

    __all__ = ["special"]

