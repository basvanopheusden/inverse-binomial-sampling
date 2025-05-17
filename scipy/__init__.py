"""Minimal stub of the :mod:`scipy` package.

This lightweight implementation only exposes the parts of SciPy required by
this repository.  Currently it provides the :mod:`scipy.special` submodule with
basic implementations of :func:`digamma` and :func:`polygamma`.  It exists so
that code written for SciPy can run in environments where the real library is
unavailable.
"""

from . import special

__all__ = ["special"]

